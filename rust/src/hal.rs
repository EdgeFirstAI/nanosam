//! EdgeFirst HAL integration for optimized image preprocessing and mask rendering.
//!
//! Uses `edgefirst-image` for hardware-accelerated letterbox resize (OpenGL/G2D/CPU)
//! and mask overlay rendering via the semantic segmentation path.

use edgefirst_decoder::{BoundingBox, DetectBox, Segmentation};
use edgefirst_image::{
    ColorMode, Crop, ImageProcessor, ImageProcessorTrait, MaskOverlay, Rect,
};
use edgefirst_tensor::{DType, PixelFormat, TensorDyn, TensorMapTrait, TensorTrait};

use crate::types::Mask;

/// ImageNet mean (RGB, 0-255 scale).
const MEAN: [f32; 3] = [123.675, 116.28, 103.53];
/// ImageNet std (RGB, 0-255 scale).
const STD: [f32; 3] = [58.395, 57.12, 57.375];

/// Encoder model input size.
const MODEL_SIZE: usize = 1024;

/// Calculate letterbox dimensions preserving aspect ratio.
///
/// Returns (left_offset, top_offset, scaled_width, scaled_height).
pub fn calculate_letterbox(
    src_w: usize,
    src_h: usize,
    dst_w: usize,
    dst_h: usize,
) -> (usize, usize, usize, usize) {
    let src_aspect = src_w as f64 / src_h as f64;
    let dst_aspect = dst_w as f64 / dst_h as f64;

    let (new_w, new_h) = if src_aspect > dst_aspect {
        let new_h = (dst_w as f64 / src_aspect).round() as usize;
        (dst_w, new_h)
    } else {
        let new_w = (dst_h as f64 * src_aspect).round() as usize;
        (new_w, dst_h)
    };

    // NanoSAM places the image at top-left origin (NOT centered)
    // to match the Python preprocess_image() convention.
    let left = 0;
    let top = 0;

    (left, top, new_w, new_h)
}

/// Per-channel linear coefficients for fused ImageNet-norm + INT8 quantize.
///
/// Maps U8 pixel directly to INT8: `i8 = clamp(round(pixel * a + b), -128, 127)`
/// Derived from: `i8 = round(((pixel - mean) / std) / scale + zp)`
///   → `a = 1 / (std * scale)`, `b = -mean / (std * scale) + zp`
#[derive(Clone)]
pub struct QuantCoeffs {
    pub a: [f32; 3],
    pub b: [f32; 3],
}

impl QuantCoeffs {
    /// Compute coefficients from ImageNet normalization constants and
    /// the TFLite model's input quantization parameters.
    pub fn new(scale: f32, zero_point: f32) -> Self {
        let mut a = [0.0f32; 3];
        let mut b = [0.0f32; 3];
        for c in 0..3 {
            a[c] = 1.0 / (STD[c] * scale);
            b[c] = -MEAN[c] / (STD[c] * scale) + zero_point;
        }
        Self { a, b }
    }

    /// Fused U8→I8 quantization for one pixel (3 channels).
    #[inline]
    pub fn quantize_pixel(&self, r: u8, g: u8, b_val: u8) -> (i8, i8, i8) {
        let qr = (r as f32 * self.a[0] + self.b[0]).round().clamp(-128.0, 127.0) as i8;
        let qg = (g as f32 * self.a[1] + self.b[1]).round().clamp(-128.0, 127.0) as i8;
        let qb = (b_val as f32 * self.a[2] + self.b[2]).round().clamp(-128.0, 127.0) as i8;
        (qr, qg, qb)
    }
}

/// Preprocess an image for the NanoSAM encoder using the HAL.
///
/// 1. HAL GPU: letterbox resize to 1024×1024 RGB U8 (GL-accelerated)
/// 2. Fused ImageNet-norm + INT8 quantize: U8 → I8 directly (no float alloc)
///
/// Returns NHWC INT8 tensor (1024*1024*3 elements) and original (h, w).
pub fn preprocess_image_hal(
    processor: &mut ImageProcessor,
    image_path: &std::path::Path,
    quant: &QuantCoeffs,
) -> Result<(Vec<i8>, (u32, u32)), Box<dyn std::error::Error>> {
    // Load image via HAL's load_image (supports JPEG/PNG from bytes)
    let image_bytes = std::fs::read(image_path)?;
    let src_tensor = edgefirst_image::load_image(&image_bytes, Some(PixelFormat::Rgb), None)?;
    let src_w = src_tensor.width().ok_or("image has no width")?;
    let src_h = src_tensor.height().ok_or("image has no height")?;
    let original_hw = (src_h as u32, src_w as u32);

    // GPU-accelerated letterbox resize to 1024×1024 RGB U8
    let mut dst_tensor =
        processor.create_image(MODEL_SIZE, MODEL_SIZE, PixelFormat::Rgb, DType::U8, None)?;

    // NanoSAM: top-left placement (NOT centered) to match training preprocessing
    let scale = (MODEL_SIZE as f32 / src_w as f32).min(MODEL_SIZE as f32 / src_h as f32);
    let new_w = (src_w as f32 * scale) as usize;
    let new_h = (src_h as f32 * scale) as usize;

    let crop = Crop::new()
        .with_dst_rect(Some(Rect::new(0, 0, new_w, new_h)))
        .with_dst_color(Some([0, 0, 0, 255])); // zero padding

    processor.convert(
        &src_tensor,
        &mut dst_tensor,
        edgefirst_image::Rotation::None,
        edgefirst_image::Flip::None,
        crop,
    )?;

    // Fused U8→I8: ImageNet normalize + quantize in one pass, no float32 alloc
    let t = dst_tensor.as_u8().ok_or("expected U8 tensor")?;
    let map = t.map()?;
    let rgb_bytes = map.as_slice();

    let total = MODEL_SIZE * MODEL_SIZE * 3;
    let mut i8_buf = vec![0i8; total];
    for (chunk, out) in rgb_bytes.chunks_exact(3).zip(i8_buf.chunks_exact_mut(3)) {
        let (qr, qg, qb) = quant.quantize_pixel(chunk[0], chunk[1], chunk[2]);
        out[0] = qr;
        out[1] = qg;
        out[2] = qb;
    }

    Ok((i8_buf, original_hw))
}

/// Convert a NanoSAM mask to the HAL's `Segmentation` format.
///
/// NanoSAM produces a full-image mask (like semantic segmentation) —
/// the HAL's `draw_decoded_masks` handles this via its semantic seg path
/// when `segmentation.shape[2] > 1` or single-channel full-image masks.
pub fn mask_to_segmentation(mask: &Mask) -> (DetectBox, Segmentation) {
    let w = mask.width as usize;
    let h = mask.height as usize;

    // Convert float mask → u8 (0 = background, 255 = foreground)
    let seg_data: Vec<u8> = mask
        .data
        .iter()
        .map(|&v| if v > 0.0 { 255u8 } else { 0u8 })
        .collect();

    let segmentation = Segmentation {
        xmin: 0.0,
        ymin: 0.0,
        xmax: 1.0,
        ymax: 1.0,
        segmentation: ndarray::Array3::from_shape_vec((h, w, 1), seg_data)
            .expect("mask shape mismatch"),
    };

    // Full-image bounding box
    let detect = DetectBox {
        bbox: BoundingBox::new(0.0, 0.0, 1.0, 1.0),
        score: mask.iou_score,
        label: 0,
    };

    (detect, segmentation)
}

/// Low-res mask from the decoder (256×256, not yet upscaled).
pub struct LowResMask {
    /// Raw 256×256 float mask (positive = foreground), cropped to content region.
    pub data_u8: Vec<u8>,
    /// Cropped mask dimensions (height, width) — content only, no padding.
    pub crop_h: usize,
    pub crop_w: usize,
    /// IoU score from the model.
    pub iou_score: f32,
}

/// Extract low-res masks from decoder output, cropping padding.
///
/// Returns 4 masks at decoder resolution (cropped to content region,
/// typically ~165×256 for landscape images). These are ready for
/// HAL GPU rendering without CPU upscaling.
pub fn extract_low_res_masks(
    iou: &[f32],
    masks_flat: &[f32],
    original_hw: (u32, u32),
) -> Vec<LowResMask> {
    extract_low_res_masks_threshold(iou, masks_flat, original_hw, 0.0)
}

/// Extract low-res masks with a configurable logit threshold.
///
/// `threshold` shifts the u8 midpoint: 0.0 = standard (positive logits → opaque),
/// negative values (e.g. -2.0) recover borderline pixels lost to INT8 quantization.
pub fn extract_low_res_masks_threshold(
    iou: &[f32],
    masks_flat: &[f32],
    original_hw: (u32, u32),
    threshold: f32,
) -> Vec<LowResMask> {
    let (orig_h, orig_w) = original_hw;

    // Compute content crop (undo letterbox padding)
    let (lim_x, lim_y) = if orig_w > orig_h {
        (256usize, (256.0 * orig_h as f32 / orig_w as f32) as usize)
    } else {
        ((256.0 * orig_w as f32 / orig_h as f32) as usize, 256usize)
    };

    let mut masks = Vec::with_capacity(4);
    for i in 0..4 {
        // Extract mask i from (65536, 4) interleaved and crop to content region
        let mut data_u8 = vec![0u8; lim_y * lim_x];
        for y in 0..lim_y {
            for x in 0..lim_x {
                let val = masks_flat[(y * 256 + x) * 4 + i] - threshold;
                // Sigmoid-like mapping to u8: threshold → 128
                data_u8[y * lim_x + x] = if val > 0.0 {
                    (128.0 + (val.min(5.0) / 5.0 * 127.0)) as u8
                } else {
                    (128.0 + (val.max(-5.0) / 5.0 * 128.0)) as u8
                };
            }
        }
        masks.push(LowResMask {
            data_u8,
            crop_h: lim_y,
            crop_w: lim_x,
            iou_score: iou[i],
        });
    }
    masks
}

/// Render a low-res mask overlay onto an image using the HAL.
///
/// The HAL handles bilinear upscaling from the decoder's ~165×256 mask
/// to the full output resolution on the GPU, avoiding the 131ms CPU
/// bilinear upscale.
pub fn render_mask_hal(
    processor: &mut ImageProcessor,
    dst: &mut TensorDyn,
    mask: &LowResMask,
    opacity: f32,
) -> Result<(), Box<dyn std::error::Error>> {
    let seg = Segmentation {
        xmin: 0.0,
        ymin: 0.0,
        xmax: 1.0,
        ymax: 1.0,
        segmentation: ndarray::Array3::from_shape_vec(
            (mask.crop_h, mask.crop_w, 1),
            mask.data_u8.clone(),
        )?,
    };

    let detect = DetectBox {
        bbox: BoundingBox::new(0.0, 0.0, 1.0, 1.0),
        score: mask.iou_score,
        label: 0,
    };

    let overlay = MaskOverlay {
        background: None,
        opacity,
        letterbox: None,
        color_mode: ColorMode::Class,
    };

    processor.draw_decoded_masks(dst, &[detect], &[seg], overlay)?;
    Ok(())
}

/// Load an image into a GPU-backed HAL tensor for rendering.
///
/// Uses `processor.create_image()` to allocate a DMA-BUF backed tensor,
/// then `convert()` to decode + copy the image into it. This ensures the
/// tensor is GPU-resident for fast mask overlay rendering.
pub fn load_image_tensor(
    processor: &mut ImageProcessor,
    image_path: &std::path::Path,
) -> Result<(TensorDyn, u32, u32), Box<dyn std::error::Error>> {
    let image_bytes = std::fs::read(image_path)?;
    let src = edgefirst_image::load_image(&image_bytes, Some(PixelFormat::Rgb), None)?;
    let w = src.width().ok_or("no width")?;
    let h = src.height().ok_or("no height")?;

    // Create GPU-backed destination (RGBA for GL mask rendering compatibility)
    let mut dst = processor.create_image(w, h, PixelFormat::Rgba, DType::U8, None)?;
    processor.convert(
        &src, &mut dst,
        edgefirst_image::Rotation::None,
        edgefirst_image::Flip::None,
        Crop::new(),
    )?;

    Ok((dst, w as u32, h as u32))
}

/// Save a HAL tensor as JPEG.
pub fn save_tensor_jpeg(
    tensor: &TensorDyn,
    path: &std::path::Path,
) -> Result<(), Box<dyn std::error::Error>> {
    edgefirst_image::save_jpeg(tensor, path.to_str().ok_or("invalid path")?, 95)?;
    Ok(())
}
