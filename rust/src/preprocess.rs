//! Image preprocessing for the NanoSAM encoder.
//!
//! Ported from `nanosam/utils/preprocess.py`. Handles:
//! - Aspect-ratio-preserving resize to 1024×1024
//! - ImageNet normalization
//! - Zero-padding
//! - Point coordinate scaling

use image::DynamicImage;

/// Encoder input size (square).
const ENCODER_SIZE: u32 = 1024;

/// ImageNet mean (RGB, 0-255 scale).
const IMAGE_MEAN: [f32; 3] = [123.675, 116.28, 103.53];
/// ImageNet std (RGB, 0-255 scale).
const IMAGE_STD: [f32; 3] = [58.395, 57.12, 57.375];

/// Preprocessed image ready for the encoder.
pub struct PreprocessedImage {
    /// NHWC float32 tensor: (1, 1024, 1024, 3) — ready for TFLite INT8 encoder.
    pub nhwc: Vec<f32>,
    /// Original image dimensions (height, width).
    pub original_hw: (u32, u32),
}

/// Preprocess an image for the NanoSAM encoder.
///
/// Performs aspect-ratio-preserving resize, ImageNet normalization, and
/// zero-padding to 1024×1024. Returns NHWC layout for TFLite.
pub fn preprocess_image(img: &DynamicImage) -> PreprocessedImage {
    let (orig_w, orig_h) = (img.width(), img.height());
    let original_hw = (orig_h, orig_w);

    // Compute resize dimensions preserving aspect ratio
    let (resize_w, resize_h) = if orig_w >= orig_h {
        (ENCODER_SIZE, (ENCODER_SIZE as f32 * orig_h as f32 / orig_w as f32) as u32)
    } else {
        ((ENCODER_SIZE as f32 * orig_w as f32 / orig_h as f32) as u32, ENCODER_SIZE)
    };

    // Resize using Lanczos3 (high quality, matches PIL's default LANCZOS)
    let resized = img.resize_exact(resize_w, resize_h, image::imageops::FilterType::Lanczos3);
    let rgb = resized.to_rgb8();

    // Build NHWC tensor with zero-padding: (1, 1024, 1024, 3)
    let total = (ENCODER_SIZE * ENCODER_SIZE * 3) as usize;
    let mut nhwc = vec![0.0f32; total];

    for y in 0..resize_h {
        for x in 0..resize_w {
            let pixel = rgb.get_pixel(x, y);
            let idx = ((y * ENCODER_SIZE + x) * 3) as usize;
            nhwc[idx] = (pixel[0] as f32 - IMAGE_MEAN[0]) / IMAGE_STD[0];
            nhwc[idx + 1] = (pixel[1] as f32 - IMAGE_MEAN[1]) / IMAGE_STD[1];
            nhwc[idx + 2] = (pixel[2] as f32 - IMAGE_MEAN[2]) / IMAGE_STD[2];
        }
    }

    PreprocessedImage { nhwc, original_hw }
}

/// Scale point coordinates from original image space to encoder input space.
///
/// `coords` is a flat slice of (x, y) pairs: `[x0, y0, x1, y1, ...]`.
/// Returns scaled coordinates in the same flat format.
pub fn scale_points(coords: &[f32], image_hw: (u32, u32)) -> Vec<f32> {
    let scale = ENCODER_SIZE as f32 / image_hw.0.max(image_hw.1) as f32;
    coords.iter().map(|&v| v * scale).collect()
}

/// Low-resolution mask size from the decoder.
const MASK_SIZE: u32 = 256;

/// Upscale a low-res 256×256 decoder mask to original image dimensions.
///
/// Crops the aspect-ratio-adjusted region, then bilinear-upsamples.
pub fn upscale_mask(mask_256: &[f32], original_hw: (u32, u32)) -> Vec<f32> {
    let (orig_h, orig_w) = original_hw;

    // Compute crop limits (undo the padding from preprocessing)
    let (lim_x, lim_y) = if orig_w > orig_h {
        (MASK_SIZE, (MASK_SIZE as f32 * orig_h as f32 / orig_w as f32) as u32)
    } else {
        ((MASK_SIZE as f32 * orig_w as f32 / orig_h as f32) as u32, MASK_SIZE)
    };

    // Bilinear upsample from (lim_y, lim_x) to (orig_h, orig_w)
    let mut out = vec![0.0f32; (orig_h * orig_w) as usize];

    for dy in 0..orig_h {
        for dx in 0..orig_w {
            // Map output pixel to source position in cropped mask
            let sx = dx as f32 * (lim_x as f32 - 1.0) / (orig_w as f32 - 1.0).max(1.0);
            let sy = dy as f32 * (lim_y as f32 - 1.0) / (orig_h as f32 - 1.0).max(1.0);

            // Bilinear interpolation
            let x0 = sx.floor() as u32;
            let y0 = sy.floor() as u32;
            let x1 = (x0 + 1).min(lim_x - 1);
            let y1 = (y0 + 1).min(lim_y - 1);
            let fx = sx - x0 as f32;
            let fy = sy - y0 as f32;

            let v00 = mask_256[(y0 * MASK_SIZE + x0) as usize];
            let v10 = mask_256[(y0 * MASK_SIZE + x1) as usize];
            let v01 = mask_256[(y1 * MASK_SIZE + x0) as usize];
            let v11 = mask_256[(y1 * MASK_SIZE + x1) as usize];

            let val = v00 * (1.0 - fx) * (1.0 - fy)
                + v10 * fx * (1.0 - fy)
                + v01 * (1.0 - fx) * fy
                + v11 * fx * fy;

            out[(dy * orig_w + dx) as usize] = val;
        }
    }

    out
}

/// Transpose NCHW flat tensor to NHWC.
pub fn nchw_to_nhwc(data: &[f32], shape: [usize; 4]) -> Vec<f32> {
    let (n, c, h, w) = (shape[0], shape[1], shape[2], shape[3]);
    let mut out = vec![0.0f32; data.len()];
    for ni in 0..n {
        for ci in 0..c {
            for hi in 0..h {
                for wi in 0..w {
                    let src = ni * (c * h * w) + ci * (h * w) + hi * w + wi;
                    let dst = ni * (h * w * c) + hi * (w * c) + wi * c + ci;
                    out[dst] = data[src];
                }
            }
        }
    }
    out
}

/// Transpose NHWC flat tensor to NCHW.
pub fn nhwc_to_nchw(data: &[f32], shape: [usize; 4]) -> Vec<f32> {
    let (n, h, w, c) = (shape[0], shape[1], shape[2], shape[3]);
    let mut out = vec![0.0f32; data.len()];
    for ni in 0..n {
        for hi in 0..h {
            for wi in 0..w {
                for ci in 0..c {
                    let src = ni * (h * w * c) + hi * (w * c) + wi * c + ci;
                    let dst = ni * (c * h * w) + ci * (h * w) + hi * w + wi;
                    out[dst] = data[src];
                }
            }
        }
    }
    out
}

/// Transpose a 3D tensor with given shape and permutation.
pub fn transpose_3d(data: &[f32], shape: [usize; 3], perm: [usize; 3]) -> Vec<f32> {
    let out_shape = [shape[perm[0]], shape[perm[1]], shape[perm[2]]];
    let mut out = vec![0.0f32; data.len()];

    let strides_in = [shape[1] * shape[2], shape[2], 1];
    let strides_out = [out_shape[1] * out_shape[2], out_shape[2], 1];

    for i0 in 0..shape[0] {
        for i1 in 0..shape[1] {
            for i2 in 0..shape[2] {
                let src = i0 * strides_in[0] + i1 * strides_in[1] + i2 * strides_in[2];
                let coords = [i0, i1, i2];
                let dst = coords[perm[0]] * strides_out[0]
                    + coords[perm[1]] * strides_out[1]
                    + coords[perm[2]] * strides_out[2];
                out[dst] = data[src];
            }
        }
    }
    out
}
