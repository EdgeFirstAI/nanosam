//! NanoSAM: Real-time Segment Anything on edge devices.
//!
//! This library provides a high-level API for running the NanoSAM segmentation
//! pipeline (ResNet18 encoder + split MobileSAM decoder) using TFLite with
//! optional NPU acceleration.
//!
//! # Usage
//!
//! ```no_run
//! use nanosam::{Session, SessionConfig, Prompt, BBox};
//!
//! let config = SessionConfig {
//!     model_archive: "nanosam_imx95.zip".into(),
//!     delegates: vec!["/usr/lib/libneutron_delegate.so".into()],
//!     xnnpack_library: Some("/usr/lib/libtensorflow-lite.so".into()),
//!     threads: 6,
//! };
//!
//! let mut session = Session::new(config)?;
//! let image = image::open("photo.jpg")?;
//! let prompt = Prompt::Box(BBox { x1: 100.0, y1: 100.0, x2: 850.0, y2: 759.0 });
//! let result = session.segment(&image, &prompt)?;
//! println!("Best mask IoU: {:.3}", result.masks[result.best_mask_index].iou_score);
//! ```

pub mod archive;
pub mod decoder;
pub mod encoder;
pub mod hal;
pub mod preprocess;
pub mod prompt_encoder;
pub mod tflite_runner;
pub mod types;

pub use types::*;

use std::time::Instant;

use edgefirst_tflite::{Library, Model};

/// A loaded NanoSAM session ready for inference.
///
/// Holds all TFLite models and interpreters. Create once at startup,
/// then call `encode()` + `decode()` or `segment()` per frame.
///
/// The session takes `&mut self` because TFLite interpreters are not
/// thread-safe.
pub struct Session {
    // We store the raw components and use unsafe lifetime extension
    // because the Library → Model → Interpreter lifetime chain requires
    // them all to live in the same scope. Using Box<> + Pin<> here.
    _lib: Box<Library>,
    _models: Box<SessionModels>,
    pub encoder: encoder::Encoder<'static>,
    decoder: decoder::Decoder<'static>,
    /// HAL image processor for optimized preprocessing and mask rendering.
    pub hal_processor: Option<edgefirst_image::ImageProcessor>,
    /// Fused U8→I8 quantization coefficients (ImageNet norm + TFLite quant).
    pub quant_coeffs: hal::QuantCoeffs,
}

/// Owned model buffers loaded from the archive.
struct SessionModels {
    encoder: Model<'static>,
    attention: Model<'static>,
    heads_a: Model<'static>,
    heads_b: Model<'static>,
    tokens: Model<'static>,
}

impl Session {
    /// Create a new session by loading models from a ZIP archive.
    pub fn new(config: SessionConfig) -> Result<Self, Box<dyn std::error::Error>> {
        // Load all model bytes from ZIP
        let mut archive = archive::ModelArchive::load(&config.model_archive)?;

        let prompt_bytes = archive.get("prompt_encoder.safetensors")?.to_vec();

        // Load TFLite library (runtime dlopen)
        let lib = Box::new(Library::new()?);
        let lib_ref: &'static Library = unsafe { &*(lib.as_ref() as *const Library) };

        // Load models from bytes
        let encoder_bytes = archive.take("encoder.tflite")?;
        let attention_bytes = archive.take("attention.tflite")?;
        let heads_a_bytes = archive.take("heads_a.tflite")?;
        let heads_b_bytes = archive.take("heads_b.tflite")?;
        let tokens_bytes = archive.take("tokens.tflite")?;

        let models = Box::new(SessionModels {
            encoder: Model::from_bytes(lib_ref, encoder_bytes)?,
            attention: Model::from_bytes(lib_ref, attention_bytes)?,
            heads_a: Model::from_bytes(lib_ref, heads_a_bytes)?,
            heads_b: Model::from_bytes(lib_ref, heads_b_bytes)?,
            tokens: Model::from_bytes(lib_ref, tokens_bytes)?,
        });
        let models_ref: &'static SessionModels =
            unsafe { &*(models.as_ref() as *const SessionModels) };

        let delegate_strs: Vec<String> = config
            .delegates
            .iter()
            .map(|p| p.to_string_lossy().to_string())
            .collect();
        let delegate_refs: Vec<&str> = delegate_strs.iter().map(|s| s.as_str()).collect();

        let enc = encoder::Encoder::new(
            lib_ref,
            &models_ref.encoder,
            config.threads,
            &delegate_refs,
        )?;

        let dec = decoder::Decoder::new(
            lib_ref,
            &prompt_bytes,
            &models_ref.attention,
            &models_ref.heads_a,
            &models_ref.heads_b,
            &models_ref.tokens,
            config.threads,
            &delegate_refs,
            config.use_xnnpack,
        )?;

        // Read encoder input quantization params for fused U8→I8 conversion
        // Default to ImageNet-typical values if we can't read them
        let quant_coeffs = hal::QuantCoeffs::new(0.018658448, -14.0);

        // Try to create HAL image processor for optimized preprocessing.
        let hal_processor = match edgefirst_image::ImageProcessor::new() {
            Ok(proc) => Some(proc),
            Err(e) => {
                eprintln!("HAL image processor not available ({}), using fallback", e);
                None
            }
        };

        Ok(Self {
            _lib: lib,
            _models: models,
            encoder: enc,
            decoder: dec,
            hal_processor,
            quant_coeffs,
        })
    }

    /// Encode an image. The embedding can be reused with multiple prompts.
    pub fn encode(
        &mut self,
        image: &image::DynamicImage,
    ) -> Result<ImageEmbedding, Box<dyn std::error::Error>> {
        let (embedding, _, _) = self.encoder.encode(image)?;
        Ok(embedding)
    }

    /// Decode a prompt against a previously computed image embedding.
    ///
    /// Returns full-resolution CPU-upscaled masks (fallback path).
    pub fn decode(
        &mut self,
        embedding: &ImageEmbedding,
        prompt: &Prompt,
    ) -> Result<SegmentationResult, Box<dyn std::error::Error>> {
        let (iou, masks_flat, mut timings) = self.decoder.decode(embedding, prompt)?;

        // Build upscaled masks (CPU bilinear — 131ms fallback)
        let t_post = Instant::now();
        let masks = decoder::build_masks(&iou, &masks_flat, embedding.original_hw);
        timings.postprocess_ms = t_post.elapsed().as_secs_f64() * 1000.0;

        let best_mask_index = iou
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);

        Ok(SegmentationResult {
            masks,
            best_mask_index,
            timings,
        })
    }

    /// Decode and return low-res masks for HAL GPU rendering.
    ///
    /// Skips CPU upscaling; returns cropped 256×256 masks ready for
    /// `hal::render_mask_hal` which does GPU bilinear upscale + overlay.
    pub fn decode_low_res(
        &mut self,
        embedding: &ImageEmbedding,
        prompt: &Prompt,
    ) -> Result<(Vec<hal::LowResMask>, usize, Timings), Box<dyn std::error::Error>> {
        let (iou, masks_flat, timings) = self.decoder.decode(embedding, prompt)?;

        let masks = hal::extract_low_res_masks(&iou, &masks_flat, embedding.original_hw);

        let best = iou
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);

        Ok((masks, best, timings))
    }

    /// Encode an image using the HAL for preprocessing when available.
    ///
    /// This is the preferred path — the HAL uses hardware-accelerated resize
    /// (OpenGL/G2D) which is significantly faster than the `image` crate.
    pub fn encode_path(
        &mut self,
        image_path: &std::path::Path,
    ) -> Result<(ImageEmbedding, f64), Box<dyn std::error::Error>> {
        if let Some(ref mut proc) = self.hal_processor {
            let t = Instant::now();
            let (i8_data, original_hw) =
                hal::preprocess_image_hal(proc, image_path, &self.quant_coeffs)?;
            let preprocess_ms = t.elapsed().as_secs_f64() * 1000.0;
            let (emb, encoder_ms) = self.encoder.encode_i8(&i8_data, original_hw)?;
            Ok((emb, preprocess_ms + encoder_ms))
        } else {
            let img = image::open(image_path)?;
            let (emb, pre_ms, enc_ms) = self.encoder.encode(&img)?;
            Ok((emb, pre_ms + enc_ms))
        }
    }

    /// Convenience: encode an image and decode a prompt in one call.
    pub fn segment(
        &mut self,
        image: &image::DynamicImage,
        prompt: &Prompt,
    ) -> Result<SegmentationResult, Box<dyn std::error::Error>> {
        let (embedding, preprocess_ms, encoder_ms) = self.encoder.encode(image)?;
        let mut result = self.decode(&embedding, prompt)?;
        result.timings.preprocess_ms = preprocess_ms;
        result.timings.encoder_ms = encoder_ms;
        Ok(result)
    }

    /// Convenience: encode from path (HAL-accelerated) + decode in one call.
    ///
    /// Uses the HAL for GPU-accelerated letterbox resize, then fused
    /// U8→I8 quantization (ImageNet norm + quant in one pass, no float alloc).
    pub fn segment_path(
        &mut self,
        image_path: &std::path::Path,
        prompt: &Prompt,
    ) -> Result<SegmentationResult, Box<dyn std::error::Error>> {
        if let Some(ref mut proc) = self.hal_processor {
            let t_pre = Instant::now();
            let (i8_data, original_hw) =
                hal::preprocess_image_hal(proc, image_path, &self.quant_coeffs)?;
            let preprocess_ms = t_pre.elapsed().as_secs_f64() * 1000.0;

            let (embedding, encoder_ms) = self.encoder.encode_i8(&i8_data, original_hw)?;
            let mut result = self.decode(&embedding, prompt)?;
            result.timings.preprocess_ms = preprocess_ms;
            result.timings.encoder_ms = encoder_ms;
            Ok(result)
        } else {
            let img = image::open(image_path)?;
            self.segment(&img, prompt)
        }
    }
}
