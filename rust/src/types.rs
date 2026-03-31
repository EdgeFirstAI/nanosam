//! Core types for the NanoSAM segmentation pipeline.

use std::path::PathBuf;

/// Bounding box prompt in original image pixel coordinates.
#[derive(Debug, Clone)]
pub struct BBox {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
}

/// Point prompt in original image pixel coordinates.
#[derive(Debug, Clone)]
pub struct Point {
    pub x: f32,
    pub y: f32,
    pub label: PointLabel,
}

/// Label for a point prompt.
#[derive(Debug, Clone, Copy)]
pub enum PointLabel {
    Background = 0,
    Foreground = 1,
}

/// A segmentation prompt (box or points).
#[derive(Debug, Clone)]
pub enum Prompt {
    Box(BBox),
    Points(Vec<Point>),
}

/// A single segmentation mask at original image resolution.
#[derive(Debug)]
pub struct Mask {
    /// Raw float values (positive = foreground). Shape: height × width, row-major.
    pub data: Vec<f32>,
    pub width: u32,
    pub height: u32,
    /// Model-predicted IoU quality score for this mask.
    pub iou_score: f32,
}

/// Per-stage timing breakdown for a single inference.
#[derive(Debug, Clone, Default)]
pub struct Timings {
    pub preprocess_ms: f64,
    pub encoder_ms: f64,
    pub prompt_encoder_ms: f64,
    pub attention_ms: f64,
    pub heads_ms: f64,
    pub tokens_ms: f64,
    pub mask_assembly_ms: f64,
    pub postprocess_ms: f64,
}

impl Timings {
    /// Total pipeline time in milliseconds.
    pub fn total_ms(&self) -> f64 {
        self.preprocess_ms
            + self.encoder_ms
            + self.prompt_encoder_ms
            + self.attention_ms
            + self.heads_ms
            + self.tokens_ms
            + self.mask_assembly_ms
            + self.postprocess_ms
    }

    /// Decoder-only time (everything after encoder).
    pub fn decoder_ms(&self) -> f64 {
        self.prompt_encoder_ms
            + self.attention_ms
            + self.heads_ms
            + self.tokens_ms
            + self.mask_assembly_ms
    }
}

/// Result of a segmentation operation.
#[derive(Debug)]
pub struct SegmentationResult {
    /// 4 masks from the decoder, with IoU scores.
    pub masks: Vec<Mask>,
    /// Index of the mask with the highest IoU score.
    pub best_mask_index: usize,
    /// Per-stage timing.
    pub timings: Timings,
}

/// Image embedding produced by the encoder (opaque to callers).
pub struct ImageEmbedding {
    /// Flat NCHW float32 data: (1, 256, 64, 64) = 1,048,576 elements.
    pub(crate) data: Vec<f32>,
    /// Original image dimensions (height, width) for mask upscaling.
    pub(crate) original_hw: (u32, u32),
}

/// Session configuration.
pub struct SessionConfig {
    /// Path to the model archive ZIP.
    pub model_archive: PathBuf,
    /// Delegate shared libraries to try for INT8 models (e.g. libneutron_delegate.so).
    /// Tried in order; falls back to CPU if none succeed.
    pub delegates: Vec<PathBuf>,
    /// Enable built-in XNNPACK delegate for attention model (multi-threaded FP16).
    pub use_xnnpack: bool,
    /// Number of TFLite interpreter threads (default: available CPUs).
    pub threads: usize,
}
