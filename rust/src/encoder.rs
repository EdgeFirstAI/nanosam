//! TFLite image encoder (INT8 quantized, supports NPU delegates).

use std::time::Instant;

use edgefirst_tflite::{Library, Model};
use image::DynamicImage;

use crate::preprocess::{self, PreprocessedImage};
use crate::tflite_runner::TfLiteRunner;
use crate::types::ImageEmbedding;

/// NanoSAM image encoder using TFLite INT8 model.
pub struct Encoder<'a> {
    runner: TfLiteRunner<'a>,
}

impl<'a> Encoder<'a> {
    /// Create an encoder from a loaded TFLite model.
    pub fn new(
        lib: &'a Library,
        model: &'a Model<'a>,
        threads: usize,
        delegate_paths: &[&str],
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let runner = TfLiteRunner::new(lib, model, threads, delegate_paths)?;
        Ok(Self { runner })
    }

    /// Encode an image, returning embeddings and timing.
    ///
    /// Handles preprocessing (resize, pad, normalize) and the NHWC↔NCHW
    /// layout conversions required by the TFLite encoder.
    pub fn encode(
        &mut self,
        image: &DynamicImage,
    ) -> Result<(ImageEmbedding, f64, f64), Box<dyn std::error::Error>> {
        // Preprocess: resize, normalize, pad → NHWC
        let t_pre = Instant::now();
        let preprocessed: PreprocessedImage = preprocess::preprocess_image(image);
        let preprocess_ms = t_pre.elapsed().as_secs_f64() * 1000.0;

        // Set input (quantized from float to INT8 internally)
        self.runner.set_input_quantized(&preprocessed.nhwc)?;

        // Run encoder
        let encode_dur = self.runner.invoke()?;
        let encoder_ms = encode_dur.as_secs_f64() * 1000.0;

        // Read output and dequantize
        let output = self.runner.output_dequantized()?;

        // Determine layout: output may be NCHW (1,256,64,64) or NHWC (1,64,64,256)
        // If output length is 1*256*64*64 = 1048576, check first dim after batch
        let data = if output.len() == 1_048_576 {
            // Check if NHWC by seeing if channels-last = 256
            // NHWC: positions [0..3] are spatial, position 3 is channel
            // NCHW: positions [0..256*64*64] are channel 0
            // Heuristic: the twin encoder outputs NHWC (1,64,64,256)
            // Convert to NCHW for the prompt encoder which expects (1,256,64,64)
            preprocess::nhwc_to_nchw(&output, [1, 64, 64, 256])
        } else {
            output
        };

        Ok((
            ImageEmbedding {
                data,
                original_hw: preprocessed.original_hw,
            },
            preprocess_ms,
            encoder_ms,
        ))
    }

    /// Encode from pre-processed NHWC float32 data (for HAL integration).
    ///
    /// Caller handles preprocessing; this just quantizes, runs inference,
    /// dequantizes, and transposes.
    pub fn encode_nhwc(
        &mut self,
        nhwc: &[f32],
        original_hw: (u32, u32),
    ) -> Result<(ImageEmbedding, f64), Box<dyn std::error::Error>> {
        self.runner.set_input_quantized(nhwc)?;
        self.run_and_read(original_hw)
    }

    /// Encode from pre-quantized NHWC INT8 data (zero-copy from HAL).
    ///
    /// The caller has already done the fused U8→I8 ImageNet-norm + quantize.
    /// This copies the I8 bytes directly into the TFLite input tensor.
    pub fn encode_i8(
        &mut self,
        i8_data: &[i8],
        original_hw: (u32, u32),
    ) -> Result<(ImageEmbedding, f64), Box<dyn std::error::Error>> {
        self.runner.set_input_i8(i8_data)?;
        self.run_and_read(original_hw)
    }

    fn run_and_read(
        &mut self,
        original_hw: (u32, u32),
    ) -> Result<(ImageEmbedding, f64), Box<dyn std::error::Error>> {
        let encode_dur = self.runner.invoke()?;
        let encoder_ms = encode_dur.as_secs_f64() * 1000.0;

        let output = self.runner.output_dequantized()?;
        let data = if output.len() == 1_048_576 {
            preprocess::nhwc_to_nchw(&output, [1, 64, 64, 256])
        } else {
            output
        };

        Ok((ImageEmbedding { data, original_hw }, encoder_ms))
    }
}
