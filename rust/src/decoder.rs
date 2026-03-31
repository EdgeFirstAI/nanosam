//! Split decoder pipeline: prompt encoder → attention → heads → tokens → masks.
//!
//! Ported from sam-decoder `src/bench/pipeline.rs` and the Python `run_pipeline.py`.

use std::time::Instant;

use edgefirst_tflite::{Library, Model};

use crate::preprocess;
use crate::prompt_encoder::PromptEncoder;
use crate::tflite_runner::TfLiteRunner;
use crate::types::{ImageEmbedding, Mask, PointLabel, Prompt, Timings};

/// Split decoder: runs prompt encoding through mask assembly.
pub struct Decoder<'a> {
    prompt_encoder: PromptEncoder<f32>,
    attention: TfLiteRunner<'a>,
    heads_a: TfLiteRunner<'a>,
    heads_b: TfLiteRunner<'a>,
    tokens: TfLiteRunner<'a>,
}

impl<'a> Decoder<'a> {
    /// Create a decoder from loaded models and prompt encoder weights.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        lib: &'a Library,
        prompt_weights_bytes: &[u8],
        attn_model: &'a Model<'a>,
        heads_a_model: &'a Model<'a>,
        heads_b_model: &'a Model<'a>,
        tokens_model: &'a Model<'a>,
        threads: usize,
        delegate_paths: &[&str],
        use_xnnpack: bool,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        // Load prompt encoder weights from safetensors bytes
        let prompt_encoder = PromptEncoder::from_safetensors_bytes(prompt_weights_bytes)?;

        // Attention uses built-in XNNPACK for multi-threaded FP16, or falls back to CPU
        let attention = if use_xnnpack {
            TfLiteRunner::with_xnnpack(lib, attn_model, threads)?
        } else {
            TfLiteRunner::new(lib, attn_model, threads, &[])?
        };

        // Heads are INT8 — try NPU delegates for acceleration.
        // Twin-model heads (tanh GELU, fixed batch, no float islands) are
        // compatible with both Neutron and VeriSilicon NPUs.
        let heads_a = TfLiteRunner::new(lib, heads_a_model, threads, delegate_paths)?;
        let heads_b = TfLiteRunner::new(lib, heads_b_model, threads, delegate_paths)?;

        // Tokens run on CPU (dynamic range quantized, small)
        let tokens = TfLiteRunner::new(lib, tokens_model, threads, &[])?;

        Ok(Self {
            prompt_encoder,
            attention,
            heads_a,
            heads_b,
            tokens,
        })
    }

    /// Run the decoder pipeline on an image embedding with a prompt.
    ///
    /// Returns 4 low-res masks (256×256) as NCHW, IoU scores, and timing.
    pub fn decode(
        &mut self,
        embedding: &ImageEmbedding,
        prompt: &Prompt,
    ) -> Result<(Vec<f32>, Vec<f32>, Timings), Box<dyn std::error::Error>> {
        let mut timings = Timings::default();
        let original_hw = embedding.original_hw;

        // Build point coords and labels from prompt
        let (point_coords, point_labels) = prompt_to_coords(prompt, original_hw);

        // --- Stage 1: Prompt encoder (Rust FP32) ---
        let t = Instant::now();
        let (sparse_emb, image_src) = self.prompt_encoder.encode(
            &point_coords,
            &point_labels,
            &embedding.data,
            0.0, // has_mask_input = false
        );
        timings.prompt_encoder_ms = t.elapsed().as_secs_f64() * 1000.0;

        // Transpose for TFLite layout
        // sparse: (1, N, 256) → (1, 256, N) for onnx2tf attention model
        let n_tokens = sparse_emb.len() / 256;
        let sparse_transposed =
            preprocess::transpose_3d(&sparse_emb, [1, n_tokens, 256], [0, 2, 1]);
        // image: (1, 256, 64, 64) NCHW → (1, 64, 64, 256) NHWC
        let image_transposed = preprocess::nchw_to_nhwc(&image_src, [1, 256, 64, 64]);

        // --- Stage 2: Attention (TFLite XNNPACK) ---
        self.attention.set_input_f32_at(0, &sparse_transposed)?;
        self.attention.set_input_f32_at(1, &image_transposed)?;
        let d_attn = self.attention.invoke()?;
        timings.attention_ms = d_attn.as_secs_f64() * 1000.0;

        // Find token output (1792 elements) and image output (1048576 elements)
        let attn_outputs = self.attention.all_outputs_f32()?;
        let (token_out, image_out) = find_attention_outputs(&attn_outputs)?;

        // --- Stage 3: Heads Part A (INT8 quantized, NPU) ---
        self.heads_a.set_input_quantized(&image_out)?;
        let d_ha = self.heads_a.invoke()?;
        let mid = self.heads_a.output_dequantized()?;

        // --- Stage 4: Heads Part B (INT8 quantized, NPU) ---
        self.heads_b.set_input_quantized(&mid)?;
        let d_hb = self.heads_b.invoke()?;
        let features = self.heads_b.output_dequantized()?;
        timings.heads_ms = (d_ha + d_hb).as_secs_f64() * 1000.0;

        // --- Stage 5: Tokens (FP32) ---
        self.tokens.set_input_f32(&token_out)?;
        let d_tok = self.tokens.invoke()?;
        timings.tokens_ms = d_tok.as_secs_f64() * 1000.0;

        // Find hyper (128 elements) and iou (4 elements)
        let tok_outputs = self.tokens.all_outputs_f32()?;
        let (hyper, iou) = find_token_outputs(&tok_outputs)?;

        // --- Stage 6: Mask assembly ---
        let t = Instant::now();
        let masks_flat = mask_assembly(&features, &hyper);
        timings.mask_assembly_ms = t.elapsed().as_secs_f64() * 1000.0;

        Ok((iou, masks_flat, timings))
    }
}

/// Convert a Prompt to the flat (coords, labels) format expected by the prompt encoder.
fn prompt_to_coords(prompt: &Prompt, image_hw: (u32, u32)) -> (Vec<f32>, Vec<f32>) {
    match prompt {
        Prompt::Box(bbox) => {
            let raw_coords = [bbox.x1, bbox.y1, bbox.x2, bbox.y2];
            let scaled = preprocess::scale_points(&raw_coords, image_hw);
            // Box prompt: 2 points with labels 2 (top-left) and 3 (bottom-right)
            (scaled, vec![2.0, 3.0])
        }
        Prompt::Points(points) => {
            let raw_coords: Vec<f32> = points.iter().flat_map(|p| [p.x, p.y]).collect();
            let scaled = preprocess::scale_points(&raw_coords, image_hw);
            let labels: Vec<f32> = points
                .iter()
                .map(|p| match p.label {
                    PointLabel::Background => 0.0,
                    PointLabel::Foreground => 1.0,
                })
                .collect();
            (scaled, labels)
        }
    }
}

/// Mask assembly: features (65536, 32) @ hyper^T (32, 4) → masks (65536, 4).
fn mask_assembly(features: &[f32], hyper: &[f32]) -> Vec<f32> {
    let rows = 65536;
    let inner = 32;
    let cols = 4;
    let mut out = vec![0.0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            let mut sum = 0.0f32;
            for k in 0..inner {
                sum += features[r * inner + k] * hyper[c * inner + k];
            }
            out[r * cols + c] = sum;
        }
    }
    out
}

/// Find token output (1792 elements) and image output (1048576 elements) from attention.
fn find_attention_outputs(
    outputs: &[(String, Vec<f32>, Vec<usize>)],
) -> Result<(Vec<f32>, Vec<f32>), Box<dyn std::error::Error>> {
    let mut token_out = None;
    let mut image_out = None;

    for (_name, data, _shape) in outputs {
        match data.len() {
            1792 => token_out = Some(data.clone()),
            1_048_576 => image_out = Some(data.clone()),
            _ => {}
        }
    }

    let token = token_out.ok_or("Attention model missing token output (1792 elements)")?;
    let image = image_out.ok_or("Attention model missing image output (1048576 elements)")?;
    Ok((token, image))
}

/// Find hyper (128 elements) and iou (4 elements) from tokens model.
fn find_token_outputs(
    outputs: &[(String, Vec<f32>, Vec<usize>)],
) -> Result<(Vec<f32>, Vec<f32>), Box<dyn std::error::Error>> {
    let mut hyper_out = None;
    let mut iou_out = None;

    for (_name, data, _shape) in outputs {
        match data.len() {
            128 => hyper_out = Some(data.clone()),
            4 => iou_out = Some(data.clone()),
            _ => {}
        }
    }

    let hyper = hyper_out.ok_or("Tokens model missing hyper output (128 elements)")?;
    let iou = iou_out.ok_or("Tokens model missing iou output (4 elements)")?;
    Ok((hyper, iou))
}

/// Build Mask structs from flat decoder output.
pub fn build_masks(
    iou: &[f32],
    masks_flat: &[f32],
    original_hw: (u32, u32),
) -> Vec<Mask> {
    let (orig_h, orig_w) = original_hw;
    let mut masks = Vec::with_capacity(4);

    for i in 0..4 {
        // Extract mask i from (65536, 4) → (256, 256)
        let mut mask_256 = vec![0.0f32; 256 * 256];
        for px in 0..65536 {
            mask_256[px] = masks_flat[px * 4 + i];
        }

        // Debug: save raw 256×256 mask as grayscale PNG
        if std::env::var("NANOSAM_DEBUG_MASKS").is_ok() {
            let debug_path = format!("/tmp/nanosam_mask256_{i}.png");
            let mut img = image::GrayImage::new(256, 256);
            for y in 0..256u32 {
                for x in 0..256u32 {
                    let v = mask_256[(y * 256 + x) as usize];
                    // Map: negative=0 (black), 0=128 (gray), positive=255 (white)
                    let px_val = ((v.clamp(-5.0, 5.0) / 5.0 * 127.0) + 128.0) as u8;
                    img.put_pixel(x, y, image::Luma([px_val]));
                }
            }
            let _ = img.save(&debug_path);
            let pos_count = mask_256.iter().filter(|&&v| v > 0.0).count();
            let min = mask_256.iter().cloned().fold(f32::INFINITY, f32::min);
            let max = mask_256.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            eprintln!(
                "DEBUG mask {i}: pos={pos_count}/65536, range=[{min:.2}, {max:.2}], saved {debug_path}"
            );
        }

        // Upscale to original resolution
        let data = preprocess::upscale_mask(&mask_256, original_hw);

        masks.push(Mask {
            data,
            width: orig_w,
            height: orig_h,
            iou_score: iou[i],
        });
    }

    masks
}
