# NanoSAM x86_64 Evaluation: ONNX Export, TFLite INT8, and COCO mIoU

**Date:** 2026-03-25
**Ticket:** [EDGEAI-1204 — NanoSAM encoder/decoder export and x86_64 validation](https://au-zone.atlassian.net/browse/EDGEAI-1204)
**Parent:** [EDGEAI-1193 — Source model export and x86_64 validation baseline](https://au-zone.atlassian.net/browse/EDGEAI-1193)
**Epic:** [EDGEAI-1138 — HuggingFace SAM Models](https://au-zone.atlassian.net/browse/EDGEAI-1138)

## Summary

Initial evaluation of NanoSAM (NVIDIA's distilled Segment Anything Model) for export to ONNX and quantized TFLite, validated on x86_64 before deployment to edge platforms. NanoSAM uses a split encoder/decoder architecture: a ResNet18 image encoder (pure CNN, ~11M params) distilled from MobileSAM, paired with the MobileSAM mask decoder (attention-based, shared unchanged).

## Target Edge Platforms

| Platform | Accelerator | Encoder Runtime | Decoder Runtime |
|----------|------------|----------------|----------------|
| Hailo-8L (RPi5) | Hailo NPU | HEF (INT8) | CPU (TFLite/ONNX) |
| Ara-2 A40 | Arm Ethos-U NPU | TFLite INT8 | CPU (TFLite) |
| i.MX 8M Plus | Arm Ethos-U NPU (eIQ/VX Delegate) | TFLite INT8 | CPU (TFLite) |
| i.MX 95 | Arm Ethos-U/eIQ NPU | TFLite INT8 | CPU (TFLite) |
| Jetson Orin Nano | NVIDIA GPU | TensorRT FP16/INT8 | TensorRT FP32 |

Encoder on NPU/GPU; decoder on CPU or wherever demonstrated to work best.

## Architecture

### Model Components

- **Encoder:** `TimmImageEncoder` wrapping `timm.create_model("resnet18", features_only=True)` with a custom upsampling neck (`up_1`: 3x Conv2d + 1x ConvTranspose2d, all with GELU), a projection head (`proj`: 2x Conv2d), and a learned positional embedding added to the output. Pure CNN, no attention. The positional embedding is a distillation artifact matching the MobileSAM teacher's embedding space — it must not be zeroed out or removed.
  - Input: `1x3x1024x1024` (NCHW, ImageNet-normalized, aspect-ratio-preserving resize with zero-padding)
  - Output: `1x256x64x64` (image embeddings, includes positional embedding)
  - **Note:** Verify the NVIDIA pre-exported ONNX has a static batch dimension of 1 using `onnx.load()` / `model.graph.input` before TFLite conversion.
- **Decoder:** MobileSAM mask decoder (prompt encoder + two-way transformer with cross-attention). Exported in **multimask mode** (no `--return-single-mask`), using `--model-type vit_t` for the MobileSAM checkpoint.
  - Inputs: `image_embeddings` (1x256x64x64), `point_coords` (1xNx2), `point_labels` (1xN), `mask_input` (1x1x256x256), `has_mask_input` (1,)
  - Outputs: `iou_predictions` (1x4), `low_res_masks` (1x4x256x256) — returned in this order from `ort_session.run()` at indices 0 and 1 respectively
  - The 4 masks include index 0 (the "single-prompt" token) and indices 1–3 (the multimask tokens). The `argmax` over `iou_predictions` may frequently select index 0 for single-point prompts.
  - When `mask_input` is not provided, pass a zero tensor (1x1x256x256) and set `has_mask_input` to `0.0`
  - **Two ONNX exports required:** (1) `--opset 16` with dynamic `num_points` axis for ONNX Runtime flexibility; (2) `--opset 13` with fixed `num_points=2` for TFLite conversion (TFLite does not support dynamic shapes, and `onnx2tf` has best compatibility with opset 13–15)

### Preprocessing

Extracted into a new shared module `nanosam/utils/preprocess.py` (see Project Structure). The existing `nanosam/utils/predictor.py` contains the preprocessing logic but is **not importable** by the ONNX/TFLite predictors because:
1. It hard-codes `.cuda()` on the return tensor (line 89) — crashes without a CUDA device
2. `run_mask_decoder()` also hard-codes `.cuda()` throughout (lines 103–110)
3. Importing `predictor.py` triggers `from torch2trt import TRTModule` and `import tensorrt as trt` (lines 16–17), which fail without TensorRT installed

The shared `preprocess.py` reimplements the preprocessing logic in pure NumPy (no torch dependency at runtime):
- Aspect-ratio-preserving resize to fit within 1024×1024
- ImageNet normalization: mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]
- Zero-padding to 1024×1024
- Point coordinates scaled by `1024 / max(height, width)`
- **Important:** `image_size` must be passed as `(height, width)`, not PIL's `(width, height)` from `image.size`. The existing `Predictor` uses `(self.image.height, self.image.width)`.

### Mask Upscaling

The decoder outputs `low_res_masks` at 256×256. These must be upscaled to the original image dimensions with aspect-ratio-aware cropping:

1. Compute the aspect-ratio-adjusted region within the 256×256 output:
   - If `width > height`: `lim_x = 256`, `lim_y = int(256 * height / width)`
   - Else: `lim_y = 256`, `lim_x = int(256 * width / height)`
2. Crop `low_res_mask[:, :, :lim_y, :lim_x]` to remove padding
3. Bilinear upsample the cropped region to `(original_height, original_width)`

This logic lives in `preprocess.py` as `upscale_mask()`. The existing `predictor.py:upscale_mask()` has a dead line (`mask[:, :, :lim_y, :lim_x]` on line 133 with the result discarded) — the new implementation omits this.

## Project Structure

### New Files

```
nanosam/
  utils/
    preprocess.py               # Shared CPU-safe preprocessing (pure NumPy)
    onnx_predictor.py           # ONNX Runtime-based Predictor
    tflite_predictor.py         # TFLite-based Predictor
  tools/
    export_decoder_onnx_static.py  # Decoder ONNX export with fixed num_points for TFLite
    export_encoder_tflite.py    # ONNX -> TFLite INT8 for encoder
    export_decoder_tflite.py    # ONNX -> TFLite for decoder (FP32 + dynamic-range quant)
    eval_coco_onnx.py           # COCO mIoU evaluation using ONNX Runtime backend
    eval_coco_tflite.py         # COCO mIoU evaluation using TFLite backend
scripts/
  run_inference_onnx.py         # Visual inference demo (ONNX backend)
  run_inference_tflite.py       # Visual inference demo (TFLite backend)
  download_models.sh            # Download pre-exported encoder ONNX
data/                           # Created at runtime for models, outputs, COCO
```

### Dependencies (venv-only)

- `onnxruntime` — ONNX inference
- `onnx` — model inspection/validation
- `onnx2tf` + `tensorflow` (pin `tensorflow==2.16.*` — TF 2.17+ changed TFLite converter API) — ONNX to TFLite conversion with INT8 calibration
- `pillow`, `matplotlib`, `numpy`, `scipy` — image handling, visualization, and bilinear interpolation for upscaling
- `pycocotools` — COCO annotation loading and mask conversion
- `torchvision` (CPU-only: `--index-url https://download.pytorch.org/whl/cpu`) — for `CocoDetection` dataset loader in eval scripts. Alternative: replace with direct `pycocotools` + `PIL.Image.open()` loader to eliminate torch dependency in eval path.
- `timm` — encoder architecture (existing dependency, only needed for decoder ONNX export)
- `torch` (CPU-only) — decoder ONNX export from mobile_sam.pt

### Memory Budget

The encoder processes a `1x3x1024x1024` float32 tensor (~12 MB). Under ONNX Runtime's CPU provider, peak working set is approximately 1–1.5 GB (model weights ~100 MB + intermediate activations). For COCO evaluation (5K images), `torchvision.datasets.CocoDetection` loads images on demand (no preload). The ONNX Runtime `InferenceSession` must be created once before the eval loop, not per-image.

## Design Details

### Shared Preprocessing (`preprocess.py`)

Pure NumPy module imported by all three predictor implementations (`predictor.py`, `onnx_predictor.py`, `tflite_predictor.py`). Contains:

- `preprocess_image(image, size=1024)` — returns `numpy.float32` array, NCHW layout. **The default size is 1024** (unlike the existing `predictor.py` which defaults to 512).
- `preprocess_points(points, image_size, size=1024)` — `image_size` is `(height, width)`, not PIL's `(width, height)`.
- `upscale_mask(mask, image_shape, size=256)` — aspect-ratio-aware crop + bilinear upsample using `scipy.ndimage.zoom` or `PIL.Image.resize`.

### ONNX Runtime Predictor (`onnx_predictor.py`)

```python
class OnnxPredictor:
    def __init__(self, image_encoder_path: str, mask_decoder_path: str,
                 image_encoder_size: int = 1024)
    def set_image(self, image: PIL.Image.Image)
    def predict(self, points, point_labels, mask_input=None)
                -> (hi_res_mask, iou_predictions, low_res_mask)
```

- Imports preprocessing from `nanosam.utils.preprocess` (NOT from `nanosam.utils.predictor`)
- All ONNX boundary I/O as `numpy.float32`
- Uses `onnxruntime.InferenceSession` with `CPUExecutionProvider`
- **Decoder output mapping:** `ort_session.run(None, inputs)` returns `[iou_predictions, low_res_masks]` at indices 0 and 1. The predictor must:
  1. Get `iou_predictions = outputs[0]`  (shape 1x4)
  2. Get `low_res_masks = outputs[1]`  (shape 1x4x256x256)
  3. Select best mask: `best_idx = iou_predictions[0].argmax()`; `best_mask = low_res_masks[0, best_idx]`
  4. Upscale: `hi_res_mask = upscale_mask(best_mask, (image.height, image.width))`
  5. Return `(hi_res_mask, iou_predictions, low_res_mask)`

### TFLite Export

**Encoder (INT8):**
- Source: `resnet18_image_encoder.onnx`
- Tool: `onnx2tf` (handles NCHW -> NHWC transposition)
- Calibration: 100–200 images from COCO val2017, preprocessed identically to inference
- Full INT8 post-training quantization (weights and activations)
- **Positional embedding risk:** The encoder's learned `pos_embedding` (1x256x64x64) is baked into the ONNX as a constant `Add` node. During NCHW→NHWC transposition, `onnx2tf` must transpose this constant to 1x64x64x256. Verify correctness with the intermediate encoder tensor gate (Phase B step 2b).

**Decoder — two ONNX exports:**

| Export | Opset | `num_points` | Dynamic axes | Purpose |
|--------|-------|-------------|-------------|---------|
| `mobile_sam_mask_decoder.onnx` | 16 | Dynamic | `point_coords:{1}`, `point_labels:{1}` | ONNX Runtime (flexible prompts) |
| `mobile_sam_mask_decoder_static.onnx` | 13 | Fixed at 2 | None | TFLite conversion (bounding box prompts only) |

The static export fixes `num_points=2` for bounding box prompts (corner points with labels `[2, 3]`). This is required because:
1. TFLite does not support dynamic shapes
2. `onnx2tf` has best compatibility with opset 13–15
3. The dynamic `num_points` axis propagates through the two-way transformer's attention sequence length, making static shape inference impossible otherwise

**Decoder TFLite conversion:**
- Start with FP32 TFLite from the static ONNX
- Then attempt dynamic-range quantization (weights INT8, activations FP32)
- Full INT8 is a stretch goal — validate IoU before committing
- **Risk:** The decoder contains a two-way transformer with cross-attention and 4D batch matmuls in the attention layers (`q @ k.permute(0,1,3,2)` and `attn @ v`). The mask prediction head also has a batch matmul: `hyper_in @ upscaled_embedding.view(b, c, h*w)`. `onnx2tf` may struggle with these ops. If FP32 TFLite conversion fails, fall back to `ai-edge-torch`. If conversion fails entirely, the decoder remains ONNX-only for x86_64 validation and we evaluate TFLite conversion feasibility per-platform.

**NCHW vs NHWC:** `onnx2tf` transposes to NHWC automatically. The TFLite predictor handles transposing inputs/outputs. Always verify via `interpreter.get_input_details()`.

### TFLite Predictor (`tflite_predictor.py`)

```python
class TflitePredictor:
    def __init__(self, encoder_path: str, decoder_path: str,
                 image_encoder_size: int = 1024)
    def set_image(self, image: PIL.Image.Image)
    def predict(self, points, point_labels, mask_input=None)
                -> (hi_res_mask, iou_predictions, low_res_mask)
```

- Same API as OnnxPredictor
- Imports preprocessing from `nanosam.utils.preprocess`
- Uses `tf.lite.Interpreter`
- Handles NHWC layout at TFLite boundary, NCHW internally for consistency

### COCO mIoU Evaluation

**Methodology:** Prompt NanoSAM with ground-truth bounding boxes from COCO val2017 annotations. Compute IoU between predicted mask and ground-truth COCO segmentation mask. Average over all objects, broken down by size (all/small/medium/large).

**`iscrowd` filtering:** The existing `eval_coco.py` does not filter `iscrowd=1` annotations. The new eval scripts **must filter these out** — crowd annotations represent groups of objects and produce poor IoU that drags down mIoU. If the ONNX FP32 baseline mIoU falls outside the ±0.01 gate of NVIDIA's 0.706, check whether the discrepancy is due to crowd filtering before concluding there is a regression.

**Evaluation matrix:**

| Backend | Encoder | Decoder | Purpose |
|---------|---------|---------|---------|
| ONNX FP32 | ONNX FP32 | ONNX FP32 | Baseline — target: match NVIDIA's 0.706 mIoU |
| TFLite INT8 | TFLite INT8 | TFLite FP32 | Quantization impact on encoder |
| TFLite INT8 | TFLite INT8 | TFLite dyn-quant | Full edge deployment quality |

**Reuse strategy:** The existing `eval_coco.py` contains `iou()`, `box_xywh_to_xyxy()`, and the evaluation loop. The `compute_eval_coco_metrics.py` computes summary statistics from JSON results. The new eval scripts will:
- Reuse `iou()` and `box_xywh_to_xyxy()` directly (pure numpy, no backend coupling)
- Rewrite `predict_box()` as a standalone function operating on numpy arrays: use `iou_preds[0].argmax()` (explicit batch dimension) instead of `iou_preds.argmax()`, remove `.detach().cpu()` calls
- Add `iscrowd` filtering in the eval loop
- Add a zero-division guard in metrics computation for empty result sets after size filtering
- Load COCO data via `torchvision.datasets.CocoDetection` (returns PIL Images on demand, compatible with all predictors)

## Execution Phases

### Phase A: ONNX Export & Validation

1. Set up venv with dependencies
2. Download `resnet18_image_encoder.onnx` from NVIDIA (Google Drive)
3. Export decoder ONNX (two exports):
   - **For ONNX Runtime:** `python3 -m nanosam.tools.export_sam_mask_decoder_onnx --model-type=vit_t --checkpoint=assets/mobile_sam.pt --output=data/mobile_sam_mask_decoder.onnx` (opset 16, dynamic `num_points`, no `--return-single-mask`)
   - **For TFLite (Phase B):** `python3 -m nanosam.tools.export_decoder_onnx_static --model-type=vit_t --checkpoint=assets/mobile_sam.pt --output=data/mobile_sam_mask_decoder_static.onnx` (opset 13, fixed `num_points=2`)
4. Build `OnnxPredictor`
5. Run inference on `assets/dogs.jpg` with bounding box [100, 100, 850, 759], selecting the best mask via `iou_predictions[0].argmax()`
6. **Gate:** Visual comparison of ONNX output (`data/basic_usage_out.jpg`) against the committed reference in `assets/basic_usage_out.jpg`. Manual visual inspection — the same dog should be segmented with comparable mask quality. Note: the reference was generated with a TRT engine whose mask selection strategy is unknown; minor differences are expected.

### Phase B: TFLite INT8 Export & Validation

1. Download COCO val2017 images and annotations
2. Convert encoder ONNX -> TFLite INT8 with COCO calibration
   - 2b. **Intermediate encoder tensor gate:** Run a canonical test image through both the ONNX encoder and TFLite encoder. Compare raw output embeddings (before decoder) to verify the positional embedding constant was transposed correctly during NCHW→NHWC conversion. Cosine similarity should be > 0.95.
3. Convert static decoder ONNX (`mobile_sam_mask_decoder_static.onnx`) -> TFLite FP32, then try dynamic-range quantization
4. Build `TflitePredictor`
5. Run on same `assets/dogs.jpg` test case
6. **Gate:** Per-pixel IoU between TFLite mask and ONNX mask. Encoder INT8: IoU > 0.90 (GELU activations in the neck are sensitive to quantization; if IoU is between 0.85–0.90, increase calibration set size before declaring failure). Decoder dynamic-quant: IoU > 0.90 (else stay FP32).

### Phase C: COCO mIoU Evaluation

1. Adapt eval pipeline for ONNX and TFLite backends (with `iscrowd` filtering)
2. Run full COCO val2017 evaluation for each backend config
3. Produce comparison table against NVIDIA's published numbers
4. **Gate:** ONNX FP32 mIoU within ±0.02 of NVIDIA's reported 0.706 (wider margin to account for potential `iscrowd` filtering difference; investigate if outside this range)

## Out of Scope

- Retraining or fine-tuning
- Hailo HEF compilation (separate per-platform step)
- TensorRT engine builds (covered by upstream repo)
- Resolution experiments (1024×1024 baseline only)
- Edge platform deployment (follows after successful x86_64 evaluation)

## Pre-trained Weights

- **Encoder:** `resnet18_image_encoder.onnx` — NVIDIA pre-exported ONNX from [Google Drive](https://drive.google.com/file/d/14-SsvoaTl-esC3JOzomHDnI9OGgdO2OR/view?usp=drive_link) (file ID: `14-SsvoaTl-esC3JOzomHDnI9OGgdO2OR`). This is a public link from the NanoSAM README. If the link becomes unavailable, we cannot proceed without a PyTorch encoder checkpoint (none publicly available) or retraining.
- **Decoder:** Exported from `assets/mobile_sam.pt` using existing `export_sam_mask_decoder_onnx.py` with `--model-type vit_t`. Two variants: opset 16 dynamic (ONNX Runtime) and opset 13 static (TFLite).
- **No PyTorch encoder checkpoint available** — working directly from NVIDIA's ONNX export

## Training Background (Reference)

NanoSAM's encoder is trained via knowledge distillation on COCO 2017 train images (no annotations needed). The MobileSAM TinyViT encoder serves as the teacher; the ResNet18 student learns to produce matching 256×64×64 embeddings. The decoder is shared from MobileSAM and never retrained. This is encoder-only feature distillation, not full SAM task training.
