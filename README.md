# NanoSAM — EdgeFirstAI Edge Platform Fork

> **This is the [EdgeFirstAI](https://edgefirst.ai) fork of [NVIDIA-AI-IOT/nanosam](https://github.com/NVIDIA-AI-IOT/nanosam).**
> Focus: porting NanoSAM to run on edge hardware beyond Jetson — NXP i.MX platforms,
> Raspberry Pi 5 with Hailo-8L, and discrete NPU accelerators.
>
> **Work in progress.** Complete LVIS validation and end-to-end optimized benchmarks will be
> published on [EdgeFirst Models](https://huggingface.co/spaces/EdgeFirst/Models) once available.

---

## Platform Support

The ResNet18 encoder is a pure CNN — it compiles and runs on every tested NPU.
The MobileSAM decoder contains attention layers and runs on CPU (ONNX) on all platforms.

| Platform | NPU / Accelerator | Encoder Format | Decoder Format |
|----------|-------------------|----------------|----------------|
| Jetson Orin Nano Super | NVIDIA GPU | TensorRT FP16 | TensorRT FP32 |
| Raspberry Pi 5 + Hailo-8L HAT | Hailo-8L NPU | HEF INT8 | ONNX CPU FP32 |
| NXP i.MX 95 | Neutron NPU | TFLite INT8 | ONNX CPU FP32 |
| NXP i.MX 95 + Ara240 (M.2 PCIe) | Kinara Ara-2 NPU | DVM INT8 | ONNX CPU FP32 |
| NXP i.MX 8M Plus | VeriSilicon NPU | TFLite INT8 | ONNX CPU FP32 |

---

## Benchmark Results

> Results validate that each platform reproduces the **ONNX FP32 reference output** (Mask IoU vs ONNX).
> These use ImageNet-pretrained ResNet18 backbone weights with an untrained distillation neck —
> NVIDIA's distilled weights are inaccessible (private). When distilled weights are available,
> actual segmentation accuracy can be measured with the same exported models.

### Encoder-only benchmarks (ResNet18, monolithic ONNX decoder on CPU)

| Platform | Accelerator | Format | Encoder (ms) | Mask IoU vs ONNX | Decoder CPU (ms) | Total (ms) |
|----------|-------------|--------|:------------:|:----------------:|:----------------:|:----------:|
| x86_64 | ONNX Runtime | ONNX FP32 (61 MB) | ~200 | baseline | 15.8 | ~216 |
| Jetson Orin Nano | NVIDIA GPU | TRT FP16 (33 MB) | 12.8 | 0.9999 | 63.3 | ~76 |
| Hailo-8L (RPi5) | Hailo-8L NPU | HEF INT8 (36 MB) | 46 | 0.9654 | 125 | ~171 |
| i.MX 95 + Ara240 | Kinara Ara-2 NPU | DVM INT8 (27 MB) | 45.3 | 0.9961 | 430 | ~475 |
| i.MX 8M Plus | VeriSilicon NPU | TFLite INT8 (16 MB) | 336 | 0.9972 | 421 | ~757 |

> **Note:** Naive `onnx2tf` → Neutron NPU conversion of the encoder **fails** due to
> float islands from GELU/erf — masks are broken (IoU 0.747). See the EdgeFirst
> optimized results below and [BENCHMARKS.md](BENCHMARKS.md) for details.

### EdgeFirst optimized pipeline (decomposed decoder, ResNet18 e200)

With the EdgeFirst twin model encoder (clean INT8, zero float islands),
decomposed decoder (prompt encoder in Rust, attention via XNNPACK, heads on
NPU, mask assembly on CPU), and HAL preprocessing:

| Platform | Encoder (ms) | Decoder (ms) | Total (ms) | IoU | Notes |
|----------|:------------:|:------------:|:----------:|:---:|-------|
| Jetson Orin Nano (MAXN_SUPER) | 15.3 | 6.3 | ~21.6 | 0.99 | TRT FP16 encoder + decoder |
| i.MX 95 (Neutron NPU) | 104.4 | 146.2 | ~331 | 0.986 | Twin model INT8, XNNPACK attention |
| i.MX 95 (CPU only, ONNX) | 2953 | 402 | ~3459 | 0.99 | Baseline without NPU |

See [BENCHMARKS.md](BENCHMARKS.md) for full methodology, per-stage
breakdown, and visual comparisons.

---

## Getting Started

### Environment Setup

Create a virtual environment and install PyTorch with the CUDA variant that matches your
driver. Check your driver version with `nvidia-smi` and pick the appropriate index URL:

| nvidia-smi CUDA column | Driver | PyTorch index |
|------------------------|--------|---------------|
| 12.x | ≥ 525 | `cu124` (recommended) |
| 11.x | ≥ 450 | `cu118` |
| CPU only | — | `cpu` |

```bash
git clone https://github.com/au-zone/nanosam -b edgefirst
cd nanosam
python3 -m venv venv && source venv/bin/activate

# Install PyTorch first — adjust cu124 to match your system (see table above)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Install remaining dependencies
pip install -r requirements.txt

# Install the nanosam package in editable mode
pip install -e .
```

### ONNX Runtime inference (CPU, any platform)

```bash
python3 scripts/run_inference_onnx.py \
    --image_encoder data/resnet18_image_encoder.onnx \
    --mask_decoder  data/mobile_sam_mask_decoder.onnx \
    --image         assets/dogs.jpg
```

### Raspberry Pi 5 + Hailo-8L

Requires the `hailo-all` system package (provides `hailo_platform` Python bindings).

```bash
python3 -m venv venv --system-site-packages
source venv/bin/activate
pip install onnxruntime pillow matplotlib numpy

python3 scripts/rpi5_hailo_inference.py \
    --encoder data/resnet18_encoder_h8l.hef \
    --decoder data/mobile_sam_mask_decoder.onnx \
    --image   assets/dogs.jpg \
    --output  out.jpg
```

### NXP i.MX 8M Plus (VeriSilicon NPU)

Requires `tflite-runtime` and `onnxruntime` on the target (both available via pip).
The VX delegate (`/usr/lib/libvx_delegate.so`) is provided by the NXP BSP and is
loaded automatically when present.

Copy files to the target, then run:

```bash
python3 scripts/imx8mp_tflite_inference.py \
    --encoder encoder_fixed_integer_quant.tflite \
    --decoder mobile_sam_mask_decoder.onnx \
    --image   dogs.jpg \
    --output  out.jpg
```

Use `--no-npu` to force CPU-only execution for debugging.

### Compile Hailo-8L HEF from ONNX

Requires [Hailo DFC 3.33.1](https://hailo.ai/developer-zone/) and COCO val2017 images.

```bash
source ~/path/to/hailo_model_zoo/venv/bin/activate

python3 nanosam/tools/compile_encoder_hailo.py \
    --onnx  data/resnet18_image_encoder.onnx \
    --coco  data/coco/val2017 \
    --output data/
```

---

## Distillation Training

Train a new ResNet18 encoder from scratch using MobileSAM (ViT-T) as the teacher.
Features are pre-extracted once on a GPU machine and cached as `.npy` files — no
TensorRT dependency, and training can resume on any CUDA machine.

### Step 1 — Export the mask decoder (once)

```bash
source venv/bin/activate

python3 nanosam/tools/export_sam_mask_decoder_onnx.py \
    --checkpoint  assets/mobile_sam.pt \
    --model-type  vit_t \
    --output      data/mobile_sam_mask_decoder.onnx \
    --return-single-mask
```

### Step 2 — Extract teacher features (run once, GPU required)

The teacher is `assets/mobile_sam.pt` (MobileSAM ViT-T). Features are saved in FP16
to keep storage manageable: ~118k COCO images produce approximately **230 GB** of `.npy` files.
Store them on a data volume, not `/home`.

```bash
python3 scripts/extract_sam_features.py \
    --checkpoint assets/mobile_sam.pt \
    --model_type  vit_t \
    --img_dir     /path/to/coco/train2017 \
    --out_dir     /datax/mobilesam-features-coco \
    --fp16
```

> **Note:** `--model_type vit_t` routes through the MobileSAM registry.
> The EfficientViT model types (`l0`, `l1`, `l2`, `xl0`, `xl1`) require a separate
> `efficientvit` package and a different checkpoint — do not use them with `mobile_sam.pt`.

### Step 3 — Train student encoder

It is recommended to validate the pipeline with a short smoke test before committing
to a full training run.

**Smoke test** — end-to-end pipeline check (~2 min):

```bash
python3 -m nanosam.tools.train \
    --images      /path/to/coco/train2017 \
    --features    /datax/mobilesam-features-coco \
    --output_dir  runs/resnet18_smoke \
    --model_name  resnet18 \
    --num_images  200 \
    --num_epochs  3 \
    --batch_size  16
```

**Convergence check** — verify loss is decreasing with a representative sample:

```bash
python3 -m nanosam.tools.train \
    --images      /path/to/coco/train2017 \
    --features    /datax/mobilesam-features-coco \
    --output_dir  runs/resnet18_mini \
    --model_name  resnet18 \
    --num_images  5000 \
    --num_epochs  20 \
    --batch_size  16
```

Loss should drop meaningfully by epoch 5–10. Check `runs/*/images/` for per-epoch
teacher vs student embedding visualisations.

**Full training run** (200 epochs, all 118k images):

```bash
python3 -m nanosam.tools.train \
    --images      /path/to/coco/train2017 \
    --features    /datax/mobilesam-features-coco \
    --output_dir  runs/resnet18_distill \
    --model_name  resnet18 \
    --num_epochs  200 \
    --batch_size  16
```

Training saves a checkpoint after every epoch and resumes automatically if interrupted.

Available student models: `resnet18`, `resnet34`, `resnet50`, `efficientvit_b0`, `efficientvit_b1`.

### Step 4 — Export to ONNX

```bash
python3 -m nanosam.tools.export_image_encoder_onnx \
    --model_name resnet18 \
    --checkpoint runs/resnet18_distill/checkpoint.pth \
    --output     data/resnet18_image_encoder.onnx
```

The exported ONNX model is ~61 MB with input shape `[1, 3, 1024, 1024]` and output
shape `[1, 256, 64, 64]`.

### Step 5 — Export to TFLite INT8 (for i.MX NPU targets)

Requires `onnx2tf`, which pulls in TensorFlow and will downgrade `onnx` and `numpy`
to versions compatible with TF. Install it in a separate step after training is complete,
or in a dedicated conversion environment.

```bash
pip install onnx2tf
```

Then convert with 200 COCO val images for calibration:

```bash
python3 -m nanosam.tools.export_encoder_tflite \
    --input          data/resnet18_image_encoder.onnx \
    --output_dir     data/encoder_tflite \
    --coco_root      /path/to/coco/val2017 \
    --num_calibration 200 \
    --int8
```

This produces several TFLite variants. The file for NPU deployment is
`encoder_fixed_integer_quant.tflite` (~16 MB). The model has float32 input/output
boundaries with INT8 internal quantisation.

> **Layout note:** onnx2tf converts internal ops to NHWC but preserves the original
> ONNX tensor names and NCHW layout at the model boundaries. The TFLite model
> therefore accepts NHWC input and returns NCHW output — no post-inference transpose
> is required before passing the embedding to the ONNX decoder.

---

## Evaluation

Evaluate with ONNX Runtime (no GPU required):

```bash
python3 -m nanosam.tools.eval_coco_onnx \
    --coco_root data/coco/val2017 \
    --coco_ann  data/coco/annotations/instances_val2017.json \
    --encoder   data/resnet18_image_encoder.onnx \
    --decoder   data/mobile_sam_mask_decoder.onnx \
    --output    data/resnet18_coco_results.json

python3 -m nanosam.tools.compute_eval_coco_metrics \
    data/resnet18_coco_results.json --size all
```

---

## Known Limitations

- **Decoder not convertible to TFLite or DVM** — the TwoWayTransformer uses 5D attention tensors that block NHWC transposition. Decoder runs ONNX on CPU on all NPU platforms.
- **Ara240 requires graph surgery** — ConvTranspose with `output_padding=[1,1]` crashes the Kinara scheduler. Workaround replaces it with Resize(nearest)+Conv and substitutes tanh for GELU.
- **NVIDIA distilled ResNet18 weights** — the original Google Drive link is private. Using ImageNet backbone weights for conversion/deployment validation.
- **TFLite full INT8 decoder fails on CPU** — Div op in GELU polynomial lacks an INT8 kernel. Works on i.MX NPU targets only.
- **onnx2tf degrades numpy and onnx** — installing `onnx2tf` downgrades `onnx` to 1.19.x and `numpy` to 1.26.x due to TensorFlow's dependency constraints. Use a separate conversion environment or install it last.

---

## Acknowledgements

- [NVIDIA-AI-IOT/nanosam](https://github.com/NVIDIA-AI-IOT/nanosam) — original NanoSAM
- [binh234/nanosam](https://github.com/binh234/nanosam) — EfficientViT encoders, ONNX Runtime support, SA-1B tooling (selectively integrated)
- [dragonSwing/nanosam](https://huggingface.co/dragonSwing/nanosam) — PPHGV2-B4 distilled encoder weights
- [SAM](https://github.com/facebookresearch/segment-anything) — Segment Anything Model (Meta AI)
- [MobileSAM](https://github.com/ChaoningZhang/MobileSAM) — distilled TinyViT SAM encoder
