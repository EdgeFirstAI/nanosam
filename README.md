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

### Encoder + Decoder (ResNet18, initial measurements)

| Platform | Accelerator | Format | Encoder (ms) | Mask IoU vs ONNX | Decoder CPU (ms) | Total (ms) |
|----------|-------------|--------|:------------:|:----------------:|:----------------:|:----------:|
| x86_64 | ONNX Runtime | ONNX FP32 (61 MB) | ~200 | baseline | 15.8 | ~216 |
| Jetson Orin Nano Super | NVIDIA GPU | TRT FP16 (33 MB) | 12.8 | 0.9999 | 63.3 | ~76 |
| Hailo-8L (RPi5) | Hailo-8L NPU | HEF INT8 (36 MB) | 46 | 0.9654 | 125 | ~171 |
| i.MX 95 + Ara240 | Kinara Ara-2 NPU | DVM INT8 (27 MB) | 45.3 | 0.9961 | 430 | ~475 |
| i.MX 95 | Neutron NPU | TFLite INT8 (16 MB) | 178 | 0.9784 | 430 | ~608 |
| i.MX 8M Plus | VeriSilicon NPU | TFLite INT8 (15 MB) | 335 | 0.9972 | 421 | ~756 |

---

## Getting Started

### ONNX Runtime (CPU, any platform)

```bash
git clone https://github.com/au-zone/nanosam -b edgefirst
cd nanosam
python3 -m venv venv && source venv/bin/activate
pip install -e ".[onnx]"

python3 scripts/basic_usage.py \
    --image_encoder data/resnet18_image_encoder_legacy.onnx \
    --mask_decoder  data/mobile_sam_mask_decoder.onnx
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

### Compile Hailo-8L HEF from ONNX

Requires [Hailo DFC 3.33.1](https://hailo.ai/developer-zone/) and COCO val2017 images.

```bash
source ~/path/to/hailo_model_zoo/venv/bin/activate

python3 nanosam/tools/compile_encoder_hailo.py \
    --onnx  data/resnet18_image_encoder_legacy.onnx \
    --coco  data/coco/val2017 \
    --output data/
```

---

## Distillation Training

Train a new encoder without TensorRT. Pre-extract teacher features once, then train anywhere.

**Step 1 — Extract teacher features** (GPU machine, run once)

```bash
python3 scripts/extract_sam_features.py \
    --checkpoint assets/mobile_sam.pt \
    --model_type  vit_t \
    --img_dir     data/coco/train2017 \
    --out_dir     data/coco/features_vit_t
```

**Step 2 — Train student encoder**

```bash
python3 -m nanosam.tools.train \
    --images      data/coco/train2017 \
    --features    data/coco/features_vit_t \
    --output_dir  runs/resnet18_distill \
    --model_name  resnet18 \
    --batch_size  16
```

Available student models: `resnet18`, `efficientvit_b0_sam`, `efficientvit_b1_sam`, `efficientvit_b2_sam`.

**Step 3 — Export to ONNX**

```bash
python3 -m nanosam.tools.export_image_encoder_onnx \
    --model_name resnet18 \
    --checkpoint runs/resnet18_distill/checkpoint.pth \
    --output     data/resnet18_image_encoder.onnx
```

---

## Evaluation

Evaluate with ONNX Runtime (no GPU required):

```bash
python3 -m nanosam.tools.eval_coco_onnx \
    --coco_root data/coco/val2017 \
    --coco_ann  data/coco/annotations/instances_val2017.json \
    --encoder   data/resnet18_image_encoder_legacy.onnx \
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

---

## Acknowledgements

- [NVIDIA-AI-IOT/nanosam](https://github.com/NVIDIA-AI-IOT/nanosam) — original NanoSAM
- [binh234/nanosam](https://github.com/binh234/nanosam) — EfficientViT encoders, ONNX Runtime support, SA-1B tooling (selectively integrated)
- [dragonSwing/nanosam](https://huggingface.co/dragonSwing/nanosam) — PPHGV2-B4 distilled encoder weights
- [SAM](https://github.com/facebookresearch/segment-anything) — Segment Anything Model (Meta AI)
- [MobileSAM](https://github.com/ChaoningZhang/MobileSAM) — distilled TinyViT SAM encoder
