# NanoSAM — EdgeFirstAI Fork

## Project Context

- **Epic:** [EDGEAI-1138 — HuggingFace SAM Models](https://au-zone.atlassian.net/browse/EDGEAI-1138)
- **This repo's ticket:** [EDGEAI-1204 — NanoSAM encoder/decoder export and x86_64 validation](https://au-zone.atlassian.net/browse/EDGEAI-1204)
- **Upstream:** https://github.com/NVIDIA-AI-IOT/nanosam
- **Branch:** `edgefirst` — all EdgeFirstAI work; `main` tracks NVIDIA upstream clean

## Workspace

`~/Software/SAM/` contains per-model clones for the SAM/segmentation evaluation sprint.
The Hailo DFC venv is at `~/Software/Studio/hailo/hailo_model_zoo/venv/` (not `./venv/`).

## Architecture

| Component | Architecture | Edge deployment |
|-----------|-------------|----------------|
| Encoder | ResNet18 (~11M) | NPU (Hailo HEF, future: i.MX NPU, Ara240) |
| Decoder | MobileSAM TwoWayTransformer | CPU only — attention layers block NPU conversion |

Split design: encoder runs once per image, decoder runs per prompt. Decoder is the
bottleneck on slow ARM cores (430ms on Cortex-A55, 63ms on Cortex-A78AE).

## Completed Work

- ONNX export: encoder (ResNet18, PPHGV2-B4) + decoder, validated IoU > 0.99 vs PyTorch
- TFLite INT8 conversion (encoder + decoder)
- Hailo-8L HEF compilation via DFC 3.33.1 (`nanosam/tools/compile_encoder_hailo.py`)
- RPi5 + Hailo-8L inference script (`scripts/rpi5_hailo_inference.py`)
- COCO mIoU evaluation with ONNX Runtime (`nanosam/tools/eval_coco_onnx.py`)
- Decoder latency benchmarked across: x86_64, Jetson Orin Nano, RPi5, i.MX 8M Plus, i.MX 95
- Distillation training pipeline (TRT-free, pre-extracted features) from binh234 fork

## Next: Distillation

Goal: train a new encoder targeting PPHGV2-B4 quality with Hailo-8L compatibility.
Pipeline: `scripts/extract_sam_features.py` (GPU, once) → `nanosam/tools/train.py` (CPU/GPU).
No TRT dependency. Teacher features cached as `.npy` files.

## i.MX NPU / Ara240 Status

Not yet attempted. Encoder is a pure CNN (no attention) — suitable candidates.
Decoder requires CPU fallback on all NPU platforms.

## Environment

- Local Python venv: `source venv/bin/activate` (created with `--system-site-packages` on RPi5)
- Hailo DFC: `source ~/Software/Studio/hailo/hailo_model_zoo/venv/bin/activate`
