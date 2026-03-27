# CLAUDE.md — NanoSAM

## Project Context

- **Epic:** [EDGEAI-1138 — HuggingFace SAM Models](https://au-zone.atlassian.net/browse/EDGEAI-1138)
- **This repo's ticket:** [EDGEAI-1204 — NanoSAM encoder/decoder export and x86_64 validation](https://au-zone.atlassian.net/browse/EDGEAI-1204)
- **Parent task:** [EDGEAI-1193 — Source model export and x86_64 validation baseline](https://au-zone.atlassian.net/browse/EDGEAI-1193)
- **Upstream:** https://github.com/NVIDIA-AI-IOT/nanosam

## Workspace

This repo is part of `~/Software/SAM/` which contains per-model clones for the SAM/segmentation evaluation sprint.

## Model

| Component | Architecture | Notes |
|-----------|-------------|-------|
| Encoder | ResNet18 (~11M total) | Pure CNN — best candidate for Hailo NPU compilation |
| Decoder | MobileSAM decoder | Contains attention layers — CPU fallback on NPU platforms |

Split encoder/decoder architecture: encoder runs once per image, decoder runs per prompt. Distilled from MobileSAM, optimized for NVIDIA TensorRT on Jetson.

## Per-Model Checklist

1. Install from source repo
2. Run inference on 5–10 test images with point/box prompts, save output masks
3. Export encoder (ResNet18) and decoder as **separate** ONNX files
4. Validate ONNX output matches PyTorch output (binary mask IoU > 0.99)
5. Record: export command, input/output tensor shapes per component, file sizes
6. Build TensorRT FP16 engines for both encoder and decoder
7. Build TensorRT INT8 engines
8. Validate TRT output against PyTorch reference

## Known Issues

- MIT license — no distribution concerns
- ResNet18 encoder is pure CNN — **novel attempt** to compile on Hailo NPU (nobody has publicly tried)
- Decoder contains attention layers — will need CPU fallback on NPU platforms
- Low maintenance activity in the repo — expect to work with the code as-is

## Environment

- Always use a local Python venv: `python -m venv venv && source venv/bin/activate`
