# EdgeFirst NanoSAM: End-to-End Benchmarks

## Summary

| Platform | Config | Total Latency | Notes |
|----------|--------|--------------|-------|
| Jetson Orin Nano | TensorRT FP16 | 21.6 ms | GPU reference |
| i.MX 95 | Vanilla CPU (ONNX) | 3,459 ms | No NPU — baseline |
| i.MX 95 | Naive onnx2tf → NPU | 697 ms | Runs but garbage masks |
| i.MX 95 | **EdgeFirst NPU** | **331 ms** | **10.5x speedup** |

| Vanilla CPU (Correct) | Naive onnx2tf → NPU (Broken) | EdgeFirst NPU (Correct) |
|:---:|:---:|:---:|
| ![CPU](assets/benchmark/imx95_vanilla_cpu.jpg) | ![Naive NPU](assets/benchmark/imx95_naive_npu.jpg) | ![EdgeFirst](assets/benchmark/imx95_edgefirst.jpg) |

> **Note:** The INT8 encoder mask has visible quality loss compared to
> the float model — missing coverage on parts of the dog's body.

## Methodology

- **Test image:** `dogs.jpg` (1180x760 RGB)
- **Prompt:** Bounding box [100, 100, 850, 759]
- **Protocol:** 10 warmup + 100 timed runs
- **Timing:** `time.perf_counter()` (Python) / `std::time::Instant` (Rust), excludes JPEG decode

| | Jetson Orin Nano | NXP i.MX 95 EVK |
|---|---|---|
| **CPU** | 6-core Cortex-A78AE | 6-core Cortex-A55 |
| **Accelerator** | Ampere GPU | Neutron NPU |
| **Memory** | 8 GB LPDDR5 | 16 GB LPDDR5 |
| **Power Mode** | MAXN_SUPER + jetson_clocks | performance governor |
| **SW** | JetPack 6.2, TensorRT 10.3, CUDA 12.5 | BSP 6.12-walnascar, Neutron v1.0.0 |

## Decoder Decomposition

The standard SAM decoder is a monolithic block that works on GPU (TensorRT)
but cannot be quantized to INT8 for NPU deployment. Full INT8 quantization
fails — softmax and LayerNorm amplify quantization error, achieving only
48.2% binary mask IoU vs 1.0 for FP32.

EdgeFirst splits the decoder at the transformer boundary:

```mermaid
graph TD
    A[Input Image] --> B[HAL Preprocess<br/>GPU letterbox + fused quant]
    B --> C[Image Encoder<br/>INT8 on NPU]

    E[Point/Box Prompt] --> F[Prompt Encoder<br/>Rust native]

    C --> G[Attention<br/>CPU / XNNPACK FP16]
    F --> G

    G --> H[Heads A+B<br/>INT8 on NPU]
    H --> I[Tokens + Mask Assembly<br/>CPU]
    I --> J[Postprocess<br/>HAL GPU upscale]

    style B fill:#0066cc,color:#fff
    style C fill:#cc3300,color:#fff
    style F fill:#666,color:#fff
    style G fill:#666,color:#fff
    style H fill:#cc3300,color:#fff
    style I fill:#666,color:#fff
    style J fill:#0066cc,color:#fff
```

The encoder uses a **twin model** (Keras rebuild with tanh-approximate GELU,
BN fusion) to eliminate the float islands that break naive onnx2tf conversion.

## Jetson Orin Nano — TensorRT FP16

100 runs, 10 warmup, MAXN_SUPER.

| Stage | Mean (ms) | Median (ms) | Std (ms) | Min (ms) | Max (ms) | P95 (ms) | P99 (ms) |
|-------|-----------|-------------|----------|----------|----------|----------|----------|
| Encoder (TRT FP16) | 15.27 | 15.26 | 0.10 | 15.08 | 15.63 | 15.42 | 15.62 |
| Decoder (TRT FP16) | 6.29 | 6.25 | 0.22 | 6.07 | 8.15 | 6.48 | 6.89 |
| **Total** | **21.56** | **21.52** | **0.26** | **21.15** | **23.53** | **21.87** | **22.08** |

## NXP i.MX 95

### Vanilla CPU (ONNX Runtime)

50 runs, 5 warmup.

| Stage | Mean (ms) | Median (ms) | Std (ms) | Min (ms) | Max (ms) | P95 (ms) | P99 (ms) |
|-------|-----------|-------------|----------|----------|----------|----------|----------|
| Preprocess | 103.47 | 103.41 | 1.79 | 101.25 | 106.91 | 105.60 | 106.45 |
| Encoder (ONNX CPU) | 2953.25 | 2952.12 | 5.23 | 2947.20 | 2979.98 | 2962.60 | 2972.08 |
| Decoder (ONNX CPU) | 402.07 | 402.05 | 2.70 | 396.68 | 409.30 | 406.23 | 408.04 |
| **Total** | **3458.79** | **3458.16** | **5.73** | **3450.30** | **3484.65** | **3467.87** | **3476.66** |

### Naive onnx2tf → Neutron

Naive conversion produces 37 float32 islands from GELU/erf. After Neutron
SDK compilation, 8 float ops remain on CPU (96.4% conversion ratio). The
model runs (encoder 198 ms, decoder 366 ms, total 697 ms) but produces
garbage masks (coverage 4.8% vs ~29%).

### EdgeFirst Optimized (Neutron NPU + XNNPACK)

100 runs, 10 warmup. Averages (per-run distribution not available from Rust CLI).

| Stage | ms |
|-------|----|
| Preprocess (HAL GPU) | 67.4 |
| Encoder (Neutron INT8) | 104.4 |
| Prompt Encoder (Rust) | 1.3 |
| Attention (XNNPACK FP16) | 103.2 |
| Heads A+B (Neutron INT8) | 5.8 |
| Tokens (CPU) | 0.6 |
| Mask Assembly (CPU) | 35.3 |
| Postprocess (HAL GPU) | 12.8 |
| **Total** | **330.7** |

## Reproduction

### Jetson Orin Nano

```bash
sudo nvpmodel -m 2 && sudo jetson_clocks

trtexec --onnx=data/resnet18_image_encoder.onnx \
  --saveEngine=data/resnet18_image_encoder.engine --fp16
trtexec --onnx=data/mobile_sam_mask_decoder.onnx \
  --saveEngine=data/mobile_sam_mask_decoder.engine --fp16 \
  --optShapes=point_coords:1x2x2,point_labels:1x2

python3 scripts/benchmark_jetson_nvidia.py \
  --image assets/dogs.jpg \
  --encoder data/resnet18_image_encoder.engine \
  --decoder data/mobile_sam_mask_decoder.engine \
  --box 100 100 850 759 --warmup 10 --runs 100
```

### NXP i.MX 95

```bash
echo performance | sudo tee /sys/devices/system/cpu/cpufreq/policy*/scaling_governor

# Vanilla CPU
python3 scripts/benchmark_imx95_vanilla.py cpu \
  --image assets/dogs.jpg \
  --encoder data/resnet18_image_encoder.onnx \
  --decoder data/mobile_sam_mask_decoder.onnx \
  --box 100 100 850 759 --warmup 5 --runs 50

# EdgeFirst optimized
./nanosam bench \
  --models nanosam_imx95.zip \
  --image assets/dogs.jpg \
  --box 100 100 850 759 \
  --delegate /usr/lib/libneutron_delegate.so \
  --xnnpack \
  --warmup 10 --runs 100
```
