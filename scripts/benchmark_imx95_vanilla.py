#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 Au-Zone Technologies
# SPDX-License-Identifier: Apache-2.0
#
# Vanilla NanoSAM benchmark on NXP i.MX 95 EVK.
#
# Two modes:
#   cpu  — Full ONNX pipeline on Cortex-A55 CPU (M1a: slow but correct baseline)
#   npu  — Naive onnx2tf INT8 encoder on Neutron NPU + ONNX decoder (M1b: broken)
#
# Usage:
#   python3 scripts/benchmark_imx95_vanilla.py cpu \
#       --image assets/dogs.jpg \
#       --encoder data/resnet18_image_encoder.onnx \
#       --decoder data/mobile_sam_mask_decoder.onnx \
#       --box 100 100 850 759 --warmup 5 --runs 50
#
#   python3 scripts/benchmark_imx95_vanilla.py npu \
#       --image assets/dogs.jpg \
#       --encoder-tflite data/resnet18_encoder_naive_int8.tflite \
#       --decoder data/mobile_sam_mask_decoder.onnx \
#       --box 100 100 850 759 --warmup 3 --runs 20

import argparse
import os
import sys
import time

import numpy as np
import PIL.Image
import PIL.ImageDraw

# ---------------------------------------------------------------------------
# ImageNet normalisation constants
# ---------------------------------------------------------------------------
IMAGE_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
IMAGE_STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)


# ---------------------------------------------------------------------------
# Preprocessing — self-contained, no nanosam package dependency
# ---------------------------------------------------------------------------

def preprocess_image_nchw(image: PIL.Image.Image, size: int = 1024) -> np.ndarray:
    """Aspect-ratio resize, ImageNet normalise, zero-pad.  Returns NCHW float32."""
    w, h = image.size
    aspect = w / h
    if aspect >= 1:
        rw, rh = size, int(size / aspect)
    else:
        rw, rh = int(size * aspect), size

    resized = image.resize((rw, rh), PIL.Image.BILINEAR)
    arr = np.asarray(resized, dtype=np.float32).transpose(2, 0, 1)  # HWC -> CHW
    mean = IMAGE_MEAN.reshape(3, 1, 1)
    std = IMAGE_STD.reshape(3, 1, 1)
    arr = (arr - mean) / std

    padded = np.zeros((1, 3, size, size), dtype=np.float32)
    padded[0, :, :rh, :rw] = arr
    return padded


def preprocess_image_nhwc(image: PIL.Image.Image, size: int = 1024) -> np.ndarray:
    """Aspect-ratio resize, ImageNet normalise, zero-pad.  Returns NHWC float32."""
    w, h = image.size
    aspect = w / h
    if aspect >= 1:
        rw, rh = size, int(size / aspect)
    else:
        rw, rh = int(size * aspect), size

    resized = image.resize((rw, rh), PIL.Image.BILINEAR)
    arr = np.asarray(resized, dtype=np.float32)  # HWC
    arr = (arr - IMAGE_MEAN) / IMAGE_STD

    padded = np.zeros((1, size, size, 3), dtype=np.float32)
    padded[0, :rh, :rw, :] = arr
    return padded


def preprocess_points(points: np.ndarray, image_size: tuple,
                      size: int = 1024) -> np.ndarray:
    """Scale (x, y) point coords to encoder input space."""
    scale = size / max(image_size)
    return (points * scale).astype(np.float32)


def upscale_mask(low_res_mask: np.ndarray, image_shape: tuple,
                 size: int = 256) -> np.ndarray:
    """Upscale decoder low-res mask (size x size) to original image dims."""
    h, w = image_shape
    if w > h:
        lim_x, lim_y = size, int(size * h / w)
    else:
        lim_x, lim_y = int(size * w / h), size

    cropped = low_res_mask[:lim_y, :lim_x]
    pil_mask = PIL.Image.fromarray(cropped, mode="F")
    upscaled = pil_mask.resize((w, h), PIL.Image.BILINEAR)
    return np.asarray(upscaled, dtype=np.float32)


# ---------------------------------------------------------------------------
# Statistics helpers (matches Jetson benchmark format)
# ---------------------------------------------------------------------------

def compute_stats(times_ms: list) -> dict:
    arr = np.array(times_ms)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
    }


def print_stats(label: str, stats: dict):
    print(
        f"{label:>15s}: "
        f"mean={stats['mean']:.2f}ms  "
        f"median={stats['median']:.2f}ms  "
        f"std={stats['std']:.2f}ms  "
        f"min={stats['min']:.2f}ms  "
        f"max={stats['max']:.2f}ms  "
        f"p95={stats['p95']:.2f}ms  "
        f"p99={stats['p99']:.2f}ms"
    )


# ---------------------------------------------------------------------------
# Decoder wrapper (ONNX, always on CPU)
# ---------------------------------------------------------------------------

class OnnxDecoder:
    """MobileSAM mask decoder via ONNX Runtime."""

    def __init__(self, decoder_path: str):
        import onnxruntime as ort
        self.session = ort.InferenceSession(
            decoder_path, providers=["CPUExecutionProvider"]
        )
        print(f"  Decoder inputs:  {[i.name for i in self.session.get_inputs()]}")
        print(f"  Decoder outputs: {[o.name for o in self.session.get_outputs()]}")

    def predict(self, embedding: np.ndarray, points: np.ndarray,
                point_labels: np.ndarray) -> tuple:
        inputs = {
            "image_embeddings": embedding,
            "point_coords": np.array([points], dtype=np.float32),
            "point_labels": np.array([point_labels], dtype=np.float32),
            "mask_input": np.zeros((1, 1, 256, 256), dtype=np.float32),
            "has_mask_input": np.array([0.0], dtype=np.float32),
        }
        iou_preds, low_res_masks = self.session.run(None, inputs)
        best_idx = int(iou_preds[0].argmax())
        return iou_preds, low_res_masks[0, best_idx]


# ---------------------------------------------------------------------------
# Save result overlay
# ---------------------------------------------------------------------------

def save_result(image: PIL.Image.Image, mask: np.ndarray,
                bbox: list, output_path: str, color: tuple = (255, 220, 0)):
    """Overlay mask (50% opacity) and bounding box on the original image."""
    overlay = PIL.Image.new("RGBA", image.size, (0, 0, 0, 0))
    mask_pil = PIL.Image.fromarray((mask * 180).astype(np.uint8), mode="L")
    color_layer = PIL.Image.new("RGBA", image.size, (*color, 0))
    color_layer.putalpha(mask_pil)
    overlay = PIL.Image.alpha_composite(overlay, color_layer)

    result = image.convert("RGBA")
    result = PIL.Image.alpha_composite(result, overlay)
    result = result.convert("RGB")

    draw = PIL.ImageDraw.Draw(result)
    draw.rectangle([bbox[0], bbox[1], bbox[2], bbox[3]],
                    outline=(0, 200, 0), width=3)

    result.save(output_path, quality=95)
    print(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# M1a: Full ONNX pipeline on CPU
# ---------------------------------------------------------------------------

def run_cpu_benchmark(args):
    import onnxruntime as ort

    print(f"\n{'='*80}")
    print("M1a: Vanilla NanoSAM — ONNX Runtime on i.MX 95 CPU (Cortex-A55)")
    print(f"{'='*80}")

    # Load models
    print(f"\nLoading encoder: {args.encoder}")
    enc_session = ort.InferenceSession(
        args.encoder, providers=["CPUExecutionProvider"]
    )
    enc_input_name = enc_session.get_inputs()[0].name
    enc_input_shape = enc_session.get_inputs()[0].shape
    print(f"  Encoder input:  {enc_input_name} {enc_input_shape}")
    print(f"  Encoder output: {enc_session.get_outputs()[0].name} "
          f"{enc_session.get_outputs()[0].shape}")

    print(f"\nLoading decoder: {args.decoder}")
    decoder = OnnxDecoder(args.decoder)

    # Load and preprocess
    image = PIL.Image.open(args.image).convert("RGB")
    print(f"\nImage: {image.width}x{image.height}")

    image_tensor = preprocess_image_nchw(image)
    print(f"Preprocessed: {image_tensor.shape} {image_tensor.dtype}")

    # Prepare box prompt
    bbox = args.box
    points = np.array([[bbox[0], bbox[1]], [bbox[2], bbox[3]]], dtype=np.float32)
    point_labels = np.array([2, 3], dtype=np.float32)
    scaled_points = preprocess_points(points, (image.height, image.width))

    # Warmup
    print(f"\nWarming up ({args.warmup} iterations)...")
    for i in range(args.warmup):
        t0 = time.perf_counter()
        enc_out = enc_session.run(None, {enc_input_name: image_tensor})
        embedding = enc_out[0]
        _, _ = decoder.predict(embedding, scaled_points, point_labels)
        dt = (time.perf_counter() - t0) * 1000
        print(f"  warmup {i+1}: {dt:.0f} ms")

    # Benchmark
    print(f"\nRunning benchmark ({args.runs} iterations)...")
    preprocess_times = []
    encoder_times = []
    decoder_times = []
    total_times = []

    for i in range(args.runs):
        t_total_start = time.perf_counter()

        # Preprocess (re-run to include in timing)
        t0 = time.perf_counter()
        img_t = preprocess_image_nchw(image)
        t1 = time.perf_counter()

        # Encoder
        enc_out = enc_session.run(None, {enc_input_name: img_t})
        embedding = enc_out[0]
        t2 = time.perf_counter()

        # Decoder
        iou_preds, best_mask_lr = decoder.predict(embedding, scaled_points,
                                                   point_labels)
        t3 = time.perf_counter()

        pre_ms = (t1 - t0) * 1000
        enc_ms = (t2 - t1) * 1000
        dec_ms = (t3 - t2) * 1000
        tot_ms = (t3 - t_total_start) * 1000

        preprocess_times.append(pre_ms)
        encoder_times.append(enc_ms)
        decoder_times.append(dec_ms)
        total_times.append(tot_ms)

        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{i+1:3d}/{args.runs}] "
                  f"pre={pre_ms:.0f}ms  enc={enc_ms:.0f}ms  "
                  f"dec={dec_ms:.0f}ms  total={tot_ms:.0f}ms")

    # Results
    print(f"\n{'='*80}")
    print(f"M1a: Vanilla NanoSAM — ONNX Runtime CPU Benchmark Results")
    print(f"  Encoder: {args.encoder}")
    print(f"  Decoder: {args.decoder}")
    print(f"  Image: {args.image} ({image.width}x{image.height})")
    print(f"  Bounding box: {bbox}")
    print(f"  Warmup: {args.warmup}, Runs: {args.runs}")
    print(f"{'='*80}")

    pre_stats = compute_stats(preprocess_times)
    enc_stats = compute_stats(encoder_times)
    dec_stats = compute_stats(decoder_times)
    tot_stats = compute_stats(total_times)

    print_stats("preprocess", pre_stats)
    print_stats("encoder", enc_stats)
    print_stats("decoder", dec_stats)
    print_stats("total", tot_stats)

    fps = 1000.0 / tot_stats["mean"]
    print(f"\n  Effective FPS (mean): {fps:.2f}")

    # Embedding diagnostics
    print(f"\n  Embedding shape: {embedding.shape}")
    print(f"  Embedding range: [{embedding.min():.4f}, {embedding.max():.4f}]")
    print(f"  Embedding mean:  {embedding.mean():.4f}")
    print(f"  Best IoU: {iou_preds[0].max():.4f}")
    print(f"{'='*80}")

    # Save mask overlay (yellow = correct baseline)
    hi_res = upscale_mask(best_mask_lr, (image.height, image.width))
    mask = (hi_res > 0).astype(np.float32)
    coverage = mask.sum() / mask.size * 100
    print(f"  Mask coverage: {mask.sum():.0f}/{mask.size} pixels ({coverage:.1f}%)")

    output_path = args.output or "vanilla_cpu_mask.jpg"
    save_result(image, mask, bbox, output_path, color=(255, 220, 0))


# ---------------------------------------------------------------------------
# M1b: Naive onnx2tf INT8 on Neutron NPU
# ---------------------------------------------------------------------------

def run_npu_benchmark(args):
    from tflite_runtime.interpreter import Interpreter, load_delegate

    print(f"\n{'='*80}")
    print("M1b: Vanilla NanoSAM — Naive onnx2tf INT8 on i.MX 95 Neutron NPU")
    print(f"{'='*80}")

    encoder_path = args.encoder_tflite
    if not encoder_path or not os.path.exists(encoder_path):
        print(f"\nERROR: Naive INT8 TFLite encoder not found: {encoder_path}")
        print("The onnx2tf conversion of NanoSAM's ResNet18 encoder produces")
        print("float islands in GELU activation conversion, resulting in:")
        print("  - Cosine similarity: 0.177 (vs 0.925+ for twin architecture)")
        print("  - Broken/garbage segmentation masks")
        print("  - Neutron delegate rejects or partially offloads the model")
        print("\nThis is the expected failure that motivates the EdgeFirst twin.")
        sys.exit(1)

    # Find Neutron delegate
    neutron_paths = [
        "/usr/lib/libneutron_delegate.so",
        "/usr/lib/liblitert_neutron_delegate.so",
    ]
    neutron_lib = None
    for p in neutron_paths:
        if os.path.exists(p):
            neutron_lib = p
            break

    if not neutron_lib:
        print(f"\nERROR: Neutron delegate not found at: {neutron_paths}")
        sys.exit(1)

    print(f"\nNeutron delegate: {neutron_lib}")
    print(f"Loading encoder:  {encoder_path}")

    # Try loading with Neutron delegate
    try:
        delegates = [load_delegate(neutron_lib)]
        print("  Neutron delegate loaded successfully")
    except Exception as e:
        print(f"  Neutron delegate failed to load: {e}")
        print("  Falling back to CPU-only TFLite (will be very slow)")
        delegates = []

    try:
        interpreter = Interpreter(
            model_path=encoder_path,
            experimental_delegates=delegates,
        )
        interpreter.allocate_tensors()
    except Exception as e:
        print(f"\nFATAL: Failed to load/allocate TFLite model: {e}")
        print("This is the expected failure — naive onnx2tf INT8 models contain")
        print("float islands that the Neutron delegate cannot handle.")
        sys.exit(1)

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    print(f"  Input:  {input_details['name']}  shape={input_details['shape']}  "
          f"dtype={input_details['dtype'].__name__}")
    print(f"  Output: {output_details['name']}  shape={output_details['shape']}  "
          f"dtype={output_details['dtype'].__name__}")

    # Check if input is quantized
    input_dtype = input_details['dtype']
    input_quant = input_details.get('quantization_parameters', {})
    input_scales = input_quant.get('scales', np.array([1.0]))
    input_scale = input_scales[0] if len(input_scales) > 0 else 1.0
    input_zps = input_quant.get('zero_points', np.array([0]))
    input_zp = input_zps[0] if len(input_zps) > 0 else 0
    output_quant = output_details.get('quantization_parameters', {})
    output_scales = output_quant.get('scales', np.array([1.0]))
    output_scale = output_scales[0] if len(output_scales) > 0 else 1.0
    output_zps = output_quant.get('zero_points', np.array([0]))
    output_zp = output_zps[0] if len(output_zps) > 0 else 0

    if input_dtype == np.int8:
        print(f"  Input quantization: scale={input_scale}, zero_point={input_zp}")
        print(f"  Output quantization: scale={output_scale}, zero_point={output_zp}")

    # Load decoder
    print(f"\nLoading decoder: {args.decoder}")
    decoder = OnnxDecoder(args.decoder)

    # Load and preprocess
    image = PIL.Image.open(args.image).convert("RGB")
    print(f"\nImage: {image.width}x{image.height}")

    # onnx2tf models expect NHWC
    image_tensor = preprocess_image_nhwc(image)

    # Quantize input if model expects INT8
    if input_dtype == np.int8:
        image_tensor = np.clip(
            np.round(image_tensor / input_scale + input_zp), -128, 127
        ).astype(np.int8)
        print(f"Preprocessed (NHWC, INT8): {image_tensor.shape}")
    else:
        print(f"Preprocessed (NHWC): {image_tensor.shape}")

    # Prepare box prompt
    bbox = args.box
    points = np.array([[bbox[0], bbox[1]], [bbox[2], bbox[3]]], dtype=np.float32)
    point_labels = np.array([2, 3], dtype=np.float32)
    scaled_points = preprocess_points(points, (image.height, image.width))

    # Warmup
    print(f"\nWarming up ({args.warmup} iterations)...")
    for i in range(args.warmup):
        t0 = time.perf_counter()
        interpreter.set_tensor(input_details['index'], image_tensor)
        interpreter.invoke()
        raw_out = interpreter.get_tensor(output_details['index'])
        dt = (time.perf_counter() - t0) * 1000
        if i == 0:
            # Diagnostic: show raw output stats
            print(f"  raw output dtype={raw_out.dtype}  shape={raw_out.shape}  "
                  f"range=[{raw_out.min()}, {raw_out.max()}]")
        print(f"  warmup {i+1}: {dt:.0f} ms")

    # Benchmark
    print(f"\nRunning benchmark ({args.runs} iterations)...")
    preprocess_times = []
    encoder_times = []
    decoder_times = []
    total_times = []

    for i in range(args.runs):
        t_total_start = time.perf_counter()

        # Preprocess
        t0 = time.perf_counter()
        img_t = preprocess_image_nhwc(image)
        if input_dtype == np.int8:
            img_t = np.clip(
                np.round(img_t / input_scale + input_zp), -128, 127
            ).astype(np.int8)
        t1 = time.perf_counter()

        # Encoder (TFLite, hopefully on NPU)
        interpreter.set_tensor(input_details['index'], img_t)
        interpreter.invoke()
        embedding = interpreter.get_tensor(output_details['index'])
        # Dequantize output if needed
        if output_details['dtype'] == np.int8:
            embedding = ((embedding.astype(np.float32) - float(output_zp))
                         * np.float32(output_scale))
        else:
            embedding = embedding.astype(np.float32)
        t2 = time.perf_counter()

        # Transpose NHWC->NCHW if needed for ONNX decoder
        # onnx2tf may produce NHWC output: (1, H, W, C) where C=256, H=W=64
        if (embedding.ndim == 4 and embedding.shape[1] != 256
                and embedding.shape[-1] == 256):
            embedding = embedding.transpose(0, 3, 1, 2)

        # Decoder (ONNX CPU)
        iou_preds, best_mask_lr = decoder.predict(embedding, scaled_points,
                                                   point_labels)
        t3 = time.perf_counter()

        pre_ms = (t1 - t0) * 1000
        enc_ms = (t2 - t1) * 1000
        dec_ms = (t3 - t2) * 1000
        tot_ms = (t3 - t_total_start) * 1000

        preprocess_times.append(pre_ms)
        encoder_times.append(enc_ms)
        decoder_times.append(dec_ms)
        total_times.append(tot_ms)

        if (i + 1) % 5 == 0 or i == 0:
            print(f"  [{i+1:3d}/{args.runs}] "
                  f"pre={pre_ms:.0f}ms  enc={enc_ms:.0f}ms  "
                  f"dec={dec_ms:.0f}ms  total={tot_ms:.0f}ms")

    # Results
    print(f"\n{'='*80}")
    print(f"M1b: Naive onnx2tf INT8 on Neutron — Benchmark Results")
    print(f"  Encoder: {encoder_path}")
    print(f"  Decoder: {args.decoder}")
    print(f"  Neutron: {neutron_lib}")
    print(f"  Image: {args.image} ({image.width}x{image.height})")
    print(f"  Bounding box: {bbox}")
    print(f"  Warmup: {args.warmup}, Runs: {args.runs}")
    print(f"{'='*80}")

    pre_stats = compute_stats(preprocess_times)
    enc_stats = compute_stats(encoder_times)
    dec_stats = compute_stats(decoder_times)
    tot_stats = compute_stats(total_times)

    print_stats("preprocess", pre_stats)
    print_stats("encoder", enc_stats)
    print_stats("decoder", dec_stats)
    print_stats("total", tot_stats)

    fps = 1000.0 / tot_stats["mean"]
    print(f"\n  Effective FPS (mean): {fps:.2f}")

    # Embedding diagnostics
    print(f"\n  Embedding shape: {embedding.shape}")
    print(f"  Embedding range: [{embedding.min():.4f}, {embedding.max():.4f}]")
    print(f"  Embedding mean:  {embedding.mean():.4f}")
    print(f"  Best IoU: {iou_preds[0].max():.4f}")

    # Quality assessment
    best_iou = float(iou_preds[0].max())
    if best_iou < 0.5:
        print(f"\n  WARNING: Best IoU {best_iou:.4f} < 0.5 — mask quality is BROKEN")
        print(f"  This confirms naive onnx2tf INT8 produces garbage masks")
    print(f"{'='*80}")

    # Save mask overlay (red = broken/naive NPU)
    hi_res = upscale_mask(best_mask_lr, (image.height, image.width))
    mask = (hi_res > 0).astype(np.float32)
    coverage = mask.sum() / mask.size * 100
    print(f"  Mask coverage: {mask.sum():.0f}/{mask.size} pixels ({coverage:.1f}%)")

    output_path = args.output or "naive_npu_mask.jpg"
    save_result(image, mask, bbox, output_path, color=(255, 60, 60))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Vanilla NanoSAM benchmark on NXP i.MX 95 EVK"
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    # CPU subcommand
    cpu_parser = sub.add_parser("cpu", help="M1a: ONNX Runtime on CPU")
    cpu_parser.add_argument("--image", type=str, required=True)
    cpu_parser.add_argument("--encoder", type=str, required=True,
                            help="ONNX encoder model")
    cpu_parser.add_argument("--decoder", type=str, required=True,
                            help="ONNX decoder model")
    cpu_parser.add_argument("--box", type=int, nargs=4,
                            default=[100, 100, 850, 759],
                            metavar=("X0", "Y0", "X1", "Y1"))
    cpu_parser.add_argument("--warmup", type=int, default=5)
    cpu_parser.add_argument("--runs", type=int, default=50)
    cpu_parser.add_argument("--output", type=str, default=None)

    # NPU subcommand
    npu_parser = sub.add_parser("npu", help="M1b: Naive INT8 on Neutron NPU")
    npu_parser.add_argument("--image", type=str, required=True)
    npu_parser.add_argument("--encoder-tflite", type=str, required=True,
                            help="Naive onnx2tf INT8 TFLite encoder")
    npu_parser.add_argument("--decoder", type=str, required=True,
                            help="ONNX decoder model")
    npu_parser.add_argument("--box", type=int, nargs=4,
                            default=[100, 100, 850, 759],
                            metavar=("X0", "Y0", "X1", "Y1"))
    npu_parser.add_argument("--warmup", type=int, default=3)
    npu_parser.add_argument("--runs", type=int, default=20)
    npu_parser.add_argument("--output", type=str, default=None)

    args = parser.parse_args()

    if args.mode == "cpu":
        run_cpu_benchmark(args)
    elif args.mode == "npu":
        run_npu_benchmark(args)


if __name__ == "__main__":
    main()
