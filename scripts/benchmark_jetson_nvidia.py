#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 Au-Zone Technologies
# SPDX-License-Identifier: Apache-2.0
#
# Benchmark script for NanoSAM on Jetson using TensorRT engines.
# Uses the pure TensorRT Python API (no torch2trt or PyTorch required).
#
# Usage:
#   python3 scripts/benchmark_jetson_nvidia.py \
#       --image assets/dogs.jpg \
#       --encoder ~/models/nanosam-nvidia/data/resnet18_image_encoder.engine \
#       --decoder ~/models/nanosam-nvidia/data/mobile_sam_mask_decoder.engine \
#       --box 100 100 850 759 \
#       --warmup 10 --runs 100

import argparse
import time
import sys
import os

import numpy as np
import PIL.Image
import tensorrt as trt


# ---------------------------------------------------------------------------
# TensorRT engine helpers
# ---------------------------------------------------------------------------

class TRTEngine:
    """Minimal TensorRT engine wrapper using the Python API."""

    def __init__(self, engine_path: str):
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
            self.runtime = trt.Runtime(self.logger)
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        # Discover I/O tensor names
        self.input_names = []
        self.output_names = []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
            else:
                self.output_names.append(name)

    def get_shape(self, name: str):
        return tuple(self.engine.get_tensor_shape(name))

    def __repr__(self):
        lines = [f"TRTEngine(inputs={self.input_names}, outputs={self.output_names})"]
        for n in self.input_names + self.output_names:
            lines.append(f"  {n}: {self.get_shape(n)}")
        return "\n".join(lines)


def trt_infer(engine: TRTEngine, inputs: dict) -> dict:
    """Run synchronous TRT inference with numpy arrays.

    Uses CUDA via the tensorrt / pycuda-free path: we allocate device memory
    through ``cudart`` that ships with the TensorRT Python wheel.
    """
    import ctypes

    # Use ctypes to call cudart directly
    _cudart = ctypes.CDLL("libcudart.so")
    _cudart.cudaMalloc.restype = ctypes.c_int
    _cudart.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
    _cudart.cudaMemcpy.restype = ctypes.c_int
    _cudart.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                    ctypes.c_size_t, ctypes.c_int]
    _cudart.cudaFree.restype = ctypes.c_int
    _cudart.cudaFree.argtypes = [ctypes.c_void_p]
    _cudart.cudaDeviceSynchronize.restype = ctypes.c_int

    def cuda_malloc(nbytes):
        ptr = ctypes.c_void_p()
        err = _cudart.cudaMalloc(ctypes.byref(ptr), ctypes.c_size_t(nbytes))
        assert err == 0, f"cudaMalloc failed with error {err}"
        return ptr.value  # return as int

    def cuda_free(ptr):
        _cudart.cudaFree(ctypes.c_void_p(ptr))

    def cuda_memcpy_h2d(dst, src_np):
        _cudart.cudaMemcpy(ctypes.c_void_p(dst),
                           src_np.ctypes.data_as(ctypes.c_void_p),
                           ctypes.c_size_t(src_np.nbytes), ctypes.c_int(1))

    def cuda_memcpy_d2h(dst_np, src):
        _cudart.cudaMemcpy(dst_np.ctypes.data_as(ctypes.c_void_p),
                           ctypes.c_void_p(src),
                           ctypes.c_size_t(dst_np.nbytes), ctypes.c_int(2))

    def cuda_sync():
        _cudart.cudaDeviceSynchronize()

    ctx = engine.context

    # Set input shapes (handles dynamic axes)
    for name in engine.input_names:
        arr = inputs[name]
        ctx.set_input_shape(name, arr.shape)

    # Allocate device buffers
    d_inputs = {}
    d_outputs = {}
    h_outputs = {}

    for name in engine.input_names:
        arr = np.ascontiguousarray(inputs[name].astype(np.float32))
        d_ptr = cuda_malloc(arr.nbytes)
        cuda_memcpy_h2d(d_ptr, arr)
        d_inputs[name] = d_ptr
        ctx.set_tensor_address(name, d_ptr)

    for name in engine.output_names:
        shape = tuple(ctx.get_tensor_shape(name))
        h_out = np.empty(shape, dtype=np.float32)
        d_ptr = cuda_malloc(h_out.nbytes)
        d_outputs[name] = d_ptr
        h_outputs[name] = h_out
        ctx.set_tensor_address(name, d_ptr)

    # Execute
    ctx.execute_async_v3(0)  # stream_handle=0 (default stream)
    cuda_sync()

    # Copy outputs back
    for name in engine.output_names:
        cuda_memcpy_d2h(h_outputs[name], d_outputs[name])

    # Free device memory
    for d in list(d_inputs.values()) + list(d_outputs.values()):
        cuda_free(d)

    return h_outputs


# ---------------------------------------------------------------------------
# Image preprocessing (matches NVIDIA NanoSAM predictor.py)
# ---------------------------------------------------------------------------

def preprocess_image(image: PIL.Image.Image, size: int = 1024) -> np.ndarray:
    """Preprocess image for the encoder: resize, normalize, pad to square."""
    image_mean = np.array([123.675, 116.28, 103.53], dtype=np.float32).reshape(3, 1, 1)
    image_std = np.array([58.395, 57.12, 57.375], dtype=np.float32).reshape(3, 1, 1)

    w, h = image.size
    aspect = w / h
    if aspect >= 1:
        rw, rh = size, int(size / aspect)
    else:
        rh, rw = size, int(size * aspect)

    resized = image.resize((rw, rh), PIL.Image.BILINEAR)
    arr = np.asarray(resized, dtype=np.float32).transpose(2, 0, 1)  # HWC -> CHW
    arr = (arr - image_mean) / image_std

    padded = np.zeros((1, 3, size, size), dtype=np.float32)
    padded[0, :, :rh, :rw] = arr
    return padded


def preprocess_points(points: np.ndarray, image_size: tuple, size: int = 1024) -> np.ndarray:
    """Scale point coordinates to the encoder's coordinate space."""
    scale = size / max(image_size)
    return (points * scale).astype(np.float32)


# ---------------------------------------------------------------------------
# Decoder inputs
# ---------------------------------------------------------------------------

def build_decoder_inputs(
    features: np.ndarray,
    points: np.ndarray,
    point_labels: np.ndarray,
) -> dict:
    """Build the input dict for the mask decoder TRT engine."""
    return {
        "image_embeddings": features,
        "point_coords": np.array([points], dtype=np.float32),
        "point_labels": np.array([point_labels], dtype=np.float32),
        "mask_input": np.zeros((1, 1, 256, 256), dtype=np.float32),
        "has_mask_input": np.array([0], dtype=np.float32),
    }


# ---------------------------------------------------------------------------
# Upscale mask
# ---------------------------------------------------------------------------

def upscale_mask(low_res_mask: np.ndarray, image_shape: tuple, size: int = 256) -> np.ndarray:
    """Upscale low-res mask to original image size using nearest-neighbor."""
    h, w = image_shape
    if w > h:
        lim_x, lim_y = size, int(size * h / w)
    else:
        lim_x, lim_y = int(size * w / h), size

    cropped = low_res_mask[:, :, :lim_y, :lim_x]
    # Simple upscale via PIL
    mask_2d = cropped[0, 0]
    mask_img = PIL.Image.fromarray(mask_2d)
    mask_img = mask_img.resize((w, h), PIL.Image.BILINEAR)
    return np.asarray(mask_img)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def compute_stats(times_ms: list) -> dict:
    arr = np.array(times_ms)
    return {
        "mean": np.mean(arr),
        "median": np.median(arr),
        "std": np.std(arr),
        "min": np.min(arr),
        "max": np.max(arr),
        "p95": np.percentile(arr, 95),
        "p99": np.percentile(arr, 99),
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
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="NanoSAM Jetson TRT Benchmark")
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--encoder", type=str, required=True)
    parser.add_argument("--decoder", type=str, required=True)
    parser.add_argument("--box", type=int, nargs=4, default=[100, 100, 850, 759],
                        metavar=("X0", "Y0", "X1", "Y1"))
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--output", type=str, default="nvidia_baseline_mask.jpg")
    args = parser.parse_args()

    # Load engines
    print(f"Loading encoder engine: {args.encoder}")
    enc_engine = TRTEngine(args.encoder)
    print(f"Loading decoder engine: {args.decoder}")
    dec_engine = TRTEngine(args.decoder)
    print(f"Encoder: {enc_engine}")
    print(f"Decoder: {dec_engine}")

    # Load and preprocess image
    image = PIL.Image.open(args.image).convert("RGB")
    image_tensor = preprocess_image(image)
    print(f"Image: {image.size[0]}x{image.size[1]} -> preprocessed {image_tensor.shape}")

    # Prepare bounding box prompt
    bbox = args.box
    points = np.array([[bbox[0], bbox[1]], [bbox[2], bbox[3]]], dtype=np.float32)
    point_labels = np.array([2, 3], dtype=np.float32)  # 2=bbox top-left, 3=bbox bottom-right
    scaled_points = preprocess_points(points, (image.height, image.width))

    # Power mode
    try:
        import subprocess
        result = subprocess.run(["nvpmodel", "-q"], capture_output=True, text=True)
        print(f"Power mode: {result.stdout.strip()}")
    except Exception:
        pass

    # ---------- Warmup ----------
    print(f"\nWarming up ({args.warmup} iterations)...")
    for i in range(args.warmup):
        enc_out = trt_infer(enc_engine, {"image": image_tensor})
        features = enc_out["image_embeddings"]
        dec_inputs = build_decoder_inputs(features, scaled_points, point_labels)
        dec_out = trt_infer(dec_engine, dec_inputs)

    # ---------- Benchmark ----------
    print(f"Running benchmark ({args.runs} iterations)...")
    encoder_times = []
    decoder_times = []
    total_times = []

    for i in range(args.runs):
        # Encoder (preprocess is numpy, included in encoder time as it would be in real usage)
        t0 = time.perf_counter()
        # Preprocess is already done (same image), but in a real pipeline it would
        # run each frame. We time just the TRT inference here.
        enc_out = trt_infer(enc_engine, {"image": image_tensor})
        features = enc_out["image_embeddings"]
        t1 = time.perf_counter()

        # Decoder
        dec_inputs = build_decoder_inputs(features, scaled_points, point_labels)
        dec_out = trt_infer(dec_engine, dec_inputs)
        t2 = time.perf_counter()

        enc_ms = (t1 - t0) * 1000
        dec_ms = (t2 - t1) * 1000
        total_ms = (t2 - t0) * 1000

        encoder_times.append(enc_ms)
        decoder_times.append(dec_ms)
        total_times.append(total_ms)

    # ---------- Results ----------
    print(f"\n{'='*80}")
    print(f"NanoSAM Jetson TRT Benchmark Results")
    print(f"  Encoder engine: {args.encoder}")
    print(f"  Decoder engine: {args.decoder}")
    print(f"  Image: {args.image} ({image.size[0]}x{image.size[1]})")
    print(f"  Bounding box: {bbox}")
    print(f"  Warmup: {args.warmup}, Runs: {args.runs}")
    print(f"{'='*80}")

    enc_stats = compute_stats(encoder_times)
    dec_stats = compute_stats(decoder_times)
    tot_stats = compute_stats(total_times)

    print_stats("encoder", enc_stats)
    print_stats("decoder", dec_stats)
    print_stats("total", tot_stats)

    fps = 1000.0 / tot_stats["mean"]
    print(f"\n  Effective FPS (mean): {fps:.1f}")
    print(f"{'='*80}")

    # ---------- Save mask overlay ----------
    low_res_masks = dec_out["low_res_masks"]
    mask = upscale_mask(low_res_masks, (image.height, image.width))
    binary_mask = (mask > 0).astype(np.uint8)

    # Create overlay
    overlay = np.array(image, dtype=np.float32)
    mask_color = np.array([30, 144, 255], dtype=np.float32)  # dodger blue
    alpha = 0.5
    for c in range(3):
        overlay[:, :, c] = np.where(
            binary_mask,
            overlay[:, :, c] * (1 - alpha) + mask_color[c] * alpha,
            overlay[:, :, c],
        )

    # Draw bounding box
    x0, y0, x1, y1 = bbox
    green = np.array([0, 255, 0], dtype=np.float32)
    thickness = 3
    overlay[y0:y0+thickness, x0:x1, :] = green
    overlay[y1-thickness:y1, x0:x1, :] = green
    overlay[y0:y1, x0:x0+thickness, :] = green
    overlay[y0:y1, x1-thickness:x1, :] = green

    out_img = PIL.Image.fromarray(overlay.astype(np.uint8))
    out_img.save(args.output, quality=95)
    print(f"\nMask overlay saved to: {args.output}")


if __name__ == "__main__":
    main()
