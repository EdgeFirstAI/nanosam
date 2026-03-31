#!/usr/bin/env python3
"""Diagnose Neutron NPU encoder accuracy vs INT8 TFLite (CPU).

Runs the same preprocessed image through both models and compares:
  1. Output quantization parameters (scale, zero_point)
  2. Raw INT8 output tensor values (before dequantization)
  3. Dequantized float32 output values
  4. Channel-wise statistics (mean, std, min, max)
  5. Cosine similarity between embeddings

This script is designed to run on the imx95-evk target to determine
whether the accuracy issue is a computation error (NPU producing wrong
INT8 values) or a dequantization parameter error (wrong scale/zp).

Usage (on imx95-evk):
    python diagnose_neutron_encoder.py \\
        --int8 /opt/sam/encoder_int8.tflite \\
        --neutron /opt/sam/encoder_neutron.tflite \\
        --neutron-lib /usr/lib/libneutron_delegate.so \\
        --image /opt/sam/dogs.jpg

Usage (host, INT8 only, no Neutron):
    python diagnose_neutron_encoder.py \\
        --int8 runs/resnet18_e200/tflite_e100/encoder_fixed_full_integer_quant.tflite \\
        --image assets/dogs.jpg
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

# Set before any TFLite/Neutron delegate load
os.environ.setdefault("NEUTRON_ENABLE_ZERO_COPY", "0")

try:
    from PIL import Image
except ImportError:
    print("ERROR: Pillow not available. Install with: pip install pillow")
    sys.exit(1)

# Try to import preprocess from nanosam package, then fall back to local copy
try:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from nanosam.utils.preprocess import preprocess_image
except ImportError:
    try:
        from preprocess import preprocess_image
    except ImportError:
        print("ERROR: Cannot find preprocess_image. Ensure nanosam is on the path or copy preprocess.py here.")
        sys.exit(1)


def load_tflite(model_path: Path, delegate_path: Path | None = None):
    """Load TFLite interpreter, optionally with Neutron delegate."""
    try:
        import tflite_runtime.interpreter as tflite
    except ImportError:
        import tensorflow.lite as tflite

    delegates = []
    if delegate_path is not None:
        print(f"  Loading Neutron delegate: {delegate_path}")
        delegates = [tflite.load_delegate(str(delegate_path))]

    interp = tflite.Interpreter(
        model_path=str(model_path),
        experimental_delegates=delegates,
        num_threads=6,
    )
    interp.allocate_tensors()
    return interp


def get_tensor_detail(interp, index: int) -> dict:
    """Get full tensor detail including quant params."""
    details = interp.get_tensor_details()
    for d in details:
        if d["index"] == index:
            return d
    return {}


def print_quant_params(label: str, detail: dict):
    """Print quantization parameters for a tensor."""
    qp = detail.get("quantization_parameters", {})
    scales = qp.get("scales", [])
    zps = qp.get("zero_points", [])
    dtype = detail.get("dtype", "unknown")
    shape = detail.get("shape", [])
    print(f"  [{label}] dtype={dtype}, shape={list(shape)}")
    if len(scales) == 0:
        print(f"    No quantization (float tensor)")
    elif len(scales) == 1:
        print(f"    scale={scales[0]:.8e}, zero_point={zps[0]}")
    else:
        print(f"    Per-channel: {len(scales)} scales, "
              f"range=[{min(scales):.4e}, {max(scales):.4e}], "
              f"zp range=[{min(zps)}, {max(zps)}]")


def run_encoder(interp, image_nhwc_int8: np.ndarray):
    """Run TFLite encoder and return (raw_output, dequant_output, elapsed_ms)."""
    inp_detail = interp.get_input_details()[0]
    out_detail = interp.get_output_details()[0]

    # Set input
    interp.set_tensor(inp_detail["index"], image_nhwc_int8)

    # Time inference
    t0 = time.perf_counter()
    interp.invoke()
    elapsed_ms = (time.perf_counter() - t0) * 1000

    # Get raw INT8 output (do not dequantize yet)
    raw = interp.get_tensor(out_detail["index"]).copy()

    # Dequantize manually
    qp = out_detail.get("quantization_parameters", {})
    scales = qp.get("scales", [])
    zps = qp.get("zero_points", [])
    if len(scales) == 1:
        scale = float(scales[0])
        zp = float(zps[0])
        dequant = (raw.astype(np.float32) - zp) * scale
    elif len(scales) > 1:
        # Per-channel dequantization
        # raw shape: (1, C, H, W) or (1, H, W, C)
        dequant = raw.astype(np.float32)
        # determine channel axis (onnx2tf: output may be NCHW for named tensors)
        if raw.shape[1] == 256:  # NCHW
            for c in range(raw.shape[1]):
                dequant[0, c, :, :] = (raw[0, c, :, :].astype(np.float32) - float(zps[c])) * float(scales[c])
        else:  # NHWC
            for c in range(raw.shape[3]):
                dequant[0, :, :, c] = (raw[0, :, :, c].astype(np.float32) - float(zps[c])) * float(scales[c])
    else:
        dequant = raw.astype(np.float32)

    return raw, dequant, elapsed_ms


def normalize_to_nchw(tensor: np.ndarray) -> np.ndarray:
    """Ensure tensor is NCHW (1, 256, 64, 64)."""
    if tensor.ndim == 4 and tensor.shape[1] != 256:
        return tensor.transpose(0, 3, 1, 2)
    return tensor


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Flat cosine similarity between two arrays."""
    a_flat = a.flatten().astype(np.float32)
    b_flat = b.flatten().astype(np.float32)
    return float(np.dot(a_flat, b_flat) / (np.linalg.norm(a_flat) * np.linalg.norm(b_flat) + 1e-8))


def channel_stats(arr_nchw: np.ndarray) -> dict:
    """Per-channel mean/std/min/max, returned as (C,) arrays."""
    # arr_nchw: (1, 256, 64, 64)
    flat = arr_nchw[0].reshape(arr_nchw.shape[1], -1)  # (256, 4096)
    return {
        "mean": flat.mean(axis=1),
        "std": flat.std(axis=1),
        "min": flat.min(axis=1),
        "max": flat.max(axis=1),
    }


def quantize_image(image: Image.Image, interp) -> np.ndarray:
    """Preprocess and quantize image to INT8 NHWC for TFLite input."""
    # Preprocess to NCHW float32 (ImageNet normalized)
    img_tensor = preprocess_image(image, size=1024)  # (1, 3, 1024, 1024) float32

    # Get input quantization params
    inp_detail = interp.get_input_details()[0]
    inp_dtype = inp_detail["dtype"]

    if inp_dtype == np.int8 or inp_dtype == np.uint8:
        qp = inp_detail.get("quantization_parameters", {})
        scales = qp.get("scales", [])
        zps = qp.get("zero_points", [])
        if len(scales) == 1:
            scale = float(scales[0])
            zp = float(zps[0])
        else:
            # Per-channel (should not happen for input but handle gracefully)
            scale = float(scales[0])
            zp = float(zps[0])
            print(f"  WARNING: Per-channel input quantization, using channel 0 params")

        # Convert NCHW → NHWC for TFLite
        img_nhwc = img_tensor.transpose(0, 2, 3, 1)  # (1, 1024, 1024, 3)

        # Quantize: q = round(x / scale + zp)
        q = np.round(img_nhwc / scale + zp).clip(-128, 127).astype(np.int8)
        return q
    else:
        # Float input — convert NCHW → NHWC
        return img_tensor.transpose(0, 2, 3, 1).astype(np.float32)


def compare_raw_int8(raw_int8: np.ndarray, raw_neutron: np.ndarray):
    """Compare raw INT8 output tensors before dequantization."""
    print("\n--- Raw INT8 Output Comparison (before dequantization) ---")

    # Normalize shapes
    if raw_int8.shape != raw_neutron.shape:
        print(f"  Shape mismatch: INT8={raw_int8.shape}, Neutron={raw_neutron.shape}")
        if raw_int8.size == raw_neutron.size:
            raw_neutron = raw_neutron.reshape(raw_int8.shape)
            print(f"  Reshaped Neutron to {raw_int8.shape}")
        else:
            print("  Cannot compare — different sizes")
            return

    diff = raw_int8.astype(np.int16) - raw_neutron.astype(np.int16)
    abs_diff = np.abs(diff)

    print(f"  INT8    : min={raw_int8.min():4d}, max={raw_int8.max():4d}, "
          f"mean={raw_int8.mean():7.2f}, std={raw_int8.std():6.2f}")
    print(f"  Neutron : min={raw_neutron.min():4d}, max={raw_neutron.max():4d}, "
          f"mean={raw_neutron.mean():7.2f}, std={raw_neutron.std():6.2f}")
    print(f"  Diff    : max_abs={abs_diff.max():4d}, mean_abs={abs_diff.mean():.2f}, "
          f"pct_identical={100*(diff==0).mean():.1f}%")

    # Check if values are completely wrong (e.g. all same value, or clamped)
    unique_neutron = len(np.unique(raw_neutron))
    unique_int8 = len(np.unique(raw_int8))
    print(f"  Unique values: INT8={unique_int8}, Neutron={unique_neutron}")
    if unique_neutron < 10:
        print(f"  WARNING: Neutron output has very few unique values — likely computation error")
        print(f"  Neutron unique values: {np.unique(raw_neutron)}")


def main():
    parser = argparse.ArgumentParser(
        description="Diagnose Neutron NPU encoder accuracy vs INT8 TFLite"
    )
    parser.add_argument("--int8", type=Path, required=True,
                        help="INT8 TFLite encoder (CPU reference)")
    parser.add_argument("--neutron", type=Path, default=None,
                        help="Neutron TFLite encoder (optional — skipped if not provided)")
    parser.add_argument("--neutron-lib", type=Path,
                        default=Path("/usr/lib/libneutron_delegate.so"),
                        help="Path to libneutron_delegate.so")
    parser.add_argument("--onnx", type=Path, default=None,
                        help="ONNX encoder (optional, float32 reference)")
    parser.add_argument("--image", type=Path, required=True,
                        help="Input image for diagnostics")
    parser.add_argument("--runs", type=int, default=3,
                        help="Number of inference runs for timing (default: 3)")
    parser.add_argument("--save-outputs", type=Path, default=None,
                        help="Save output tensors as .npz for offline analysis")
    args = parser.parse_args()

    print(f"NEUTRON_ENABLE_ZERO_COPY = {os.environ.get('NEUTRON_ENABLE_ZERO_COPY', 'not set')}")

    # Load image
    print(f"\nLoading image: {args.image}")
    image = Image.open(args.image).convert("RGB")
    print(f"  Size: {image.width}x{image.height}")

    # ---- Load INT8 TFLite (CPU reference) ----
    print(f"\n[1] Loading INT8 TFLite encoder: {args.int8}")
    interp_int8 = load_tflite(args.int8, delegate_path=None)

    inp_detail_int8 = interp_int8.get_input_details()[0]
    out_detail_int8 = interp_int8.get_output_details()[0]
    print_quant_params("Input", inp_detail_int8)
    print_quant_params("Output", out_detail_int8)

    # Quantize image using INT8 model's params
    img_q = quantize_image(image, interp_int8)
    print(f"  Quantized input: shape={img_q.shape}, dtype={img_q.dtype}, "
          f"range=[{img_q.min()}, {img_q.max()}]")

    # Warmup
    print("  Warming up...")
    run_encoder(interp_int8, img_q)

    # Timed runs
    times_int8 = []
    for i in range(args.runs):
        raw_int8, dequant_int8, t = run_encoder(interp_int8, img_q)
        times_int8.append(t)
    print(f"  Timing: {np.mean(times_int8):.1f} ± {np.std(times_int8):.1f} ms "
          f"(over {args.runs} runs)")

    # Normalize output to NCHW
    raw_int8_nchw = normalize_to_nchw(raw_int8)
    dequant_int8_nchw = normalize_to_nchw(dequant_int8)
    print(f"  Output (raw INT8): shape={raw_int8_nchw.shape}")
    print(f"  Output (dequant): min={dequant_int8_nchw.min():.3f}, "
          f"max={dequant_int8_nchw.max():.3f}, "
          f"mean={dequant_int8_nchw.mean():.4f}, "
          f"std={dequant_int8_nchw.std():.4f}")

    stats_int8 = channel_stats(dequant_int8_nchw)

    # ---- Load ONNX reference (optional) ----
    dequant_onnx_nchw = None
    if args.onnx is not None:
        print(f"\n[2] Loading ONNX encoder (float32 reference): {args.onnx}")
        import onnxruntime as ort
        sess = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
        inp_name = sess.get_inputs()[0].name
        # Preprocess to NCHW float32
        img_float = preprocess_image(image, size=1024)

        # Warmup
        sess.run(None, {inp_name: img_float})
        times_onnx = []
        for _ in range(args.runs):
            t0 = time.perf_counter()
            (onnx_out,) = sess.run(None, {inp_name: img_float})
            times_onnx.append((time.perf_counter() - t0) * 1000)

        dequant_onnx_nchw = onnx_out
        print(f"  Timing: {np.mean(times_onnx):.1f} ± {np.std(times_onnx):.1f} ms")
        print(f"  Output: shape={dequant_onnx_nchw.shape}, "
              f"min={dequant_onnx_nchw.min():.3f}, max={dequant_onnx_nchw.max():.3f}, "
              f"mean={dequant_onnx_nchw.mean():.4f}, std={dequant_onnx_nchw.std():.4f}")

        cos_int8_onnx = cosine_similarity(dequant_int8_nchw, dequant_onnx_nchw)
        print(f"  Cosine (INT8 vs ONNX): {cos_int8_onnx:.4f}")

    # ---- Load Neutron TFLite (NPU) ----
    raw_neutron_nchw = None
    dequant_neutron_nchw = None
    if args.neutron is not None:
        print(f"\n[3] Loading Neutron TFLite encoder: {args.neutron}")
        if not args.neutron_lib.exists():
            print(f"  WARNING: Neutron delegate not found at {args.neutron_lib} — skipping NPU")
        else:
            interp_neutron = load_tflite(args.neutron, delegate_path=args.neutron_lib)

            inp_detail_neu = interp_neutron.get_input_details()[0]
            out_detail_neu = interp_neutron.get_output_details()[0]
            print_quant_params("Input", inp_detail_neu)
            print_quant_params("Output", out_detail_neu)

            # Check if quant params differ from INT8 model
            qp_int8 = out_detail_int8.get("quantization_parameters", {})
            qp_neu = out_detail_neu.get("quantization_parameters", {})
            s_int8 = list(qp_int8.get("scales", []))
            s_neu = list(qp_neu.get("scales", []))
            zp_int8 = list(qp_int8.get("zero_points", []))
            zp_neu = list(qp_neu.get("zero_points", []))

            print("\n  --- Quantization Parameter Comparison ---")
            if s_int8 == s_neu and zp_int8 == zp_neu:
                print("  Output quant params: IDENTICAL between INT8 and Neutron")
            else:
                print("  Output quant params: DIFFER between INT8 and Neutron!")
                if len(s_int8) == 1:
                    print(f"    INT8   : scale={s_int8[0]:.8e}, zp={zp_int8[0]}")
                    print(f"    Neutron: scale={s_neu[0]:.8e}, zp={zp_neu[0]}")
                else:
                    diff_s = [abs(a-b) for a,b in zip(s_int8[:10], s_neu[:10])]
                    print(f"    First 10 scale diffs: {diff_s}")

            # Quantize image using Neutron model's input params
            img_q_neu = quantize_image(image, interp_neutron)

            # Warmup
            print("  Warming up Neutron...")
            run_encoder(interp_neutron, img_q_neu)

            # Timed runs
            times_neu = []
            for i in range(args.runs):
                raw_neutron, dequant_neutron, t = run_encoder(interp_neutron, img_q_neu)
                times_neu.append(t)
            print(f"  Timing: {np.mean(times_neu):.1f} ± {np.std(times_neu):.1f} ms "
                  f"(over {args.runs} runs)")

            raw_neutron_nchw = normalize_to_nchw(raw_neutron)
            dequant_neutron_nchw = normalize_to_nchw(dequant_neutron)
            print(f"  Output (raw INT8): shape={raw_neutron_nchw.shape}")
            print(f"  Output (dequant): min={dequant_neutron_nchw.min():.3f}, "
                  f"max={dequant_neutron_nchw.max():.3f}, "
                  f"mean={dequant_neutron_nchw.mean():.4f}, "
                  f"std={dequant_neutron_nchw.std():.4f}")

            # Compare raw INT8 values
            compare_raw_int8(raw_int8_nchw, raw_neutron_nchw)

            # Cosine similarities
            cos_neu_int8 = cosine_similarity(dequant_neutron_nchw, dequant_int8_nchw)
            print(f"\n  Cosine (Neutron dequant vs INT8 dequant): {cos_neu_int8:.4f}")
            if dequant_onnx_nchw is not None:
                cos_neu_onnx = cosine_similarity(dequant_neutron_nchw, dequant_onnx_nchw)
                print(f"  Cosine (Neutron dequant vs ONNX float32): {cos_neu_onnx:.4f}")

            # Channel-wise analysis: find worst channels
            stats_neu = channel_stats(dequant_neutron_nchw)
            stats_int8 = channel_stats(dequant_int8_nchw)
            channel_cos = np.array([
                cosine_similarity(dequant_int8_nchw[0, c:c+1], dequant_neutron_nchw[0, c:c+1])
                for c in range(dequant_int8_nchw.shape[1])
            ])
            worst_channels = np.argsort(channel_cos)[:10]
            best_channels = np.argsort(channel_cos)[-5:][::-1]

            print(f"\n  --- Per-Channel Cosine Similarity ---")
            print(f"  Overall: mean={channel_cos.mean():.4f}, "
                  f"min={channel_cos.min():.4f} (ch {channel_cos.argmin()}), "
                  f"max={channel_cos.max():.4f} (ch {channel_cos.argmax()})")
            print(f"  Worst 10 channels: {list(worst_channels)}")
            print(f"    INT8  mean: {stats_int8['mean'][worst_channels]}")
            print(f"    Neutron mean: {stats_neu['mean'][worst_channels]}")
            print(f"  Best 5 channels: {list(best_channels)}")

    # ---- Summary ----
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    if dequant_onnx_nchw is not None:
        print(f"ONNX (float32 ref):  mean={dequant_onnx_nchw.mean():.4f}, "
              f"std={dequant_onnx_nchw.std():.4f}")

    print(f"INT8 TFLite (CPU):   mean={dequant_int8_nchw.mean():.4f}, "
          f"std={dequant_int8_nchw.std():.4f}")

    if dequant_onnx_nchw is not None:
        cos = cosine_similarity(dequant_int8_nchw, dequant_onnx_nchw)
        print(f"Cosine INT8 vs ONNX: {cos:.4f}")

    if dequant_neutron_nchw is not None:
        print(f"Neutron TFLite (NPU): mean={dequant_neutron_nchw.mean():.4f}, "
              f"std={dequant_neutron_nchw.std():.4f}")
        cos = cosine_similarity(dequant_neutron_nchw, dequant_int8_nchw)
        print(f"Cosine Neutron vs INT8: {cos:.4f}")
        if dequant_onnx_nchw is not None:
            cos2 = cosine_similarity(dequant_neutron_nchw, dequant_onnx_nchw)
            print(f"Cosine Neutron vs ONNX: {cos2:.4f}")

    # ---- Save for offline analysis ----
    if args.save_outputs is not None:
        save_dict = {
            "int8_raw": raw_int8_nchw,
            "int8_dequant": dequant_int8_nchw,
        }
        if dequant_onnx_nchw is not None:
            save_dict["onnx_float"] = dequant_onnx_nchw
        if raw_neutron_nchw is not None:
            save_dict["neutron_raw"] = raw_neutron_nchw
            save_dict["neutron_dequant"] = dequant_neutron_nchw
        np.savez(str(args.save_outputs), **save_dict)
        print(f"\nSaved outputs to: {args.save_outputs}")
        print(f"  Keys: {list(save_dict.keys())}")
        print(f"  Load with: data = np.load('{args.save_outputs}')")


if __name__ == "__main__":
    main()
