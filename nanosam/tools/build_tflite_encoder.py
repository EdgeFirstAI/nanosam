#!/usr/bin/env python3
"""Build clean INT8 TFLite encoder from PyTorch checkpoint (twin model).

Rebuilds the NanoSAM ResNet18 encoder natively in Keras/TensorFlow, loads
weights from the PyTorch checkpoint with proper transposition, fuses
BatchNorm into Conv2D weights offline, verifies FP32 accuracy, then
quantizes to INT8 using the TFLite converter.

This eliminates the onnx2tf dequant→float→quant "float islands" that cause
Neutron NPU accuracy collapse (cosine 0.177 vs ONNX). The Keras model uses
tanh-approximate GELU which decomposes to fully INT8-quantizable TFLite ops
(MUL, ADD, TANH — all bounded), unlike the erf polynomial replacement.

Architecture (TimmImageEncoder):
  Input:  (1, 1024, 1024, 3) NHWC — ImageNet-normalized
  ResNet18 backbone (17 conv+BN fused → Conv2D+bias + ReLU)
  Head: 3× Conv+GELU → ConvTranspose+GELU → Conv+GELU → Conv1×1
  + pos_embedding
  Output: (1, 64, 64, 256) NHWC

Usage:
    source venv/bin/activate
    python nanosam/tools/build_tflite_encoder.py \\
        --checkpoint runs/resnet18_e200/checkpoint_epoch_100.pth \\
        --onnx runs/resnet18_e200/resnet18_e100.onnx \\
        --calibration runs/resnet18_e200/tflite_e100/calibration/image.npy \\
        --image assets/dogs.jpg \\
        --output-dir runs/resnet18_e200/tflite_twin
"""

import argparse
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Weight loading & BN fusion
# ---------------------------------------------------------------------------

def load_pytorch_weights(checkpoint_path):
    """Load state_dict from PyTorch checkpoint."""
    import torch
    ckpt = torch.load(str(checkpoint_path), map_location="cpu")
    state_dict = ckpt["model"] if "model" in ckpt else ckpt
    # Convert all tensors to numpy
    return {k: v.numpy() for k, v in state_dict.items()}


def fuse_conv_bn(conv_weight, gamma, beta, bn_mean, bn_var, conv_bias=None,
                 epsilon=1e-5):
    """Fuse BatchNorm into preceding Conv2D weights (PyTorch layout).

    BN(Conv(x)) = gamma * (Conv(x) - mean) / sqrt(var + eps) + beta
                 = fused_conv(x)  with:
                   W_fused = W * scale[:, None, None, None]
                   b_fused = (b - mean) * scale + beta

    Args:
        conv_weight: (Cout, Cin, kH, kW) — PyTorch layout
        gamma, beta, bn_mean, bn_var: (Cout,) — BN parameters
        conv_bias: (Cout,) or None — if None, treated as zeros

    Returns:
        fused_weight: (Cout, Cin, kH, kW) — PyTorch layout
        fused_bias: (Cout,)
    """
    scale = gamma / np.sqrt(bn_var + epsilon)  # (Cout,)
    fused_weight = conv_weight * scale[:, None, None, None]
    b = conv_bias if conv_bias is not None else np.zeros_like(bn_mean)
    fused_bias = (b - bn_mean) * scale + beta
    return fused_weight, fused_bias


def fuse_all_backbone_bn(sd):
    """Fuse all backbone Conv+BN pairs, return dict of fused weights.

    Returns dict mapping layer names to (weight_nhwc, bias) tuples,
    ready for Keras assignment.
    """
    fused = {}

    def _fuse(conv_key, bn_key, name):
        w = sd[f"{conv_key}.weight"]
        gamma = sd[f"{bn_key}.weight"]
        beta = sd[f"{bn_key}.bias"]
        mean = sd[f"{bn_key}.running_mean"]
        var = sd[f"{bn_key}.running_var"]
        fw, fb = fuse_conv_bn(w, gamma, beta, mean, var)
        # Transpose to Keras NHWC: (Cout,Cin,kH,kW) → (kH,kW,Cin,Cout)
        fused[name] = (fw.transpose(2, 3, 1, 0), fb)

    # Initial conv + bn
    _fuse("backbone.conv1", "backbone.bn1", "conv1")

    # Layer blocks
    channels = [(1, 64, 64), (2, 128, 64), (3, 256, 128), (4, 512, 256)]
    for layer_idx, out_ch, in_ch in channels:
        for block_idx in range(2):
            prefix = f"backbone.layer{layer_idx}.{block_idx}"
            _fuse(f"{prefix}.conv1", f"{prefix}.bn1",
                  f"layer{layer_idx}_{block_idx}_conv1")
            _fuse(f"{prefix}.conv2", f"{prefix}.bn2",
                  f"layer{layer_idx}_{block_idx}_conv2")

            # Downsample shortcut (only first block of layers 2-4)
            ds_key = f"{prefix}.downsample.0"
            ds_bn_key = f"{prefix}.downsample.1"
            if f"{ds_key}.weight" in sd:
                _fuse(ds_key, ds_bn_key,
                      f"layer{layer_idx}_{block_idx}_downsample")

    return fused


# ---------------------------------------------------------------------------
# Keras model builder
# ---------------------------------------------------------------------------

def build_encoder_model():
    """Build the full NanoSAM ResNet18 encoder in Keras (NHWC, fused BN)."""
    import tensorflow as tf

    def gelu_approx(x):
        return tf.nn.gelu(x, approximate=True)

    # Fixed batch=1 avoids SHAPE/STRIDED_SLICE/PACK dynamic shape ops
    inp = tf.keras.Input(shape=(1024, 1024, 3), batch_size=1, name="image")

    # ---- ResNet18 Backbone (all BN fused into conv) ----
    # Use explicit ZeroPadding2D to match PyTorch's symmetric padding
    # (TF padding='same' is asymmetric when stride > 1, causing edge diffs)

    # Initial: Pad3 + Conv7×7/s2 + ReLU + Pad1 + MaxPool3×3/s2
    x = tf.keras.layers.ZeroPadding2D(padding=3, name="conv1_pad")(inp)
    x = tf.keras.layers.Conv2D(
        64, 7, strides=2, padding="valid", use_bias=True, name="conv1")(x)
    x = tf.keras.layers.ReLU(name="relu1")(x)
    x = tf.keras.layers.ZeroPadding2D(padding=1, name="maxpool_pad")(x)
    x = tf.keras.layers.MaxPool2D(
        3, strides=2, padding="valid", name="maxpool")(x)

    def basic_block(x, out_ch, stride, name, has_downsample=False):
        """Fused BasicBlock: pad+conv+relu → pad+conv → add(shortcut) → relu.

        Uses explicit ZeroPadding2D(1) before each 3×3 conv to match
        PyTorch's symmetric padding=1 behavior.
        """
        shortcut = x
        x = tf.keras.layers.ZeroPadding2D(
            padding=1, name=f"{name}_conv1_pad")(x)
        x = tf.keras.layers.Conv2D(
            out_ch, 3, strides=stride, padding="valid", use_bias=True,
            name=f"{name}_conv1")(x)
        x = tf.keras.layers.ReLU(name=f"{name}_relu1")(x)
        x = tf.keras.layers.ZeroPadding2D(
            padding=1, name=f"{name}_conv2_pad")(x)
        x = tf.keras.layers.Conv2D(
            out_ch, 3, padding="valid", use_bias=True,
            name=f"{name}_conv2")(x)
        if has_downsample:
            shortcut = tf.keras.layers.Conv2D(
                out_ch, 1, strides=stride, padding="valid", use_bias=True,
                name=f"{name}_downsample")(shortcut)
        x = tf.keras.layers.Add(name=f"{name}_add")([x, shortcut])
        x = tf.keras.layers.ReLU(name=f"{name}_relu2")(x)
        return x

    # Layer1: 64→64, no stride, no downsample
    x = basic_block(x, 64, 1, "layer1_0")
    x = basic_block(x, 64, 1, "layer1_1")

    # Layer2: 64→128, stride=2, downsample on first
    x = basic_block(x, 128, 2, "layer2_0", has_downsample=True)
    x = basic_block(x, 128, 1, "layer2_1")

    # Layer3: 128→256, stride=2, downsample on first
    x = basic_block(x, 256, 2, "layer3_0", has_downsample=True)
    x = basic_block(x, 256, 1, "layer3_1")

    # Layer4: 256→512, stride=2, downsample on first
    x = basic_block(x, 512, 2, "layer4_0", has_downsample=True)
    x = basic_block(x, 512, 1, "layer4_1")
    # Backbone output: (1, 32, 32, 512)

    # ---- Head: up_1 (3× Pad+Conv+GELU → ConvT+Crop+GELU) ----
    x = tf.keras.layers.ZeroPadding2D(padding=1, name="up1_conv0_pad")(x)
    x = tf.keras.layers.Conv2D(
        256, 3, padding="valid", use_bias=True, name="up1_conv0")(x)
    x = tf.keras.layers.Lambda(gelu_approx, name="up1_gelu0")(x)

    x = tf.keras.layers.ZeroPadding2D(padding=1, name="up1_conv1_pad")(x)
    x = tf.keras.layers.Conv2D(
        256, 3, padding="valid", use_bias=True, name="up1_conv1")(x)
    x = tf.keras.layers.Lambda(gelu_approx, name="up1_gelu1")(x)

    x = tf.keras.layers.ZeroPadding2D(padding=1, name="up1_conv2_pad")(x)
    x = tf.keras.layers.Conv2D(
        256, 3, padding="valid", use_bias=True, name="up1_conv2")(x)
    x = tf.keras.layers.Lambda(gelu_approx, name="up1_gelu2")(x)

    # ConvTranspose: 32→65 (valid) → crop top-left → 64
    # Matches PyTorch ConvTranspose2d(k=3, s=2, padding=1, output_padding=1)
    # which keeps rows/cols [1:65] from the full 65-element output.
    # Neutron NPU does not support TRANSPOSE_CONV; this stays on CPU (~3ms).
    x = tf.keras.layers.Conv2DTranspose(
        256, 3, strides=2, padding="valid", use_bias=True,
        name="up1_convtranspose")(x)
    x = tf.keras.layers.Cropping2D(
        cropping=((1, 0), (1, 0)), name="up1_crop")(x)
    x = tf.keras.layers.Lambda(gelu_approx, name="up1_gelu3")(x)

    # ---- Head: proj (Pad+Conv+GELU → Conv1×1) ----
    x = tf.keras.layers.ZeroPadding2D(padding=1, name="proj_conv0_pad")(x)
    x = tf.keras.layers.Conv2D(
        256, 3, padding="valid", use_bias=True, name="proj_conv0")(x)
    x = tf.keras.layers.Lambda(gelu_approx, name="proj_gelu0")(x)

    x = tf.keras.layers.Conv2D(
        256, 1, padding="valid", use_bias=True, name="proj_conv1")(x)
    # pos_embedding is handled by adding a bias-like layer; see assign step

    model = tf.keras.Model(inputs=inp, outputs=x, name="nanosam_encoder")
    return model


def assign_weights(model, fused_backbone, sd):
    """Assign fused backbone + head weights to the Keras model.

    Args:
        model: Keras model from build_encoder_model()
        fused_backbone: dict from fuse_all_backbone_bn()
        sd: PyTorch state_dict (numpy arrays)
    """
    # ---- Backbone (fused) ----
    for name, (weight, bias) in fused_backbone.items():
        model.get_layer(name).set_weights([weight, bias])

    # ---- Head: up_1 ----
    # PyTorch Sequential indices: 0, 2, 4 = conv; 6 = convtranspose
    for keras_idx, pt_idx in [(0, 0), (1, 2), (2, 4)]:
        w = sd[f"up_1.{pt_idx}.weight"].transpose(2, 3, 1, 0)
        b = sd[f"up_1.{pt_idx}.bias"]
        model.get_layer(f"up1_conv{keras_idx}").set_weights([w, b])

    # ConvTranspose2d: (Cin, Cout, kH, kW) → (kH, kW, Cout, Cin)
    w = sd["up_1.6.weight"].transpose(2, 3, 1, 0)
    b = sd["up_1.6.bias"]
    model.get_layer("up1_convtranspose").set_weights([w, b])

    # ---- Head: proj ----
    for keras_idx, pt_idx in [(0, 0), (1, 2)]:
        w = sd[f"proj.{pt_idx}.weight"].transpose(2, 3, 1, 0)
        b = sd[f"proj.{pt_idx}.bias"]
        model.get_layer(f"proj_conv{keras_idx}").set_weights([w, b])


def add_pos_embedding(model, sd):
    """Add positional embedding as a final bias addition.

    The pos_embedding is a fixed (1, 256, 64, 64) NCHW parameter added to
    the output. We fold it into the last conv layer's bias for simplicity.
    """
    pos_emb = sd["pos_embedding"]  # (1, 256, 64, 64) NCHW
    pos_emb_nhwc = pos_emb.transpose(0, 2, 3, 1)  # (1, 64, 64, 256)

    # Add pos_embedding to the last conv's bias
    # proj_conv1 is the final 1×1 conv with shape (1, 1, 256, 256) weights
    last_layer = model.get_layer("proj_conv1")
    w, b = last_layer.get_weights()
    # pos_embedding is (1, 64, 64, 256) — spatially varying, cannot fold into
    # a per-channel bias. Instead, we add it as a separate constant.

    # Since pos_embedding is small (~1e-5 * randn), check magnitude
    print(f"  pos_embedding: range=[{pos_emb.min():.6f}, {pos_emb.max():.6f}], "
          f"std={pos_emb.std():.6f}")
    return pos_emb_nhwc


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def cosine_similarity(a, b):
    a_flat = a.flatten().astype(np.float64)
    b_flat = b.flatten().astype(np.float64)
    return float(np.dot(a_flat, b_flat) / (
        np.linalg.norm(a_flat) * np.linalg.norm(b_flat) + 1e-12))


def verify_fp32(model, pos_emb_nhwc, onnx_path, image_path):
    """Compare Keras FP32 output vs ONNX reference."""
    import PIL.Image
    import onnxruntime as ort

    from nanosam.utils.preprocess import preprocess_image

    image = PIL.Image.open(image_path).convert("RGB")
    tensor_nchw = preprocess_image(image, size=1024)  # (1, 3, 1024, 1024)
    tensor_nhwc = tensor_nchw.transpose(0, 2, 3, 1)   # (1, 1024, 1024, 3)

    # Keras inference
    keras_out_nhwc = model.predict(tensor_nhwc, verbose=0)  # (1, 64, 64, 256)
    keras_out_nhwc = keras_out_nhwc + pos_emb_nhwc  # add pos embedding
    keras_out_nchw = keras_out_nhwc.transpose(0, 3, 1, 2)  # (1, 256, 64, 64)

    # ONNX reference
    sess = ort.InferenceSession(
        str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_out = sess.run(None, {"image": tensor_nchw})[0]  # (1, 256, 64, 64)

    cos = cosine_similarity(keras_out_nchw, onnx_out)
    max_diff = float(np.abs(keras_out_nchw - onnx_out).max())
    mean_diff = float(np.abs(keras_out_nchw - onnx_out).mean())

    print(f"\n--- FP32 Verification (Keras vs ONNX) ---")
    print(f"  Keras:  shape={keras_out_nchw.shape}, "
          f"range=[{keras_out_nchw.min():.4f}, {keras_out_nchw.max():.4f}], "
          f"mean={keras_out_nchw.mean():.4f}, std={keras_out_nchw.std():.4f}")
    print(f"  ONNX:   shape={onnx_out.shape}, "
          f"range=[{onnx_out.min():.4f}, {onnx_out.max():.4f}], "
          f"mean={onnx_out.mean():.4f}, std={onnx_out.std():.4f}")
    print(f"  Cosine similarity: {cos:.6f}")
    print(f"  Max diff: {max_diff:.6f}")
    print(f"  Mean diff: {mean_diff:.6f}")

    passed = cos > 0.999
    print(f"  {'PASS' if passed else 'FAIL'} (threshold: cosine > 0.999)")
    return passed, cos


# ---------------------------------------------------------------------------
# INT8 Quantization
# ---------------------------------------------------------------------------

def convert_int8(model, pos_emb_nhwc, calibration_path, output_path):
    """Convert Keras model to INT8 TFLite with calibration data."""
    import tensorflow as tf

    # Build a model that includes pos_embedding addition
    # Create a wrapper model that adds the constant
    inp = model.input
    x = model.output
    pos_const = tf.constant(pos_emb_nhwc, dtype=tf.float32)
    x = tf.keras.layers.Add(name="add_pos_emb")([x, pos_const])
    full_model = tf.keras.Model(inputs=inp, outputs=x,
                                name="nanosam_encoder_full")

    calib_data = np.load(str(calibration_path))
    print(f"  Calibration data: shape={calib_data.shape}, "
          f"dtype={calib_data.dtype}")

    def representative_dataset():
        for i in range(calib_data.shape[0]):
            yield [calib_data[i:i+1].astype(np.float32)]

    converter = tf.lite.TFLiteConverter.from_keras_model(full_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8
    ]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    print("  Converting to INT8 TFLite...")
    tflite_model = converter.convert()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(tflite_model)
    size_mb = len(tflite_model) / 1024 / 1024
    print(f"  Saved: {output_path} ({size_mb:.1f} MB)")

    # Check for float islands
    interp = tf.lite.Interpreter(model_content=tflite_model)
    interp.allocate_tensors()
    details = interp.get_tensor_details()
    float_tensors = [d for d in details
                     if d["dtype"] == np.float32
                     and len(d["shape"]) > 1]  # skip scalar constants
    print(f"  Total tensors: {len(details)}, "
          f"float32 tensors (non-scalar): {len(float_tensors)}")
    if float_tensors:
        print("  WARNING: Float islands detected!")
        for d in float_tensors[:5]:
            print(f"    {d['name']}: shape={list(d['shape'])}")
    else:
        print("  PASS: No float islands — fully quantized INT8 model")

    return tflite_model


def verify_int8(tflite_path, onnx_path, image_path):
    """Compare INT8 TFLite output vs ONNX reference."""
    import PIL.Image
    import onnxruntime as ort

    from nanosam.utils.preprocess import preprocess_image

    try:
        import tflite_runtime.interpreter as tflite
    except ImportError:
        import tensorflow.lite as tflite

    image = PIL.Image.open(image_path).convert("RGB")
    tensor_nchw = preprocess_image(image, size=1024)
    tensor_nhwc = tensor_nchw.transpose(0, 2, 3, 1)

    # Load TFLite
    interp = tflite.Interpreter(model_path=str(tflite_path), num_threads=6)
    interp.allocate_tensors()
    inp_detail = interp.get_input_details()[0]
    out_detail = interp.get_output_details()[0]

    # Quantize input
    inp_qp = inp_detail["quantization_parameters"]
    scale = float(inp_qp["scales"][0])
    zp = float(inp_qp["zero_points"][0])
    inp_q = np.clip(
        np.round(tensor_nhwc / scale + zp), -128, 127).astype(np.int8)

    interp.set_tensor(inp_detail["index"], inp_q)
    interp.invoke()

    # Dequantize output
    raw = interp.get_tensor(out_detail["index"])
    out_qp = out_detail["quantization_parameters"]
    out_scale = float(out_qp["scales"][0])
    out_zp = float(out_qp["zero_points"][0])
    tflite_out = (raw.astype(np.float32) - out_zp) * out_scale

    # Normalize to NCHW if needed
    if tflite_out.shape[1] != 256:
        tflite_out_nchw = tflite_out.transpose(0, 3, 1, 2)
    else:
        tflite_out_nchw = tflite_out

    # ONNX reference
    sess = ort.InferenceSession(
        str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_out = sess.run(None, {"image": tensor_nchw})[0]

    cos = cosine_similarity(tflite_out_nchw, onnx_out)
    max_diff = float(np.abs(tflite_out_nchw - onnx_out).max())

    print(f"\n--- INT8 Verification (TFLite vs ONNX) ---")
    print(f"  TFLite: shape={tflite_out_nchw.shape}, "
          f"range=[{tflite_out_nchw.min():.4f}, {tflite_out_nchw.max():.4f}]")
    print(f"  Output quant: scale={out_scale:.4e}, zp={int(out_zp)}")
    print(f"  Cosine similarity: {cos:.4f}")
    print(f"  Max diff: {max_diff:.4f}")

    passed = cos >= 0.93
    print(f"  {'PASS' if passed else 'FAIL'} (threshold: cosine >= 0.93)")
    return passed, cos


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Build clean INT8 TFLite NanoSAM encoder (twin model)")
    parser.add_argument(
        "--checkpoint", type=Path, required=True,
        help="PyTorch checkpoint (e.g. checkpoint_epoch_100.pth)")
    parser.add_argument(
        "--onnx", type=Path, required=True,
        help="ONNX encoder for FP32 verification")
    parser.add_argument(
        "--calibration", type=Path, required=True,
        help="Calibration .npy file (200, 1024, 1024, 3) NHWC float32")
    parser.add_argument(
        "--image", type=Path, required=True,
        help="Test image for verification")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("runs/resnet18_e200/tflite_twin"),
        help="Output directory")
    parser.add_argument(
        "--verify-only", action="store_true",
        help="Only verify FP32 model, skip INT8 conversion")
    args = parser.parse_args()

    assert args.checkpoint.exists(), f"Not found: {args.checkpoint}"
    assert args.onnx.exists(), f"Not found: {args.onnx}"
    assert args.image.exists(), f"Not found: {args.image}"

    # Step 1: Load PyTorch weights
    print("Step 1: Loading PyTorch weights...")
    sd = load_pytorch_weights(args.checkpoint)
    print(f"  Loaded {len(sd)} parameters")

    # Step 2: Fuse BatchNorm into Conv
    print("\nStep 2: Fusing BatchNorm into Conv weights...")
    fused = fuse_all_backbone_bn(sd)
    print(f"  Fused {len(fused)} conv+BN pairs")

    # Step 3: Build Keras model
    print("\nStep 3: Building Keras model...")
    model = build_encoder_model()
    model.summary(print_fn=lambda s: None)  # suppress verbose output
    print(f"  Parameters: {model.count_params():,}")

    # Step 4: Assign weights
    print("\nStep 4: Assigning weights...")
    assign_weights(model, fused, sd)
    pos_emb_nhwc = add_pos_embedding(model, sd)
    print("  Weights assigned")

    # Step 5: FP32 verification
    print("\nStep 5: FP32 verification...")
    fp32_pass, fp32_cos = verify_fp32(model, pos_emb_nhwc, args.onnx,
                                       args.image)

    if args.verify_only:
        print("\n--verify-only: skipping INT8 conversion")
        return

    if not args.calibration.exists():
        print(f"\nCalibration not found: {args.calibration}")
        print("Skipping INT8 conversion")
        return

    # Step 6: INT8 quantization
    print("\nStep 6: INT8 quantization...")
    output_path = args.output_dir / "encoder_int8.tflite"
    convert_int8(model, pos_emb_nhwc, args.calibration, output_path)

    # Step 7: INT8 verification
    print("\nStep 7: INT8 verification...")
    int8_pass, int8_cos = verify_int8(output_path, args.onnx, args.image)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"FP32 Keras vs ONNX:  cosine={fp32_cos:.6f}  "
          f"{'PASS' if fp32_pass else 'FAIL'}")
    print(f"INT8 TFLite vs ONNX: cosine={int8_cos:.4f}  "
          f"{'PASS' if int8_pass else 'FAIL'}")
    print(f"Output: {output_path}")
    print(f"\nNext: run neutron-converter on {output_path}")


if __name__ == "__main__":
    main()
