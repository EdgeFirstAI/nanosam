"""Convert encoder ONNX to TFLite INT8 with COCO calibration.

onnx2tf INT8 calibration requires pre-saved .npy files referenced via
the `quant_calib_input_op_name_np_data_path` parameter. This script:
1. Pre-processes the ONNX model to fix known onnx2tf compatibility issues:
   - Adds missing `kernel_shape` attribute to ConvTranspose nodes
   - Patches onnx.helper with bfloat16 stub required by onnx-graphsurgeon 0.5.x
2. Generates calibration .npy files from COCO val2017 images (INT8 only)
3. Runs FP32 conversion to verify basic conversion works
4. Runs INT8 conversion with calibration data

Known compatibility issues resolved here:
- onnx-graphsurgeon 0.5.x references onnx.helper.float32_to_bfloat16 which
  does not exist in onnx 1.20.x; patched via ml_dtypes.
- ConvTranspose nodes exported by PyTorch may omit the kernel_shape attribute;
  onnx2tf requires it to compute the weight transposition table.
- GELU activation is expressed via Erf nodes; TFLite does not support tf.math.erf
  natively (becomes FlexErf), so replace_to_pseudo_operators=['erf'] is used to
  substitute a polynomial approximation that uses only built-in TFLite ops.
"""

import argparse
import numpy as np
import PIL.Image
import glob
import os

from nanosam.utils.preprocess import preprocess_image


# ---------------------------------------------------------------------------
# Compatibility patch — must happen BEFORE onnx_graphsurgeon is imported.
# onnx-graphsurgeon 0.5.x references onnx.helper.float32_to_bfloat16 which
# was never part of the public onnx 1.x API.
# ---------------------------------------------------------------------------
def _patch_onnx_helper():
    """Inject a float32_to_bfloat16 stub into onnx.helper if missing."""
    import onnx
    import onnx.helper

    if not hasattr(onnx.helper, "float32_to_bfloat16"):
        import ml_dtypes

        def _float32_to_bfloat16(x):
            return ml_dtypes.bfloat16(np.float32(x)).view(np.uint16).item()

        onnx.helper.float32_to_bfloat16 = _float32_to_bfloat16


# ---------------------------------------------------------------------------
# ONNX pre-processing — fix issues before handing to onnx2tf
# ---------------------------------------------------------------------------
def fix_convtranspose_kernel_shape(model):
    """Add missing kernel_shape attribute to ConvTranspose nodes.

    PyTorch may omit kernel_shape from ConvTranspose ONNX exports because the
    shape is implied by the weight tensor. onnx2tf's weight-transposition logic
    reads kernel_shape to build the axis permutation table; without it the table
    has rank 2 regardless of the actual weight rank, causing a ValueError.
    """
    import onnx
    import onnx.numpy_helper

    init_map = {i.name: i for i in model.graph.initializer}
    patched = 0
    for node in model.graph.node:
        if node.op_type != "ConvTranspose":
            continue
        has_kernel_shape = any(a.name == "kernel_shape" for a in node.attribute)
        if has_kernel_shape:
            continue
        if len(node.input) < 2:
            continue
        w_name = node.input[1]
        if w_name not in init_map:
            continue
        arr = onnx.numpy_helper.to_array(init_map[w_name])
        # ONNX ConvTranspose weight layout: [C_in, C_out/group, *kernel_dims]
        kernel_shape = list(arr.shape[2:])
        node.attribute.append(
            onnx.helper.make_attribute("kernel_shape", kernel_shape)
        )
        print(f"  Patched kernel_shape={kernel_shape} on node {node.name}")
        patched += 1
    return model, patched


def prepare_onnx(input_path, output_path):
    """Load, fix, validate, and save an ONNX model ready for onnx2tf."""
    import onnx

    model = onnx.load(input_path)
    model, n = fix_convtranspose_kernel_shape(model)
    if n:
        print(f"Fixed {n} ConvTranspose node(s) — kernel_shape added.")
    onnx.checker.check_model(model)
    onnx.save(model, output_path)
    return output_path


# ---------------------------------------------------------------------------
# Calibration data
# ---------------------------------------------------------------------------
def generate_calibration_npy(coco_root, output_dir, num_images=200, size=1024):
    """Pre-save calibration images as .npy files for onnx2tf.

    onnx2tf INT8 calibration expects NHWC layout (matching the TFLite model
    input after NCHW→NHWC transposition).
    """
    calib_dir = os.path.join(output_dir, "calibration")
    os.makedirs(calib_dir, exist_ok=True)

    image_paths = sorted(glob.glob(os.path.join(coco_root, "*.jpg")))[:num_images]
    if not image_paths:
        raise FileNotFoundError(
            f"No JPEG images found in {coco_root}. "
            "Download COCO val2017 with: "
            "wget http://images.cocodataset.org/zips/val2017.zip"
        )

    calib_images = []
    for path in image_paths:
        image = PIL.Image.open(path).convert("RGB")
        # preprocess_image returns NCHW; onnx2tf calibration uses NHWC
        tensor_nchw = preprocess_image(image, size=size)
        tensor_nhwc = np.transpose(tensor_nchw[0], (1, 2, 0))  # CHW -> HWC
        calib_images.append(tensor_nhwc)

    calib_array = np.stack(calib_images, axis=0)  # (N, H, W, C)
    npy_path = os.path.join(calib_dir, "image.npy")
    np.save(npy_path, calib_array)
    print(
        f"Saved {len(calib_images)} calibration images to {npy_path} "
        f"shape={calib_array.shape} dtype={calib_array.dtype}"
    )
    return npy_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Export NanoSAM encoder ONNX to TFLite FP32 or INT8."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Encoder ONNX path (e.g. data/resnet18_image_encoder.onnx)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/encoder_tflite",
        help="Directory for TFLite output files.",
    )
    parser.add_argument(
        "--coco_root",
        type=str,
        default="data/coco/val2017",
        help="Path to COCO val2017 JPEG images (required for INT8).",
    )
    parser.add_argument(
        "--num_calibration",
        type=int,
        default=200,
        help="Number of COCO images to use for INT8 calibration.",
    )
    parser.add_argument(
        "--int8",
        action="store_true",
        help="Export INT8 quantized model (requires --coco_root).",
    )
    args = parser.parse_args()

    # Patch must happen before any onnx_graphsurgeon / onnx2tf import
    _patch_onnx_helper()

    import onnx2tf

    os.makedirs(args.output_dir, exist_ok=True)

    # Fix ONNX issues before conversion
    fixed_onnx = os.path.join(args.output_dir, "encoder_fixed.onnx")
    print(f"Pre-processing ONNX: {args.input} -> {fixed_onnx}")
    prepare_onnx(args.input, fixed_onnx)

    if not args.int8:
        print("Converting encoder ONNX to TFLite FP32...")
        onnx2tf.convert(
            input_onnx_file_path=fixed_onnx,
            output_folder_path=args.output_dir,
            copy_onnx_input_output_names_to_tflite=True,
            # Erf -> polynomial approximation to avoid FlexErf in TFLite
            replace_to_pseudo_operators=["erf"],
            non_verbose=True,
        )
        tflite_files = glob.glob(os.path.join(args.output_dir, "*.tflite"))
        print(f"FP32 TFLite saved to {args.output_dir}/")
        for f in tflite_files:
            size_mb = os.path.getsize(f) / 1024 / 1024
            print(f"  {os.path.basename(f)}: {size_mb:.1f} MB")
    else:
        print(
            f"Generating calibration data from {args.num_calibration} COCO images..."
        )
        calib_npy = generate_calibration_npy(
            args.coco_root, args.output_dir, num_images=args.num_calibration
        )
        print("Converting encoder to INT8...")
        # quant_calib_input_op_name_np_data_path: list of [input_op_name, npy_path]
        # The input op name must match the ONNX graph input name.
        onnx2tf.convert(
            input_onnx_file_path=fixed_onnx,
            output_folder_path=args.output_dir,
            copy_onnx_input_output_names_to_tflite=True,
            replace_to_pseudo_operators=["erf"],
            non_verbose=True,
            quant_calib_input_op_name_np_data_path=[["image", calib_npy]],
        )
        tflite_files = glob.glob(os.path.join(args.output_dir, "*.tflite"))
        print(f"INT8 TFLite saved to {args.output_dir}/")
        for f in tflite_files:
            size_mb = os.path.getsize(f) / 1024 / 1024
            print(f"  {os.path.basename(f)}: {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
