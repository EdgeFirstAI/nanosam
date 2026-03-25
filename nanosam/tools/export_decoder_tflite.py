"""Convert static decoder ONNX to TFLite (FP32 + optional dynamic-range quantization).

The MobileSAM mask decoder contains a two-way transformer with cross-attention
and 4D batch matmuls.  Known compatibility issues (same as encoder):

- onnx-graphsurgeon 0.5.x references onnx.helper.float32_to_bfloat16 which
  does not exist in onnx 1.20.x; patched via ml_dtypes.
- ConvTranspose nodes exported by PyTorch may omit the kernel_shape attribute;
  onnx2tf requires it to compute the weight transposition table.
- GELU activation is expressed via Erf nodes; TFLite does not support tf.math.erf
  natively (becomes FlexErf), so replace_to_pseudo_operators=['erf'] is used to
  substitute a polynomial approximation that uses only built-in TFLite ops.

Additional decoder-specific issues encountered during conversion:
- /Div node: onnx2tf's correction_process_for_accuracy_errors builds a
  tf_keras.Model with a scalar constant (1024.0) as an output, triggering
  ValueError. Fixed with disable_strict_mode=True.
- /pe_layer/MatMul node: onnx2tf incorrectly treats the [64,64,2] positional
  encoding constant as NCHW and transposes it to [64,2,64], causing an
  InvalidArgumentError in BatchMatMulV2. Fixed by constant-folding the
  pe_layer MatMul (both inputs are constants) into a single [64,64,128] tensor.
- /Reshape_1 node: onnx2tf incorrectly transposes a 5D tensor [1,1,256,64,64]
  (image embeddings tiled for attention) when computing the reshape target shape
  from a dynamic Shape node, producing a wildly wrong shape. This is a
  fundamental limitation of onnx2tf's NCHW->NHWC heuristic with non-image 5D
  tensors in the transformer attention layers.

CONVERSION STATUS: FP32 and dynamic-range quantization both FAIL on this version
of onnx2tf (1.28.8) due to the 5D tensor reshape issue in the attention layers.
The decoder remains ONNX-only for TFLite deployment.

Dynamic-range quantization: weights INT8, activations FP32 — no calibration data
needed.
"""

import argparse
import glob
import os

import numpy as np


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
    shape is implied by the weight tensor.  onnx2tf's weight-transposition logic
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


def fold_pe_layer_matmul(model):
    """Constant-fold the /pe_layer/MatMul node.

    onnx2tf incorrectly transposes the [64,64,2] positional encoding constant
    (treating it as NCHW) to [64,2,64], causing a shape mismatch when multiplied
    by the [2,128] Gaussian matrix.

    Both inputs to /pe_layer/MatMul are constants, so the result [64,64,128] can
    be pre-computed and inserted as a single Constant node, bypassing onnx2tf's
    broken NCHW transpose heuristic for this particular non-image 3D tensor.
    """
    import onnx
    import onnx.numpy_helper
    import onnx.helper

    # Find the Constant node that produces /pe_layer/Constant_output_0
    const_val = None
    const_node_name = None
    const_node_idx = None
    for i, node in enumerate(model.graph.node):
        if node.op_type == "Constant":
            for out in node.output:
                if out == "/pe_layer/Constant_output_0":
                    for attr in node.attribute:
                        if attr.name == "value":
                            const_val = onnx.numpy_helper.to_array(attr.t)
                    const_node_name = node.name
                    const_node_idx = i
                    break

    if const_val is None:
        print("  /pe_layer/Constant not found — skipping pe_layer fold.")
        return model, 0

    # Find the Gaussian matrix initializer
    init_map = {}
    for init in model.graph.initializer:
        init_map[init.name] = onnx.numpy_helper.to_array(init)

    gauss_key = "model.prompt_encoder.pe_layer.positional_encoding_gaussian_matrix"
    if gauss_key not in init_map:
        print("  pe_layer Gaussian matrix not found — skipping pe_layer fold.")
        return model, 0

    gauss = init_map[gauss_key]

    # Pre-compute: [64,64,2] @ [2,128] -> [64,64,128]
    result = np.matmul(const_val, gauss).astype(np.float32)
    print(f"  Pre-computed /pe_layer/MatMul: {const_val.shape} @ {gauss.shape} -> {result.shape}")

    # Find the MatMul node index
    matmul_node_idx = None
    for i, node in enumerate(model.graph.node):
        if node.name == "/pe_layer/MatMul":
            matmul_node_idx = i
            break

    if matmul_node_idx is None:
        print("  /pe_layer/MatMul not found — skipping pe_layer fold.")
        return model, 0

    # Build replacement Constant node
    new_const = onnx.helper.make_node(
        "Constant",
        inputs=[],
        outputs=["/pe_layer/MatMul_output_0"],
        name="/pe_layer/MatMul_folded",
        value=onnx.numpy_helper.from_array(result, name="/pe_layer/MatMul_output_0"),
    )

    # Remove old Constant and MatMul nodes, insert replacement
    nodes = list(model.graph.node)
    for idx in sorted([const_node_idx, matmul_node_idx], reverse=True):
        del nodes[idx]

    insert_at = min(const_node_idx, matmul_node_idx)
    nodes.insert(insert_at, new_const)

    del model.graph.node[:]
    model.graph.node.extend(nodes)

    return model, 1


def prepare_onnx(input_path, output_path):
    """Load, fix, validate, and save an ONNX model ready for onnx2tf."""
    import onnx

    model = onnx.load(input_path)
    model, n = fix_convtranspose_kernel_shape(model)
    if n:
        print(f"Fixed {n} ConvTranspose node(s) — kernel_shape added.")
    model, n = fold_pe_layer_matmul(model)
    if n:
        print(f"Folded pe_layer MatMul into a constant.")
    onnx.checker.check_model(model)
    onnx.save(model, output_path)
    return output_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Export NanoSAM decoder ONNX to TFLite FP32 or dynamic-range quantized."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Static decoder ONNX path (e.g. data/mobile_sam_mask_decoder_static.onnx)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/decoder_tflite",
        help="Directory for TFLite output files.",
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Apply dynamic-range quantization (weights INT8, activations FP32). "
             "No calibration data needed.",
    )
    args = parser.parse_args()

    # Patch must happen before any onnx_graphsurgeon / onnx2tf import
    _patch_onnx_helper()

    import onnx2tf

    os.makedirs(args.output_dir, exist_ok=True)

    # Fix ONNX issues before conversion
    fixed_onnx = os.path.join(args.output_dir, "decoder_fixed.onnx")
    print(f"Pre-processing ONNX: {args.input} -> {fixed_onnx}")
    prepare_onnx(args.input, fixed_onnx)

    quant_label = "(dynamic-range quant)" if args.quantize else "(FP32)"
    print(f"Converting decoder ONNX to TFLite {quant_label}...")

    convert_kwargs = dict(
        input_onnx_file_path=fixed_onnx,
        output_folder_path=args.output_dir,
        copy_onnx_input_output_names_to_tflite=True,
        # Erf -> polynomial approximation to avoid FlexErf in TFLite
        replace_to_pseudo_operators=["erf"],
        # disable_strict_mode skips correction_process_for_accuracy_errors which
        # crashes on the /Div node (scalar constant used as Keras model output)
        disable_strict_mode=True,
        non_verbose=True,
    )

    if args.quantize:
        # Dynamic-range quantization: weights INT8, activations FP32
        # No calibration data needed — only quantizes weights statically
        convert_kwargs["quant_type"] = "dynamic"

    onnx2tf.convert(**convert_kwargs)

    tflite_files = glob.glob(os.path.join(args.output_dir, "*.tflite"))
    print(f"TFLite saved to {args.output_dir}/")
    for f in tflite_files:
        size_mb = os.path.getsize(f) / 1024 / 1024
        print(f"  {os.path.basename(f)}: {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
