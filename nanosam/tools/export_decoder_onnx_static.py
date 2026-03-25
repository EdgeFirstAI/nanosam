import torch
import argparse
import warnings

from nanosam.mobile_sam import sam_model_registry
from nanosam.mobile_sam.utils.onnx import SamOnnxModel

try:
    import onnxruntime
    onnxruntime_exists = True
except ImportError:
    onnxruntime_exists = False


def run_export(model_type, checkpoint, output, opset, gelu_approximate=False):
    print("Loading model...")
    sam = sam_model_registry[model_type](checkpoint=checkpoint)

    onnx_model = SamOnnxModel(
        model=sam,
        return_single_mask=False,
        use_stability_score=False,
        return_extra_metrics=False,
    )

    if gelu_approximate:
        for _, m in onnx_model.named_modules():
            if isinstance(m, torch.nn.GELU):
                m.approximate = "tanh"

    embed_dim = sam.prompt_encoder.embed_dim
    embed_size = sam.prompt_encoder.image_embedding_size
    mask_input_size = [4 * x for x in embed_size]

    # Fixed num_points=2 for bounding box prompts (labels 2=top-left, 3=bottom-right)
    dummy_inputs = {
        "image_embeddings": torch.randn(1, embed_dim, *embed_size, dtype=torch.float),
        "point_coords": torch.randint(low=0, high=1024, size=(1, 2, 2), dtype=torch.float),
        "point_labels": torch.randint(low=0, high=4, size=(1, 2), dtype=torch.float),
        "mask_input": torch.randn(1, 1, *mask_input_size, dtype=torch.float),
        "has_mask_input": torch.tensor([1], dtype=torch.float),
    }

    _ = onnx_model(**dummy_inputs)

    output_names = ["iou_predictions", "low_res_masks"]

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        with open(output, "wb") as f:
            print(f"Exporting static ONNX model to {output}...")
            torch.onnx.export(
                onnx_model,
                tuple(dummy_inputs.values()),
                f,
                export_params=True,
                verbose=False,
                opset_version=opset,
                do_constant_folding=True,
                input_names=list(dummy_inputs.keys()),
                output_names=output_names,
                # No dynamic_axes — all shapes are fixed
                dynamo=False,
            )

    if onnxruntime_exists:
        ort_inputs = {k: v.cpu().numpy() for k, v in dummy_inputs.items()}
        providers = ["CPUExecutionProvider"]
        ort_session = onnxruntime.InferenceSession(output, providers=providers)
        _ = ort_session.run(None, ort_inputs)
        print("Static model has successfully been run with ONNXRuntime.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export SAM mask decoder to static ONNX (fixed num_points=2, for TFLite conversion)."
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--model-type", type=str, required=True)
    parser.add_argument("--opset", type=int, default=13)
    parser.add_argument("--gelu-approximate", action="store_true")
    args = parser.parse_args()

    run_export(
        model_type=args.model_type,
        checkpoint=args.checkpoint,
        output=args.output,
        opset=args.opset,
        gelu_approximate=args.gelu_approximate,
    )
