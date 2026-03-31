#!/usr/bin/env python3
"""End-to-end NanoSAM inference pipeline.

Runs NanoSAM (ResNet18 encoder) with the optimized split MobileSAM decoder.
Supports two modes:

  ONNX mode (development/host):
    encoder ONNX → full decoder ONNX → mask

  Split mode (i.MX95 target):
    encoder ONNX → prompt encoder (augmented ONNX) → attention (ONNX or TFLite)
    → heads Part A + B (INT8/Neutron TFLite) → tokens (TFLite) → mask assembly

Usage:
    # ONNX mode (simple, no TFLite required):
    python nanosam/scripts/run_pipeline.py \\
        --encoder data/resnet18_image_encoder.onnx \\
        --decoder data/mobile_sam_mask_decoder_4mask.onnx \\
        --image assets/dogs.jpg --box 100 100 850 759 --output out.jpg

    # Split mode (ONNX attention + INT8 heads, no TFLite delegate needed):
    python nanosam/scripts/run_pipeline.py \\
        --encoder data/resnet18_image_encoder.onnx \\
        --decoder data/mobile_sam_mask_decoder_4mask.onnx \\
        --attn ../sam-decoder/models/sam_decoder_attn.onnx \\
        --heads-a ../sam-decoder/models/heads_part_a_int8.tflite \\
        --heads-b ../sam-decoder/models/heads_part_b_int8.tflite \\
        --tokens ../sam-decoder/models/heads_tokens_dynq.tflite \\
        --image assets/dogs.jpg --box 100 100 850 759 --output out.jpg

    # Split mode with Neutron NPU heads (on imx95-evk):
    python nanosam/scripts/run_pipeline.py \\
        --encoder /opt/sam/resnet18_e100.onnx \\
        --decoder /opt/sam/mobile_sam_mask_decoder_4mask.onnx \\
        --attn /opt/sam/models/sam_decoder_attn.onnx \\
        --heads-a /opt/sam/models/heads_part_a_neutron.tflite \\
        --heads-b /opt/sam/models/heads_part_b_neutron.tflite \\
        --tokens /opt/sam/models/heads_tokens_dynq.tflite \\
        --neutron /usr/lib/libneutron_delegate.so \\
        --image /opt/sam/dogs.jpg --box 100 100 850 759 --output /opt/sam/out.jpg

Requires: onnxruntime, safetensors, numpy, Pillow
Split mode additionally requires: tensorflow (tflite_runtime on target)
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import PIL.Image
import PIL.ImageDraw

# NanoSAM preprocessing utilities — try package import first, fall back to
# local preprocess.py (for deployment environments without the full package).
try:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from nanosam.utils.preprocess import preprocess_image, preprocess_points, upscale_mask
except ModuleNotFoundError:
    # Deployed standalone: expect preprocess.py alongside this script
    _here = Path(__file__).resolve().parent
    sys.path.insert(0, str(_here))
    from preprocess import preprocess_image, preprocess_points, upscale_mask  # noqa: E402

# --- Decoder graph boundaries (from split_decoder.py) ---
ATTN_INPUTS = [
    "/Concat_2_output_0",                          # sparse_embeddings (1, 7, 256)
    "/Add_7_output_0",                             # image_src (1, 256, 64, 64)
]
ATTN_OUTPUTS = [
    "/transformer/norm_final_attn/Add_1_output_0", # token_output (1, 7, 256)
    "/transformer/layers.1/norm4/Add_1_output_0",  # image_output (1, 4096, 256)
]


# ---------------------------------------------------------------------------
# ONNX utilities
# ---------------------------------------------------------------------------

def _build_augmented_session(decoder_path: Path, extra_outputs: list) -> ort.InferenceSession:
    """Load a decoder ONNX and add extra intermediate tensors to its outputs.

    Requires the `onnx` package (available on host, may not be on target).
    Use save_prompt_encoder() to pre-build the augmented model for deployment.
    """
    import onnx  # lazy import — not available on all targets
    model = onnx.load(str(decoder_path))
    existing = {o.name for o in model.graph.output}
    for name in extra_outputs:
        if name not in existing:
            model.graph.output.append(
                onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, None)
            )
    return ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )


def save_prompt_encoder(decoder_path: Path, output_path: Path) -> None:
    """Extract and save a pruned prompt encoder ONNX for deployment.

    Extracts only the subgraph needed to compute ATTN_INPUTS from the full
    decoder, removing all transformer/heads nodes.  This reduces the model
    from ~16 MB / 1187 nodes to ~55 KB / 146 nodes and avoids ONNX Runtime
    executing dead code (which caused 509 ms → 164 ms on i.MX95).

    Deploy this to targets where the `onnx` package is not available.

    Usage:
        python run_pipeline.py --save-prompt-encoder data/prompt_encoder.onnx \\
            --decoder data/mobile_sam_mask_decoder_4mask.onnx
    """
    import onnx
    from onnx import shape_inference
    from onnx.utils import Extractor

    print(f"Building prompt encoder from {decoder_path}...")
    model = onnx.load(str(decoder_path))

    # Shape inference populates value_info so Extractor can find intermediates
    model = shape_inference.infer_shapes(model)

    input_names = [i.name for i in model.graph.input]
    e = Extractor(model)
    extracted = e.extract_model(input_names, ATTN_INPUTS)

    print(f"  Pruned: {len(model.graph.node)} → {len(extracted.graph.node)} nodes")
    onnx.save(extracted, str(output_path))
    print(f"  Saved: {output_path} ({output_path.stat().st_size / 1024:.1f} KB)")


class NanoSAMEncoder:
    """ONNX ResNet18 image encoder."""

    def __init__(self, model_path: Path):
        self._sess = ort.InferenceSession(
            str(model_path), providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        self._input_name = self._sess.get_inputs()[0].name
        self._output_name = self._sess.get_outputs()[0].name
        provider = self._sess.get_providers()[0]
        print(f"  Encoder running on: {provider}")

    def encode(self, image: PIL.Image.Image) -> tuple:
        """Returns (image_embeddings (1, 256, 64, 64) float32, timings dict)."""
        t0 = time.perf_counter()
        tensor = preprocess_image(image, size=1024, normalize=True)
        t_pre = time.perf_counter()
        emb = self._sess.run([self._output_name], {self._input_name: tensor})[0]
        t_enc = time.perf_counter()
        timings = {
            "preprocess_ms": (t_pre - t0) * 1e3,
            "encoder_ms": (t_enc - t_pre) * 1e3,
        }
        return emb, timings


class TFLiteEncoder:
    """TFLite INT8 ResNet18 encoder (Neutron NPU or XNNPACK CPU).

    onnx2tf transposes the model to NHWC, so this class handles:
    - Input:  NCHW float32 from preprocess_image → quantize → NHWC INT8
    - Output: NHWC INT8 → dequantize → transpose to NCHW float32 for prompt encoder
    """

    def __init__(self, model_path: Path, delegate_path: Path = None):
        try:
            import tflite_runtime.interpreter as tflite
            Interpreter = tflite.Interpreter
            load_delegate = tflite.load_delegate
        except ImportError:
            import tensorflow as tf
            Interpreter = tf.lite.Interpreter
            load_delegate = tf.lite.experimental.load_delegate

        kwargs = {}
        if delegate_path:
            try:
                delegate = load_delegate(str(delegate_path))
                kwargs["experimental_delegates"] = [delegate]
                print(f"  Loaded delegate: {delegate_path.name}")
            except Exception as e:
                print(f"  Warning: could not load delegate {delegate_path}: {e}")

        self._interp = Interpreter(model_path=str(model_path), **kwargs)
        self._interp.allocate_tensors()
        self._inp = self._interp.get_input_details()[0]
        self._out = self._interp.get_output_details()[0]
        print(f"  TFLite encoder input:  {self._inp['shape'].tolist()} dtype={self._inp['dtype'].__name__}")
        print(f"  TFLite encoder output: {self._out['shape'].tolist()} dtype={self._out['dtype'].__name__}")

    def encode(self, image) -> tuple:
        """Returns (image_embeddings (1, 256, 64, 64) NCHW float32, timings dict)."""
        t0 = time.perf_counter()
        tensor_nchw = preprocess_image(image, size=1024, normalize=True)
        # Transpose NCHW → NHWC for TFLite
        tensor_nhwc = tensor_nchw.transpose(0, 2, 3, 1)  # (1, 1024, 1024, 3)
        t_pre = time.perf_counter()

        # Quantize input if INT8
        data = tensor_nhwc
        if self._inp["dtype"] == np.int8:
            scale = self._inp["quantization_parameters"]["scales"][0]
            zp = self._inp["quantization_parameters"]["zero_points"][0]
            data = np.clip(np.round(data / scale + zp), -128, 127).astype(np.int8)

        self._interp.set_tensor(self._inp["index"], data)
        self._interp.invoke()
        t_enc = time.perf_counter()

        # Dequantize output if INT8 (explicit float32 to avoid int32 zero_point promotion)
        raw = self._interp.get_tensor(self._out["index"])
        if self._out["dtype"] == np.int8:
            scale = float(self._out["quantization_parameters"]["scales"][0])
            zp = float(self._out["quantization_parameters"]["zero_points"][0])
            raw = (raw.astype(np.float32) - zp) * scale
        else:
            raw = raw.astype(np.float32)

        # Transpose NHWC → NCHW only if output is in NHWC layout.
        # onnx2tf may preserve NCHW for named output tensors.
        # NHWC: (1, 64, 64, 256)   → transpose to (1, 256, 64, 64)
        # NCHW: (1, 256, 64, 64)   → already correct, no transpose
        if raw.shape[1] != 256:
            emb = raw.transpose(0, 3, 1, 2)
        else:
            emb = raw

        timings = {
            "preprocess_ms": (t_pre - t0) * 1e3,
            "encoder_ms": (t_enc - t_pre) * 1e3,
        }
        return emb, timings


class PromptEncoder:
    """Extract prompt-encoding outputs from the decoder ONNX.

    Accepts either:
    - A pre-built prompt_encoder.onnx (built with save_prompt_encoder(); no onnx
      package required at runtime — suitable for target deployment)
    - The full decoder ONNX, augmented at load time to emit ATTN_INPUTS (requires
      the onnx package; convenient for host-side development)
    """

    def __init__(self, model_path: Path, is_prebuilt: bool = False):
        if is_prebuilt:
            self._sess = ort.InferenceSession(
                str(model_path), providers=["CPUExecutionProvider"]
            )
        else:
            self._sess = _build_augmented_session(model_path, ATTN_INPUTS)
        self._out_names = [o.name for o in self._sess.get_outputs()]

    def encode(
        self,
        image_embeddings: np.ndarray,
        point_coords: np.ndarray,
        point_labels: np.ndarray,
    ) -> tuple:
        """Run prompt encoding.

        Args:
            image_embeddings: (1, 256, 64, 64) float32
            point_coords: (1, N, 2) float32 — scaled to encoder space
            point_labels: (1, N) float32 — 2/3 for box corners, 0/1 for bg/fg points

        Returns:
            sparse_embeddings: (1, M, 256) float32 where M = 1+4+N tokens
            image_src: (1, 256, 64, 64) float32
        """
        mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
        has_mask_input = np.array([0.0], dtype=np.float32)

        t0 = time.perf_counter()
        outputs = self._sess.run(None, {
            "image_embeddings": image_embeddings,
            "point_coords": point_coords,
            "point_labels": point_labels,
            "mask_input": mask_input,
            "has_mask_input": has_mask_input,
        })
        prompt_ms = (time.perf_counter() - t0) * 1e3
        out_map = dict(zip(self._out_names, outputs))

        sparse = out_map[ATTN_INPUTS[0]]  # (1, M, 256)
        image_src = out_map[ATTN_INPUTS[1]]  # (1, 256, 64, 64)
        return sparse, image_src, {"prompt_enc_ms": prompt_ms}


class ONNXDecoder:
    """Full MobileSAM decoder (ONNX) for the simple ONNX-only mode."""

    def __init__(self, decoder_path: Path):
        self._sess = ort.InferenceSession(
            str(decoder_path), providers=["CPUExecutionProvider"]
        )

    def decode(
        self,
        image_embeddings: np.ndarray,
        point_coords: np.ndarray,
        point_labels: np.ndarray,
    ) -> tuple:
        """Run the full decoder.

        Returns:
            iou_predictions: (1, 4) float32
            low_res_masks: (1, 4, 256, 256) float32
        """
        mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
        has_mask_input = np.array([0.0], dtype=np.float32)

        t0 = time.perf_counter()
        outputs = self._sess.run(None, {
            "image_embeddings": image_embeddings,
            "point_coords": point_coords,
            "point_labels": point_labels,
            "mask_input": mask_input,
            "has_mask_input": has_mask_input,
        })
        decoder_ms = (time.perf_counter() - t0) * 1e3
        return outputs[0], outputs[1], {"decoder_ms": decoder_ms}


class ONNXAttentionRunner:
    """Run the split attention ONNX sub-model."""

    def __init__(self, attn_path: Path):
        self._sess = ort.InferenceSession(
            str(attn_path), providers=["CPUExecutionProvider"]
        )
        self._in_names = [i.name for i in self._sess.get_inputs()]
        self._out_names = [o.name for o in self._sess.get_outputs()]

    def run(self, sparse: np.ndarray, image_src: np.ndarray) -> tuple:
        """Returns (token_out (1,M,256), image_out (1,4096,256))."""
        inputs = {n: v for n, v in zip(self._in_names, [sparse, image_src])}
        outputs = self._sess.run(None, inputs)
        out_map = dict(zip(self._out_names, outputs))
        token_out = out_map[ATTN_OUTPUTS[0]]   # (1, M, 256)
        image_out = out_map[ATTN_OUTPUTS[1]]   # (1, 4096, 256)
        return token_out, image_out


class TFLiteRunner:
    """TFLite model runner with transparent INT8 quantize/dequantize handling.

    Also handles shape adaptation for models produced by onnx2tf:
    - 3D tensors (1, N, C) may be transposed to (1, C, N)
    - 4D tensors (1, C, H, W) are transposed to (1, H, W, C) NHWC
    """

    def __init__(self, model_path: Path, delegate_path: Path = None):
        try:
            import tflite_runtime.interpreter as tflite
            Interpreter = tflite.Interpreter
            load_delegate = tflite.load_delegate
        except ImportError:
            import tensorflow as tf
            Interpreter = tf.lite.Interpreter
            load_delegate = tf.lite.experimental.load_delegate

        kwargs = {}
        if delegate_path:
            try:
                delegate = load_delegate(str(delegate_path))
                kwargs["experimental_delegates"] = [delegate]
                print(f"  Loaded delegate: {delegate_path.name}")
            except Exception as e:
                print(f"  Warning: could not load delegate {delegate_path}: {e}")

        self._interp = Interpreter(model_path=str(model_path), **kwargs)
        self._interp.allocate_tensors()
        self._inp = self._interp.get_input_details()
        self._out = self._interp.get_output_details()

    def input_shapes(self) -> list:
        return [d["shape"].tolist() for d in self._inp]

    def output_shapes(self) -> list:
        return [d["shape"].tolist() for d in self._out]

    def _adapt_input(self, data: np.ndarray, detail: dict) -> np.ndarray:
        """Reshape/transpose data to match TFLite expected input shape."""
        expected = list(detail["shape"])
        actual = list(data.shape)

        if actual == expected:
            return data

        # Same number of elements — try to figure out the transposition
        if np.prod(actual) != np.prod(expected):
            raise ValueError(
                f"Shape mismatch: data {actual} vs TFLite input {expected} "
                f"(different element count)"
            )

        # 3D: (1, N, C) → (1, C, N)
        if len(actual) == 3 and len(expected) == 3:
            transposed = data.transpose(0, 2, 1)
            if list(transposed.shape) == expected:
                return transposed

        # 4D NCHW → NHWC: (1, C, H, W) → (1, H, W, C)
        if len(actual) == 4 and len(expected) == 4:
            transposed = data.transpose(0, 2, 3, 1)
            if list(transposed.shape) == expected:
                return transposed

        # Fall back to reshape (preserves data order, use when shapes are equivalent)
        return data.reshape(expected)

    def run(self, *inputs: np.ndarray) -> list:
        """Run inference. Returns list of output arrays (all float32)."""
        for detail, data in zip(self._inp, inputs):
            data = self._adapt_input(data, detail)

            if detail["dtype"] == np.int8:
                scale = detail["quantization_parameters"]["scales"][0]
                zp = detail["quantization_parameters"]["zero_points"][0]
                data = np.clip(np.round(data / scale + zp), -128, 127).astype(np.int8)
            else:
                data = data.astype(detail["dtype"])

            self._interp.set_tensor(detail["index"], data)

        self._interp.invoke()

        results = []
        for detail in self._out:
            raw = self._interp.get_tensor(detail["index"])
            if detail["dtype"] == np.int8:
                scale = float(detail["quantization_parameters"]["scales"][0])
                zp = float(detail["quantization_parameters"]["zero_points"][0])
                raw = (raw.astype(np.float32) - zp) * scale
            results.append(raw.astype(np.float32))
        return results


class SplitDecoder:
    """Orchestrate the split decoder pipeline: attn → heads A/B → tokens → masks."""

    def __init__(
        self,
        attn_path: Path,
        heads_a_path: Path,
        heads_b_path: Path,
        tokens_path: Path,
        delegate_path: Path = None,
    ):
        print(f"  Loading attention: {attn_path.name}")
        self._attn = ONNXAttentionRunner(attn_path)

        print(f"  Loading heads Part A: {heads_a_path.name}")
        self._heads_a = TFLiteRunner(heads_a_path, delegate_path)

        print(f"  Loading heads Part B: {heads_b_path.name}")
        self._heads_b = TFLiteRunner(heads_b_path, delegate_path)

        print(f"  Loading tokens: {tokens_path.name}")
        self._tokens = TFLiteRunner(tokens_path)

    def decode(self, sparse: np.ndarray, image_src: np.ndarray) -> tuple:
        """Run the split pipeline from attention inputs to final masks.

        Args:
            sparse: (1, M, 256) float32 — from PromptEncoder
            image_src: (1, 256, 64, 64) float32 — from PromptEncoder

        Returns:
            iou_predictions: (1, 4) float32
            low_res_masks: (1, 4, 256, 256) float32
        """
        timings = {}

        # Attention
        t0 = time.perf_counter()
        token_out, image_out = self._attn.run(sparse, image_src)
        timings["attention_ms"] = (time.perf_counter() - t0) * 1e3
        # token_out: (1, M, 256), image_out: (1, 4096, 256)

        # Heads Part A: image_out → upscaled features
        t0 = time.perf_counter()
        (features_a,) = self._heads_a.run(image_out)
        timings["heads_a_ms"] = (time.perf_counter() - t0) * 1e3
        # features_a: (1, 128, 128, 64) NHWC from TFLite

        # Heads Part B: upscaled features → dense features
        t0 = time.perf_counter()
        (features_b,) = self._heads_b.run(features_a)
        timings["heads_b_ms"] = (time.perf_counter() - t0) * 1e3
        # features_b: (1, 256, 256, 32) NHWC from TFLite

        # Tokens: token_out → iou + hyper_in_tokens
        t0 = time.perf_counter()
        tok_outputs = self._tokens.run(token_out)
        timings["tokens_ms"] = (time.perf_counter() - t0) * 1e3
        iou_predictions = next(v for v in tok_outputs if v.shape == (1, 4))
        hyper = next(v for v in tok_outputs if v.ndim == 3)
        # hyper: (1, 4, 32) — mask token projections

        # Mask assembly: dense features @ hyper^T → (1, 256, 256, 4) → NCHW
        t0 = time.perf_counter()
        features_flat = features_b.reshape(1, 256 * 256, 32)       # (1, 65536, 32)
        masks_flat = np.matmul(features_flat, hyper.transpose(0, 2, 1))  # (1, 65536, 4)
        masks_nhwc = masks_flat.reshape(1, 256, 256, 4)
        low_res_masks = masks_nhwc.transpose(0, 3, 1, 2)            # (1, 4, 256, 256)
        timings["mask_assembly_ms"] = (time.perf_counter() - t0) * 1e3

        return iou_predictions, low_res_masks, timings


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def build_box_prompt(
    bbox: list, image: PIL.Image.Image
) -> tuple:
    """Build point_coords and point_labels for a bounding box prompt.

    Args:
        bbox: [x1, y1, x2, y2] in original image pixel coordinates
        image: original PIL image (for size)

    Returns:
        point_coords: (1, 2, 2) float32 — scaled to encoder space
        point_labels: (1, 2) float32 — [2, 3] for box corners
    """
    x1, y1, x2, y2 = bbox
    points = np.array([[x1, y1], [x2, y2]], dtype=np.float32)
    scaled = preprocess_points(points, image_size=(image.height, image.width))
    point_coords = scaled[np.newaxis, :, :]   # (1, 2, 2)
    point_labels = np.array([[2.0, 3.0]], dtype=np.float32)
    return point_coords, point_labels


def build_point_prompt(
    points_xy: list, labels: list, image: PIL.Image.Image
) -> tuple:
    """Build point_coords and point_labels for one or more point prompts.

    Args:
        points_xy: list of (x, y) tuples
        labels: list of int (0=background, 1=foreground)
        image: original PIL image

    Returns:
        point_coords: (1, N, 2) float32 — scaled to encoder space
        point_labels: (1, N) float32
    """
    pts = np.array(points_xy, dtype=np.float32)
    scaled = preprocess_points(pts, image_size=(image.height, image.width))
    point_coords = scaled[np.newaxis, :, :]
    point_labels = np.array([labels], dtype=np.float32)
    return point_coords, point_labels


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def draw_mask(
    image: PIL.Image.Image,
    mask: np.ndarray,
    color: tuple = (0, 255, 0),
    alpha: float = 0.4,
) -> PIL.Image.Image:
    """Overlay a binary mask on the image.

    Args:
        image: Original PIL RGB image
        mask: 2D array, positive values = foreground
        color: RGB tuple for mask overlay
        alpha: Blend factor for the mask overlay
    """
    result = image.copy().convert("RGBA")
    overlay = PIL.Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = PIL.ImageDraw.Draw(overlay)

    binary = (mask > 0).astype(np.uint8)
    mask_pil = PIL.Image.fromarray(binary * 255, mode="L")
    # Use the mask to paint the overlay
    mask_rgba = PIL.Image.new("RGBA", image.size, (*color, int(alpha * 255)))
    overlay.paste(mask_rgba, mask=mask_pil)

    result = PIL.Image.alpha_composite(result, overlay)
    return result.convert("RGB")


def draw_box(image: PIL.Image.Image, bbox: list, color: str = "red", width: int = 3) -> PIL.Image.Image:
    """Draw a bounding box on the image."""
    result = image.copy()
    draw = PIL.ImageDraw.Draw(result)
    x1, y1, x2, y2 = bbox
    draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="NanoSAM end-to-end inference (ONNX or split TFLite pipeline)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Models
    parser.add_argument("--encoder", type=Path,
                        help="ResNet18 ONNX encoder")
    parser.add_argument("--encoder-tflite", type=Path,
                        help="ResNet18 TFLite INT8 encoder (Neutron NPU on target; "
                             "overrides --encoder for the encode stage)")
    parser.add_argument("--decoder", type=Path,
                        help="Full MobileSAM decoder ONNX (4-mask, required for prompt encoding)")

    # Split pipeline (optional — if not given, runs ONNX-only mode)
    split = parser.add_argument_group("split pipeline (TFLite heads)")
    split.add_argument("--prompt-encoder", type=Path,
                       help="Pre-built prompt_encoder.onnx for target deployment "
                            "(avoids onnx package requirement; build with --save-prompt-encoder)")
    split.add_argument("--attn", type=Path,
                       help="Attention ONNX model (sam_decoder_attn.onnx)")
    split.add_argument("--heads-a", type=Path,
                       help="Heads Part A TFLite (INT8 or Neutron)")
    split.add_argument("--heads-b", type=Path,
                       help="Heads Part B TFLite (INT8 or Neutron)")
    split.add_argument("--tokens", type=Path,
                       help="Tokens TFLite (heads_tokens_dynq.tflite)")
    split.add_argument("--neutron", type=Path,
                       help="Path to libneutron_delegate.so for NPU execution")
    split.add_argument("--xnnpack", type=Path,
                       help="Path to XNNPACK/GPU delegate .so for TFLite attention")

    # Utility: build prompt encoder for deployment
    parser.add_argument("--save-prompt-encoder", type=Path, metavar="OUTPUT",
                        help="Build and save prompt_encoder.onnx from --decoder, then exit")

    # Input
    parser.add_argument("--image", type=Path,
                        help="Input image (required for inference)")
    prompt = parser.add_mutually_exclusive_group(required=False)
    prompt.add_argument("--box", nargs=4, type=float, metavar=("X1", "Y1", "X2", "Y2"),
                        help="Bounding box prompt")
    prompt.add_argument("--point", nargs=2, type=float, action="append",
                        metavar=("X", "Y"),
                        help="Point prompt (use multiple --point for multi-point; "
                             "pair with --label; default label=1 foreground)")

    parser.add_argument("--label", type=int, action="append", default=None,
                        help="Label for each --point (0=bg, 1=fg); default all 1")

    # Output
    parser.add_argument("--output", type=Path, default=Path("output.jpg"),
                        help="Output image path (default: output.jpg)")
    parser.add_argument("--mask-index", type=int, default=None,
                        help="Which of the 4 masks to use (0-3); default: highest IoU")
    parser.add_argument("--show-all-masks", action="store_true",
                        help="Also save all 4 individual masks as output_mask_N.jpg")

    # Timing
    parser.add_argument("--warmup", type=int, default=1, metavar="N",
                        help="Warmup runs before timing (default: 1)")
    parser.add_argument("--runs", type=int, default=1, metavar="N",
                        help="Timed inference runs to average (default: 1)")

    return parser.parse_args()


def main():
    args = parse_args()

    # Utility mode: build prompt encoder for deployment
    if args.save_prompt_encoder:
        if not args.decoder:
            print("Error: --save-prompt-encoder requires --decoder", file=sys.stderr)
            sys.exit(1)
        if not args.decoder.exists():
            print(f"Error: not found: {args.decoder}", file=sys.stderr)
            sys.exit(1)
        save_prompt_encoder(args.decoder, args.save_prompt_encoder)
        return

    # Set NEUTRON_ENABLE_ZERO_COPY=0 before any delegate is loaded to avoid NPU bugs
    if args.neutron:
        os.environ.setdefault("NEUTRON_ENABLE_ZERO_COPY", "0")

    # Validate required args for inference
    if not args.encoder and not args.encoder_tflite:
        print("Error: --encoder or --encoder-tflite is required", file=sys.stderr)
        sys.exit(1)
    if not args.decoder and not args.prompt_encoder:
        print("Error: --decoder is required (or --prompt-encoder for pre-built model)", file=sys.stderr)
        sys.exit(1)
    if not args.image:
        print("Error: --image is required", file=sys.stderr)
        sys.exit(1)
    if not args.box and not args.point:
        print("Error: a prompt is required: --box X1 Y1 X2 Y2 or --point X Y", file=sys.stderr)
        sys.exit(1)
    for p in [p for p in [args.encoder, args.encoder_tflite, args.image] if p]:
        if not p.exists():
            print(f"Error: not found: {p}", file=sys.stderr)
            sys.exit(1)
    if args.decoder and not args.decoder.exists():
        print(f"Error: not found: {args.decoder}", file=sys.stderr)
        sys.exit(1)

    use_split = args.attn is not None or args.heads_a is not None
    if use_split:
        missing = [n for n, p in [("--attn", args.attn), ("--heads-a", args.heads_a),
                                   ("--heads-b", args.heads_b), ("--tokens", args.tokens)]
                   if p is None]
        if missing:
            print(f"Error: split mode requires {', '.join(missing)}", file=sys.stderr)
            sys.exit(1)
        for n, p in [("--attn", args.attn), ("--heads-a", args.heads_a),
                     ("--heads-b", args.heads_b), ("--tokens", args.tokens)]:
            if not p.exists():
                print(f"Error: {n}: not found: {p}", file=sys.stderr)
                sys.exit(1)
        if args.prompt_encoder and not args.prompt_encoder.exists():
            print(f"Error: --prompt-encoder: not found: {args.prompt_encoder}", file=sys.stderr)
            sys.exit(1)

    # Load image
    print(f"Loading image: {args.image}")
    image = PIL.Image.open(args.image).convert("RGB")
    print(f"  Size: {image.width}x{image.height}")

    # Build prompt
    if args.box:
        bbox = [int(v) for v in args.box]
        point_coords, point_labels = build_box_prompt(bbox, image)
        print(f"  Prompt: box {bbox}")
    else:
        pts = [(int(x), int(y)) for x, y in args.point]
        labels = args.label if args.label else [1] * len(pts)
        if len(labels) < len(pts):
            labels += [1] * (len(pts) - len(labels))
        point_coords, point_labels = build_point_prompt(pts, labels, image)
        print(f"  Prompt: {len(pts)} point(s) {pts} labels={labels}")

    # Load models
    print("\nLoading models...")
    if args.encoder_tflite:
        print(f"  Using TFLite encoder: {args.encoder_tflite.name}")
        encoder = TFLiteEncoder(args.encoder_tflite, delegate_path=args.neutron)
    else:
        encoder = NanoSAMEncoder(args.encoder)

    if use_split:
        print("  Loading prompt encoder...")
        if args.prompt_encoder:
            prompt_enc = PromptEncoder(args.prompt_encoder, is_prebuilt=True)
        else:
            prompt_enc = PromptEncoder(args.decoder, is_prebuilt=False)

        print("  Loading split decoder...")
        split_dec = SplitDecoder(
            attn_path=args.attn,
            heads_a_path=args.heads_a,
            heads_b_path=args.heads_b,
            tokens_path=args.tokens,
            delegate_path=args.neutron,
        )
    else:
        if not args.decoder:
            print("Error: --decoder is required for ONNX-only mode", file=sys.stderr)
            sys.exit(1)
        onnx_dec = ONNXDecoder(args.decoder)

    def run_inference():
        emb, enc_timings = encoder.encode(image)
        if use_split:
            sparse, image_src, pe_timings = prompt_enc.encode(
                emb, point_coords, point_labels)
            iou, masks, dec_timings = split_dec.decode(sparse, image_src)
            all_timings = {**enc_timings, **pe_timings, **dec_timings}
        else:
            iou, masks, dec_timings = onnx_dec.decode(emb, point_coords, point_labels)
            all_timings = {**enc_timings, **dec_timings}
        return iou, masks, all_timings

    # Warmup
    if args.warmup > 0:
        print(f"\nWarmup ({args.warmup} run{'s' if args.warmup > 1 else ''})...", flush=True)
        for _ in range(args.warmup):
            run_inference()

    # Timed runs
    print(f"\nTiming ({args.runs} run{'s' if args.runs > 1 else ''})...", flush=True)
    accumulated = {}
    for run_i in range(args.runs):
        iou_predictions, low_res_masks, timings = run_inference()
        for k, v in timings.items():
            accumulated[k] = accumulated.get(k, 0.0) + v

    # Average timings
    avg = {k: v / args.runs for k, v in accumulated.items()}

    # Compute totals
    if use_split:
        decode_total = sum(avg.get(k, 0) for k in (
            "prompt_enc_ms", "attention_ms", "heads_a_ms",
            "heads_b_ms", "tokens_ms", "mask_assembly_ms"))
    else:
        decode_total = avg.get("decoder_ms", 0)
    pipeline_total = avg.get("preprocess_ms", 0) + avg.get("encoder_ms", 0) + decode_total

    # Print timing table
    enc_label = (f"Encoder ({args.encoder_tflite.stem})" if args.encoder_tflite
                 else "Encoder (ResNet18 ONNX)")

    print(f"\n{'─' * 44}")
    print(f"  {'Stage':<28} {'ms':>8}")
    print(f"{'─' * 44}")
    print(f"  {'Preprocess (resize+pad)':<28} {avg['preprocess_ms']:>8.1f}")
    print(f"  {enc_label:<28} {avg['encoder_ms']:>8.1f}")
    if use_split:
        print(f"  {'Prompt encoder (ONNX)':<28} {avg['prompt_enc_ms']:>8.1f}")
        print(f"  {'Attention (ONNX)':<28} {avg['attention_ms']:>8.1f}")
        print(f"  {'Heads Part A (TFLite)':<28} {avg['heads_a_ms']:>8.1f}")
        print(f"  {'Heads Part B (TFLite)':<28} {avg['heads_b_ms']:>8.1f}")
        print(f"  {'Tokens (TFLite)':<28} {avg['tokens_ms']:>8.1f}")
        print(f"  {'Mask assembly (NumPy)':<28} {avg['mask_assembly_ms']:>8.1f}")
    else:
        print(f"  {'Decoder (ONNX)':<28} {avg['decoder_ms']:>8.1f}")
    print(f"{'─' * 44}")
    print(f"  {'Decoder total':<28} {decode_total:>8.1f}")
    print(f"  {'Pipeline total':<28} {pipeline_total:>8.1f}")
    if args.runs > 1:
        print(f"  (averaged over {args.runs} runs)")
    print(f"{'─' * 44}")

    print(f"\n  IoU: {[f'{v:.3f}' for v in iou_predictions.flatten().tolist()]}")

    # Select primary mask
    if args.mask_index is not None:
        mask_idx = args.mask_index
    else:
        mask_idx = int(np.argmax(iou_predictions.flatten()))
    print(f"  Best mask: index {mask_idx} (IoU={iou_predictions.flatten()[mask_idx]:.3f})")

    # Upscale and save
    stem = args.output.stem
    suffix = args.output.suffix or ".jpg"
    parent = args.output.parent
    parent.mkdir(parents=True, exist_ok=True)

    def upscale_and_draw(idx: int) -> PIL.Image.Image:
        mask = upscale_mask(low_res_masks[0, idx], (image.height, image.width))
        vis = draw_mask(image, mask)
        if args.box:
            vis = draw_box(vis, bbox)
        return vis

    t0 = time.perf_counter()

    # Always save the selected mask as the primary output
    upscale_and_draw(mask_idx).save(str(args.output))
    print(f"\n  Primary output (mask {mask_idx}): {args.output}")

    # Always save all 4 masks for visual inspection
    print("  All masks:")
    for i in range(low_res_masks.shape[1]):
        out_path = parent / f"{stem}_mask_{i}{suffix}"
        upscale_and_draw(i).save(str(out_path))
        iou = iou_predictions.flatten()[i]
        marker = " ◀ selected" if i == mask_idx else ""
        print(f"    mask_{i} (IoU={iou:.3f}): {out_path}{marker}")

    vis_ms = (time.perf_counter() - t0) * 1e3
    print(f"  Visualization + save: {vis_ms:.1f} ms")


if __name__ == "__main__":
    main()
