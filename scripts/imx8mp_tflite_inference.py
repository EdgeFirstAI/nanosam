"""NanoSAM inference on NXP i.MX 8M Plus with VeriSilicon NPU.

Encoder  → TFLite INT8 (VeriSilicon NPU via libvx_delegate)
Decoder  → ONNX Runtime (CPU, FP32)

Usage (on i.MX 8M Plus):
    python3 imx8mp_tflite_inference.py \\
        --encoder resnet18_encoder_int8.tflite \\
        --decoder mobile_sam_mask_decoder.onnx \\
        --image   dogs.jpg \\
        --output  out.jpg

The TFLite model was exported by onnx2tf and expects NHWC float32 input
with ImageNet normalisation applied.  The NPU delegate is loaded from
/usr/lib/libvx_delegate.so (present in the NXP BSP).
"""

import argparse
import time
import numpy as np
import PIL.Image
import PIL.ImageDraw

import onnxruntime as ort
from tflite_runtime.interpreter import Interpreter, load_delegate


# ---------------------------------------------------------------------------
# ImageNet normalisation constants (must match training / ONNX export)
# ---------------------------------------------------------------------------
_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
_STD  = np.array([58.395,  57.12,  57.375], dtype=np.float32)


# ---------------------------------------------------------------------------
# Preprocessing — self-contained, no nanosam package dependency
# ---------------------------------------------------------------------------

def preprocess_image(image: PIL.Image.Image, size: int = 1024) -> np.ndarray:
    """Aspect-ratio-preserving resize + ImageNet normalise + zero-pad.

    Returns NHWC float32 array of shape (1, size, size, 3).
    onnx2tf transposes the ONNX model from NCHW to NHWC, so the TFLite
    model input is NHWC.
    """
    aspect = image.width / image.height
    if aspect >= 1:
        rw, rh = size, int(size / aspect)
    else:
        rw, rh = int(size * aspect), size

    img_resized = image.resize((rw, rh), PIL.Image.BILINEAR)
    arr = np.asarray(img_resized, dtype=np.float32)          # HWC [0, 255]
    arr = (arr - _MEAN) / _STD                               # HWC normalised

    tensor = np.zeros((1, size, size, 3), dtype=np.float32)  # NHWC
    tensor[0, :rh, :rw, :] = arr
    return tensor


def preprocess_points(points: np.ndarray, image_size: tuple,
                      size: int = 1024) -> np.ndarray:
    """Scale (x, y) point coords to encoder input space."""
    scale = size / max(image_size[0], image_size[1])
    return points * scale


def upscale_mask(mask: np.ndarray, image_shape: tuple,
                 size: int = 256) -> np.ndarray:
    """Upscale low-res decoder mask (size×size) to original image dimensions."""
    height, width = image_shape
    if width > height:
        lim_x, lim_y = size, int(size * height / width)
    else:
        lim_x, lim_y = int(size * width / height), size

    cropped = mask[:lim_y, :lim_x]
    pil_mask = PIL.Image.fromarray(cropped, mode="F")
    upscaled = pil_mask.resize((width, height), PIL.Image.BILINEAR)
    return np.asarray(upscaled, dtype=np.float32)


# ---------------------------------------------------------------------------
# TFLite encoder (VeriSilicon NPU)
# ---------------------------------------------------------------------------

class TFLiteEncoder:
    """Wraps TFLite INT8 encoder with optional VX NPU delegate."""

    VX_DELEGATE = "/usr/lib/libvx_delegate.so"

    def __init__(self, model_path: str, use_npu: bool = True):
        delegates = []
        if use_npu:
            try:
                delegates = [load_delegate(self.VX_DELEGATE)]
                print(f"  NPU delegate loaded: {self.VX_DELEGATE}")
            except Exception as e:
                print(f"  NPU delegate unavailable ({e}), falling back to CPU")

        self.interpreter = Interpreter(
            model_path=model_path,
            experimental_delegates=delegates,
        )
        self.interpreter.allocate_tensors()

        self._input  = self.interpreter.get_input_details()[0]
        self._output = self.interpreter.get_output_details()[0]
        print(f"  Encoder input : {self._input['name']}  "
              f"shape={self._input['shape']}  dtype={self._input['dtype'].__name__}")
        print(f"  Encoder output: {self._output['name']}  "
              f"shape={self._output['shape']}  dtype={self._output['dtype'].__name__}")

    def infer(self, image_nhwc: np.ndarray) -> np.ndarray:
        """Run encoder on a pre-processed NHWC image.

        Returns embedding as NCHW float32 (1, 256, 64, 64) for the ONNX decoder.
        """
        self.interpreter.set_tensor(self._input['index'], image_nhwc)
        self.interpreter.invoke()
        # onnx2tf preserves the original ONNX output layout (NCHW) at the model
        # boundary even though internal ops run NHWC — no transpose needed.
        embedding = self.interpreter.get_tensor(self._output['index'])
        return embedding.astype(np.float32)


# ---------------------------------------------------------------------------
# ONNX decoder (CPU)
# ---------------------------------------------------------------------------

class OnnxDecoder:
    """Thin wrapper around the MobileSAM mask decoder ONNX model."""

    def __init__(self, decoder_path: str):
        self.session = ort.InferenceSession(
            decoder_path, providers=["CPUExecutionProvider"]
        )

    def predict(self, embedding: np.ndarray, points: np.ndarray,
                point_labels: np.ndarray) -> tuple:
        inputs = {
            "image_embeddings": embedding,
            "point_coords":     np.array([points],       dtype=np.float32),
            "point_labels":     np.array([point_labels], dtype=np.float32),
            "mask_input":       np.zeros((1, 1, 256, 256), dtype=np.float32),
            "has_mask_input":   np.array([0.0],            dtype=np.float32),
        }
        iou_preds, low_res_masks = self.session.run(None, inputs)
        best_idx = iou_preds[0].argmax()
        return iou_preds, low_res_masks[0, best_idx]


# ---------------------------------------------------------------------------
# Output — PIL-based mask overlay (no matplotlib dependency)
# ---------------------------------------------------------------------------

def save_result(image: PIL.Image.Image, mask: np.ndarray,
                bbox: list, output_path: str,
                t_enc: float, t_dec: float):
    """Overlay mask and bounding box on the original image, save as JPEG."""
    # Yellow mask overlay at 50% opacity
    overlay = PIL.Image.new("RGBA", image.size, (0, 0, 0, 0))
    mask_pil = PIL.Image.fromarray((mask * 180).astype(np.uint8), mode="L")
    yellow_layer = PIL.Image.new("RGBA", image.size, (255, 220, 0, 0))
    yellow_layer.putalpha(mask_pil)
    overlay = PIL.Image.alpha_composite(overlay, yellow_layer)

    result = image.convert("RGBA")
    result = PIL.Image.alpha_composite(result, overlay)
    result = result.convert("RGB")

    # Bounding box
    draw = PIL.ImageDraw.Draw(result)
    draw.rectangle([bbox[0], bbox[1], bbox[2], bbox[3]], outline=(0, 200, 0), width=3)

    result.save(output_path, quality=95)
    print(f"Saved: {output_path}  "
          f"(encoder {t_enc:.0f} ms, decoder {t_dec:.0f} ms, "
          f"total {t_enc + t_dec:.0f} ms)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="NanoSAM inference on i.MX 8M Plus (TFLite NPU + ONNX CPU)"
    )
    parser.add_argument("--encoder",      type=str, required=True,
                        help="INT8 TFLite encoder path")
    parser.add_argument("--decoder",      type=str, required=True,
                        help="ONNX mask decoder path")
    parser.add_argument("--image",        type=str, required=True,
                        help="Input image path")
    parser.add_argument("--output",       type=str, default="imx8mp_out.jpg",
                        help="Output image path")
    parser.add_argument("--encoder-size", type=int, default=1024)
    parser.add_argument("--no-npu",       action="store_true",
                        help="Disable NPU delegate, run on CPU only")
    args = parser.parse_args()

    print(f"Loading encoder: {args.encoder}")
    encoder = TFLiteEncoder(args.encoder, use_npu=not args.no_npu)

    print(f"Loading decoder: {args.decoder}")
    decoder = OnnxDecoder(args.decoder)

    image = PIL.Image.open(args.image).convert("RGB")
    print(f"Image: {image.width}x{image.height}")

    # Preprocess
    image_tensor = preprocess_image(image, size=args.encoder_size)

    # Warmup
    encoder.infer(image_tensor)

    # Encoder (NPU)
    t0 = time.perf_counter()
    embedding = encoder.infer(image_tensor)
    t_enc = (time.perf_counter() - t0) * 1000
    print(f"Encoder: {t_enc:.1f} ms  |  embedding {embedding.shape}  "
          f"range [{embedding.min():.4f}, {embedding.max():.4f}]")

    # Bounding-box prompt (same as basic_usage.py / run_inference_onnx.py)
    bbox = [100, 100, 850, 759]
    points = np.array([[bbox[0], bbox[1]], [bbox[2], bbox[3]]])
    point_labels = np.array([2, 3])
    scaled_pts = preprocess_points(
        points, image_size=(image.height, image.width), size=1024
    )

    # Decoder (CPU)
    t1 = time.perf_counter()
    iou_preds, best_mask_lr = decoder.predict(embedding, scaled_pts, point_labels)
    t_dec = (time.perf_counter() - t1) * 1000
    print(f"Decoder: {t_dec:.1f} ms  |  best IoU: {iou_preds[0].max():.4f}")

    # Upscale and save
    hi_res = upscale_mask(best_mask_lr, (image.height, image.width))
    mask = hi_res > 0
    print(f"Mask coverage: {mask.sum()}/{mask.size} pixels "
          f"({100 * mask.sum() / mask.size:.1f}%)")

    save_result(image, mask, bbox, args.output, t_enc, t_dec)


if __name__ == "__main__":
    main()
