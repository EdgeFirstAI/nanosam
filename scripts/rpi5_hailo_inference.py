"""NanoSAM inference on Raspberry Pi 5 with Hailo-8L HAT.

Encoder  → Hailo-8L HEF (NPU, INT8)
Decoder  → ONNX Runtime (CPU, FP32)

Usage (on RPi5):
    python rpi5_hailo_inference.py \\
        --encoder resnet18_encoder_h8l.hef \\
        --decoder mobile_sam_mask_decoder.onnx \\
        --image dogs.jpg \\
        --output out.jpg

The HEF expects raw [0, 255] float32 NHWC input — normalization is folded in.
The ONNX decoder is unchanged from the x86_64 baseline.
"""

import argparse
import time
import numpy as np
import PIL.Image
import matplotlib.pyplot as plt

# ONNX Runtime — installed on RPi5 via pip
import onnxruntime as ort

# HailoRT Python bindings — installed system-wide via hailo-all package
from hailo_platform import (
    HEF,
    VDevice,
    HailoStreamInterface,
    InferVStreams,
    ConfigureParams,
    InputVStreamParams,
    OutputVStreamParams,
    FormatType,
)


# ---------------------------------------------------------------------------
# Preprocessing (mirrors nanosam/utils/preprocess.py — no package dependency)
# ---------------------------------------------------------------------------

def preprocess_image_raw(image: PIL.Image.Image, size: int = 1024) -> np.ndarray:
    """Aspect-ratio-preserving resize + zero-pad → NHWC float32 [0, 255].

    Returns array shape (1, size, size, 3) — Hailo expects NHWC.
    Normalization is handled by the HEF (folded from .alls model script).
    """
    aspect = image.width / image.height
    if aspect >= 1:
        rw, rh = size, int(size / aspect)
    else:
        rw, rh = int(size * aspect), size
    img_resized = image.resize((rw, rh), PIL.Image.BILINEAR)
    arr = np.array(img_resized, dtype=np.float32)   # HWC [0, 255]
    tensor = np.zeros((1, size, size, 3), dtype=np.float32)
    tensor[0, :rh, :rw, :] = arr
    return tensor


def preprocess_points(points: np.ndarray, image_size: tuple, size: int = 1024) -> np.ndarray:
    """Scale (x, y) point coords to decoder coordinate space (always 1024)."""
    scale = size / max(image_size[0], image_size[1])
    return points * scale


def upscale_mask(mask: np.ndarray, image_shape: tuple, size: int = 256) -> np.ndarray:
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
# Hailo encoder inference
# ---------------------------------------------------------------------------

class HailoEncoder:
    """Wraps Hailo-8L HEF inference for the NanoSAM ResNet18 encoder.

    Keeps VDevice alive across calls to avoid per-inference setup overhead.
    Use as a context manager:
        with HailoEncoder(hef_path) as enc:
            embedding = enc.infer(image_nhwc)
    """

    def __init__(self, hef_path: str):
        self.hef = HEF(hef_path)
        self.input_name = self.hef.get_input_vstream_infos()[0].name
        self.output_name = self.hef.get_output_vstream_infos()[0].name
        input_shape = self.hef.get_input_vstream_infos()[0].shape
        output_shape = self.hef.get_output_vstream_infos()[0].shape
        print(f"  Encoder input : {self.input_name} {input_shape}")
        print(f"  Encoder output: {self.output_name} {output_shape}")
        self._vdevice = None
        self._pipeline = None
        self._network_group = None

    def __enter__(self):
        params = VDevice.create_params()
        self._vdevice_cm = VDevice(params)
        self._vdevice = self._vdevice_cm.__enter__()

        configure_params = ConfigureParams.create_from_hef(
            hef=self.hef, interface=HailoStreamInterface.PCIe
        )
        network_groups = self._vdevice.configure(self.hef, configure_params)
        self._network_group = network_groups[0]

        input_params = InputVStreamParams.make(
            self._network_group, format_type=FormatType.FLOAT32
        )
        output_params = OutputVStreamParams.make(
            self._network_group, format_type=FormatType.FLOAT32
        )

        # Keep all three contexts alive for the duration
        self._activate_cm = self._network_group.activate()
        self._activate_cm.__enter__()

        self._pipeline_cm = InferVStreams(self._network_group, input_params, output_params)
        self._pipeline = self._pipeline_cm.__enter__()

        return self

    def __exit__(self, *args):
        if self._pipeline_cm:
            self._pipeline_cm.__exit__(*args)
        if self._activate_cm:
            self._activate_cm.__exit__(*args)
        if self._vdevice_cm:
            self._vdevice_cm.__exit__(*args)

    def infer(self, image_nhwc: np.ndarray) -> np.ndarray:
        """Run encoder on a single preprocessed image.

        Args:
            image_nhwc: float32 (1, H, W, 3) raw [0, 255] pixel values.

        Returns:
            float32 array — image embedding, shape (1, C, H, W) NCHW.
        """
        input_data = {self.input_name: image_nhwc}
        output_data = self._pipeline.infer(input_data)
        embedding_nhwc = output_data[self.output_name]   # (1, 64, 64, 256)
        # Transpose NHWC → NCHW to match ONNX decoder expectation
        return np.transpose(embedding_nhwc, (0, 3, 1, 2))   # (1, 256, 64, 64)


# ---------------------------------------------------------------------------
# ONNX decoder
# ---------------------------------------------------------------------------

class OnnxDecoder:
    """Thin wrapper around the MobileSAM mask decoder ONNX model."""

    def __init__(self, decoder_path: str):
        self.session = ort.InferenceSession(
            decoder_path, providers=["CPUExecutionProvider"]
        )

    def predict(self, embedding: np.ndarray, points: np.ndarray,
                point_labels: np.ndarray) -> tuple:
        point_coords = np.array([points], dtype=np.float32)        # (1, N, 2)
        point_labels_arr = np.array([point_labels], dtype=np.float32)  # (1, N)
        mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
        has_mask = np.array([0.0], dtype=np.float32)

        inputs = {
            "image_embeddings": embedding,
            "point_coords": point_coords,
            "point_labels": point_labels_arr,
            "mask_input": mask_input,
            "has_mask_input": has_mask,
        }
        iou_preds, low_res_masks = self.session.run(None, inputs)
        best_idx = iou_preds[0].argmax()
        return iou_preds, low_res_masks[0, best_idx]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="NanoSAM on RPi5 Hailo-8L")
    parser.add_argument("--encoder", type=str, default="resnet18_encoder_h8l.hef")
    parser.add_argument("--decoder", type=str, default="mobile_sam_mask_decoder.onnx")
    parser.add_argument("--image", type=str, default="dogs.jpg")
    parser.add_argument("--output", type=str, default="hailo_out.jpg")
    parser.add_argument("--encoder-size", type=int, default=1024)
    parser.add_argument("--save-embedding", type=str, default=None,
                        help="Save encoder embedding as .npy for cross-platform comparison")
    args = parser.parse_args()

    # Load models
    print(f"Loading encoder HEF: {args.encoder}")
    encoder = HailoEncoder(args.encoder)

    print(f"Loading decoder ONNX: {args.decoder}")
    decoder = OnnxDecoder(args.decoder)

    # Load image
    image = PIL.Image.open(args.image).convert("RGB")
    print(f"Image: {image.width}x{image.height}")

    # Preprocess
    image_tensor = preprocess_image_raw(image, size=args.encoder_size)  # NHWC [0,255]

    # Encoder (Hailo NPU) — keep VDevice alive for the inference
    with encoder:
        # Warmup run (first inference includes pipeline setup overhead)
        encoder.infer(image_tensor)

        t0 = time.perf_counter()
        embedding = encoder.infer(image_tensor)   # (1, 256, 64, 64)
        t_enc = (time.perf_counter() - t0) * 1000
    print(f"Encoder: {t_enc:.1f} ms  |  embedding {embedding.shape}  "
          f"range [{embedding.min():.4f}, {embedding.max():.4f}]")
    if args.save_embedding:
        np.save(args.save_embedding, embedding)
        print(f"Saved embedding: {args.save_embedding}")

    # Bounding-box prompt (same as basic_usage.py)
    bbox = [100, 100, 850, 759]
    points = np.array([[bbox[0], bbox[1]], [bbox[2], bbox[3]]])
    point_labels = np.array([2, 3])
    scaled_pts = preprocess_points(
        points,
        image_size=(image.height, image.width),
        size=1024,   # decoder coord space is always 1024
    )

    # Decoder (ONNX CPU)
    t1 = time.perf_counter()
    iou_preds, best_mask_lr = decoder.predict(embedding, scaled_pts, point_labels)
    t_dec = (time.perf_counter() - t1) * 1000
    print(f"Decoder: {t_dec:.1f} ms  |  best IoU: {iou_preds[0].max():.4f}")

    # Upscale mask
    hi_res_mask = upscale_mask(best_mask_lr, (image.height, image.width))
    mask = hi_res_mask > 0
    coverage = 100 * mask.sum() / mask.size
    print(f"Mask coverage: {mask.sum()}/{mask.size} pixels ({coverage:.1f}%)")
    print(f"Total latency: {t_enc + t_dec:.1f} ms")

    # Save output
    plt.figure(figsize=(10, 8))
    plt.imshow(image)
    plt.imshow(mask, alpha=0.5)
    x = [bbox[0], bbox[2], bbox[2], bbox[0], bbox[0]]
    y = [bbox[1], bbox[1], bbox[3], bbox[3], bbox[1]]
    plt.plot(x, y, "g-")
    plt.title(f"NanoSAM — Hailo-8L encoder ({t_enc:.0f} ms) + ONNX decoder ({t_dec:.0f} ms)")
    plt.savefig(args.output, dpi=100, bbox_inches="tight")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
