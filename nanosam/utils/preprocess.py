"""Shared CPU-safe preprocessing for NanoSAM (pure NumPy, no torch dependency)."""

import numpy as np
import PIL.Image

IMAGE_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32).reshape(3, 1, 1)
IMAGE_STD = np.array([58.395, 57.12, 57.375], dtype=np.float32).reshape(3, 1, 1)


def preprocess_image(image, size: int = 1024, normalize: bool = True) -> np.ndarray:
    """Preprocess an image for the NanoSAM encoder.

    Aspect-ratio-preserving resize, optional ImageNet normalization, zero-padding.
    Returns NCHW float32 numpy array with batch dim.

    Args:
        image: PIL Image or numpy array.
        size: Target encoder input size.
        normalize: If True, apply ImageNet normalization. Set to False for "nonorm"
            encoders (e.g. PPHGV2) that expect raw [0, 255] pixel values.
    """
    if isinstance(image, np.ndarray):
        image = PIL.Image.fromarray(image)

    aspect_ratio = image.width / image.height
    if aspect_ratio >= 1:
        resize_width = size
        resize_height = int(size / aspect_ratio)
    else:
        resize_height = size
        resize_width = int(size * aspect_ratio)

    image_resized = image.resize((resize_width, resize_height))
    image_np = np.asarray(image_resized, dtype=np.float32)

    # HWC -> CHW
    image_chw = np.transpose(image_np, (2, 0, 1))

    # Normalize (skip for "nonorm" encoders that expect raw [0, 255])
    if normalize:
        image_chw = (image_chw - IMAGE_MEAN) / IMAGE_STD

    # Pad to size x size
    image_tensor = np.zeros((1, 3, size, size), dtype=np.float32)
    image_tensor[0, :, :resize_height, :resize_width] = image_chw

    return image_tensor


def preprocess_points(points: np.ndarray, image_size: tuple, size: int = 1024) -> np.ndarray:
    """Scale point coordinates to encoder input space.

    Args:
        points: Nx2 array of (x, y) coordinates.
        image_size: (height, width) — NOT PIL's (width, height).
        size: Encoder input size (default 1024).
    """
    scale = size / max(image_size[0], image_size[1])
    return points * scale


def upscale_mask(mask: np.ndarray, image_shape: tuple, size: int = 256) -> np.ndarray:
    """Upscale a low-res decoder mask to original image dimensions.

    Crops the aspect-ratio-adjusted region from the size x size mask,
    then bilinear-upsamples to (height, width).

    Args:
        mask: 2D array of shape (size, size) — a single mask (no batch/channel dims).
        image_shape: (height, width) of the original image.
        size: Low-res mask size (default 256).
    """
    height, width = image_shape

    if width > height:
        lim_x = size
        lim_y = int(size * height / width)
    else:
        lim_x = int(size * width / height)
        lim_y = size

    # Crop to remove padding region
    cropped = mask[:lim_y, :lim_x]

    # Bilinear upsample to original resolution
    cropped_pil = PIL.Image.fromarray(cropped, mode="F")
    upscaled = cropped_pil.resize((width, height), resample=PIL.Image.BILINEAR)

    return np.asarray(upscaled, dtype=np.float32)
