import numpy as np
import PIL.Image
import pytest


def test_preprocess_image_output_shape():
    from nanosam.utils.preprocess import preprocess_image
    image = PIL.Image.new("RGB", (640, 480))
    result = preprocess_image(image, size=1024)
    assert result.shape == (1, 3, 1024, 1024)
    assert result.dtype == np.float32


def test_preprocess_image_landscape_padding():
    """Landscape image: width=1024, height padded."""
    from nanosam.utils.preprocess import preprocess_image
    image = PIL.Image.new("RGB", (800, 600))
    result = preprocess_image(image, size=1024)
    resize_height = int(1024 * 600 / 800)  # 768
    assert np.allclose(result[0, :, resize_height:, :], 0.0, atol=1e-6)


def test_preprocess_image_portrait_padding():
    """Portrait image: height=1024, width padded."""
    from nanosam.utils.preprocess import preprocess_image
    image = PIL.Image.new("RGB", (400, 600))
    result = preprocess_image(image, size=1024)
    resize_width = int(1024 * 400 / 600)  # 682
    assert np.allclose(result[0, :, :, resize_width:], 0.0, atol=1e-6)


def test_preprocess_points_scaling():
    from nanosam.utils.preprocess import preprocess_points
    points = np.array([[100.0, 200.0], [300.0, 400.0]])
    # image_size = (height=600, width=800), max=800, scale=1024/800=1.28
    result = preprocess_points(points, image_size=(600, 800), size=1024)
    expected_scale = 1024.0 / 800.0
    np.testing.assert_allclose(result, points * expected_scale)


def test_upscale_mask_output_shape():
    from nanosam.utils.preprocess import upscale_mask
    mask = np.random.randn(256, 256).astype(np.float32)
    result = upscale_mask(mask, image_shape=(480, 640), size=256)
    assert result.shape == (480, 640)


def test_upscale_mask_landscape_crops_correctly():
    """For a landscape image (w>h), lim_x=256 and lim_y < 256."""
    from nanosam.utils.preprocess import upscale_mask
    mask = np.ones((256, 256), dtype=np.float32)
    result = upscale_mask(mask, image_shape=(480, 640), size=256)
    assert result.shape == (480, 640)
