import numpy as np
import PIL.Image
import pytest
import os

ENCODER_PATH = "data/resnet18_image_encoder.onnx"
DECODER_PATH = "data/mobile_sam_mask_decoder.onnx"

pytestmark = pytest.mark.skipif(
    not (os.path.exists(ENCODER_PATH) and os.path.exists(DECODER_PATH)),
    reason="ONNX model files not available"
)


def test_onnx_predictor_set_image():
    from nanosam.utils.onnx_predictor import OnnxPredictor
    predictor = OnnxPredictor(ENCODER_PATH, DECODER_PATH)
    image = PIL.Image.open("assets/dogs.jpg")
    predictor.set_image(image)
    assert predictor.features is not None
    assert predictor.features.shape == (1, 256, 64, 64)


def test_onnx_predictor_predict_box():
    from nanosam.utils.onnx_predictor import OnnxPredictor
    predictor = OnnxPredictor(ENCODER_PATH, DECODER_PATH)
    image = PIL.Image.open("assets/dogs.jpg")
    predictor.set_image(image)

    bbox = [100, 100, 850, 759]
    points = np.array([[bbox[0], bbox[1]], [bbox[2], bbox[3]]])
    point_labels = np.array([2, 3])

    hi_res_mask, iou_preds, low_res_mask = predictor.predict(points, point_labels)

    assert hi_res_mask.shape == (image.height, image.width)
    assert iou_preds.shape[1] == 4  # multimask mode
    assert low_res_mask.shape[2:] == (256, 256)
    # Mask should have nonzero content (something was segmented)
    binary_mask = hi_res_mask > 0
    assert binary_mask.sum() > 0
    assert binary_mask.sum() < binary_mask.size  # not everything
