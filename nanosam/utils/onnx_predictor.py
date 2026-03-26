"""ONNX Runtime-based predictor for NanoSAM (CPU, no TensorRT dependency)."""

import numpy as np
import PIL.Image
import onnxruntime as ort

from nanosam.utils.preprocess import preprocess_image, preprocess_points, upscale_mask


class OnnxPredictor:
    def __init__(
        self,
        image_encoder_path: str,
        mask_decoder_path: str,
        image_encoder_size: int = 1024,
        decoder_coord_size: int = 1024,
        normalize_input: bool = True,
    ):
        providers = ["CPUExecutionProvider"]
        self.encoder_session = ort.InferenceSession(image_encoder_path, providers=providers)
        self.decoder_session = ort.InferenceSession(mask_decoder_path, providers=providers)
        self.image_encoder_size = image_encoder_size
        # The SAM mask decoder always expects point coordinates scaled to the
        # decoder's training coordinate space (1024 for standard SAM decoders).
        # This may differ from the image encoder's input resolution.
        self.decoder_coord_size = decoder_coord_size
        # Set to False for "nonorm" encoders (e.g. PPHGV2) that expect raw [0,255]
        self.normalize_input = normalize_input
        self.image = None
        self.features = None

    def set_image(self, image):
        """Run the encoder on an image, caching the embeddings."""
        if isinstance(image, np.ndarray):
            image = PIL.Image.fromarray(image)
        self.image = image
        image_tensor = preprocess_image(
            image, size=self.image_encoder_size, normalize=self.normalize_input
        )
        outputs = self.encoder_session.run(None, {"image": image_tensor})
        self.features = outputs[0]  # shape: (1, 256, 64, 64)

    def predict(self, points, point_labels, mask_input=None):
        """Run the decoder with point/box prompts.

        Returns: (hi_res_mask, iou_predictions, low_res_masks)
        """
        assert self.features is not None, "Call set_image() first"

        scaled_points = preprocess_points(
            points,
            image_size=(self.image.height, self.image.width),
            size=self.decoder_coord_size,
        )

        # Prepare decoder inputs as float32 numpy arrays
        point_coords = np.array([scaled_points], dtype=np.float32)  # (1, N, 2)
        point_labels_arr = np.array([point_labels], dtype=np.float32)  # (1, N)

        if mask_input is None:
            mask_input_arr = np.zeros((1, 1, 256, 256), dtype=np.float32)
            has_mask_input = np.array([0.0], dtype=np.float32)
        else:
            mask_input_arr = np.array(mask_input, dtype=np.float32)
            has_mask_input = np.array([1.0], dtype=np.float32)

        decoder_inputs = {
            "image_embeddings": self.features,
            "point_coords": point_coords,
            "point_labels": point_labels_arr,
            "mask_input": mask_input_arr,
            "has_mask_input": has_mask_input,
        }

        # ort_session.run returns [iou_predictions, low_res_masks]
        # iou_predictions: (1, 4), low_res_masks: (1, 4, 256, 256)
        outputs = self.decoder_session.run(None, decoder_inputs)
        iou_predictions = outputs[0]
        low_res_masks = outputs[1]

        # Select best mask by IoU score
        best_idx = iou_predictions[0].argmax()
        best_mask = low_res_masks[0, best_idx]  # (256, 256)

        # Upscale to original image dimensions
        hi_res_mask = upscale_mask(
            best_mask,
            image_shape=(self.image.height, self.image.width),
        )

        return hi_res_mask, iou_predictions, low_res_masks
