"""Visual inference demo using ONNX Runtime backend."""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image

from nanosam.utils.onnx_predictor import OnnxPredictor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_encoder", type=str, default="data/resnet18_image_encoder.onnx")
    parser.add_argument("--mask_decoder", type=str, default="data/mobile_sam_mask_decoder.onnx")
    parser.add_argument("--image", type=str, default="assets/dogs.jpg")
    parser.add_argument("--output", type=str, default="data/basic_usage_onnx_out.jpg")
    parser.add_argument("--encoder_size", type=int, default=1024)
    parser.add_argument("--no-normalize", action="store_true",
                        help="Skip ImageNet normalization (for 'nonorm' encoders like PPHGV2)")
    args = parser.parse_args()

    predictor = OnnxPredictor(
        args.image_encoder, args.mask_decoder,
        image_encoder_size=args.encoder_size,
        normalize_input=not args.no_normalize,
    )

    image = PIL.Image.open(args.image)
    predictor.set_image(image)

    # Same bounding box as examples/basic_usage.py
    bbox = [100, 100, 850, 759]
    points = np.array([[bbox[0], bbox[1]], [bbox[2], bbox[3]]])
    point_labels = np.array([2, 3])

    hi_res_mask, iou_preds, _ = predictor.predict(points, point_labels)
    mask = hi_res_mask > 0

    print(f"Best IoU score: {iou_preds[0].max():.4f}")
    print(f"Mask coverage: {mask.sum()}/{mask.size} pixels ({100*mask.sum()/mask.size:.1f}%)")

    plt.figure(figsize=(10, 8))
    plt.imshow(image)
    plt.imshow(mask, alpha=0.5)
    x = [bbox[0], bbox[2], bbox[2], bbox[0], bbox[0]]
    y = [bbox[1], bbox[1], bbox[3], bbox[3], bbox[1]]
    plt.plot(x, y, "g-")
    plt.title("NanoSAM ONNX Runtime — Bounding Box Segmentation")
    plt.savefig(args.output, dpi=100, bbox_inches="tight")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
