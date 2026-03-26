"""COCO mIoU evaluation using ONNX Runtime backend."""

import json
import argparse
import numpy as np
import tqdm
from torchvision.datasets import CocoDetection

from nanosam.utils.onnx_predictor import OnnxPredictor


def box_xywh_to_xyxy(box):
    return [box[0], box[1], box[0] + box[2], box[1] + box[3]]


def iou(mask_a, mask_b):
    intersection = np.count_nonzero(mask_a & mask_b)
    union = np.count_nonzero(mask_a | mask_b)
    if union == 0:
        return 0.0
    return intersection / union


def predict_box(predictor, image, box, set_image=True):
    if set_image:
        predictor.set_image(image)

    points = np.array([[box[0], box[1]], [box[2], box[3]]])
    point_labels = np.array([2, 3])

    hi_res_mask, iou_preds, _ = predictor.predict(points, point_labels)
    mask = hi_res_mask > 0
    return mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco_root", type=str, default="data/coco/val2017")
    parser.add_argument("--coco_ann", type=str, default="data/coco/annotations/instances_val2017.json")
    parser.add_argument("--image_encoder", type=str, default="data/resnet18_image_encoder.onnx")
    parser.add_argument("--mask_decoder", type=str, default="data/mobile_sam_mask_decoder.onnx")
    parser.add_argument("--encoder_size", type=int, default=1024)
    parser.add_argument("--decoder_coord_size", type=int, default=1024,
                        help="Coordinate space the mask decoder expects (1024 for standard SAM decoders).")
    parser.add_argument("--output", type=str, default="data/nanosam_onnx_coco_results.json")
    args = parser.parse_args()

    dataset = CocoDetection(root=args.coco_root, annFile=args.coco_ann)

    predictor = OnnxPredictor(
        image_encoder_path=args.image_encoder,
        mask_decoder_path=args.mask_decoder,
        image_encoder_size=args.encoder_size,
        decoder_coord_size=args.decoder_coord_size,
    )

    results = []

    for i in tqdm.tqdm(range(len(dataset))):
        image, anns = dataset[i]

        image_set = False
        for ann in anns:
            # Filter crowd annotations
            if ann.get("iscrowd", 0):
                continue

            box = box_xywh_to_xyxy(ann["bbox"])
            mask_coco = dataset.coco.annToMask(ann) > 0
            mask_sam = predict_box(predictor, image, box, set_image=not image_set)
            image_set = True

            results.append({
                "id": ann["id"],
                "area": ann["area"],
                "category_id": ann["category_id"],
                "iscrowd": ann["iscrowd"],
                "image_id": ann["image_id"],
                "box": box,
                "iou": iou(mask_sam, mask_coco),
            })

    with open(args.output, "w") as f:
        json.dump(results, f)

    # Print summary
    if results:
        miou = sum(r["iou"] for r in results) / len(results)
        print(f"\nmIoU (all, n={len(results)}): {miou:.4f}")
    else:
        print("\nNo results.")


if __name__ == "__main__":
    main()
