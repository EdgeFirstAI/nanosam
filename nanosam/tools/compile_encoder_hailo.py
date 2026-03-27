"""Compile NanoSAM ResNet18 encoder to Hailo-8L HEF using DFC ClientRunner API.

Run from the Hailo DFC venv (Python 3.11):
    cd ~/Software/Studio/hailo
    source venv/bin/activate
    python ~/Software/SAM/nanosam/nanosam/tools/compile_encoder_hailo.py

Requires:
    hailo-dataflow-compiler 3.33.1 installed in the active venv.

Outputs (relative to nanosam repo root):
    data/resnet18_encoder_h8l.hef   — HEF binary for RPi5 / Hailo-8L
    data/resnet18_encoder_h8l.har   — HAR archive (quantized weights, metadata)
"""

import os
import sys
import glob
import argparse
import numpy as np
from pathlib import Path
from PIL import Image

# ---------------------------------------------------------------------------
# Paths — adjust if your checkout locations differ
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
ONNX_PATH = REPO_ROOT / "data" / "resnet18_image_encoder_legacy.onnx"
ALLS_PATH = Path(__file__).resolve().parent / "resnet18_encoder.alls"
OUTPUT_DIR = REPO_ROOT / "data"
COCO_VAL_DIR = REPO_ROOT / "data" / "coco" / "val2017"

HW_ARCH = "hailo8l"
NET_NAME = "resnet18_encoder"
INPUT_NAME = "image"
INPUT_SIZE = 1024   # encoder input resolution (square)
CALIB_COUNT = 200   # calibration images


def load_calibration_images(coco_dir: Path, count: int, size: int) -> np.ndarray:
    """Load COCO val2017 images as raw [0, 255] float32 NHWC array.

    The Hailo DFC expects calibration data in the same format as runtime input,
    i.e. RAW pixels (normalization is folded into the .alls model script).
    """
    jpg_files = sorted(coco_dir.glob("*.jpg"))[:count]
    if not jpg_files:
        raise FileNotFoundError(f"No JPEG files found in {coco_dir}")

    print(f"Loading {len(jpg_files)} calibration images at {size}x{size}...")
    calib = np.zeros((len(jpg_files), size, size, 3), dtype=np.float32)
    for i, p in enumerate(jpg_files):
        img = Image.open(p).convert("RGB")
        # Aspect-ratio-preserving resize + zero-pad (matches preprocess_image behaviour)
        aspect = img.width / img.height
        if aspect >= 1:
            rw, rh = size, int(size / aspect)
        else:
            rw, rh = int(size * aspect), size
        img = img.resize((rw, rh), Image.BILINEAR)
        arr = np.array(img, dtype=np.float32)   # HWC [0,255]
        calib[i, :rh, :rw, :] = arr
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(jpg_files)}")

    print(f"Calibration array: {calib.shape}, min={calib.min():.0f}, max={calib.max():.0f}")
    return calib


def compile_encoder(onnx_path: Path, alls_path: Path, output_dir: Path,
                    calib_data: np.ndarray, hw_arch: str) -> Path:
    from hailo_sdk_client import ClientRunner

    print(f"\n{'='*60}")
    print(f"Compiling NanoSAM ResNet18 encoder → {hw_arch}")
    print(f"  ONNX : {onnx_path}")
    print(f"  ALLS : {alls_path}")
    print(f"{'='*60}")

    # Step 1 — Translate ONNX
    print("\n[1/4] Translating ONNX model...")
    runner = ClientRunner(hw_arch=hw_arch)
    runner.translate_onnx_model(
        model=str(onnx_path),
        net_name=NET_NAME,
        net_input_shapes={INPUT_NAME: [1, 3, INPUT_SIZE, INPUT_SIZE]},
        # No end_node_names needed — encoder has a single output node
    )
    print("  Translation complete")

    # Step 2 — Load model script
    print(f"\n[2/4] Loading model script: {alls_path}")
    runner.load_model_script(str(alls_path))
    print("  Model script loaded")

    # Step 3 — Optimize (quantize)
    print(f"\n[3/4] Optimizing ({calib_data.shape[0]} calibration images)...")
    runner.optimize(calib_data)
    print("  Optimization complete")

    # Save HAR
    har_path = output_dir / f"{NET_NAME}_h8l.har"
    runner.save_har(str(har_path))
    print(f"  Saved HAR: {har_path} ({har_path.stat().st_size / 1e6:.1f} MB)")

    # Step 4 — Compile to HEF
    print("\n[4/4] Compiling to HEF...")
    hef_bytes = runner.compile()
    hef_path = output_dir / f"{NET_NAME}_h8l.hef"
    hef_path.write_bytes(hef_bytes)
    print(f"  Saved HEF: {hef_path} ({hef_path.stat().st_size / 1e6:.1f} MB)")

    return hef_path


def main():
    parser = argparse.ArgumentParser(description="Compile NanoSAM encoder for Hailo-8L")
    parser.add_argument("--onnx", type=Path, default=ONNX_PATH, help="Encoder ONNX path")
    parser.add_argument("--alls", type=Path, default=ALLS_PATH, help=".alls model script path")
    parser.add_argument("--coco", type=Path, default=COCO_VAL_DIR, help="COCO val2017 image dir")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR, help="Output directory")
    parser.add_argument("--calib-count", type=int, default=CALIB_COUNT,
                        help="Number of calibration images")
    args = parser.parse_args()

    for p, label in [(args.onnx, "ONNX"), (args.alls, ".alls"), (args.coco, "COCO dir")]:
        if not p.exists():
            print(f"ERROR: {label} not found: {p}", file=sys.stderr)
            sys.exit(1)

    args.output.mkdir(parents=True, exist_ok=True)

    calib = load_calibration_images(args.coco, args.calib_count, INPUT_SIZE)
    hef_path = compile_encoder(args.onnx, args.alls, args.output, calib, HW_ARCH)

    print(f"\nDone. HEF ready for deployment: {hef_path}")


if __name__ == "__main__":
    main()
