#!/usr/bin/env bash
set -euo pipefail

# Require an active venv to avoid installing into global environment
if [ -z "${VIRTUAL_ENV:-}" ]; then
    echo "ERROR: No virtual environment active. Run: source venv/bin/activate"
    exit 1
fi

mkdir -p data

# Encoder ONNX from NVIDIA (Google Drive file ID: 14-SsvoaTl-esC3JOzomHDnI9OGgdO2OR)
ENCODER_ID="14-SsvoaTl-esC3JOzomHDnI9OGgdO2OR"
ENCODER_OUT="data/resnet18_image_encoder.onnx"

if [ ! -f "$ENCODER_OUT" ]; then
    echo "Downloading resnet18_image_encoder.onnx from Google Drive..."
    pip install gdown -q
    gdown "https://drive.google.com/uc?id=${ENCODER_ID}" -O "$ENCODER_OUT"
    echo "Downloaded: $ENCODER_OUT ($(du -h "$ENCODER_OUT" | cut -f1))"
else
    echo "Encoder ONNX already exists: $ENCODER_OUT"
fi
