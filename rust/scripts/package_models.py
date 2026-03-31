#!/usr/bin/env python3
"""Package NanoSAM model artifacts into a deployment ZIP archive.

Creates an uncompressed ZIP with canonical model names. The packager selects
the appropriate per-target model variant at packaging time — the same binary
works with different target-specific archives.

Canonical archive layout:
    encoder.tflite              — Image encoder (INT8, platform-specific)
    attention.tflite            — Transformer attention (FP16 or FP32)
    heads_a.tflite              — Decoder heads Part A (INT8, platform-specific)
    heads_b.tflite              — Decoder heads Part B (INT8, platform-specific)
    tokens.tflite               — Token MLP (dynamic range quantized)
    prompt_encoder.safetensors  — Prompt encoder weights

Usage:
    # i.MX95 (Neutron NPU)
    python scripts/package_models.py \\
        --encoder ../runs/resnet18_e200/tflite_twin/encoder_neutron.tflite \\
        --attention ../../sam-decoder/models/sam_decoder_attn_xnnpack_fp16.tflite \\
        --heads-a ../../sam-decoder/models/heads_part_a_neutron.tflite \\
        --heads-b ../../sam-decoder/models/heads_part_b_neutron.tflite \\
        --tokens ../../sam-decoder/models/heads_tokens_dynq.tflite \\
        --prompt-weights ../../sam-decoder/models/prompt_encoder_weights.safetensors \\
        --output nanosam_imx95.zip

    # i.MX 8M Plus (standard INT8 NPU)
    python scripts/package_models.py \\
        --encoder ../runs/resnet18_e200/tflite_twin/encoder_int8.tflite \\
        --attention ../../sam-decoder/models/sam_decoder_attn_static_float32.tflite \\
        --heads-a ../../sam-decoder/models/heads_part_a_int8.tflite \\
        --heads-b ../../sam-decoder/models/heads_part_b_int8.tflite \\
        --tokens ../../sam-decoder/models/heads_tokens_dynq.tflite \\
        --prompt-weights ../../sam-decoder/models/prompt_encoder_weights.safetensors \\
        --output nanosam_imx8mp.zip
"""

import argparse
import os
import zipfile
from pathlib import Path


CANONICAL_NAMES = {
    "encoder": "encoder.tflite",
    "attention": "attention.tflite",
    "heads_a": "heads_a.tflite",
    "heads_b": "heads_b.tflite",
    "tokens": "tokens.tflite",
    "prompt_weights": "prompt_encoder.safetensors",
}


def main():
    parser = argparse.ArgumentParser(
        description="Package NanoSAM models into a deployment ZIP archive"
    )
    parser.add_argument("--encoder", type=Path, required=True,
                        help="INT8 TFLite encoder (platform-specific)")
    parser.add_argument("--attention", type=Path, required=True,
                        help="Attention TFLite model (FP16 or FP32)")
    parser.add_argument("--heads-a", type=Path, required=True,
                        help="Heads Part A TFLite (INT8, platform-specific)")
    parser.add_argument("--heads-b", type=Path, required=True,
                        help="Heads Part B TFLite (INT8, platform-specific)")
    parser.add_argument("--tokens", type=Path, required=True,
                        help="Tokens TFLite (dynamic range quantized)")
    parser.add_argument("--prompt-weights", type=Path, required=True,
                        help="Prompt encoder weights (.safetensors)")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output ZIP archive path")
    args = parser.parse_args()

    # Validate all inputs exist
    inputs = {
        "encoder": args.encoder,
        "attention": args.attention,
        "heads_a": args.heads_a,
        "heads_b": args.heads_b,
        "tokens": args.tokens,
        "prompt_weights": args.prompt_weights,
    }

    for name, path in inputs.items():
        if not path.exists():
            print(f"Error: {name} not found: {path}")
            raise SystemExit(1)

    # Create uncompressed ZIP
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(str(args.output), "w", compression=zipfile.ZIP_STORED) as zf:
        total = 0
        print(f"Packaging models into {args.output}:")
        for key, source_path in inputs.items():
            canonical = CANONICAL_NAMES[key]
            size = source_path.stat().st_size
            total += size
            size_str = f"{size / 1048576:.1f} MB" if size > 1_000_000 else f"{size / 1024:.1f} KB"
            print(f"  {canonical:<36} {size_str:>8}  ← {source_path.name}")
            zf.write(str(source_path), canonical)

    archive_size = args.output.stat().st_size
    print(f"\nArchive: {args.output} ({archive_size / 1048576:.1f} MB)")
    print(f"  Models total: {total / 1048576:.1f} MB")


if __name__ == "__main__":
    main()
