#!/usr/bin/env python
"""Evaluate a trained U-Net on held-out image/mask pairs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running this file directly from scripts/ without installing the package first.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from segmenteverygrain.experiments.evaluation import evaluate_unet_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--eval-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--dice-weight", type=float, default=1.0)
    parser.add_argument("--count-weight", type=float, default=0.4)
    parser.add_argument("--no-overlays", action="store_true")
    args = parser.parse_args()

    # This scores a trained model on held-out paired images and optionally saves overlays.
    summary = evaluate_unet_model(
        model_path=args.model,
        eval_dir=args.eval_dir,
        output_dir=args.output_dir,
        tile_size=args.tile_size,
        dice_weight=args.dice_weight,
        count_weight=args.count_weight,
        save_overlays=not args.no_overlays,
    )
    print(f"Saved metrics to {args.output_dir}/metrics.json")
    print(f"objective = {summary['objective']}")
    print(f"dice_loss = {summary['dice_loss']}")
    print(f"count_penalty = {summary['count_penalty']}")


if __name__ == "__main__":
    main()
