#!/usr/bin/env python
"""Build an experiment training set and train/fine-tune a U-Net."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running this file directly from scripts/ without installing the package first.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from segmenteverygrain.experiments.training import train_unet_experiment


def str_to_bool(value: str) -> bool:
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean, got {value!r}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--clean-dir", default=None)
    parser.add_argument("--synthetic-dir", default=None)
    parser.add_argument("--real-noisy-dir", default=None)
    parser.add_argument("--output-model", default=None)
    parser.add_argument("--pretrained-model", default="models/seg_model.keras")
    parser.add_argument("--model-type", choices=["unet", "unet_modified"], default="unet")
    parser.add_argument("--augmentation", type=str_to_bool, default=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--use-reduce-lr", type=str_to_bool, default=True)
    args = parser.parse_args()

    # This builds the requested training mixture, patchifies it, and fine-tunes the U-Net.
    summary = train_unet_experiment(
        run_dir=args.run_dir,
        clean_dir=args.clean_dir,
        synthetic_dir=args.synthetic_dir,
        real_noisy_dir=args.real_noisy_dir,
        output_model=args.output_model,
        pretrained_model=args.pretrained_model,
        model_type=args.model_type,
        augmentation=args.augmentation,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        use_reduce_lr=args.use_reduce_lr,
    )
    print(f"Saved model to {summary['training']['output_model']}")
    print(f"Dataset counts: {summary['dataset']['counts_by_source']}")


if __name__ == "__main__":
    main()
