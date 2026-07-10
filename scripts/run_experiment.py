#!/usr/bin/env python
"""Run a configured Segment Every Grain experiment end-to-end."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Allow running this file directly from scripts/ without installing the package first.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from segmenteverygrain.experiments.io import read_json, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to an experiment JSON config.")
    parser.add_argument("--skip-fit", action="store_true")
    parser.add_argument("--skip-generate", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    args = parser.parse_args()

    cfg = read_json(args.config)
    run_dir = Path(cfg["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "config.json", cfg)

    theta_path = cfg.get("theta_path") or str(run_dir / "theta.json")
    synthetic_dir = cfg.get("synthetic_dir") or str(run_dir / "synthetic_noisy")

    if cfg.get("fit_noise", False) and not args.skip_fit:
        # Heavy ML/noise imports are delayed so config checks and skipped stages start fast.
        from segmenteverygrain.experiments.noise_fit import fit_noise_parameters

        # Optional stage: learn theta from clean images and reference non-clean images.
        fit_cfg = cfg.get("fit_noise_options", {})
        fit_noise_parameters(
            cfg["clean_dir"],
            cfg["reference_noisy_dir"],
            output_path=theta_path,
            maxiter=fit_cfg.get("maxiter", 20),
            popsize=fit_cfg.get("popsize", 10),
            seed=fit_cfg.get("seed", 2),
            max_images=fit_cfg.get("max_images"),
            polish=fit_cfg.get("polish", True),
        )

    if cfg.get("generate_synthetic", False) and not args.skip_generate:
        # Optional stage: use the fitted theta to create synthetic noisy training examples.
        from segmenteverygrain.experiments.synthetic_generation import generate_synthetic_dataset

        generate_synthetic_dataset(
            clean_dir=cfg["clean_dir"],
            output_dir=synthetic_dir,
            theta_path=theta_path,
            reference_noisy_dir=cfg["reference_noisy_dir"],
            seed=cfg.get("synthetic_seed", 42),
        )

    model_path = cfg.get("output_model") or str(run_dir / "model.keras")
    if not args.skip_train:
        # Training stage: choose the configured clean/synthetic/real-noisy mixture and train.
        from segmenteverygrain.experiments.training import train_unet_experiment

        train_unet_experiment(
            run_dir=run_dir,
            clean_dir=cfg.get("train_clean_dir", cfg.get("clean_dir")),
            synthetic_dir=synthetic_dir if cfg.get("use_synthetic", False) else cfg.get("train_synthetic_dir"),
            real_noisy_dir=cfg.get("real_noisy_dir"),
            output_model=model_path,
            pretrained_model=cfg.get("pretrained_model", "models/seg_model.keras"),
            model_type=cfg.get("model_type", "unet"),
            augmentation=cfg.get("augmentation", True),
            epochs=cfg.get("epochs", 50),
            learning_rate=cfg.get("learning_rate", 1e-4),
            use_reduce_lr=cfg.get("use_reduce_lr", True),
        )

    if cfg.get("eval_dir") and not args.skip_eval:
        # Evaluation stage: score the trained model on held-out paired images/masks.
        from segmenteverygrain.experiments.evaluation import evaluate_unet_model

        evaluate_unet_model(
            model_path=model_path,
            eval_dir=cfg["eval_dir"],
            output_dir=run_dir / "evaluation",
            tile_size=cfg.get("tile_size", 256),
            dice_weight=cfg.get("dice_weight", 1.0),
            count_weight=cfg.get("count_weight", 0.4),
            save_overlays=cfg.get("save_overlays", True),
        )

    print(f"Finished configured experiment: {cfg.get('name', args.config)}")


if __name__ == "__main__":
    main()
