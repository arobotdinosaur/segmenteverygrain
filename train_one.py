"""Train multiple iterations of a given theta, each saved as name_N.

Usage:
    python train_one.py --name my_experiment --n-runs 5 0.002 0.002 0.001 0.76 0.07

Trains 5 models with tags my_experiment_0, my_experiment_1, ..., my_experiment_4
for the same theta. Results are collected and summarized.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from surrogate_gp import (
    black_box,
    CLEAN_PATH,
    TARGET_PATH,
    PREDICT_PATH,
    bounds,
)


def main():
    parser = argparse.ArgumentParser(
        description="Train multiple iterations of a given theta."
    )
    parser.add_argument("theta", type=float, nargs=5,
                        help="Five theta values: a b sigma_r l k")
    parser.add_argument("--name", type=str, default=None,
                        help="Base name for runs (suffixed with _N)")
    parser.add_argument("--n-runs", type=int, default=1,
                        help="Number of repeated training runs (default: 1)")
    parser.add_argument("--model-family", type=str, default="unet",
                        choices=["unet", "unet_modified", "resnext"])
    parser.add_argument("--model-weights", type=str, default="./models/seg_model.keras",
                        help="Pretrained weights path")
    parser.add_argument("--no-pretrained", action="store_false", dest="use_pretrained",
                        help="Train from scratch instead of fine-tuning")
    parser.add_argument("--combine-with-clean", action="store_true",
                        help="Include pre-injection clean images in training")
    parser.add_argument("--n-synthetic-variants", type=int, default=8,
                        help="Number of noisy variants per clean image")
    parser.add_argument("--n-crop-views", type=int, default=8,
                        help="Replicate each real-noisy crop region for more views")
    parser.add_argument("--dice-weight", type=float, default=1.0)
    parser.add_argument("--count-weight", type=float, default=0.4)
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save the combined results JSON")

    args = parser.parse_args()
    theta = args.theta

    base_name = args.name if args.name is not None else "_".join(f"{v:.4g}" for v in theta)

    print(f"Theta: a={theta[0]:.6f}, b={theta[1]:.6f}, sigma_r={theta[2]:.6f}, "
          f"l={theta[3]:.4f}, k={theta[4]:.4f}")
    print(f"Base name: {base_name}")
    print(f"Runs: {args.n_runs}")
    print(f"Model family: {args.model_family}")

    summaries = []
    for run_idx in range(args.n_runs):
        tag = f"{base_name}_{run_idx}"
        print(f"\n--- Run {run_idx + 1}/{args.n_runs}: {tag} ---")
        summary = black_box(
            theta,
            model_family=args.model_family,
            model_weights_file=args.model_weights,
            use_pretrained=args.use_pretrained,
            dice_weight=args.dice_weight,
            count_weight=args.count_weight,
            tag=tag,
            combine_with_clean=args.combine_with_clean,
            n_synthetic_variants=args.n_synthetic_variants,
            n_crop_views=args.n_crop_views,
        )
        summaries.append(summary)

    print("\n" + "=" * 60)
    print("=== Combined Results ===")
    print("=" * 60)
    objectives = [s["objective"] for s in summaries]
    dice_losses = [s["dice_loss"] for s in summaries]
    count_penalties = [s["count_penalty"] for s in summaries]
    val_losses = [s["metrics"]["val_loss"] for s in summaries]
    val_accs = [s["metrics"]["val_accuracy"] for s in summaries]
    test_losses = [s["metrics"]["test_loss"] for s in summaries]
    test_accs = [s["metrics"]["test_accuracy"] for s in summaries]

    for i, s in enumerate(summaries):
        print(f"  [{i}] objective={s['objective']:.6f}  dice={s['dice_loss']:.6f}  "
              f"count={s['count_penalty']:.6f}  val_loss={s['metrics']['val_loss']:.6f}  "
              f"model={s['metrics']['model_path']}")

    def _stats(vals):
        return min(vals), max(vals), sum(vals) / len(vals)

    lo, hi, avg = _stats(objectives)
    print(f"\n  Objective:  min={lo:.6f}  max={hi:.6f}  avg={avg:.6f}")
    lo, hi, avg = _stats(dice_losses)
    print(f"  Dice loss:  min={lo:.6f}  max={hi:.6f}  avg={avg:.6f}")
    lo, hi, avg = _stats(count_penalties)
    print(f"  Count pen:  min={lo:.6f}  max={hi:.6f}  avg={avg:.6f}")
    lo, hi, avg = _stats(val_losses)
    print(f"  Val loss:   min={lo:.6f}  max={hi:.6f}  avg={avg:.6f}")
    lo, hi, avg = _stats(test_losses)
    print(f"  Test loss:  min={lo:.6f}  max={hi:.6f}  avg={avg:.6f}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump({
                "theta": list(theta),
                "base_name": base_name,
                "n_runs": args.n_runs,
                "runs": summaries,
                "summary": {
                    "objective": {
                        "min": min(objectives), "max": max(objectives),
                        "mean": sum(objectives) / len(objectives),
                    },
                    "dice_loss": {
                        "min": min(dice_losses), "max": max(dice_losses),
                        "mean": sum(dice_losses) / len(dice_losses),
                    },
                    "count_penalty": {
                        "min": min(count_penalties), "max": max(count_penalties),
                        "mean": sum(count_penalties) / len(count_penalties),
                    },
                },
            }, f, indent=2)
        print(f"\nFull results written to {args.output}")


if __name__ == "__main__":
    main()
