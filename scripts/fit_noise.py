#!/usr/bin/env python
"""Fit synthetic-noise parameters from clean and reference non-clean images."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running this file directly from scripts/ without installing the package first.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from segmenteverygrain.experiments.noise_fit import fit_noise_parameters


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-dir", required=True)
    parser.add_argument("--reference-noisy-dir", required=True)
    parser.add_argument("--output", default="runs/theta.json")
    parser.add_argument("--maxiter", type=int, default=20)
    parser.add_argument("--popsize", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--no-polish", action="store_true")
    args = parser.parse_args()

    # This launches the BO/noise-fitting step and writes theta so later stages can reuse it.
    summary = fit_noise_parameters(
        args.clean_dir,
        args.reference_noisy_dir,
        output_path=args.output,
        maxiter=args.maxiter,
        popsize=args.popsize,
        seed=args.seed,
        max_images=args.max_images,
        polish=not args.no_polish,
    )
    print(f"Saved theta to {args.output}")
    print(f"theta = {summary['theta']}")
    print(f"loss = {summary['loss']}")


if __name__ == "__main__":
    main()
