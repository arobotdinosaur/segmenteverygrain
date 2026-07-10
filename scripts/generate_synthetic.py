#!/usr/bin/env python
"""Generate synthetic noisy image/mask pairs from clean pairs and theta."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running this file directly from scripts/ without installing the package first.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from segmenteverygrain.experiments.synthetic_generation import generate_synthetic_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-dir", required=True)
    parser.add_argument("--theta", dest="theta_path", required=True)
    parser.add_argument("--reference-noisy-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--manifest", default=None)
    args = parser.parse_args()

    # This turns clean image/mask pairs into noisy-looking training pairs using saved theta.
    summary = generate_synthetic_dataset(
        clean_dir=args.clean_dir,
        output_dir=args.output_dir,
        theta_path=args.theta_path,
        reference_noisy_dir=args.reference_noisy_dir,
        seed=args.seed,
        manifest_path=args.manifest,
    )
    print(f"Generated {summary['n_pairs']} synthetic pairs in {summary['output_dir']}")


if __name__ == "__main__":
    main()
