#!/usr/bin/env python
"""Stage clean/synthetic/real non-clean image-mask pairs into one training folder."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running this file directly from scripts/ without installing the package first.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from segmenteverygrain.experiments.dataset_builder import build_training_set


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--clean-dir", default=None)
    parser.add_argument("--synthetic-dir", default=None)
    parser.add_argument("--real-noisy-dir", default=None)
    parser.add_argument("--no-reset", action="store_true")
    args = parser.parse_args()

    # This copies the selected data sources into one staged folder that training can read.
    summary = build_training_set(
        output_dir=args.output_dir,
        clean_dir=args.clean_dir,
        synthetic_dir=args.synthetic_dir,
        real_noisy_dir=args.real_noisy_dir,
        reset=not args.no_reset,
    )
    print(f"Staged {summary['n_pairs']} pairs into {summary['staged_dir']}")
    print(f"Counts by source: {summary['counts_by_source']}")


if __name__ == "__main__":
    main()
