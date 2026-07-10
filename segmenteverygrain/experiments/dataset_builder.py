"""Build explicit training sets from clean, synthetic, and real non-clean data."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from .io import (
    ImageMaskPair,
    copy_pairs_to_folder,
    count_by_source,
    load_image_mask_pairs,
    pairs_to_dicts,
    write_json,
)


def collect_training_pairs(
    *,
    clean_dir: str | Path | None = None,
    synthetic_dir: str | Path | None = None,
    real_noisy_dir: str | Path | None = None,
) -> list[ImageMaskPair]:
    """Collect pairs from any non-empty combination of data sources."""

    # This is the switchboard for the four experiment types: choose clean only,
    # clean plus synthetic, clean plus real noisy, or all of them together.
    pairs: list[ImageMaskPair] = []
    pairs.extend(load_image_mask_pairs(clean_dir, source="clean"))
    pairs.extend(load_image_mask_pairs(synthetic_dir, source="synthetic"))
    pairs.extend(load_image_mask_pairs(real_noisy_dir, source="real_noisy"))
    if not pairs:
        raise ValueError(
            "No training pairs found. Provide at least one non-empty clean, "
            "synthetic, or real non-clean directory."
        )
    return pairs


def build_training_set(
    *,
    output_dir: str | Path,
    clean_dir: str | Path | None = None,
    synthetic_dir: str | Path | None = None,
    real_noisy_dir: str | Path | None = None,
    reset: bool = True,
) -> dict:
    """Stage selected image/mask pairs into one folder and save a manifest."""

    # Training becomes simpler after staging because the U-Net code only needs one folder.
    # The manifest keeps the original source information so we can audit the mixture later.
    pairs = collect_training_pairs(
        clean_dir=clean_dir,
        synthetic_dir=synthetic_dir,
        real_noisy_dir=real_noisy_dir,
    )
    staged_dir, staged_manifest = copy_pairs_to_folder(pairs, output_dir, reset=reset)
    summary = {
        "staged_dir": str(staged_dir),
        "clean_dir": str(clean_dir) if clean_dir else None,
        "synthetic_dir": str(synthetic_dir) if synthetic_dir else None,
        "real_noisy_dir": str(real_noisy_dir) if real_noisy_dir else None,
        "n_pairs": len(pairs),
        "counts_by_source": count_by_source(pairs),
        "input_pairs": pairs_to_dicts(pairs),
        "staged_pairs": staged_manifest,
    }
    write_json(Path(staged_dir) / "training_manifest.json", summary)
    return summary


def split_real_noisy_pairs_for_r_experiment(
    pairs: Sequence[ImageMaskPair],
    r: int,
) -> tuple[list[ImageMaskPair], list[ImageMaskPair]]:
    """Split real non-clean pairs into theta-fit and training subsets."""

    # Experiment 4 uses this idea: fit theta on r real images, then train with the remaining
    # 3-r real images so fitting data and training data are not accidentally mixed.
    if r < 0:
        raise ValueError("r must be non-negative")
    if r > len(pairs):
        raise ValueError(f"r={r} is larger than the number of pairs ({len(pairs)})")
    return list(pairs[:r]), list(pairs[r:])
