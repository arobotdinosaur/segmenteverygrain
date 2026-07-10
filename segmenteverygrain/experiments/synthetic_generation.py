"""Synthetic noisy image generation stage."""

from __future__ import annotations

from pathlib import Path

from create_synthetic_images import generate_synthetic_images

from .io import load_image_mask_pairs, pairs_to_dicts, write_json
from .noise_fit import load_theta


def generate_synthetic_dataset(
    clean_dir: str | Path,
    output_dir: str | Path,
    *,
    theta_path: str | Path | None = None,
    theta_values=None,
    reference_noisy_dir: str | Path,
    seed: int = 42,
    manifest_path: str | Path | None = None,
) -> dict:
    """Generate synthetic noisy image/mask pairs from clean image/mask pairs."""

    # Theta is the fitted noise recipe from BO; it controls how clean images are corrupted.
    # The clean masks are preserved so the U-Net still has correct labels to learn from.
    theta = load_theta(theta_path, theta_values)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # This calls the existing image generator, but wraps it in a reproducible experiment
    # stage that also writes a manifest describing what was created.
    generate_synthetic_images(
        theta=theta,
        input_folder=str(clean_dir),
        output_folder=str(output_dir),
        noise_reference_folder=str(reference_noisy_dir),
        seed=seed,
    )

    pairs = load_image_mask_pairs(output_dir, source="synthetic", require_pairs=True)
    summary = {
        "clean_dir": str(clean_dir),
        "reference_noisy_dir": str(reference_noisy_dir),
        "output_dir": str(output_dir),
        "theta": theta,
        "seed": seed,
        "n_pairs": len(pairs),
        "pairs": pairs_to_dicts(pairs),
    }
    if manifest_path is not None:
        write_json(manifest_path, summary)
    else:
        write_json(output_dir / "synthetic_manifest.json", summary)
    return summary
