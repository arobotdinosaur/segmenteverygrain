"""Fit synthetic-noise parameters from clean and non-clean image folders."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution

from synthetic_noise import NoiseParams, load_images_from_folder, objective

from .io import write_json


DEFAULT_BOUNDS = [
    # These ranges tell Bayesian/global optimization what noise recipes are allowed.
    # Keeping the bounds explicit makes each synthetic-noise run easier to reproduce.
    (1e-6, 0.2),   # a: signal-dependent variance slope
    (1e-6, 0.05),  # b: signal-independent variance floor
    (1e-6, 0.05),  # sigma_r: row-wise offset strength
    (0.3, 8.0),    # l: correlated-noise blur length
    (1e-6, 0.1),   # k: correlated-noise strength
]


def theta_to_params(theta) -> NoiseParams:
    theta = np.asarray(theta, dtype=float)
    if theta.shape[0] != 5:
        raise ValueError("theta must have five values: [a, b, sigma_r, l, k]")
    return NoiseParams(
        a=float(theta[0]),
        b=float(theta[1]),
        sigma_r=float(theta[2]),
        l=float(theta[3]),
        k=float(theta[4]),
    )


def params_to_theta(params: NoiseParams) -> list[float]:
    return [params.a, params.b, params.sigma_r, params.l, params.k]


def fit_noise_parameters(
    clean_dir: str | Path,
    reference_noisy_dir: str | Path,
    *,
    output_path: str | Path | None = None,
    maxiter: int = 20,
    popsize: int = 10,
    seed: int = 2,
    max_images: int | None = None,
    polish: bool = True,
    bounds=DEFAULT_BOUNDS,
) -> dict:
    """Fit theta by matching synthetic noise to reference non-clean images."""

    # This stage only learns the noise settings; it does not train the U-Net.
    # It compares clean images plus simulated noise against the real non-clean examples.
    clean_images, clean_paths = load_images_from_folder(str(clean_dir))
    noisy_images, noisy_paths = load_images_from_folder(str(reference_noisy_dir))

    if max_images is not None:
        clean_images = clean_images[:max_images]
        clean_paths = clean_paths[:max_images]
        noisy_images = noisy_images[:max_images]
        noisy_paths = noisy_paths[:max_images]

    if not clean_images:
        raise ValueError(f"No clean images found in {clean_dir}")
    if not noisy_images:
        raise ValueError(f"No reference non-clean images found in {reference_noisy_dir}")

    result = differential_evolution(
        # Differential evolution tries many theta candidates and keeps improving them
        # until the synthetic images statistically resemble the reference noisy images.
        objective,
        bounds=bounds,
        args=(clean_images, noisy_images, seed),
        maxiter=maxiter,
        popsize=popsize,
        polish=polish,
        seed=seed,
        disp=True,
    )
    params = theta_to_params(result.x)
    summary = {
        "theta": params_to_theta(params),
        "params": asdict(params),
        "loss": float(result.fun),
        "success": bool(result.success),
        "message": str(result.message),
        "clean_dir": str(clean_dir),
        "reference_noisy_dir": str(reference_noisy_dir),
        "clean_images": clean_paths,
        "reference_noisy_images": noisy_paths,
        "maxiter": maxiter,
        "popsize": popsize,
        "seed": seed,
    }
    if output_path is not None:
        write_json(output_path, summary)
    return summary


def load_theta(theta_path: str | Path | None = None, theta_values=None) -> list[float]:
    """Load theta from a JSON file or direct values."""

    # Scripts can pass theta directly for quick tests, or load it from a saved BO result.
    # Both paths become the same five-number list before image generation starts.
    if theta_values is not None:
        return [float(v) for v in theta_values]
    if theta_path is None:
        raise ValueError("Provide theta_path or theta_values.")

    import json

    with open(theta_path) as f:
        data = json.load(f)
    if isinstance(data, list):
        theta = data
    elif "theta" in data:
        theta = data["theta"]
    elif "params" in data:
        params = data["params"]
        theta = [params["a"], params["b"], params["sigma_r"], params["l"], params["k"]]
    else:
        raise ValueError(f"Could not find theta in {theta_path}")
    return [float(v) for v in theta]
