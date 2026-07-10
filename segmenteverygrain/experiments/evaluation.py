"""Evaluation helpers for trained Segment Every Grain U-Net models."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import segmenteverygrain as seg

from .io import ensure_dir, load_image_mask_pairs, write_json
from .training import load_keras_unet


def dice_loss(predicted_probs, true_masks, eps=1e-7):
    """Multi-class Dice loss averaged over images. Lower is better."""

    total = 0.0
    for pred, true in zip(predicted_probs, true_masks):
        h, w = min(pred.shape[0], true.shape[0]), min(pred.shape[1], true.shape[1])
        pred, true = pred[:h, :w], true[:h, :w]
        one_hot = np.eye(pred.shape[-1])[true.astype(np.uint8)]
        intersection = np.sum(pred * one_hot, axis=(0, 1))
        union = np.sum(pred + one_hot, axis=(0, 1))
        total += float(1 - np.mean((2 * intersection + eps) / (union + eps)))
    return total / max(1, len(predicted_probs))


def count_penalty(predicted_probs, true_masks):
    """Normalized absolute difference in number of connected grain regions."""

    total = 0.0
    for pred, true in zip(predicted_probs, true_masks):
        h, w = min(pred.shape[0], true.shape[0]), min(pred.shape[1], true.shape[1])
        pred_label = np.argmax(pred[:h, :w], axis=-1).astype(np.uint8)
        true_label = true[:h, :w].astype(np.uint8)
        n_pred = max(cv2.connectedComponents((pred_label == 1).astype(np.uint8))[0] - 1, 0)
        n_true = max(cv2.connectedComponents((true_label == 1).astype(np.uint8))[0] - 1, 0)
        total += abs(n_pred - n_true) / max(n_true, 1)
    return total / max(1, len(predicted_probs))


def compute_mask_loss(predicted_probs, true_masks, dice_weight=1.0, count_weight=0.4):
    # The objective mixes pixel overlap with grain-count accuracy, so a model is rewarded
    # for drawing the right shapes and finding roughly the right number of grains.
    dice = dice_loss(predicted_probs, true_masks)
    count = count_penalty(predicted_probs, true_masks)
    return dice_weight * dice + count_weight * count


def predict_eval_pairs(model, eval_dir: str | Path, *, tile_size: int = 256):
    """Predict probability maps for all image/mask pairs in eval_dir."""

    # Evaluation uses paired held-out images and masks, then stores raw probability maps
    # so the scoring functions can compare predictions against the true masks.
    pairs = load_image_mask_pairs(eval_dir, source="eval", require_pairs=True)
    pred_probs = []
    true_masks = []
    for pair in pairs:
        image = cv2.imread(pair.image, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not read evaluation image: {pair.image}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pred = seg.predict_image_mirror(image, model, tile_size)
        pred_probs.append(pred)
        mask = cv2.imread(pair.mask, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Could not read evaluation mask: {pair.mask}")
        true_masks.append(mask.astype(np.uint8))
    return pairs, pred_probs, true_masks


def save_prediction_overlays(pairs, pred_probs, output_dir: str | Path) -> list[str]:
    """Save simple RGB overlays of predicted labels over input images."""

    # Overlays are not used for training; they are visual sanity checks so we can quickly
    # see where the model is segmenting grains well or badly.
    output_dir = ensure_dir(output_dir)
    outputs = []
    color_map = np.array(
        [
            [0, 0, 0],
            [70, 130, 180],
            [255, 165, 0],
        ],
        dtype=np.uint8,
    )
    for pair, pred in zip(pairs, pred_probs):
        image = cv2.imread(pair.image, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = np.argmax(pred, axis=-1).astype(np.uint8)
        label_rgb = color_map[label]
        h, w = min(image.shape[0], label_rgb.shape[0]), min(image.shape[1], label_rgb.shape[1])
        overlay = (0.65 * image[:h, :w] + 0.35 * label_rgb[:h, :w]).astype(np.uint8)
        out_path = output_dir / f"{pair.base_name}_prediction_overlay.png"
        Image.fromarray(overlay).save(out_path)
        outputs.append(str(out_path))
    return outputs


def evaluate_unet_model(
    *,
    model_path: str | Path,
    eval_dir: str | Path,
    output_dir: str | Path,
    tile_size: int = 256,
    dice_weight: float = 1.0,
    count_weight: float = 0.4,
    save_overlays: bool = True,
) -> dict:
    """Evaluate a saved U-Net model on held-out paired images/masks."""

    # This is the final scoring step: load the model, predict masks, compute metrics,
    # and save everything needed to compare experiment recipes.
    output_dir = ensure_dir(output_dir)
    model = load_keras_unet(model_path)
    pairs, pred_probs, true_masks = predict_eval_pairs(model, eval_dir, tile_size=tile_size)

    per_image = []
    for pair, pred, true in zip(pairs, pred_probs, true_masks):
        per_image.append(
            {
                "base_name": pair.base_name,
                "image": pair.image,
                "mask": pair.mask,
                "dice_loss": dice_loss([pred], [true]),
                "count_penalty": count_penalty([pred], [true]),
            }
        )

    summary = {
        "model_path": str(model_path),
        "eval_dir": str(eval_dir),
        "tile_size": tile_size,
        "dice_weight": dice_weight,
        "count_weight": count_weight,
        "n_images": len(pairs),
        "dice_loss": dice_loss(pred_probs, true_masks),
        "count_penalty": count_penalty(pred_probs, true_masks),
        "objective": compute_mask_loss(pred_probs, true_masks, dice_weight, count_weight),
        "per_image": per_image,
    }
    if save_overlays:
        summary["overlays"] = save_prediction_overlays(
            pairs,
            pred_probs,
            output_dir / "overlays",
        )
    write_json(output_dir / "metrics.json", summary)
    return summary
