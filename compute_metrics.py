"""
Run inference on segmentation models and compute evaluation metrics.

For each image directory:
  • General metrics  — one value per model (e.g. multi-class Dice loss)
  • Class-specific metrics — one value per (model, class) combination.

Results are printed as DataFrames and saved as per-model CSV files.
Optionally produces a grouped bar-chart comparison of general metrics.

This is the metrics-and-statistics half of model_comparison.py — no overlays.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import cv2
from skimage.measure import label, regionprops
from scipy.stats import wasserstein_distance

from surrogate_gp import evaluate_model_masks, dice_loss, count_penalty
from create_synthetic_images import load_image_mask_pairs

# ═══════════════════════════════════════════════════════════════════
# Configuration — EDIT THESE
# ═══════════════════════════════════════════════════════════════════

# Each entry: (display_name, model_path, model_family)
#   model_family options: "unet", "unet_modified", "resnext"
"""MODELS = [
    ("OneSyntheticPerRealNoisy", "./models/synthetic_blackbox_iter_208.keras", "unet"),
    ("CleanPatchesWithoutAugmentation", "./models/clean_blackbox.keras", "unet"),
    ("CleanWithClassicalAugmentation", "./clean_only_model.keras", "unet"),
    ("CleanWithDoubledClassicalAugmentation", "./clean_only_ExtraAugmodel.keras", "unet"),
    #("clean_w_augFullExtra", "./clean_only_FullExtraAugmodel.keras", "unet"),
    #("LayeredAugMinObjective0", "./synthetic_blackbox_var2_LayeredExtraAugMinObjective_0.keras", "unet"),
    ("SyntheticLayeredAugMinObjectiveMaxRecall", "./synthetic_blackbox_var2_LayeredExtraAugMinObjective_1.keras", "unet"),
    #("LayeredAugMinObjective2", "./synthetic_blackbox_var2_LayeredExtraAugMinObjective_2.keras", "unet"),
    #("LayeredAugMinObjective3", "./synthetic_blackbox_var2_LayeredExtraAugMinObjective_3.keras", "unet"),
    #("LayeredAugMinObjective4", "./synthetic_blackbox_var2_LayeredExtraAugMinObjective_4.keras", "unet"),
    #("OnTopAugMinObjective0", "./synthetic_blackbox_var2_OnTopAugMinObjective_0.keras", "unet"),
    #("OnTopAugMinObjective1", "./synthetic_blackbox_var2_OnTopAugMinObjective_1.keras", "unet"),
    #("OnTopAugMinObjective2", "./synthetic_blackbox_var2_OnTopAugMinObjective_2.keras", "unet"),
    ("SyntheticExtraAugMinObjectiveMaxDice", "./synthetic_blackbox_ExtraAugMinObjective_3.keras", "unet"),
    ("SyntheticOnTopAugMinObjectiveMaxDice", "./synthetic_blackbox_var2_OnTopAugMinObjective_3.keras", "unet"),#selected for highest dice Extra;
    #("OnTopAugMinObjective4", "./synthetic_blackbox_var2_OnTopAugMinObjective_4.keras", "unet"),
]"""
"""MODELS = [ 
    ("SyntheticAugNoClassicalOnClean1", "./NoClassicalAugBOMinObjective_1.keras","unet"),
    ("CleanPatchesWithoutAugmentation", "./models/clean_blackbox.keras", "unet"),
    ("CleanPatchesWithoutAugmentationFullInfo", "./models/clean_only_model_noaugmentation.keras", "unet"),
    ("ClassicalAugmentationOnClean", "./clean_only_model.keras", "unet"),
    ("ClassicalAugmentationOnCleanFullInfo", "./clean_only_FullSameAugmodel.keras", "unet"),
    ("SyntheticTrial2AugNoClassicalOnClean1", "./NoClassicalAugBOMinObjective2_1.keras","unet"),
    ("SyntheticTrial2AugNoClassicalOnClean6", "./NoClassicalAugBOMinObjective2_6.keras","unet"),     
          ]"""
MODELS = [
    ("CleanPatchesWithoutAugmentation", "./models/clean_blackbox.keras", "unet"),
    ("CleanPatchesWithoutAugmentationFullInfo", "./models/clean_only_model_noaugmentation.keras", "unet"),
    ("ClassicalAugmentationOnClean", "./clean_only_model.keras", "unet"),
    ("SyntheticLayeredValOnlyNoise8","./LayeredAugBOMinValLossOnlyNoise_8.keras","unet"),
    ("SyntheticValOnlyNoise0","./SyntheticAugBOMinValLossOnlyNoise_0.keras","unet"),
    ("SyntheticValOnlyNoise4","./SyntheticAugBOMinValLossOnlyNoise_4.keras","unet"),
]

MODELS = [("MultiresolutionBaseline_same_image_balance", "./NoPatchMultiresolutionBaselineT1.keras", "unet"), # Baseline
          ("no_classic_aug_noisy_and_synthetic_aug_clean_HiDice", "./SyntheticAugBOMinValLossOnlyNoiseNoRealNoisyAug_6.keras", "unet"),
          #("no_classic_aug_noisy_and_synthetic_aug_clean_HiDiceT2", "./SyntheticAugBOMinValLossOnlyNoiseNoRealNoisyAugT2_0.keras", "unet"),
          #("no_classic_aug_noisy_and_synthetic_aug_clean_LowValLossT2", "./SyntheticAugBOMinValLossOnlyNoiseNoRealNoisyAugT2_1.keras", "unet"),
          ("multipleNoiseT1ValLoss","./multipleNoise_3.keras","unet"),
          ("multipleNoiseT1ValAcc","./multipleNoise_1.keras","unet"),
          ("multipleNoiseT2ValLoss","./multipleNoiseT2_6.keras","unet"),
          ("multipleNoiseT2ValAcc","./multipleNoiseT2_2.keras","unet"),
          #("no_classic_aug_noisy_and_split_aug_clean_HiDice", "./SplitAugBOMinValLossOnlyNoiseNoRealNoisyAug_6.keras", "unet"),
          #("no_classic_aug_noisy_and_split_aug_clean_HiAcc", "./SplitAugBOMinValLossOnlyNoiseNoRealNoisyAug_4.keras", "unet"),
          #("no_classic_aug_noisy_and_synthetic_aug_clean_HiAcc", "./SyntheticAugBOMinValLossOnlyNoiseNoRealNoisyAug_1.keras", "unet"),
          #("classic_aug_noisy_and_synthetic_aug_clean_HiAcc", "./SyntheticAugBOMinValLossOnlyNoise_2.keras", "unet"),
          #("no_classic_aug_noisy_and_synthetic_aug_clean_HiAcc", "./", "unet"),
          #("SyntheticAugmentationOnMultiresolutionImages","./SyntheticOnMultiAugBOMinValLossOnlyNoiseNoRealNoisyAugT2_0.keras","unet"),
          #("cHigher?? What is this","./SyntheticAugBOMinValLossOnlyNoiseNoRealNoisyAug_1.keras","unet"),
          #("classic_aug_noisy_and_classic_aug_clean", "./clean_only_model.keras", "unet"), #train_clean_only.py and surrogate_gp.py
          ("no_classic_aug_noisy_and_classic_aug_clean", "./NoRealNoisyAugClassicalOnClean.keras", "unet"),
          #("classic_aug_noisy_and_synthetic_plus_classic_aug_clean_HiAcc", "./LayeredAugBOMinValLossOnlyNoise_8.keras", "unet"), #layered_augmentation_model_training.py
          ("no_classic_aug_noisy_and_synthetic_plus_classic_aug_clean_HiAcc", "./LayeredAugBOMinValLossOnlyNoiseNoRealNoisyAug_1.keras", "unet"),
]

MODELS = [
    ("MultiresolutionBaseline_same_image_balance", "./NoPatchMultiresolutionBaselineT1.keras", "unet"),
    ("MultiresolutionBaseline_same_image_balanceT2", "./NoPatchMultiresolutionBaseline2.keras", "unet"),
    ("MultiresolutionBaseline_same_image_balanceT3", "./NoPatchMultiresolutionBaseline3.keras", "unet"),
    #("multipleNoiseExtra0","./multipleNoiseExtra_0.keras","unet"),
    #("multipleNoiseExtra1","./multipleNoiseExtra_1.keras","unet"),
    #("multipleNoiseExtra2","./multipleNoiseExtra_2.keras","unet"),
    ("multipleNoiseHard0","./multipleNoiseHardMeanFilter_0.keras","unet"),
    ("multipleNoiseHard1","./multipleNoiseHardMeanFilter_1.keras","unet"),
    ("multipleNoiseHard2","./multipleNoiseHardMeanFilter_2.keras","unet"),
    #("multipleNoiseEasier0","./multipleNoiseAboveMeanFilter_0.keras","unet"),
    #("multipleNoiseEasier1","./multipleNoiseAboveMeanFilter_1.keras","unet"),
    #("multipleNoiseHarder0","./multipleNoiseBelowMeanFilter_0.keras","unet"),
    #("multipleNoiseHarder1","./multipleNoiseBelowMeanFilter_1.keras","unet"),
    #("multipleNoiseHarder2","./multipleNoiseBelowMeanFilter_2.keras","unet"),
    ]
"""+[(f"multipleNoise{i}",f"./multipleNoise_{i}.keras","unet") for i in range (10)
]"""

MODELS = [
        ("multipleNoiseHardOneSeed0","./reproduceNoiseTest_0.keras","unet"),
        ("multipleNoiseHardOneSeed1","./reproduceNoiseTest_1.keras","unet"),
        ("multipleNoiseHardOneSeed2","./reproduceNoiseTest_2.keras","unet"),
]

IMAGE_DIRS = ["./withheldNoisyTestImages/confident_style(model_style)/"]
PATCH_DIR = "./model_comparison_workspace"
OUTPUT_DIR = "./model_comparison_metricsReproduce5"
SHOW_COMPARISON_CHART = True

NUM_CLASSES = 3          # 0 = background, 1 = grain, 2 = boundary
CLASS_LABELS = [0, 1, 2]

# ═══════════════════════════════════════════════════════════════════
# General metric helpers
# ═══════════════════════════════════════════════════════════════════

def compute_pixel_accuracy(pred_probs, true_masks):
    per_image = []
    for pred, true in zip(pred_probs, true_masks):
        h = min(pred.shape[0], true.shape[0])
        w = min(pred.shape[1], true.shape[1])
        pred_label = np.argmax(pred[:h, :w], axis=-1)
        true_label = true[:h, :w]
        total = h * w
        if total == 0:
            continue
        per_image.append(float(np.sum(pred_label == true_label)) / total)
    return float(np.mean(per_image)) if per_image else 0.0

# ═══════════════════════════════════════════════════════════════════
# Class-specific metric helpers
# ═══════════════════════════════════════════════════════════════════

def _per_class_tp_fp_fn(pred_probs, true_masks, class_idx):
    per_image = []
    for pred, true in zip(pred_probs, true_masks):
        h = min(pred.shape[0], true.shape[0])
        w = min(pred.shape[1], true.shape[1])
        pred_label = np.argmax(pred[:h, :w], axis=-1)
        true_label = true[:h, :w]
        tp = int(np.sum((pred_label == class_idx) & (true_label == class_idx)))
        fp = int(np.sum((pred_label == class_idx) & (true_label != class_idx)))
        fn = int(np.sum((pred_label != class_idx) & (true_label == class_idx)))
        per_image.append((tp, fp, fn))
    return per_image

def compute_class_iou(pred_probs, true_masks, class_idx):
    per_image = _per_class_tp_fp_fn(pred_probs, true_masks, class_idx)
    vals = []
    for tp, fp, fn in per_image:
        denom = tp + fp + fn
        if denom > 0:
            vals.append(tp / denom)
    return float(np.mean(vals)) if vals else 0.0

def compute_class_dice(pred_probs, true_masks, class_idx):
    per_image = _per_class_tp_fp_fn(pred_probs, true_masks, class_idx)
    vals = []
    for tp, fp, fn in per_image:
        denom = 2 * tp + fp + fn
        if denom > 0:
            vals.append((2 * tp) / denom)
    return float(np.mean(vals)) if vals else 0.0

def compute_class_precision(pred_probs, true_masks, class_idx):
    per_image = _per_class_tp_fp_fn(pred_probs, true_masks, class_idx)
    vals = []
    for tp, fp, _ in per_image:
        denom = tp + fp
        if denom > 0:
            vals.append(tp / denom)
    return float(np.mean(vals)) if vals else 0.0

def compute_class_recall(pred_probs, true_masks, class_idx):
    per_image = _per_class_tp_fp_fn(pred_probs, true_masks, class_idx)
    vals = []
    for tp, _, fn in per_image:
        denom = tp + fn
        if denom > 0:
            vals.append(tp / denom)
    return float(np.mean(vals)) if vals else 0.0

# ═══════════════════════════════════════════════════════════════════
# Grain-instance metric helpers
# ═══════════════════════════════════════════════════════════════════

INTERIOR = 1
MIN_OVERLAP = 0.05

def _pred_true_labels(pred_probs, true_masks):
    pred_labels = [np.argmax(p, axis=-1).astype(np.uint8) for p in pred_probs]
    true_labels = list(true_masks)
    return pred_labels, true_labels

def _grain_instances(masks):
    all_instances = []
    all_regions = []
    for m in masks:
        grain = (m == INTERIOR)
        instances = label(grain, connectivity=1)
        all_instances.append(instances)
        all_regions.append(regionprops(instances))
    return all_instances, all_regions

def _overlap_matrix(inst_pred, inst_true):
    n_true = int(inst_true.max())
    n_pred = int(inst_pred.max())
    if n_true == 0 or n_pred == 0:
        return np.zeros((n_true, n_pred)) if n_true > 0 else np.zeros((1, n_pred))
    counts = np.zeros((n_true + 1, n_pred + 1), dtype=np.float64)
    np.add.at(counts, (inst_true.ravel(), inst_pred.ravel()), 1)
    return counts[1:, 1:]

def compute_grain_count_error(pred_probs, true_masks):
    pred_labels, true_labels = _pred_true_labels(pred_probs, true_masks)
    _, pred_regions = _grain_instances(pred_labels)
    _, true_regions = _grain_instances(true_labels)
    per_image = []
    for pred_r, true_r in zip(pred_regions, true_regions):
        n_pred = len(pred_r)
        n_true = len(true_r)
        if n_true == 0:
            continue
        per_image.append(abs(n_pred - n_true) / n_true)
    return float(np.mean(per_image)) if per_image else 0.0

def compute_signed_grain_count_error(pred_probs, true_masks):
    pred_labels, true_labels = _pred_true_labels(pred_probs, true_masks)
    _, pred_regions = _grain_instances(pred_labels)
    _, true_regions = _grain_instances(true_labels)
    per_image = []
    for pred_r, true_r in zip(pred_regions, true_regions):
        n_pred = len(pred_r)
        n_true = len(true_r)
        if n_true == 0:
            continue
        per_image.append((n_pred - n_true) / n_true)
    return float(np.mean(per_image)) if per_image else 0.0

def compute_mean_grain_area_error(pred_probs, true_masks, threshold=MIN_OVERLAP):
    pred_labels, true_labels = _pred_true_labels(pred_probs, true_masks)
    pred_insts, pred_regs = _grain_instances(pred_labels)
    true_insts, true_regs = _grain_instances(true_labels)

    per_image = []
    for inst_pred, regs_pred, inst_true, regs_true in zip(
        pred_insts, pred_regs, true_insts, true_regs
    ):
        n_true = len(regs_true)
        n_pred = len(regs_pred)
        if n_true == 0 or n_pred == 0:
            continue
        overlap = _overlap_matrix(inst_pred, inst_true)
        areas_pred = np.array([r.area for r in regs_pred])
        areas_true = np.array([r.area for r in regs_true])

        total_error = 0.0
        n_matched = 0
        for t in range(n_true):
            pred_idx = np.argmax(overlap[t])
            overlap_frac = overlap[t, pred_idx] / max(areas_true[t], 1)
            if overlap_frac >= threshold:
                total_error += abs(areas_pred[pred_idx] - areas_true[t]) / areas_true[t]
                n_matched += 1
        if n_matched > 0:
            per_image.append(total_error / n_matched)
    return float(np.mean(per_image)) if per_image else 0.0

def compute_wasserstein_grain_size_distance(pred_probs, true_masks):
    pred_labels, true_labels = _pred_true_labels(pred_probs, true_masks)
    _, pred_regs = _grain_instances(pred_labels)
    _, true_regs = _grain_instances(true_labels)

    per_image = []
    for pred_regions, true_regions in zip(pred_regs, true_regs):
        pred_areas = [r.area for r in pred_regions]
        true_areas = [r.area for r in true_regions]
        if len(pred_areas) == 0 or len(true_areas) == 0:
            continue
        per_image.append(float(wasserstein_distance(pred_areas, true_areas)))
    return float(np.mean(per_image)) if per_image else 0.0
def compute_oversegmentation(pred_probs, true_masks, threshold=MIN_OVERLAP):
    pred_labels, true_labels = _pred_true_labels(pred_probs, true_masks)
    pred_insts, pred_regs = _grain_instances(pred_labels)
    true_insts, true_regs = _grain_instances(true_labels)

    per_image = []
    for inst_pred, regs_pred, inst_true, regs_true in zip(
        pred_insts, pred_regs, true_insts, true_regs
    ):
        n_true = len(regs_true)
        n_pred = len(regs_pred)
        if n_true == 0 or n_pred == 0:
            continue
        overlap = _overlap_matrix(inst_pred, inst_true)
        areas_true = np.array([r.area for r in regs_true])

        total_split = 0
        for t in range(n_true):
            overlaps = overlap[t] / max(areas_true[t], 1)
            overlapping_pred = int(np.sum(overlaps > threshold))
            if overlapping_pred > 1:
                total_split += overlapping_pred - 1
        per_image.append(total_split / n_true)
    return float(np.mean(per_image)) if per_image else 0.0


def compute_undersegmentation(pred_probs, true_masks, threshold=MIN_OVERLAP):
    pred_labels, true_labels = _pred_true_labels(pred_probs, true_masks)
    pred_insts, pred_regs = _grain_instances(pred_labels)
    true_insts, true_regs = _grain_instances(true_labels)

    per_image = []
    for inst_pred, regs_pred, inst_true, regs_true in zip(
        pred_insts, pred_regs, true_insts, true_regs
    ):
        n_pred = len(regs_pred)
        n_true = len(regs_true)
        if n_pred == 0 or n_true == 0:
            continue
        overlap = _overlap_matrix(inst_pred, inst_true)
        areas_pred = np.array([r.area for r in regs_pred])

        total_merge = 0
        for p in range(n_pred):
            overlaps = overlap[:, p] / max(areas_pred[p], 1)
            overlapping_true = int(np.sum(overlaps > threshold))
            if overlapping_true > 1:
                total_merge += overlapping_true - 1
        per_image.append(total_merge / n_pred)
    return float(np.mean(per_image)) if per_image else 0.0

# ═══════════════════════════════════════════════════════════════════
# Function registries — add your own functions here
# ═══════════════════════════════════════════════════════════════════

def _macro_average(fn):
    def wrapper(pred_probs, true_masks):
        vals = [fn(pred_probs, true_masks, c) for c in CLASS_LABELS]
        return float(np.mean(vals))
    return wrapper

GENERAL_FUNCTIONS = {
    "Dice Loss": lambda p, t: dice_loss(p, t),
    "Count Penalty": lambda p, t: count_penalty(p, t),
    "Pixel Accuracy": lambda p, t: compute_pixel_accuracy(p, t),
    "Avg IoU": _macro_average(compute_class_iou),
    "Avg Dice": _macro_average(compute_class_dice),
    "Avg Precision": _macro_average(compute_class_precision),
    "Avg Recall": _macro_average(compute_class_recall),
    "Grain Count Error": compute_grain_count_error,
    "Signed Grain Count Error": compute_signed_grain_count_error,
    "Mean Grain Area Error": compute_mean_grain_area_error,
    "Wasserstein Grain Size Dist": compute_wasserstein_grain_size_distance,
    "Oversegmentation": compute_oversegmentation,
    "Undersegmentation": compute_undersegmentation,
}

CLASS_FUNCTIONS = {
    "IoU": compute_class_iou,
    "Dice": compute_class_dice,
    "Precision": compute_class_precision,
    "Recall": compute_class_recall,
}

# ═══════════════════════════════════════════════════════════════════
# Model loading
# ═══════════════════════════════════════════════════════════════════

def _load_model(model_path, model_family):
    model_path = Path(model_path)
    if model_family in {"unet", "unet_modified"}:
        import tensorflow as tf
        model = tf.keras.models.load_model(str(model_path), compile=False)
    elif model_family == "resnext":
        import torch
        from segmenteverygrain.resnext_model import MaskingResNeXt
        device = torch.device(
            "mps" if torch.backends.mps.is_available()
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        model = MaskingResNeXt(num_classes=3, pretrained=False)
        model.load_state_dict(torch.load(str(model_path), map_location=device))
        model.to(device)
        model.eval()
    else:
        raise ValueError(f"Unknown model_family: {model_family}")
    return model

# ═══════════════════════════════════════════════════════════════════
# Per-directory: run inference + compute + save metrics
# ═══════════════════════════════════════════════════════════════════

def process_directory(image_dir, output_dir, show_chart=SHOW_COMPARISON_CHART):
    """Load all models, run inference on image_dir, and print/save metrics."""
    image_dir = Path(image_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dir_tag = image_dir.name

    print(f"\n{'='*60}")
    print(f"Processing directory: {image_dir}")
    print(f"{'='*60}")

    # ── Inference ─────────────────────────────────────────────────
    all_predictions = {}
    for model_name, model_path, model_family in MODELS:
        print(f"\n--- {model_name} ({model_family}) ---")
        model = _load_model(model_path, model_family)
        print(f"  Loaded from {model_path}")

        pred_probs, true_masks = evaluate_model_masks(
            model, str(image_dir), PATCH_DIR, model_family,
        )
        print(f"  Processed {len(pred_probs)} image(s)")
        all_predictions[model_name] = (pred_probs, true_masks)

    # Verify all models saw the same images
    n_images = None
    for model_name, (preds, trues) in all_predictions.items():
        if n_images is None:
            n_images = len(preds)
        else:
            assert len(preds) == n_images, (
                f"Model '{model_name}' produced {len(preds)} predictions, "
                f"expected {n_images}"
            )

    # ── DataFrame 1 — General metrics ─────────────────────────────
    print(f"\n{'='*60}")
    print("General metrics (one value per model)")
    print(f"{'='*60}")

    general_rows = []
    for model_name, (preds, trues) in all_predictions.items():
        row = {"Model": model_name}
        for fn_name, fn in GENERAL_FUNCTIONS.items():
            row[fn_name] = fn(preds, trues)
        general_rows.append(row)

    general_df = pd.DataFrame(general_rows).set_index("Model")
    general_numeric_df = general_df.astype(float)
    general_df_str = general_df.map(lambda v: f"{v:.4f}")
    print(general_df_str.to_string())

    # ── DataFrame 2 — Class-specific metrics ──────────────────────
    print(f"\n{'='*60}")
    print("Class-specific metrics (per model & class)")
    print(f"{'='*60}")

    class_rows = []
    for model_name, (preds, trues) in all_predictions.items():
        for cls in CLASS_LABELS:
            row = {"Model": model_name, "Class": cls}
            for fn_name, fn in CLASS_FUNCTIONS.items():
                row[fn_name] = fn(preds, trues, cls)
            class_rows.append(row)

    class_specific_df = pd.DataFrame(class_rows).set_index(["Model", "Class"])
    class_specific_df_str = class_specific_df.map(lambda v: f"{v:.4f}")
    print(class_specific_df_str.to_string())

    # ── Save per-model CSV files ──────────────────────────────────
    general_vals = {r["Model"]: r for r in general_rows}
    for model_name in all_predictions:
        gen_path = output_dir / f"{model_name}_general_metrics.csv"
        pd.DataFrame([general_vals[model_name]]).set_index("Model").to_csv(gen_path)
        print(f"  Saved {gen_path}")

    class_vals = {(r["Model"], r["Class"]): r for r in class_rows}
    for model_name in all_predictions:
        rows = [class_vals[(model_name, cls)] for cls in CLASS_LABELS]
        cls_path = output_dir / f"{model_name}_class_metrics.csv"
        pd.DataFrame(rows).set_index(["Model", "Class"]).to_csv(cls_path)
        print(f"  Saved {cls_path}")

    # ── Optional: grouped bar chart ───────────────────────────────
    if show_chart:
        import matplotlib.pyplot as plt

        metric_groups = [
            ("Accuracy metrics",
             ["Pixel Accuracy", "Avg IoU", "Avg Dice", "Avg Precision", "Avg Recall"]),
            ("Bounded error / loss",
             ["Dice Loss", "Count Penalty", "Grain Count Error",
              "Oversegmentation", "Undersegmentation"]),
            ("Magnitude error, pixel-scale",
             ["Mean Grain Area Error", "Wasserstein Grain Size Dist"]),
        ]
        metric_groups = [
            (title, [m for m in cols if m in general_numeric_df.columns])
            for title, cols in metric_groups
        ]
        metric_groups = [(t, c) for t, c in metric_groups if c]

        n_models = len(general_numeric_df)
        fig, axes = plt.subplots(1, len(metric_groups),
                                 figsize=(6 * len(metric_groups), 7),
                                 constrained_layout=True)
        if len(metric_groups) == 1:
            axes = [axes]
        for ax, (title, metrics) in zip(axes, metric_groups):
            general_numeric_df[metrics].T.plot(kind="bar", ax=ax)
            ax.set_title(title)
            ax.set_xlabel("Metric")
            ax.set_ylabel("Value")
            ax.tick_params(axis="x", rotation=30)
            ax.grid(axis="y", alpha=0.3)
            ax.legend_.remove()
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, title="Model", loc="upper center",
                   ncol=min(n_models, 4), fontsize=8, frameon=True)
        fig.suptitle(f"{dir_tag} — Model comparison")
        chart_path = output_dir / "model_comparison_chart.png"
        plt.savefig(chart_path, dpi=150)
        print(f"\n  Saved comparison chart: {chart_path}")
        plt.show()

        class_metric_names = list(CLASS_FUNCTIONS.keys())
        class_numeric_df = class_specific_df[class_metric_names].astype(float)
        n_classes = len(CLASS_LABELS)
        fig2, axes2 = plt.subplots(1, n_classes,
                                   figsize=(6 * n_classes, 7),
                                   constrained_layout=True)
        if n_classes == 1:
            axes2 = [axes2]
        class_labels_display = ["Background", "Grain", "Boundary"]
        for ax, cls, cls_label in zip(axes2, CLASS_LABELS, class_labels_display):
            cls_data = class_numeric_df.xs(cls, level="Class")
            cls_data.T.plot(kind="bar", ax=ax)
            ax.set_title(f"Class {cls}: {cls_label}")
            ax.set_xlabel("Metric")
            ax.set_ylabel("Value")
            ax.tick_params(axis="x", rotation=30)
            ax.grid(axis="y", alpha=0.3)
            ax.legend_.remove()
        handles2, labels2 = axes2[0].get_legend_handles_labels()
        fig2.legend(handles2, labels2, title="Model", loc="upper center",
                    ncol=min(n_models, 4), fontsize=8, frameon=True)
        fig2.suptitle(f"{dir_tag} — Class-specific metrics")
        class_chart_path = output_dir / "model_comparison_class_chart.png"
        plt.savefig(class_chart_path, dpi=150)
        print(f"\n  Saved class comparison chart: {class_chart_path}")
        plt.show()

    print(f"\n  → Metrics saved to {output_dir}/")


# ═══════════════════════════════════════════════════════════════════
# Run for every directory
# ═══════════════════════════════════════════════════════════════════

for img_dir in IMAGE_DIRS:
    process_directory(img_dir, Path(OUTPUT_DIR) / Path(img_dir).name,
                      show_chart=SHOW_COMPARISON_CHART)
