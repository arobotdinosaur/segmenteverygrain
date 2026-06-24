"""
Compare multiple segmentation models on the same set of images.

Each model is loaded, run on all images in a directory, and evaluated with:
  • General metrics  — one value per model (e.g. multi-class Dice loss)
  • Class-specific metrics — one value per (model, class) combination.

Results are collected into two DataFrames and prediction overlays are displayed.
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
MODELS = [
    ("iter_208", "./models/synthetic_blackbox_iter_208.keras", "unet"),
    ("modelep30lr3", "./models/modelep30lr3.keras", "unet"),
    ("iter_293", "./models/synthetic_blackbox_iter_293.keras", "unet"),
    ("no_rock","./models/seg_model_alumina_only.keras", "unet"),
    ("blackbox_clean","./models/clean_blackbox.keras", "unet"),
]

IMAGE_DIR = "./test_only_noisy_images/"
PATCH_DIR = "./model_comparison_workspace"
OUTPUT_DIR = "./model_comparison_metrics"   # where per-model CSV files are saved
SHOW_PLOTS = True

NUM_CLASSES = 3          # 0 = background, 1 = grain, 2 = boundary
CLASS_LABELS = [0, 1, 2]

# ═══════════════════════════════════════════════════════════════════
# General metric helpers
# ═══════════════════════════════════════════════════════════════════

def compute_pixel_accuracy(pred_probs, true_masks):
    """Overall per-pixel accuracy averaged across all images."""
    correct = 0
    total = 0
    for pred, true in zip(pred_probs, true_masks):
        h = min(pred.shape[0], true.shape[0])
        w = min(pred.shape[1], true.shape[1])
        pred_label = np.argmax(pred[:h, :w], axis=-1)
        true_label = true[:h, :w]
        correct += int(np.sum(pred_label == true_label))
        total += int(h * w)
    return correct / total if total > 0 else 0.0


# ═══════════════════════════════════════════════════════════════════
# Class-specific metric helpers
# ═══════════════════════════════════════════════════════════════════

def _per_class_tp_fp_fn(pred_probs, true_masks, class_idx):
    """Aggregate true-positive, false-positive, false-negative counts for one class."""
    tp = fp = fn = 0
    for pred, true in zip(pred_probs, true_masks):
        h = min(pred.shape[0], true.shape[0])
        w = min(pred.shape[1], true.shape[1])
        pred_label = np.argmax(pred[:h, :w], axis=-1)
        true_label = true[:h, :w]
        tp += int(np.sum((pred_label == class_idx) & (true_label == class_idx)))
        fp += int(np.sum((pred_label == class_idx) & (true_label != class_idx)))
        fn += int(np.sum((pred_label != class_idx) & (true_label == class_idx)))
    return tp, fp, fn


def compute_class_iou(pred_probs, true_masks, class_idx):
    tp, fp, fn = _per_class_tp_fp_fn(pred_probs, true_masks, class_idx)
    denom = tp + fp + fn
    return tp / denom if denom > 0 else 0.0


def compute_class_dice(pred_probs, true_masks, class_idx):
    tp, fp, fn = _per_class_tp_fp_fn(pred_probs, true_masks, class_idx)
    denom = 2 * tp + fp + fn
    return (2 * tp) / denom if denom > 0 else 0.0


def compute_class_precision(pred_probs, true_masks, class_idx):
    tp, fp, _ = _per_class_tp_fp_fn(pred_probs, true_masks, class_idx)
    denom = tp + fp
    return tp / denom if denom > 0 else 0.0


def compute_class_recall(pred_probs, true_masks, class_idx):
    tp, _, fn = _per_class_tp_fp_fn(pred_probs, true_masks, class_idx)
    denom = tp + fn
    return tp / denom if denom > 0 else 0.0


# ═══════════════════════════════════════════════════════════════════
# Grain-instance metric helpers
# ═══════════════════════════════════════════════════════════════════

INTERIOR = 1
MIN_OVERLAP = 0.05


def _pred_true_labels(pred_probs, true_masks):
    """Convert probability lists to integer label lists."""
    pred_labels = [np.argmax(p, axis=-1).astype(np.uint8) for p in pred_probs]
    true_labels = list(true_masks)
    return pred_labels, true_labels


def _grain_instances(masks):
    """Return list of (instance_label_map, list_of_regionprops) per image."""
    all_instances = []
    all_regions = []
    for m in masks:
        grain = (m == INTERIOR)
        instances = label(grain, connectivity=1)
        all_instances.append(instances)
        all_regions.append(regionprops(instances))
    return all_instances, all_regions


def _overlap_matrix(inst_pred, inst_true):
    """Build overlap matrix: rows=true grains, cols=predicted grains.

    overlap[t-1, p-1] = number of pixels where inst_true == t and inst_pred == p.
    Computed as a single 2-D histogram over the labels so cost is one pass over
    the image rather than O(n_true * n_pred) full-image boolean ANDs.
    """
    n_true = int(inst_true.max())
    n_pred = int(inst_pred.max())
    if n_true == 0 or n_pred == 0:
        return np.zeros((n_true, n_pred)) if n_true > 0 else np.zeros((1, n_pred))
    counts = np.zeros((n_true + 1, n_pred + 1), dtype=np.float64)
    np.add.at(counts, (inst_true.ravel(), inst_pred.ravel()), 1)
    return counts[1:, 1:]   # drop background row/col (label 0)


def compute_grain_count_error(pred_probs, true_masks):
    """|N_pred - N_true| / N_true."""
    pred_labels, true_labels = _pred_true_labels(pred_probs, true_masks)
    _, pred_regions = _grain_instances(pred_labels)
    _, true_regions = _grain_instances(true_labels)
    n_pred = sum(len(r) for r in pred_regions)
    n_true = sum(len(r) for r in true_regions)
    if n_true == 0:
        return 0.0
    return abs(n_pred - n_true) / n_true


def compute_signed_grain_count_error(pred_probs, true_masks):
    """(N_pred - N_true) / N_true. Positive = oversegmentation."""
    pred_labels, true_labels = _pred_true_labels(pred_probs, true_masks)
    _, pred_regions = _grain_instances(pred_labels)
    _, true_regions = _grain_instances(true_labels)
    n_pred = sum(len(r) for r in pred_regions)
    n_true = sum(len(r) for r in true_regions)
    if n_true == 0:
        return 0.0
    return (n_pred - n_true) / n_true


def compute_mean_grain_area_error(pred_probs, true_masks, threshold=MIN_OVERLAP):
    """Mean relative area error across best-overlap grain matches."""
    pred_labels, true_labels = _pred_true_labels(pred_probs, true_masks)
    pred_insts, pred_regs = _grain_instances(pred_labels)
    true_insts, true_regs = _grain_instances(true_labels)

    total_error = 0.0
    n_matched = 0
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

        for t in range(n_true):
            pred_idx = np.argmax(overlap[t])
            overlap_frac = overlap[t, pred_idx] / max(areas_true[t], 1)
            if overlap_frac >= threshold:
                total_error += abs(areas_pred[pred_idx] - areas_true[t]) / areas_true[t]
                n_matched += 1

    return total_error / n_matched if n_matched > 0 else 0.0


def compute_wasserstein_grain_size_distance(pred_probs, true_masks):
    """Wasserstein distance between grain area distributions."""
    pred_labels, true_labels = _pred_true_labels(pred_probs, true_masks)
    _, pred_regs = _grain_instances(pred_labels)
    _, true_regs = _grain_instances(true_labels)

    pred_areas = [r.area for regions in pred_regs for r in regions]
    true_areas = [r.area for regions in true_regs for r in regions]

    if len(pred_areas) == 0 or len(true_areas) == 0:
        return 0.0
    return float(wasserstein_distance(pred_areas, true_areas))


def compute_oversegmentation(pred_probs, true_masks, threshold=MIN_OVERLAP):
    """sum(max(0, overlapping_pred - 1)) / N_true."""
    pred_labels, true_labels = _pred_true_labels(pred_probs, true_masks)
    pred_insts, pred_regs = _grain_instances(pred_labels)
    true_insts, true_regs = _grain_instances(true_labels)

    total_split = 0
    n_true_total = 0
    for inst_pred, regs_pred, inst_true, regs_true in zip(
        pred_insts, pred_regs, true_insts, true_regs
    ):
        n_true = len(regs_true)
        n_pred = len(regs_pred)
        n_true_total += n_true
        if n_true == 0 or n_pred == 0:
            continue
        overlap = _overlap_matrix(inst_pred, inst_true)
        areas_true = np.array([r.area for r in regs_true])

        for t in range(n_true):
            overlaps = overlap[t] / max(areas_true[t], 1)
            overlapping_pred = int(np.sum(overlaps > threshold))
            if overlapping_pred > 1:
                total_split += overlapping_pred - 1

    return total_split / n_true_total if n_true_total > 0 else 0.0


def compute_undersegmentation(pred_probs, true_masks, threshold=MIN_OVERLAP):
    """sum(max(0, overlapping_true - 1)) / N_pred."""
    pred_labels, true_labels = _pred_true_labels(pred_probs, true_masks)
    pred_insts, pred_regs = _grain_instances(pred_labels)
    true_insts, true_regs = _grain_instances(true_labels)

    total_merge = 0
    n_pred_total = 0
    for inst_pred, regs_pred, inst_true, regs_true in zip(
        pred_insts, pred_regs, true_insts, true_regs
    ):
        n_pred = len(regs_pred)
        n_true = len(regs_true)
        n_pred_total += n_pred
        if n_pred == 0 or n_true == 0:
            continue
        overlap = _overlap_matrix(inst_pred, inst_true)
        areas_pred = np.array([r.area for r in regs_pred])

        for p in range(n_pred):
            overlaps = overlap[:, p] / max(areas_pred[p], 1)
            overlapping_true = int(np.sum(overlaps > threshold))
            if overlapping_true > 1:
                total_merge += overlapping_true - 1

    return total_merge / n_pred_total if n_pred_total > 0 else 0.0


# ═══════════════════════════════════════════════════════════════════
# Function registries — add your own functions here
# ═══════════════════════════════════════════════════════════════════
#
# General functions:   callable(pred_probs_list, true_masks_list) → scalar
# Class functions:     callable(pred_probs_list, true_masks_list, class_idx) → scalar

def _macro_average(fn):
    """Average a class-specific metric across all CLASS_LABELS."""
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
# Run inference for every model
# ═══════════════════════════════════════════════════════════════════

all_predictions = {}

for model_name, model_path, model_family in MODELS:
    print(f"\n--- {model_name} ({model_family}) ---")
    model = _load_model(model_path, model_family)
    print(f"  Loaded from {model_path}")

    pred_probs, true_masks = evaluate_model_masks(
        model, IMAGE_DIR, PATCH_DIR, model_family,
    )
    print(f"  Processed {len(pred_probs)} image(s)")
    all_predictions[model_name] = (pred_probs, true_masks)

# ── Verify that all models saw the same images ────────────────────
n_images = None
for model_name, (preds, trues) in all_predictions.items():
    if n_images is None:
        n_images = len(preds)
    else:
        assert len(preds) == n_images, (
            f"Model '{model_name}' produced {len(preds)} predictions, "
            f"expected {n_images}"
        )

# ═══════════════════════════════════════════════════════════════════
# DataFrame 1 — General metrics  (index: model name)
# ═══════════════════════════════════════════════════════════════════

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
general_numeric_df = general_df.astype(float)          # keep numeric copy for plotting
general_df = general_df.map(lambda v: f"{v:.4f}")
print(general_df.to_string())

# ═══════════════════════════════════════════════════════════════════
# DataFrame 2 — Class-specific metrics  (MultiIndex: model, class)
# ═══════════════════════════════════════════════════════════════════

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
class_specific_df = class_specific_df.map(lambda v: f"{v:.4f}")
print(class_specific_df.to_string())

# ═══════════════════════════════════════════════════════════════════
# Save per-model CSV files
# ═══════════════════════════════════════════════════════════════════

output_dir = Path(OUTPUT_DIR)
output_dir.mkdir(parents=True, exist_ok=True)

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

# ═══════════════════════════════════════════════════════════════════
# Visualization — overlays for each image × model
# ═══════════════════════════════════════════════════════════════════

if SHOW_PLOTS:
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    # ── Comparison chart — metrics across all models ──────────────
    # Metrics are grouped by scale so they don't drown each other out:
    #   • accuracy-style, higher is better, all in [0, 1]
    #   • bounded error/loss, lower is better, all in [0, 1]
    #   • magnitude error, lower is better, pixel-scale (tens–hundreds)
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

    fig_c, axes_c = plt.subplots(1, len(metric_groups),
                                 figsize=(6 * len(metric_groups), 6))
    if len(metric_groups) == 1:
        axes_c = [axes_c]
    for ax_c, (title, metrics) in zip(axes_c, metric_groups):
        general_numeric_df[metrics].T.plot(kind="bar", ax=ax_c)
        ax_c.set_title(title)
        ax_c.set_xlabel("Metric")
        ax_c.set_ylabel("Value")
        ax_c.tick_params(axis="x", rotation=30)
        ax_c.legend(title="Model")
        ax_c.grid(axis="y", alpha=0.3)
    plt.suptitle("Model comparison — " + ", ".join(general_numeric_df.index))
    plt.tight_layout()
    comparison_path = output_dir / "model_comparison_chart.png"
    plt.savefig(comparison_path, dpi=150)
    print(f"\n  Saved comparison chart: {comparison_path}")
    plt.show()

    cmap = ListedColormap(['black', 'steelblue', 'orange'])
    n_models = len(MODELS)
    pairs = load_image_mask_pairs(IMAGE_DIR)

    for i, (img_path, mask_path) in enumerate(pairs):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        true = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        img_stem = Path(img_path).stem

        fig, axes = plt.subplots(1, 2 + n_models, figsize=(6 * (2 + n_models), 5))
        axes[0].imshow(img)
        axes[0].set_title("Input image")
        axes[0].axis("off")

        axes[1].imshow(true, cmap=cmap, vmin=0, vmax=2)
        axes[1].set_title("Ground truth")
        axes[1].axis("off")

        for j, (model_name, _, _) in enumerate(MODELS):
            pred_probs = all_predictions[model_name][0][i]
            pred_label = np.argmax(pred_probs, axis=-1).astype(np.uint8)
            axes[2 + j].imshow(img)
            axes[2 + j].imshow(pred_label, cmap=cmap, vmin=0, vmax=2, alpha=0.4)
            axes[2 + j].set_title(f"Overlay — {model_name}")
            axes[2 + j].axis("off")

        plt.suptitle(f"Image {i}")
        plt.tight_layout()
        plt.savefig(output_dir / f"{img_stem}_overview.png", dpi=150)
        plt.show()

        for j, (model_name, _, _) in enumerate(MODELS):
            pred_probs = all_predictions[model_name][0][i]
            pred_label = np.argmax(pred_probs, axis=-1).astype(np.uint8)
            fig2, ax2 = plt.subplots(1, 1, figsize=(6, 5))
            ax2.imshow(img)
            ax2.imshow(pred_label, cmap=cmap, vmin=0, vmax=2, alpha=0.4)
            ax2.set_title(f"{model_name} overlay")
            ax2.axis("off")
            plt.tight_layout()
            plt.savefig(output_dir / f"{img_stem}_{model_name}_overlay.png", dpi=150)
            plt.close(fig2)