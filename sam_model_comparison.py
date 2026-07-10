"""
Compare the zero-shot SAM 3 mask decoder against fine-tuned decoders on the same images.

Mirrors model_comparison.py: each model is loaded, run on all images in a directory,
and evaluated with general metrics (one value per model). Results go into a DataFrame,
per-model CSVs are written, and a grouped comparison chart plus per-image overlay
figures are saved to OUTPUT_DIR.

Unlike the U-Net script there are no class-specific metrics: SAM 3 emits binary
instance masks with no boundary class, so the per-class (0/1/2) table has no analogue.
The grain-instance metrics carry over unchanged, and detection metrics (instance
F1/precision/recall) are added since automatic mask generation actually has to *find*
the grains rather than being handed an argmax.

Usage (in the segmenteverygrain_new_SAM env):
    KMP_DUPLICATE_LIB_OK=TRUE python sam_model_comparison.py
"""

import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import copy
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy import ndimage
from scipy.stats import wasserstein_distance
from transformers import pipeline

from create_synthetic_images import load_image_mask_pairs

# ═══════════════════════════════════════════════════════════════════
# Configuration — EDIT THESE
# ═══════════════════════════════════════════════════════════════════

# Each entry: (display_name, decoder_path)
#   decoder_path None -> stock facebook/sam3 decoder (zero-shot baseline)
MODELS = [
    ("zero_shot", None),
    ("analytic_all", "./models/sam3_decoder_analytic_all_ft.pt"),
    ("analytic", "./models/sam3_decoder_analytic_ft.pt"),
    ("prac7_only", "./models/sam3_decoder_prac7_only_ft.pt"),
    ("wilson_only", "./models/sam3_decoder_wilson_only_ft.pt"),
]

IMAGE_DIR = "./real_noisy_images"
OUTPUT_DIR = "./sam_model_comparison_metrics"
SHOW_PLOTS = True
MAX_PANEL_COLS = 3       # overview figure wraps to a new row past this many panels

# Automatic mask generation settings. points_per_crop / pred_iou_thresh are the
# sweep-optimal pair from sam3_sweep_multi.py (mean instance F1 0.633).
#
# Caution: that sweep used the *stock* decoder on *clean* images. Fine-tuning shifts the
# IoU-prediction head's calibration downward (median mask score 0.81 zero-shot vs 0.69 for
# analytic_all on this eval set), so a threshold fixed across decoders silently discards
# more of a fine-tuned model's masks than the baseline's. Comparing at 0.70 measures score
# calibration as much as mask quality. Drop to ~0.60 to compare the decoders' mask output
# on more even footing.
SAM_MODEL_ID = "facebook/sam3"
POINTS_PER_CROP = 32
PRED_IOU_THRESH = 0.60
FIXED = {"points_per_batch": 64, "stability_score_thresh": 0.90, "stability_score_offset": 1.0}

GRAIN_CLASS = 1          # ground-truth mask: 0 = background, 1 = grain, 2 = boundary
MIN_GRAIN_PX = 30        # discard instances smaller than this, predicted and ground truth
MIN_OVERLAP = 0.05       # overlap fraction counted as "touching" for over/undersegmentation
MATCH_IOU = 0.5          # IoU for a predicted mask to count as a true positive


# ═══════════════════════════════════════════════════════════════════
# Ground-truth instances
# ═══════════════════════════════════════════════════════════════════

def label_instances(binary, min_px=MIN_GRAIN_PX):
    """Connected components of a binary map, with sub-min_px components dropped."""
    lbl, n = ndimage.label(binary)
    if n == 0:
        return lbl, 0
    sizes = np.bincount(lbl.ravel())
    keep = np.where(sizes >= min_px)[0]
    keep = keep[keep != 0]
    remap = np.zeros(n + 1, dtype=int)
    remap[keep] = np.arange(1, len(keep) + 1)
    return remap[lbl], len(keep)


def load_ground_truth(mask_path):
    gt = np.array(Image.open(mask_path))
    if gt.ndim == 3:   # some masks are saved RGB(A) with the class id duplicated per channel
        gt = gt[..., 0]
    inst, n_gt = label_instances(gt == GRAIN_CLASS)
    return gt, inst, n_gt


# ═══════════════════════════════════════════════════════════════════
# Inference
# ═══════════════════════════════════════════════════════════════════

def generate_masks_square(generator, image_np, **params):
    """SAM 3's automatic point grid only covers ~the top square of a non-square image,
    so on a landscape frame the bottom ~25-30% gets no masks at all. Reflect-pad to a
    square, run SAM, crop masks back, and drop any that live entirely in the padding.
    Lifted from sam3_sweep_multi.py, where the behaviour was diagnosed."""
    h, w = image_np.shape[:2]
    s = max(h, w)
    padded = np.pad(image_np, ((0, s - h), (0, s - w), (0, 0)), mode="reflect")
    out = generator(Image.fromarray(padded), **params)
    masks = []
    for m, sc in zip(out["masks"], out["scores"]):
        if float(sc) < PRED_IOU_THRESH:
            continue
        m = np.asarray(m, dtype=bool)[:h, :w]
        if int(m.sum()) >= MIN_GRAIN_PX:
            masks.append(m)
    return masks


def _load_decoder(model, decoder_path, base_state):
    if decoder_path is None:
        model.mask_decoder.load_state_dict(base_state)
    else:
        model.mask_decoder.load_state_dict(torch.load(decoder_path, map_location="cpu"))


# ═══════════════════════════════════════════════════════════════════
# Overlap bookkeeping
# ═══════════════════════════════════════════════════════════════════

def _overlap_matrix(pred_masks, gt_inst, n_gt):
    """overlap[t, p] = pixels where gt_inst == t+1 and pred_masks[p] is True.

    Predicted masks are kept as a list rather than rasterized into a label map:
    SAM's masks may overlap each other, and flattening would silently destroy that.
    """
    overlap = np.zeros((n_gt, len(pred_masks)), dtype=np.float64)
    for p, m in enumerate(pred_masks):
        counts = np.bincount(gt_inst[m], minlength=n_gt + 1)
        overlap[:, p] = counts[1:]
    return overlap


def _areas(pred_masks, gt_inst, n_gt):
    pred_areas = np.array([int(m.sum()) for m in pred_masks], dtype=np.float64)
    gt_areas = np.bincount(gt_inst.ravel(), minlength=n_gt + 1)[1:].astype(np.float64)
    return pred_areas, gt_areas


# ═══════════════════════════════════════════════════════════════════
# General metric helpers
# ═══════════════════════════════════════════════════════════════════
#
# Every metric takes (pred_masks_list, gt_list) where gt_list holds
# (gt_raw, gt_inst, n_gt) triples, and returns a scalar aggregated across images.
# Counting metrics accumulate over images (micro-average), matching the way
# model_comparison.py's _per_class_tp_fp_fn aggregates.

def _match_instances(pred_masks, gt_inst, n_gt, iou_thr=MATCH_IOU):
    """Greedy one-to-one matching, largest predicted mask first. Returns (tp, fp, fn, ious)."""
    if n_gt == 0 or not pred_masks:
        return 0, len(pred_masks), n_gt, []
    overlap = _overlap_matrix(pred_masks, gt_inst, n_gt)
    pred_areas, gt_areas = _areas(pred_masks, gt_inst, n_gt)

    matched, ious = set(), []
    for p in np.argsort(-pred_areas):
        inter = overlap[:, p]
        if not inter.any():
            continue
        iou = inter / (pred_areas[p] + gt_areas - inter)
        t = int(np.argmax(iou))
        if iou[t] >= iou_thr and t not in matched:
            matched.add(t)
            ious.append(float(iou[t]))
    tp = len(matched)
    return tp, len(pred_masks) - tp, n_gt - tp, ious


def _tp_fp_fn(pred_list, gt_list):
    tp = fp = fn = 0
    ious = []
    for masks, (_, gt_inst, n_gt) in zip(pred_list, gt_list):
        t, f, m, i = _match_instances(masks, gt_inst, n_gt)
        tp += t
        fp += f
        fn += m
        ious += i
    return tp, fp, fn, ious


def compute_instance_precision(pred_list, gt_list):
    tp, fp, _, _ = _tp_fp_fn(pred_list, gt_list)
    return tp / (tp + fp) if (tp + fp) else 0.0


def compute_instance_recall(pred_list, gt_list):
    tp, _, fn, _ = _tp_fp_fn(pred_list, gt_list)
    return tp / (tp + fn) if (tp + fn) else 0.0


def compute_instance_f1(pred_list, gt_list):
    tp, fp, fn, _ = _tp_fp_fn(pred_list, gt_list)
    denom = 2 * tp + fp + fn
    return (2 * tp) / denom if denom else 0.0


def compute_mean_matched_iou(pred_list, gt_list):
    _, _, _, ious = _tp_fp_fn(pred_list, gt_list)
    return float(np.mean(ious)) if ious else 0.0


def _foreground_counts(pred_list, gt_list):
    inter = pred_px = true_px = 0
    for masks, (gt_raw, _, _) in zip(pred_list, gt_list):
        true_fg = gt_raw == GRAIN_CLASS
        pred_fg = np.zeros(true_fg.shape, dtype=bool)
        for m in masks:
            pred_fg |= m
        inter += int((pred_fg & true_fg).sum())
        pred_px += int(pred_fg.sum())
        true_px += int(true_fg.sum())
    return inter, pred_px, true_px


def compute_fg_iou(pred_list, gt_list):
    inter, pred_px, true_px = _foreground_counts(pred_list, gt_list)
    union = pred_px + true_px - inter
    return inter / union if union else 0.0


def compute_fg_precision(pred_list, gt_list):
    inter, pred_px, _ = _foreground_counts(pred_list, gt_list)
    return inter / pred_px if pred_px else 0.0


def compute_fg_recall(pred_list, gt_list):
    inter, _, true_px = _foreground_counts(pred_list, gt_list)
    return inter / true_px if true_px else 0.0


# ── Grain-instance metrics — same definitions as model_comparison.py ──

def _grain_counts(pred_list, gt_list):
    n_pred = sum(len(m) for m in pred_list)
    n_true = sum(n_gt for _, _, n_gt in gt_list)
    return n_pred, n_true


def compute_predicted_grain_count(pred_list, gt_list):
    return float(_grain_counts(pred_list, gt_list)[0])


def compute_grain_count_error(pred_list, gt_list):
    """|N_pred - N_true| / N_true."""
    n_pred, n_true = _grain_counts(pred_list, gt_list)
    return abs(n_pred - n_true) / n_true if n_true else 0.0


def compute_signed_grain_count_error(pred_list, gt_list):
    """(N_pred - N_true) / N_true. Positive = oversegmentation."""
    n_pred, n_true = _grain_counts(pred_list, gt_list)
    return (n_pred - n_true) / n_true if n_true else 0.0


def compute_mean_grain_area_error(pred_list, gt_list, threshold=MIN_OVERLAP):
    """Mean relative area error across best-overlap grain matches."""
    total_error = 0.0
    n_matched = 0
    for masks, (_, gt_inst, n_gt) in zip(pred_list, gt_list):
        if n_gt == 0 or not masks:
            continue
        overlap = _overlap_matrix(masks, gt_inst, n_gt)
        pred_areas, gt_areas = _areas(masks, gt_inst, n_gt)
        for t in range(n_gt):
            p = int(np.argmax(overlap[t]))
            if overlap[t, p] / max(gt_areas[t], 1) >= threshold:
                total_error += abs(pred_areas[p] - gt_areas[t]) / gt_areas[t]
                n_matched += 1
    return total_error / n_matched if n_matched else 0.0


def compute_wasserstein_grain_size_distance(pred_list, gt_list):
    """Wasserstein distance between grain area distributions."""
    pred_areas = [int(m.sum()) for masks in pred_list for m in masks]
    true_areas = [
        a for (_, gt_inst, n_gt) in gt_list
        for a in np.bincount(gt_inst.ravel(), minlength=n_gt + 1)[1:]
    ]
    if not pred_areas or not true_areas:
        return 0.0
    return float(wasserstein_distance(pred_areas, true_areas))


def compute_oversegmentation(pred_list, gt_list, threshold=MIN_OVERLAP):
    """sum(max(0, overlapping_pred - 1)) / N_true."""
    total_split = 0
    n_true_total = 0
    for masks, (_, gt_inst, n_gt) in zip(pred_list, gt_list):
        n_true_total += n_gt
        if n_gt == 0 or not masks:
            continue
        overlap = _overlap_matrix(masks, gt_inst, n_gt)
        _, gt_areas = _areas(masks, gt_inst, n_gt)
        for t in range(n_gt):
            n_touch = int(np.sum(overlap[t] / max(gt_areas[t], 1) > threshold))
            if n_touch > 1:
                total_split += n_touch - 1
    return total_split / n_true_total if n_true_total else 0.0


def compute_undersegmentation(pred_list, gt_list, threshold=MIN_OVERLAP):
    """sum(max(0, overlapping_true - 1)) / N_pred."""
    total_merge = 0
    n_pred_total = 0
    for masks, (_, gt_inst, n_gt) in zip(pred_list, gt_list):
        n_pred_total += len(masks)
        if n_gt == 0 or not masks:
            continue
        overlap = _overlap_matrix(masks, gt_inst, n_gt)
        pred_areas, _ = _areas(masks, gt_inst, n_gt)
        for p in range(len(masks)):
            n_touch = int(np.sum(overlap[:, p] / max(pred_areas[p], 1) > threshold))
            if n_touch > 1:
                total_merge += n_touch - 1
    return total_merge / n_pred_total if n_pred_total else 0.0


# ═══════════════════════════════════════════════════════════════════
# Function registry — add your own functions here
# ═══════════════════════════════════════════════════════════════════
#
# General functions: callable(pred_masks_list, gt_list) → scalar

GENERAL_FUNCTIONS = {
    "Instance F1": compute_instance_f1,
    "Instance Precision": compute_instance_precision,
    "Instance Recall": compute_instance_recall,
    "Mean Matched IoU": compute_mean_matched_iou,
    "FG IoU": compute_fg_iou,
    "FG Precision": compute_fg_precision,
    "FG Recall": compute_fg_recall,
    "Grain Count Error": compute_grain_count_error,
    "Signed Grain Count Error": compute_signed_grain_count_error,
    "Predicted Grain Count": compute_predicted_grain_count,
    "Mean Grain Area Error": compute_mean_grain_area_error,
    "Wasserstein Grain Size Dist": compute_wasserstein_grain_size_distance,
    "Oversegmentation": compute_oversegmentation,
    "Undersegmentation": compute_undersegmentation,
}

# ═══════════════════════════════════════════════════════════════════
# Run inference for every model
# ═══════════════════════════════════════════════════════════════════

device = ("cuda" if torch.cuda.is_available()
          else "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
          else "cpu")
print(f"device={device}")

generator = pipeline("mask-generation", model=SAM_MODEL_ID, device=device)
base_decoder_state = copy.deepcopy(generator.model.mask_decoder.state_dict())

pairs = load_image_mask_pairs(IMAGE_DIR)
if not pairs:
    raise ValueError(f"No image/mask pairs found in {IMAGE_DIR}")

images, gt_list, stems = [], [], []
for img_path, mask_path in pairs:
    rgb = np.array(Image.open(img_path).convert("RGB"))
    gt_raw, gt_inst, n_gt = load_ground_truth(mask_path)
    images.append(rgb)
    gt_list.append((gt_raw, gt_inst, n_gt))
    stems.append(Path(img_path).stem)
    print(f"  {stems[-1]}: {rgb.shape[1]}x{rgb.shape[0]}, {n_gt} ground-truth grains")

all_predictions = {}

for model_name, decoder_path in MODELS:
    print(f"\n--- {model_name} ---")
    _load_decoder(generator.model, decoder_path, base_decoder_state)
    print(f"  Loaded decoder from {decoder_path or SAM_MODEL_ID + ' (stock)'}")

    pred_masks_list = []
    t0 = time.time()
    for rgb, stem in zip(images, stems):
        masks = generate_masks_square(
            generator, rgb, **FIXED,
            points_per_crop=POINTS_PER_CROP, pred_iou_thresh=PRED_IOU_THRESH,
        )
        pred_masks_list.append(masks)
        print(f"  {stem}: {len(masks)} masks")
    print(f"  Processed {len(pred_masks_list)} image(s) in {time.time() - t0:.0f}s")
    all_predictions[model_name] = pred_masks_list

# ── Verify that all models saw the same images ────────────────────
n_images = None
for model_name, preds in all_predictions.items():
    if n_images is None:
        n_images = len(preds)
    else:
        assert len(preds) == n_images, (
            f"Model '{model_name}' produced {len(preds)} predictions, "
            f"expected {n_images}"
        )

# ═══════════════════════════════════════════════════════════════════
# DataFrame — General metrics  (index: model name)
# ═══════════════════════════════════════════════════════════════════

print(f"\n{'='*60}")
print("General metrics (one value per model)")
print(f"{'='*60}")

general_rows = []
for model_name, preds in all_predictions.items():
    row = {"Model": model_name}
    for fn_name, fn in GENERAL_FUNCTIONS.items():
        row[fn_name] = fn(preds, gt_list)
    general_rows.append(row)

general_df = pd.DataFrame(general_rows).set_index("Model")
general_numeric_df = general_df.astype(float)          # keep numeric copy for plotting
general_df = general_df.map(lambda v: f"{v:.4f}")
print(general_df.to_string())

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
         ["Instance F1", "Instance Precision", "Instance Recall", "Mean Matched IoU",
          "FG IoU", "FG Precision", "FG Recall"]),
        ("Bounded error / loss",
         ["Grain Count Error", "Oversegmentation", "Undersegmentation"]),
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
    plt.suptitle("SAM 3 decoder comparison — " + ", ".join(general_numeric_df.index))
    plt.tight_layout()
    comparison_path = output_dir / "sam_model_comparison_chart.png"
    plt.savefig(comparison_path, dpi=150)
    print(f"\n  Saved comparison chart: {comparison_path}")
    plt.show()

    cmap = ListedColormap(['black', 'steelblue', 'orange'])
    n_models = len(MODELS)

    def render_overlay(image_np, masks, alpha=0.5, seed=0):
        """Paint each instance a random colour, largest first so small grains stay visible."""
        base = image_np.astype(float) / 255.0
        out = base.copy()
        rng = np.random.default_rng(seed)
        for m in sorted(masks, key=lambda x: int(x.sum()), reverse=True):
            out[m] = (1 - alpha) * out[m] + alpha * rng.random(3)
        return out

    for i, (img, (gt_raw, _, n_gt), img_stem) in enumerate(zip(images, gt_list, stems)):
        n_panels = 2 + n_models
        ncols = min(MAX_PANEL_COLS, n_panels)
        nrows = int(np.ceil(n_panels / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows),
                                 squeeze=False)
        axes = axes.ravel()

        axes[0].imshow(img)
        axes[0].set_title("Input image")
        axes[0].axis("off")

        axes[1].imshow(gt_raw, cmap=cmap, vmin=0, vmax=2)
        axes[1].set_title(f"Ground truth ({n_gt} grains)")
        axes[1].axis("off")

        for j, (model_name, _) in enumerate(MODELS):
            masks = all_predictions[model_name][i]
            axes[2 + j].imshow(render_overlay(img, masks))
            axes[2 + j].set_title(f"Overlay — {model_name} ({len(masks)})")
            axes[2 + j].axis("off")

        for ax in axes[n_panels:]:
            ax.axis("off")

        plt.suptitle(f"Image {i}")
        plt.tight_layout()
        plt.savefig(output_dir / f"{img_stem}_overview.png", dpi=150)
        plt.show()

        for j, (model_name, _) in enumerate(MODELS):
            masks = all_predictions[model_name][i]
            fig2, ax2 = plt.subplots(1, 1, figsize=(6, 5))
            ax2.imshow(render_overlay(img, masks))
            ax2.set_title(f"{model_name} overlay")
            ax2.axis("off")
            plt.tight_layout()
            plt.savefig(output_dir / f"{img_stem}_{model_name}_overlay.png", dpi=150)
            plt.close(fig2)
