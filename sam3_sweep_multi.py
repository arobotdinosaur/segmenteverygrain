"""Multi-image SAM 3 aggressiveness sweep with ground-truth scoring.

Runs SAM 3 automatic mask generation across a grid of aggressiveness parameters on
several clean grain images, scores each run against the ground-truth `_mask.tif`
(1 = grain, 2 = boundary) with instance-level and pixel-level metrics, and writes
per-image + aggregate results and overlay/heatmap figures.

Usage (in the segmenteverygrain_new_SAM env):
    KMP_DUPLICATE_LIB_OK=TRUE python sam3_sweep_multi.py [--smoke]
"""
import argparse
import itertools
import os
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from PIL import Image
from scipy import ndimage
from transformers import pipeline

# ---- 10 image / ground-truth-mask pairs (etched prac7 + Wilson surfaces) ----
PAIRS = [
    ("test_only_clean_images/cropped_prac7_etched_020.tif",        "test_only_clean_images/cropped_prac7_etched_020_mask.tif"),
    ("real_clean_images/cropped_prac7_etched_070_image.tif",       "real_clean_images/cropped_prac7_etched_070_mask.tif"),
    ("real_clean_images/cropped_prac7_etched_090_image.tif",       "real_clean_images/cropped_prac7_etched_090_mask.tif"),
    ("real_clean_images/cropped_prac7_etched_099_image.tif",       "real_clean_images/cropped_prac7_etched_099_mask.tif"),
    ("annotations/annotated_cropped_Wilson180_0.25__surface1_05.tif", "annotations/annotated_cropped_Wilson180_0.25__surface1_05_mask.tif"),
    ("annotations/annotated_cropped_Wilson180_0.25__surface1_06.tif", "annotations/annotated_cropped_Wilson180_0.25__surface1_06_mask.tif"),
    ("annotations/annotated_cropped_Wilson180_0.25__surface1_07.tif", "annotations/annotated_cropped_Wilson180_0.25__surface1_07_mask.tif"),
    ("Masks_and_images/cropped_Wilson180_0.25__surface1_08_image.tif", "Masks_and_images/cropped_Wilson180_0.25__surface1_08_mask.tif"),
    ("Masks_and_images/cropped_Wilson180_0.25__surface1_09_image.tif", "Masks_and_images/cropped_Wilson180_0.25__surface1_09_mask.tif"),
    ("Masks_and_images/cropped_Wilson180_0.25__surface1_10_image.tif", "Masks_and_images/cropped_Wilson180_0.25__surface1_10_mask.tif"),
]

SWEEP_AXES = {
    # F1 fell monotonically 32->64->96 on the first image, so the optimum is at/below 32;
    # extend downward (cheaper passes) to actually locate the peak. Keep 64/96 to map the slope.
    "points_per_crop": [16, 24, 32, 48, 64, 96],
    # pred_iou_thresh is a pure post-filter (one permissive pass reproduces every value), so
    # densifying this axis is free -- finer resolution on the precision/recall trade-off.
    "pred_iou_thresh": [0.60, 0.70, 0.78, 0.85, 0.90, 0.95],
}
FIXED = {"points_per_batch": 64, "stability_score_thresh": 0.90, "stability_score_offset": 1.0}
GRAIN_CLASS = 1
MIN_GRAIN_PX = 30
OUT = "sam3_sweep_multi"


def label_instances(binary, min_px):
    lbl, n = ndimage.label(binary)
    if n == 0:
        return lbl, 0
    sizes = np.bincount(lbl.ravel())
    keep = np.where(sizes >= min_px)[0]
    keep = keep[keep != 0]
    remap = np.zeros(n + 1, dtype=int)
    remap[keep] = np.arange(1, len(keep) + 1)
    return remap[lbl], len(keep)


def instance_scores(pred_masks, gt_label, gt_areas, n_gt, iou_thr=0.5, min_px=30):
    masks = [m for m in pred_masks if int(m.sum()) >= min_px]
    matched, ious, tp = set(), [], 0
    for m in sorted(masks, key=lambda x: int(x.sum()), reverse=True):
        lbls = gt_label[m]
        lbls = lbls[lbls > 0]
        if lbls.size == 0:
            continue
        cand, inter = np.unique(lbls, return_counts=True)
        pred_area = int(m.sum())
        iou = inter / (pred_area + gt_areas[cand] - inter)
        k = int(np.argmax(iou))
        if iou[k] >= iou_thr and cand[k] not in matched:
            matched.add(int(cand[k])); tp += 1; ious.append(float(iou[k]))
    n_pred = len(masks)
    fp, fn = n_pred - tp, n_gt - len(matched)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return dict(n_gt=n_gt, n_pred=n_pred, tp=tp, fp=fp, fn=fn,
               inst_precision=prec, inst_recall=rec, inst_f1=f1,
               mean_matched_iou=float(np.mean(ious)) if ious else 0.0)


def foreground_scores(pred_masks, gt, grain_class=1):
    true_fg = gt == grain_class
    pred_fg = np.zeros(true_fg.shape, dtype=bool)
    for m in pred_masks:
        pred_fg |= m
    inter = int((pred_fg & true_fg).sum())
    union = int((pred_fg | true_fg).sum())
    return dict(
        fg_iou=inter / union if union else 0.0,
        fg_recall=inter / int(true_fg.sum()) if true_fg.any() else 0.0,
        fg_precision=inter / int(pred_fg.sum()) if pred_fg.any() else 0.0,
    )


def generate_masks_square(generator, image_np, **params):
    """SAM 3's automatic point grid only covers ~the top square of a non-square image,
    so on a landscape frame the bottom ~25-30% gets *no* masks at all. Verified this is
    positional (survives a vertical flip) and not a threshold artifact (permissive
    stability/pred_iou filters don't recover it) -- the masks are never generated there.
    Fix: reflect-pad to a square so the real content occupies the covered top region,
    run SAM, crop masks back, and drop any that live entirely in the padded margin.
    Returns (masks_in_original_frame, scores) with padding-only masks removed."""
    h, w = image_np.shape[:2]
    s = max(h, w)
    padded = np.pad(image_np, ((0, s - h), (0, s - w), (0, 0)), mode="reflect")
    out = generator(Image.fromarray(padded), **params)
    masks, scores = [], []
    for m, sc in zip(out["masks"], out["scores"]):
        m = np.asarray(m, dtype=bool)[:h, :w]
        if int(m.sum()) >= MIN_GRAIN_PX:  # discards masks that fell entirely in the padding
            masks.append(m)
            scores.append(float(sc))
    return masks, np.asarray(scores)


def render_overlay(image_np, masks, alpha=0.5, seed=0):
    base = image_np.astype(float) / 255.0
    if base.ndim == 2:
        base = np.stack([base] * 3, axis=-1)
    out = base.copy()
    rng = np.random.default_rng(seed)
    for m in sorted(masks, key=lambda x: int(x.sum()), reverse=True):
        out[m] = (1 - alpha) * out[m] + alpha * rng.random(3)
    return (out * 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="one image, one combo, for timing")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
              else "cpu")
    print(f"device={device}", flush=True)
    generator = pipeline("mask-generation", model="facebook/sam3", device=device)

    axis_names = list(SWEEP_AXES)
    ppc_grid = SWEEP_AXES["points_per_crop"]
    iou_grid = sorted(SWEEP_AXES["pred_iou_thresh"])  # ascending; run once at the lowest
    pairs = PAIRS[:1] if args.smoke else PAIRS
    if args.smoke:
        ppc_grid, iou_grid = ppc_grid[:1], iou_grid[-1:]

    records = []
    t_start = time.time()
    for img_path, mask_path in pairs:
        name = os.path.splitext(os.path.basename(img_path))[0]
        image = Image.open(img_path).convert("RGB")
        image_np = np.array(image)
        gt = np.array(Image.open(mask_path))
        if gt.ndim == 3:  # some masks are saved RGB(A) with the class id duplicated per channel
            gt = gt[..., 0]
        gt_label, n_gt = label_instances(gt == GRAIN_CLASS, MIN_GRAIN_PX)
        gt_areas = np.bincount(gt_label.ravel(), minlength=n_gt + 1)
        overlays = {}
        # pred_iou_thresh is a pure post-filter on the returned per-mask scores (NMS is score-ordered),
        # so one permissive pass per points_per_crop reproduces every stricter threshold exactly. Verified.
        for ppc in ppc_grid:
            t0 = time.time()
            masks_all, scores_all = generate_masks_square(
                generator, image_np, **FIXED, points_per_crop=ppc, pred_iou_thresh=iou_grid[0])
            secs = round(time.time() - t0, 1)
            for t in iou_grid:
                masks = [m for m, s in zip(masks_all, scores_all) if s >= t]
                row = dict(image=name, points_per_crop=ppc, pred_iou_thresh=t)
                row.update(instance_scores(masks, gt_label, gt_areas, n_gt, min_px=MIN_GRAIN_PX))
                row.update(foreground_scores(masks, gt, GRAIN_CLASS))
                row["seconds"] = secs if t == iou_grid[0] else 0.0
                records.append(row)
                overlays[(ppc, t)] = render_overlay(image_np, masks)
                print(f"  {name} ppc={ppc} pred_iou={t} -> F1={row['inst_f1']:.3f} "
                      f"recall={row['inst_recall']:.3f} n_pred={row['n_pred']}", flush=True)
        pd.DataFrame(records).to_csv(f"{OUT}/per_image_results.csv", index=False)  # checkpoint

        if not args.smoke and len(axis_names) == 2:
            rv, cv = SWEEP_AXES[axis_names[0]], SWEEP_AXES[axis_names[1]]
            df = pd.DataFrame(records)
            df = df[df.image == name].set_index(axis_names)
            fig, axes = plt.subplots(len(rv), len(cv), figsize=(5 * len(cv), 5 * len(rv)), squeeze=False)
            for i, a in enumerate(rv):
                for j, b in enumerate(cv):
                    ax = axes[i][j]; ax.imshow(overlays[(a, b)]); ax.axis("off")
                    r = df.loc[(a, b)]
                    ax.set_title(f"{axis_names[0]}={a}, {axis_names[1]}={b}\n"
                                 f"F1={r.inst_f1:.2f} recall={r.inst_recall:.2f} n={int(r.n_pred)}", fontsize=9)
            fig.suptitle(f"{name}  (GT={n_gt})", fontsize=13)
            plt.tight_layout(); plt.savefig(f"{OUT}/{name}_overlays.png", dpi=110, bbox_inches="tight"); plt.close(fig)
        print(f"[done] {name} (GT={n_gt})  elapsed={time.time()-t_start:.0f}s", flush=True)

    results = pd.DataFrame(records)
    results.to_csv(f"{OUT}/per_image_results.csv", index=False)
    if args.smoke:
        print("SMOKE OK", flush=True)
        return

    metrics = ["inst_f1", "inst_recall", "inst_precision", "mean_matched_iou", "fg_iou", "fg_recall", "n_pred"]
    agg = results.groupby(axis_names)[metrics].agg(["mean", "std"])
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    agg = agg.reset_index()
    agg.to_csv(f"{OUT}/aggregate_results.csv", index=False)

    # Aggregate heatmaps (mean across images)
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for ax, metric in zip(axes.ravel(), ["inst_f1", "inst_recall", "inst_precision", "mean_matched_iou", "fg_iou", "n_pred"]):
        grid = results.groupby(axis_names)[metric].mean().unstack()
        im = ax.imshow(grid.values, cmap="viridis", aspect="auto")
        ax.set_xticks(range(len(grid.columns))); ax.set_xticklabels(grid.columns)
        ax.set_yticks(range(len(grid.index))); ax.set_yticklabels(grid.index)
        ax.set_xlabel(axis_names[1]); ax.set_ylabel(axis_names[0]); ax.set_title(f"mean {metric}")
        for (yy, xx), v in np.ndenumerate(grid.values):
            ax.text(xx, yy, f"{v:.2f}" if metric != "n_pred" else f"{v:.0f}", ha="center", va="center", color="w", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"SAM 3 sweep — mean over {len(pairs)} images", fontsize=14)
    plt.tight_layout(); plt.savefig(f"{OUT}/aggregate_heatmaps.png", dpi=120, bbox_inches="tight"); plt.close(fig)

    ranked = agg.sort_values("inst_f1_mean", ascending=False)
    with open(f"{OUT}/summary.txt", "w") as f:
        f.write(f"SAM 3 aggressiveness sweep over {len(pairs)} images\n")
        f.write(f"FIXED = {FIXED}\n\nRanked by mean instance F1:\n")
        f.write(ranked.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\n=== Ranked by mean instance F1 ===", flush=True)
    print(ranked.to_string(index=False, float_format=lambda v: f"{v:.3f}"), flush=True)
    print(f"\nTotal elapsed: {time.time()-t_start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
