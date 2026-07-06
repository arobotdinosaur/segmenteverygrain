# Mass auto-annotation of grain images with SAM3.
#
# Two engines (--engine):
#   amg  (default) SAM3's automatic mask generator with the sweep-tuned knobs and
#        the square-pad fix. Best instance F1 on clean images (0.67 on prac7_099).
#        Cannot use a fine-tuned decoder: injecting one breaks AMG's IoU/stability
#        filtering (measured F1 0.30), because fine-tuning with
#        multimask_output=False recalibrates the IoU head AMG filters on.
#   grid Dense grid of point prompts through the promptable decoder (the same path
#        Sam3Predictor / SAM3_finetune.ipynb use), so a fine-tuned decoder plugs
#        straight in via --decoder. Slightly lower instance F1 today (0.61 with the
#        clean_noaug decoder) but better per-mask IoU, ~28 s/image, and it improves
#        as the decoder is fine-tuned on more annotations.
#
# Both write training pairs in the project convention:
#   {stem}_image.tif   copy of the input image
#   {stem}_mask.tif    0/1/2 (+3) semantic mask: background/interior/boundary/ignore
#   qc_overlays/{stem}_overlay.png + summary.csv   (resumable: done images skipped)
#
# Usage (segmenteverygrain_new_SAM env):
#   python sam3_mass_annotate.py --input-folder testcleanimages
#   python sam3_mass_annotate.py --input-folder testcleanimages --engine grid \
#       --decoder models/sam3_decoder_clean_noaug_ft.pt
import argparse
import csv
import os
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage

from sam3_predictor import Sam3Predictor

BACKGROUND, GRAIN_INTERIOR, GRAIN_BOUNDARY, IGNORE = 0, 1, 2, 3


def annotate_image_amg(generator, image_rgb, args):
    """SAM3 automatic mask generation with square-pad fix (see SAM3_clean_grains.ipynb)."""
    h, w = image_rgb.shape[:2]
    s = max(h, w)
    padded = np.pad(image_rgb, ((0, s - h), (0, s - w), (0, 0)), mode="reflect")
    out = generator(Image.fromarray(padded),
                    points_per_crop=args.points_per_crop,
                    pred_iou_thresh=args.pred_iou_thresh,
                    stability_score_thresh=args.stability_score_thresh)
    masks = [np.asarray(m, bool)[:h, :w] for m in out["masks"]]
    masks = [m for m in masks if m.sum() >= args.min_size]
    return sorted(masks, key=lambda m: m.sum(), reverse=True)


def grid_points(h, w, n_per_side, margin=0.02):
    xs = np.linspace(margin * w, (1 - margin) * w, n_per_side)
    ys = np.linspace(margin * h, (1 - margin) * h, n_per_side)
    gx, gy = np.meshgrid(xs, ys)
    return np.column_stack([gx.ravel(), gy.ravel()])


@torch.no_grad()
def grid_masks(predictor, points, batch_size=64):
    """Prompt each grid point separately; return low-res bool masks + iou scores."""
    model, processor = predictor.model, predictor.processor
    osizes = predictor._original_sizes
    masks_lr, scores = [], []
    for i in range(0, len(points), batch_size):
        chunk = points[i:i + batch_size]
        inputs = processor(
            original_sizes=osizes.tolist() if torch.is_tensor(osizes) else osizes,
            input_points=[[[list(map(float, p))] for p in chunk]],
            input_labels=[[[1]] * len(chunk)],
            return_tensors="pt",
        ).to(predictor.device)
        out = model(image_embeddings=predictor._embeddings,
                    input_points=inputs["input_points"],
                    input_labels=inputs["input_labels"],
                    multimask_output=False)
        masks_lr.append((out.pred_masks[0, :, 0] > 0).cpu())
        scores.append(out.iou_scores.reshape(-1).cpu())
    return torch.cat(masks_lr).numpy(), torch.cat(scores).numpy()


def nms_masks(masks, scores, iou_thresh):
    """Greedy NMS over (stacked) bool masks, highest score first."""
    order = np.argsort(scores)[::-1]
    areas = masks.reshape(len(masks), -1).sum(1)
    keep, suppressed = [], np.zeros(len(masks), bool)
    flat = masks.reshape(len(masks), -1)
    for i in order:
        if suppressed[i]:
            continue
        keep.append(i)
        inter = flat[~suppressed] @ flat[i]
        cand = np.where(~suppressed)[0]
        iou = inter / np.maximum(areas[cand] + areas[i] - inter, 1)
        suppressed[cand[iou > iou_thresh]] = True
        suppressed[i] = True
    return keep


def masks_to_semantic(masks_bool, shape, boundary_width=2, unlabeled=IGNORE):
    label = np.full(shape, unlabeled, np.uint8)
    for m in masks_bool:
        label[m] = GRAIN_INTERIOR
    for m in masks_bool:
        band = (ndimage.binary_dilation(m, iterations=boundary_width)
                & ~ndimage.binary_erosion(m, iterations=boundary_width))
        label[band] = GRAIN_BOUNDARY
    return label


def save_overlay(image_rgb, masks_bool, out_path):
    overlay = image_rgb.astype(float) / 255.0
    rng = np.random.default_rng(0)
    for m in masks_bool:
        overlay[m] = 0.55 * overlay[m] + 0.45 * rng.random(3)
        outline = m & ~ndimage.binary_erosion(m)
        overlay[outline] = (1.0, 1.0, 1.0)
    Image.fromarray((overlay * 255).astype(np.uint8)).save(out_path)


def annotate_image(predictor, image_rgb, args):
    h, w = image_rgb.shape[:2]
    predictor.set_image(image_rgb)
    pts = grid_points(h, w, args.grid_points)
    masks_lr, scores = grid_masks(predictor, pts, args.points_per_batch)

    lh, lw = masks_lr.shape[-2:]
    min_size_lr = max(1, int(args.min_size * (lh * lw) / (h * w)))
    areas = masks_lr.reshape(len(masks_lr), -1).sum(1)
    ok = (scores >= args.iou_thresh) & (areas >= min_size_lr) & (areas < 0.5 * lh * lw)
    masks_lr, scores = masks_lr[ok], scores[ok]
    if len(masks_lr) == 0:
        return []

    keep = nms_masks(masks_lr, scores, args.nms_iou)
    kept = torch.as_tensor(masks_lr[keep], dtype=torch.float)[:, None]
    out = []
    for i in range(0, len(kept), 32):
        up = F.interpolate(kept[i:i + 32], size=(h, w), mode="bilinear", align_corners=False)
        for m in (up[:, 0] > 0.5).numpy():
            if m.sum() >= args.min_size:
                out.append(m)
    return sorted(out, key=lambda m: m.sum(), reverse=True)


def main():
    ap = argparse.ArgumentParser(description="Mass-annotate grain images with SAM3 grid prompting.")
    ap.add_argument("--input-folder", default="testcleanimages")
    ap.add_argument("--output-folder", default=None, help="default: <input>_sam3_annotated")
    ap.add_argument("--engine", default="amg", choices=["amg", "grid"])
    ap.add_argument("--decoder", default=None, help="fine-tuned mask-decoder .pt (grid engine only)")
    ap.add_argument("--pattern", default="*.tif")
    ap.add_argument("--max-images", type=int, default=None)
    # grid engine knobs
    ap.add_argument("--grid-points", type=int, default=32, help="grid points per side")
    ap.add_argument("--points-per-batch", type=int, default=64)
    ap.add_argument("--iou-thresh", type=float, default=0.7, help="min predicted IoU to keep a mask")
    ap.add_argument("--nms-iou", type=float, default=0.6)
    # amg engine knobs (defaults = sweep-tuned values from sam3_sweep_multi.py)
    ap.add_argument("--points-per-crop", type=int, default=48)
    ap.add_argument("--pred-iou-thresh", type=float, default=0.70)
    ap.add_argument("--stability-score-thresh", type=float, default=0.90)
    ap.add_argument("--min-size", type=int, default=25)
    ap.add_argument("--boundary-width", type=int, default=2)
    ap.add_argument("--unlabeled", type=int, default=IGNORE,
                    help="class for pixels no grain covers: 3=ignore (default) or 0=background")
    args = ap.parse_args()

    out_dir = Path(args.output_folder or f"{Path(args.input_folder).name}_sam3_annotated")
    qc_dir = out_dir / "qc_overlays"
    qc_dir.mkdir(parents=True, exist_ok=True)

    if args.engine == "amg":
        if args.decoder:
            raise SystemExit("--decoder only works with --engine grid "
                             "(fine-tuned decoders break AMG's proposal filtering).")
        from transformers import pipeline
        device = ("cuda" if torch.cuda.is_available()
                  else "mps" if torch.backends.mps.is_available() else "cpu")
        generator = pipeline("mask-generation", model="facebook/sam3", device=device)
        run_one = lambda image: annotate_image_amg(generator, image, args)
        mps = device == "mps"
    else:
        predictor = Sam3Predictor()
        if args.decoder:
            predictor.model.mask_decoder.load_state_dict(
                torch.load(args.decoder, map_location=predictor.device))
            print(f"loaded fine-tuned decoder: {args.decoder}")
        run_one = lambda image: annotate_image(predictor, image, args)
        mps = predictor.device == "mps"

    paths = sorted(Path(args.input_folder).glob(args.pattern))
    paths = [p for p in paths if "mask" not in p.stem.lower()][: args.max_images]
    print(f"annotating {len(paths)} images from {args.input_folder} -> {out_dir}")

    rows = []
    for i, p in enumerate(paths, 1):
        mask_out = out_dir / f"{p.stem}_mask.tif"
        if mask_out.exists():                       # resumable
            print(f"[{i:>3}/{len(paths)}] {p.stem}: already done, skipping")
            continue
        t0 = time.time()
        image = np.array(Image.open(p).convert("RGB"))
        masks = run_one(image)
        label = masks_to_semantic(masks, image.shape[:2],
                                  args.boundary_width, args.unlabeled)
        Image.open(p).save(out_dir / f"{p.stem}_image.tif")
        Image.fromarray(label).save(mask_out)
        save_overlay(image, masks, qc_dir / f"{p.stem}_overlay.png")
        cov = float((label == GRAIN_INTERIOR).mean() + (label == GRAIN_BOUNDARY).mean())
        rows.append({"image": p.name, "grains": len(masks), "coverage": round(cov, 3),
                     "seconds": round(time.time() - t0, 1)})
        print(f"[{i:>3}/{len(paths)}] {p.stem}: {len(masks)} grains, "
              f"coverage {cov:.0%} ({rows[-1]['seconds']}s)", flush=True)
        if mps:
            torch.mps.empty_cache()

    if rows:
        with open(out_dir / "summary.csv", "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=["image", "grains", "coverage", "seconds"])
            wr.writeheader(); wr.writerows(rows)
    print(f"done: outputs in {out_dir}")


if __name__ == "__main__":
    main()
