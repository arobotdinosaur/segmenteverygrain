"""Black-box objective + Bayesian optimization loop for synthetic-data search.

High-level flow:
theta -> synthetic generator -> train chosen model -> eval on real data -> score

The GP loop then uses those scores to suggest the next theta to try.
"""

import numpy as np
import cv2
import shutil
import copy
import random
import threading
from collections import defaultdict
from functools import partial
from pathlib import Path
import segmenteverygrain as seg
from create_synthetic_images import (
    generate_synthetic_images as generate_synthetic_images_from_script,
    load_image_mask_pairs,
)
from synthetic_noise import NoiseParams, synthetic_noise_model_input
import tensorflow as tf
tf.config.experimental.enable_op_determinism()
from sklearn.model_selection import train_test_split
from glob import glob

import albumentations as A
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from keras.optimizers import Adam
import keras
from segmenteverygrain.resnext_model import MaskingResNeXt, weighted_crossentropy_torch

MASTER_SEED = 42
_rng_cache = {}
_rng_lock = threading.Lock()


def _derive_seeds(master_seed):
    global MASTER_SEED, _keras_seed, _shuffle_seed, _noise_seed, _noise_master_rng
    MASTER_SEED = master_seed
    rng = np.random.default_rng(MASTER_SEED)
    _keras_seed = int(rng.integers(2**32))
    _shuffle_seed = int(rng.integers(2**63))
    _noise_seed = int(rng.integers(2**63))
    keras.utils.set_random_seed(_keras_seed)
    _noise_master_rng = np.random.default_rng(_noise_seed)
    _rng_cache.clear()


_derive_seeds(MASTER_SEED)


def boundary_gradient_alignment(
    clean: np.ndarray,
    noisy: np.ndarray,
    mask: np.ndarray,
    boundary_class: int = 2,
    dilation_size: int = 3,
    min_clean_gradient: float = 1e-3,
    eps: float = 1e-8,
) -> float:
    clean = clean.astype(np.float32)
    noisy = noisy.astype(np.float32)

    if clean.max() > 1.0:
        clean /= 255.0
    if noisy.max() > 1.0:
        noisy /= 255.0

    boundary_region = (mask == boundary_class).astype(np.uint8)

    if dilation_size > 1:
        kernel = np.ones((dilation_size, dilation_size), np.uint8)
        boundary_region = cv2.dilate(boundary_region, kernel, iterations=1)

    boundary_region = boundary_region.astype(bool)

    clean_gx = cv2.Sobel(clean, cv2.CV_32F, 1, 0, ksize=3)
    clean_gy = cv2.Sobel(clean, cv2.CV_32F, 0, 1, ksize=3)
    noisy_gx = cv2.Sobel(noisy, cv2.CV_32F, 1, 0, ksize=3)
    noisy_gy = cv2.Sobel(noisy, cv2.CV_32F, 0, 1, ksize=3)

    clean_mag = np.hypot(clean_gx, clean_gy)
    noisy_mag = np.hypot(noisy_gx, noisy_gy)

    valid = boundary_region & (clean_mag >= min_clean_gradient)

    if not np.any(valid):
        return 0.0

    cosine = (clean_gx * noisy_gx + clean_gy * noisy_gy) / (
        clean_mag * noisy_mag + eps
    )

    alignment = (cosine[valid] + 1.0) / 2.0
    weights = clean_mag[valid]

    return float(np.clip(np.average(alignment, weights=weights), 0.0, 1.0))


def _to_binary_mask(mask):
    if mask.dtype == bool:
        return mask
    mask = np.asarray(mask)
    if mask.ndim == 3:
        mask = mask[..., 0]
    return (mask > 0).astype(np.uint8)


def _sobel_gradients(image):
    image = image.astype(np.float32)
    if image.max() > 1.0:
        image = image / 255.0
    gx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)
    return gx, gy, mag


def false_edge_density(
    clean_image: np.ndarray,
    noisy_image: np.ndarray,
    boundary_mask: np.ndarray,
    exclusion_radius: int = 3,
    edge_threshold_percentile: float = 90.0,
    clean_edge_tolerance: float = 0.5,
    eps: float = 1e-8,
) -> float:
    mask = _to_binary_mask(boundary_mask).astype(np.uint8)

    kernel_size = 2 * exclusion_radius + 1
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)

    excluded_boundary_region = cv2.dilate(
        mask,
        kernel,
        iterations=1,
    ).astype(bool)

    non_boundary_region = ~excluded_boundary_region

    _, _, clean_mag = _sobel_gradients(clean_image)
    _, _, noisy_mag = _sobel_gradients(noisy_image)

    if not np.any(non_boundary_region):
        return 0.0

    threshold = np.percentile(
        noisy_mag[non_boundary_region],
        edge_threshold_percentile,
    )

    strong_noisy_edges = noisy_mag >= threshold
    weak_clean_structure = clean_mag < clean_edge_tolerance * threshold

    false_edges = (
        non_boundary_region
        & strong_noisy_edges
        & weak_clean_structure
    )

    return float(
        np.sum(false_edges)
        / (np.sum(non_boundary_region) + eps)
    )


def boundary_contrast_retention(
    clean_image: np.ndarray,
    noisy_image: np.ndarray,
    boundary_mask: np.ndarray,
    dilation_radius: int = 2,
    eps: float = 1e-8,
    clip_score: bool = False,
) -> float:
    mask = _to_binary_mask(boundary_mask).astype(np.uint8)

    if dilation_radius > 0:
        kernel_size = 2 * dilation_radius + 1
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        evaluation_mask = cv2.dilate(mask, kernel, iterations=1).astype(bool)
    else:
        evaluation_mask = mask.astype(bool)

    _, _, clean_mag = _sobel_gradients(clean_image)
    _, _, noisy_mag = _sobel_gradients(noisy_image)

    if not np.any(evaluation_mask):
        return 0.0

    clean_contrast = float(np.mean(clean_mag[evaluation_mask]))
    noisy_contrast = float(np.mean(noisy_mag[evaluation_mask]))

    score = noisy_contrast / (clean_contrast + eps)

    if clip_score:
        score = np.clip(score, 0.0, 1.0)

    return float(score)


def precompute_score_stats(region_pairs, noise_params, n_samples=50,
                           evidence_fns=None, quantile_bounds=None):
    """Pre-compute score statistics for each evidence function on every region.

    Runs once before training. Only processes train-split regions.

    Parameters
    ----------
    region_pairs:
        List of dicts with keys img_path, mask_path, x, y, split.
    noise_params:
        NoiseParams used to generate synthetic realizations.
    n_samples:
        Number of noisy realizations per region for stat estimation.
    evidence_fns:
        List of callables with signature
        (clean, noisy, mask) -> float.  Defaults to
        [boundary_gradient_alignment].
    quantile_bounds:
        Optional list with one entry per evidence function.
        Each entry is either a (lo, hi) tuple of quantile bounds
        (e.g. (0.05, 0.95)) to enable quantile-based filtering for
        that function, or None to use z-score filtering instead.
        Example: [(0.023, 0.159), None] uses quantile for the first
        function and z-score for the second.

    Returns
    -------
    dict keyed by (img_path, x, y) -> list of dicts, one per evidence
        function.  Each dict has keys 'mean', 'std', and optionally
        'q_lo', 'q_hi' (only when that function has quantile_bounds).
    """
    if evidence_fns is None:
        evidence_fns = [boundary_gradient_alignment]

    train_pairs = [p for p in region_pairs if p["split"] == "train"]
    stats = {}
    for p in train_pairs:
        img_path = p["img_path"]
        mask_path = p["mask_path"]
        rx, ry = p["x"], p["y"]

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE).astype(np.uint8)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE).astype(np.uint8)

        crop_img = img[ry:ry + 384, rx:rx + 384]
        crop_mask = mask[ry:ry + 384, rx:rx + 384]

        sy = (384 - 256) // 2
        sx = (384 - 256) // 2
        clean = crop_img[sy:sy + 256, sx:sx + 256].astype(np.float32) / 255.0
        mask_cls = crop_mask[sy:sy + 256, sx:sx + 256]

        all_scores = [[] for _ in evidence_fns]
        rng = np.random.default_rng(_noise_seed)
        for _ in range(n_samples):
            noisy = synthetic_noise_model_input(clean, noise_params, rng)
            for j, fn in enumerate(evidence_fns):
                all_scores[j].append(fn(clean, noisy, mask_cls))

        func_stats = []
        for j in range(len(evidence_fns)):
            arr = np.array(all_scores[j])
            entry = {"mean": float(arr.mean()), "std": float(arr.std())}
            if quantile_bounds is not None and quantile_bounds[j] is not None:
                lo_q, hi_q = quantile_bounds[j]
                entry["q_lo"] = float(np.percentile(arr, lo_q * 100))
                entry["q_hi"] = float(np.percentile(arr, hi_q * 100))
            func_stats.append(entry)

        stats[(img_path, rx, ry)] = func_stats

    return stats


COMBINE_WITH_CLEAN = True  # Set True to include pre-injection clean images in training
AUGMENT_CLEAN = True  # Set True to apply Albumentations spatial augmentation to real clean images
AUGMENT_NOISY = True  # Set True to apply Albumentations spatial augmentation to real noisy images
N_SYNTHETIC_VARIANTS = 8  # Number of noisy variants per clean image
N_CROP_VIEWS = 8  # Replicate each real-noisy crop region for more views per epoch

# Evidence functions used to filter synthetic noise realizations by z-score.
# Each function has signature (clean, noisy, mask) -> float.
# A realization is kept only if ALL functions produce scores within their
# pre-computed [mean - z*std, mean + z*std] bounds.
EVIDENCE_FNS = [boundary_gradient_alignment]

# Albumentations pipeline for real noisy image augmentation.
real_noisy_aug = A.Compose([
    A.Rotate(
        limit=(-45, 45),
        interpolation=cv2.INTER_LINEAR,
        mask_interpolation=cv2.INTER_NEAREST,
        border_mode=cv2.BORDER_REFLECT_101,
        p=0.8,
    ),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomCrop(height=256, width=256),
])

TARGET_PATH = "./real_noisy_images/"
CLEAN_PATH = "./real_clean_images/"
PREDICT_PATH = "./prediction_noisy_images/"


# Workspace helper: recreate a directory from scratch for each run.
def reset_dir(path):
    path = Path(path)
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# Region-grid utilities for on-the-fly real-noisy augmentation
# ---------------------------------------------------------------------------

def compute_crop_grid(height, width, crop_size=384, stride=192):
    """Return list of (x, y) top-left corners for valid crop_size regions."""
    positions = []
    for y in range(0, height - crop_size + 1, stride):
        for x in range(0, width - crop_size + 1, stride):
            positions.append((x, y))
    return positions


def load_and_augment_real_noisy(img_path, mask_path, rx, ry, augment=True):
    """Load a full image+m·ask, extract the (rx,ry) 384×384 region,
    apply Albumentations → 256×256, and return (img, mask).

    *img_path* and *mask_path* may be either Python strings/bytes or
    tf.EagerTensor; *rx* and *ry* may be Python ints or tf.EagerTensor.
    """
    if hasattr(img_path, "numpy"):
        img_path = img_path.numpy()
    if hasattr(mask_path, "numpy"):
        mask_path = mask_path.numpy()
    if hasattr(rx, "numpy"):
        rx = int(rx.numpy())
    if hasattr(ry, "numpy"):
        ry = int(ry.numpy())
    img_path = img_path.decode() if isinstance(img_path, bytes) else str(img_path)
    mask_path = mask_path.decode() if isinstance(mask_path, bytes) else str(mask_path)

    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.uint8)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE).astype(np.uint8)

    crop_img = img[ry:ry + 384, rx:rx + 384]
    crop_mask = mask[ry:ry + 384, rx:rx + 384]

    if augment:
        augmented = real_noisy_aug(image=crop_img, mask=crop_mask)
        out_img = augmented["image"].astype(np.float32) / 255.0
        out_mask = augmented["mask"]
    else:
        sy = (384 - 256) // 2
        sx = (384 - 256) // 2
        out_img = crop_img[sy:sy + 256, sx:sx + 256].astype(np.float32) / 255.0
        out_mask = crop_mask[sy:sy + 256, sx:sx + 256]

    return out_img, out_mask


# Split-selected image/mask pairs into a fresh folder on disk.
def stage_pairs(pairs, folder):
    folder = reset_dir(folder)
    for img_path, mask_path in pairs:
        shutil.copy2(img_path, folder / Path(img_path).name)
        shutil.copy2(mask_path, folder / Path(mask_path).name)
    return folder


# Multi-resolution augmentation: downsample then upsample each synthetic patch.
def create_scaled_variants(image_files, mask_files, scales, split_dir):
    """For a given split, creates downsampled+upsampled variants at each scale and saves to disk."""
    split_dir = Path(split_dir)
    all_images = list(image_files)
    all_masks = list(mask_files)

    for scale in scales:
        scale_dir = split_dir / f"res_{scale:.2f}"
        img_dir = scale_dir / "images"
        mask_dir = scale_dir / "masks"
        img_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)

        for img_path, mask_path in zip(image_files, mask_files):
            img = cv2.imread(img_path)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if img is None or mask is None:
                continue

            h, w = img.shape[:2]
            new_h, new_w = int(h * scale), int(w * scale)

            down_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            up_img = cv2.resize(down_img, (w, h), interpolation=cv2.INTER_CUBIC)

            img_name = Path(img_path).stem
            cv2.imwrite(str(img_dir / f"{img_name}_res{int(scale*100)}.png"), up_img)
            cv2.imwrite(str(mask_dir / f"{img_name}_res{int(scale*100)}.png"), mask)

        all_images += sorted(glob(str(img_dir / "*.png")))
        all_masks += sorted(glob(str(mask_dir / "*.png")))

    return all_images, all_masks


# Torch patch dataset used only for the ResNeXt path.
class PatchDataset(Dataset):
    """Simple paired patch dataset for torch models."""

    def __init__(self, image_files, mask_files, augment=False):
        self.image_files = list(image_files)
        self.mask_files = list(mask_files)
        self.augment = augment

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img = Image.open(self.image_files[idx]).convert("RGB")
        mask = Image.open(self.mask_files[idx]).convert("L")

        if self.augment:
            if random.random() > 0.5:
                img = TF.hflip(img)
                mask = TF.hflip(mask)
            if random.random() > 0.5:
                img = TF.vflip(img)
                mask = TF.vflip(mask)
            k = random.randint(0, 3)
            if k:
                img = TF.rotate(img, 90 * k)
                mask = TF.rotate(mask, 90 * k)

        img_t = torch.from_numpy(np.array(img).astype("float32") / 255.0).permute(2, 0, 1)
        mask_t = torch.from_numpy(np.array(mask).astype("int64"))
        return img_t, mask_t


# Apply synthetic noise to an existing TF dataset of (image, mask) pairs.
def add_synthetic_noise(dataset, noise_params, score_stats=None,
                        z_lower=1.5, z_upper=1.5, evidence_fns=None,
                        quantile_bounds=None):
    """Wrap a TF dataset so synthetic noise is applied to each image.

    If *score_stats* is provided (a dict from precompute_score_stats),
    the dataset must be built with include_meta=True so each element is
    (img, mask, img_path, rx, ry). Each image gets one noisy realization
    at a time; if any evidence function's score falls outside the
    accepted range the realization is regenerated until all filters
    pass (AND logic).

    Filtering is per-function:
      - If quantile_bounds[j] is not None and the precomputed entry
        has 'q_lo'/'q_hi', quantile-based filtering is used for that
        function: reject if score outside [q_lo, q_hi].
      - Otherwise, z-score filtering is used for that function:
        reject if score outside [mean - z_lower[j]*std, mean + z_upper[j]*std].

    If *score_stats* is None, expects a plain (img, mask) dataset and
    falls back to a single-pass with no filtering.

    Parameters
    ----------
    z_lower, z_upper:
        Scalar or list/tuple with one entry per evidence function.
        A scalar is broadcast to all functions.
        Used only for functions where quantile_bounds[j] is None.
    quantile_bounds:
        Optional list with one entry per evidence function.
        Each entry is either a (lo, hi) tuple to enable quantile-based
        filtering, or None to use z-score filtering instead.
    evidence_fns:
        List of callables with signature (clean, noisy, mask) -> float.
        Defaults to [boundary_gradient_alignment].
    """
    if evidence_fns is None:
        evidence_fns = [boundary_gradient_alignment]

    n_fns = len(evidence_fns)
    if np.isscalar(z_lower):
        z_lower = [z_lower] * n_fns
    if np.isscalar(z_upper):
        z_upper = [z_upper] * n_fns
    if quantile_bounds is None:
        quantile_bounds = [None] * n_fns

    has_meta = score_stats is not None

    if has_meta:
        for key in score_stats:
            if key not in _rng_cache:
                element_seed = int(_noise_master_rng.integers(0, 2**63))
                _rng_cache[key] = np.random.default_rng(element_seed)

    def _apply_noise(img, mask, img_path=None, rx=None, ry=None):
        gray = img[..., 0]
        mask_class = tf.argmax(mask, axis=-1)  # (256,256) with values 0,1,2

        def _fn_with_meta(clean, mask_idx, ip, x, y):
            clean_np = clean.numpy().astype(np.float32)
            mask_np = mask_idx.numpy().astype(np.uint8)

            ip_str = ip.numpy().decode() if hasattr(ip, 'numpy') else (ip.decode() if isinstance(ip, bytes) else str(ip))
            key = (ip_str, int(x.numpy()) if hasattr(x, 'numpy') else int(x),
                   int(y.numpy()) if hasattr(y, 'numpy') else int(y))

            with _rng_lock:
                rng = _rng_cache[key]

            noisy = synthetic_noise_model_input(clean_np, noise_params, rng)

            if score_stats is not None:
                if key in score_stats:
                    func_stats = score_stats[key]

                    while True:
                        all_pass = True
                        for j, fn in enumerate(evidence_fns):
                            entry = func_stats[j]
                            use_q = (quantile_bounds[j] is not None
                                     and "q_lo" in entry)
                            if use_q:
                                lo = entry["q_lo"]
                                hi = entry["q_hi"]
                            elif isinstance(entry, dict):
                                mean_s = entry["mean"]
                                std_s = entry["std"]
                                lo = mean_s - z_lower[j] * std_s
                                hi = mean_s + z_upper[j] * std_s
                            else:
                                mean_s, std_s = entry
                                lo = mean_s - z_lower[j] * std_s
                                hi = mean_s + z_upper[j] * std_s
                            s = fn(clean_np, noisy, mask_np)
                            if not (lo <= s <= hi):
                                all_pass = False
                                break
                        if all_pass:
                            break
                        noisy = synthetic_noise_model_input(clean_np, noise_params, rng)

            return noisy

        def _fn_plain(clean, mask_idx):
            clean_np = clean.numpy().astype(np.float32)
            rng = np.random.default_rng(_noise_seed)
            return synthetic_noise_model_input(clean_np, noise_params, rng)

        if has_meta:
            noisy = tf.py_function(_fn_with_meta, [gray, mask_class, img_path, rx, ry], tf.float32)
        else:
            noisy = tf.py_function(_fn_plain, [gray, mask_class], tf.float32)

        noisy.set_shape((256, 256))
        noisy = tf.stack([noisy, noisy, noisy], axis=-1)
        return noisy, mask

    if has_meta:
        def _wrap(img, mask, img_path, rx, ry):
            noisy, mask = _apply_noise(img, mask, img_path, rx, ry)
            return noisy, mask
    else:
        def _wrap(img, mask):
            noisy, mask = _apply_noise(img, mask)
            return noisy, mask

    return dataset.map(_wrap, num_parallel_calls=1)


# TensorFlow dataset builder for online synthetic noise.
def build_synthetic_noise_dataset(image_files, mask_files, params, n_variants=8, seed=None, batch_size=None):
    """Build a TF dataset that applies synthetic noise on-the-fly to clean patches.

    Each clean patch yields *n_variants* noisy versions per epoch,
    each with independent noise.

    If *seed* is None, noise is fresh random each time (training).
    If *seed* is an int, noise is reproducible (val/test).
    """

    def generator():
        rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
        for img_path, mask_path in zip(image_files, mask_files):
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask_onehot = np.eye(3, dtype=np.float32)[mask]
            for _ in range(n_variants):
                noisy = synthetic_noise_model_input(img, params, rng)
                noisy = np.stack([noisy] * 3, axis=-1).astype(np.float32)
                yield noisy, mask_onehot

    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(256, 256, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(256, 256, 3), dtype=tf.float32),
        ),
    )
    if image_files:
        dataset = dataset.apply(tf.data.experimental.assert_cardinality(len(image_files) * n_variants))
    if batch_size is not None:
        dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset


# TensorFlow dataset builder used by the Keras U-Net paths.
def build_dataset(image_files, mask_files, augmentation=False, batch_size=32, shuffle_buffer=1000):
    """Builds a TF dataset from image and mask file paths.

    If *batch_size* is None, returns unbatched elements.
    """
    dataset = tf.data.Dataset.from_tensor_slices((image_files, mask_files))

    if augmentation:
        dataset = tf.data.Dataset.from_tensor_slices((
            image_files,
            mask_files,
            tf.Variable([True] * len(image_files), dtype=tf.bool),
        ))

    if augmentation:
        _seed = _shuffle_seed
        dataset = dataset.map(
            lambda img, mask, *a: seg.load_and_preprocess_seeded(img, mask, _seed),
            num_parallel_calls=1,
        )
    else:
        dataset = dataset.map(seg.load_and_preprocess, num_parallel_calls=1)
    if batch_size is not None:
        dataset = dataset.shuffle(shuffle_buffer, seed=_shuffle_seed).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset


# TensorFlow dataset builder for region-based real noisy augmentation.
def build_real_noisy_dataset(pairs, split, augment=True, batch_size=32,
                             shuffle_buffer=1000, include_meta=False):
    """Build a TF dataset from region-assigned real noisy image pairs.

    *pairs* is a list of dicts with keys: img_path, mask_path, x, y, split.
    Filters to *split*, then applies load_and_augment_real_noisy via py_function.

    If *batch_size* is None, returns an unbatched (element-level) dataset
    suitable for concatenation before batching.

    If *include_meta* is True, each element is a 5-tuple
    (img, mask, img_path, rx, ry) so downstream wrappers can look up
    pre-computed stats by region key.
    """
    items = [p for p in pairs if p["split"] == split]
    if not items:
        empty_img = tf.zeros([0, 256, 256, 3], dtype=tf.float32)
        empty_mask = tf.zeros([0, 256, 256, 3], dtype=tf.float32)
        return tf.data.Dataset.from_tensor_slices((empty_img, empty_mask))

    img_paths = [p["img_path"] for p in items]
    mask_paths = [p["mask_path"] for p in items]
    xs = [p["x"] for p in items]
    ys = [p["y"] for p in items]

    def _map_fn(img_p, mask_p, rx, ry):
        img_np, mask_np = tf.py_function(
            lambda ip, mp, x, y: load_and_augment_real_noisy(ip, mp, x, y, augment),
            [img_p, mask_p, rx, ry],
            Tout=(tf.float32, tf.int64),
        )
        img_np.set_shape((256, 256, 3))
        mask_np.set_shape((256, 256))
        if include_meta:
            return img_np, mask_np, img_p, rx, ry
        return img_np, mask_np

    dataset = tf.data.Dataset.from_tensor_slices((img_paths, mask_paths, xs, ys))
    dataset = dataset.map(_map_fn, num_parallel_calls=1)

    if include_meta:
        def _onehot_meta(img, mask, img_p, rx, ry):
            mask = tf.one_hot(tf.cast(mask, tf.int32), depth=3, axis=-1)
            mask = tf.reshape(mask, (256, 256, 3))
            return img, mask, img_p, rx, ry
        dataset = dataset.map(_onehot_meta, num_parallel_calls=1)
    else:
        def _onehot(img, mask):
            mask = tf.one_hot(tf.cast(mask, tf.int32), depth=3, axis=-1)
            mask = tf.reshape(mask, (256, 256, 3))
            return img, mask
        dataset = dataset.map(_onehot, num_parallel_calls=1)

    if batch_size is not None:
        dataset = dataset.shuffle(shuffle_buffer, seed=_shuffle_seed).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset


# Shared evaluation helper for torch models.
def evaluate_torch_model(model, dataloader, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for imgs, masks in dataloader:
            imgs, masks = imgs.to(device), masks.to(device)
            preds = model(imgs)
            loss = weighted_crossentropy_torch(preds, masks, device=device)
            running_loss += loss.item() * imgs.size(0)

            pred_labels = torch.argmax(preds, dim=1)
            correct += (pred_labels == masks).sum().item()
            total += masks.numel()

    return {
        "loss": float(running_loss / max(1, len(dataloader.dataset))),
        "accuracy": float(correct / max(1, total)),
    }


def train_model_on_resolutions(
    synthetic_folder,
    real_noisy_folder=TARGET_PATH,
    model_name="layered_augmentation",
    scales=(0.5, 0.75, 1.0),
    workspace="./blackbox_workspace",
    model_family="unet",
    model_weights_file="./models/seg_model.keras",
    use_pretrained=True,
    loss = "weighted_crossentropy",
    n_crop_views=N_CROP_VIEWS,
    noise_params=None,
    combine_with_clean=COMBINE_WITH_CLEAN,
    augment_clean=AUGMENT_CLEAN,
    augment_noisy=AUGMENT_NOISY,
):
    """Train on synthetic multi-res patches and evaluate on held-out real noisy images."""
    workspace = Path(workspace)
    patch_dir = reset_dir(workspace / "patches")

    # --- Synthetic: patchify clean images → group by source → split → online noise ---
    syn_image_dir, syn_mask_dir = seg.patchify_training_data(
        CLEAN_PATH, str(patch_dir / "clean_patches"),
    )
    all_syn_images = sorted(glob(syn_image_dir + "/*.png"))
    all_syn_masks = sorted(glob(syn_mask_dir + "/*.png"))

    source_groups = defaultdict(list)
    for img, msk in zip(all_syn_images, all_syn_masks):
        src_key = Path(img).stem.rsplit("_patch", 1)[0]
        source_groups[src_key].append((img, msk))

    source_keys = list(source_groups.keys())
    train_keys, val_keys = train_test_split(source_keys, test_size=0.25, random_state=42)

    def _split_patches(keys):
        items = [p for k in keys for p in source_groups[k]]
        return [p[0] for p in items], [p[1] for p in items]

    train_syn_images, train_syn_masks = _split_patches(train_keys)
    val_syn_images, val_syn_masks = _split_patches(val_keys)

    split_dir = patch_dir / "clean_patches" / "Patches"
    train_dir = split_dir / "train"
    val_dir = split_dir / "val"

    print("Creating multi-resolution synthetic training data...")
    train_syn_images, train_syn_masks = create_scaled_variants(train_syn_images, train_syn_masks, scales, train_dir)
    print("Creating multi-resolution synthetic validation data...")
    val_syn_images, val_syn_masks = create_scaled_variants(val_syn_images, val_syn_masks, scales, val_dir)

    # --- Real noisy: region-based Albumentations pipeline instead of patchify ---
    # Assign entire images to splits to guarantee zero pixel leakage.
    pairs_with_regions = []
    if augment_noisy or augment_clean:
        real_pairs = load_image_mask_pairs(real_noisy_folder)
        n_images = len(real_pairs)
        split_point = max(1, round(n_images * 0.75))
        image_assignments = {}
        for idx, (img_path, _) in enumerate(real_pairs):
            image_assignments[img_path] = "train" if idx < split_point else "val"

        for img_path, mask_path in real_pairs:
            height, width = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE).shape
            crop_positions = compute_crop_grid(height, width, crop_size=384, stride=128)
            split_label = image_assignments[img_path]
            for cx, cy in crop_positions:
                pairs_with_regions.append({
                    "img_path": img_path,
                    "mask_path": mask_path,
                    "x": cx,
                    "y": cy,
                    "split": split_label,
                })

        # Replicate region entries so each region produces n_crop_views different
        # random views (crop position, rotation, flips) per epoch.
        if n_crop_views > 1:
            pairs_with_regions = [
                {**p, "crop_view_id": i}
                for p in pairs_with_regions
                for i in range(n_crop_views)
            ]

        n_train_real = sum(1 for p in pairs_with_regions if p["split"] == "train")
        n_val_real = sum(1 for p in pairs_with_regions if p["split"] == "val")
        print(f"Real noisy: {n_train_real} train regions + {n_val_real} val regions (from {len(real_pairs)} images, view_repeat={n_crop_views})")

    # --- Clean: region-based Albumentations augmentation (no synthetic noise) ---
    clean_region_pairs = []
    if augment_clean:
        clean_pairs = load_image_mask_pairs(CLEAN_PATH)
        for img_path, mask_path in clean_pairs:
            src_key = Path(img_path).stem.replace("_image", "").replace("_mask", "").replace("image", "").replace("mask", "")
            height, width = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE).shape
            crop_positions = compute_crop_grid(height, width, crop_size=384, stride=128)
            if src_key in train_keys:
                split_label = "train"
            elif src_key in val_keys:
                split_label = "val"
            else:
                print(f"  Warning: clean image {img_path} (key={src_key}) not found in splits, skipping")
                continue
            for cx, cy in crop_positions:
                clean_region_pairs.append({
                    "img_path": img_path,
                    "mask_path": mask_path,
                    "x": cx,
                    "y": cy,
                    "split": split_label,
                })
        if n_crop_views > 1:
            clean_region_pairs = [
                {**p, "crop_view_id": i}
                for p in clean_region_pairs
                for i in range(n_crop_views)
            ]
        n_train_clean = sum(1 for p in clean_region_pairs if p["split"] == "train")
        n_val_clean = sum(1 for p in clean_region_pairs if p["split"] == "val")
        print(f"Clean augmented: {n_train_clean} train regions + {n_val_clean} val regions (from {len(clean_pairs)} images, view_repeat={n_crop_views})")

    print(f"Synthetic: {len(train_syn_images)} train, {len(val_syn_images)} val")

    if model_family in {"unet", "unet_modified"}:
        # Keras path: either start from the repo constructors or fine-tune a saved model.
        print("Using Unet")

        syn_train_ds = build_dataset(train_syn_images, train_syn_masks, augmentation=True, batch_size=None)
        syn_val_ds = build_dataset(val_syn_images, val_syn_masks, augmentation=False, batch_size=None)
    
        #Splitting augmentation logic
        rng = np.random.default_rng(42)
        perm = rng.permutation(len(clean_region_pairs))

        clean_subsets = [
            [clean_region_pairs[i] for i in subset_indices]
            for subset_indices in np.array_split(perm, 8)
        ]

        real_noisy_train_ds = build_real_noisy_dataset(pairs_with_regions, "train", augment=False, batch_size=None)
        real_noisy_val_ds = build_real_noisy_dataset(pairs_with_regions, "val", augment=False, batch_size=None)
        
        #Add synthetic noise to the real noisy data
        """if noise_params is not None:
            real_noisy_train_ds = add_synthetic_noise(real_noisy_train_ds, noise_params)"""

        # Build training streams: synthetic + (optional clean) + (optional real noisy).
        record_noise_params = [[0.07652291241426773,0.02034248150131113,0.0498800248343853,4.230016482296793,0.0874570440048911],
                               [0.0007317985679122529,0.030369003290806544,0.04082591422871262,0.9003738842091749,0.08203474496944271],
                               [0.14569068912848973,0.024456623033781676,0.0379143819252047,5.879460532352486,0.046381700718777175],
                               [0.17328192338055962,0.04885113314764954,0.038550162917255555,6.706567984131701,0.009970874388909262],
                               [0.1410890857134039,0.010530486059497041,0.0005376122902326199,5.289880188407845,0.09547597369878381],
                               [0.00020340266028383094,0.007967854538489146,0.001486517870161773,5.010807594297073,0.003083440260853602],
                               [0.15278329586406902,0.048511615197154745,0.0008300255906465081,2.5919338311650932,0.017782477982277174],
                               [0.05666339970163336,0.049457972663613015,0.02400459899113409,3.614819767927746,0.054692281754324286]]
        streams_train = [syn_train_ds,real_noisy_train_ds]
        streams_val = [syn_val_ds,real_noisy_val_ds]
        if augment_clean:
            for i in range(len(clean_subsets)):
                clean_train_ds = build_real_noisy_dataset(
                    clean_subsets[i], "train", augment=False, batch_size=None, include_meta=False,
                )
                clean_val_ds = build_real_noisy_dataset(clean_subsets[i], "val", augment=False, batch_size=None)
                """params_i = NoiseParams(*record_noise_params[i])
                score_stats_i = precompute_score_stats(
                    clean_subsets[i], params_i, n_samples=50,
                    evidence_fns=EVIDENCE_FNS,
                )
                clean_train_ds = add_synthetic_noise(
                    clean_train_ds, params_i, score_stats=score_stats_i,
                    z_lower=0, z_upper=1, evidence_fns=EVIDENCE_FNS,
                )"""
                streams_train.append(clean_train_ds)
                streams_val.append(clean_val_ds)

        train_dataset = streams_train[0]
        for ds in streams_train[1:]:
            train_dataset = train_dataset.concatenate(ds)
        train_dataset = train_dataset.shuffle(2000, seed=_shuffle_seed).batch(32).prefetch(tf.data.AUTOTUNE)

        val_dataset = streams_val[0]
        for ds in streams_val[1:]:
            val_dataset = val_dataset.concatenate(ds)
        val_dataset = val_dataset.shuffle(1000, seed=_shuffle_seed).batch(32).prefetch(tf.data.AUTOTUNE)

        if use_pretrained:
            print("Using pretrained")
            if model_weights_file is None:
                raise ValueError("model_weights_file is required when use_pretrained=True.")
            model = seg.create_and_train_model_from_pretrained(
                model_weights_file,
                train_dataset,
                val_dataset,
                test_dataset=None,
                epochs=100,
                learning_rate=1e-2,
                model_type=model_family,
                save_plot_path=f"loss_plots/training_loss_plot_{model_name}.png",
                show_plot=False,
                use_reduce_lr=True,
                loss = loss
            )
        else:
            if model_family == "unet_modified":
                model = seg.UnetModified()
            else:
                model = seg.Unet()
            model.compile(
                optimizer=Adam(learning_rate=1e-2),
                loss=seg.weighted_crossentropy,
                metrics=["accuracy"],
            )
            model.fit(train_dataset, epochs=80, validation_data=val_dataset)

        model_path = Path("models") / f"{model_name}.keras"
        model.save(model_path)

        val_metrics = model.evaluate(val_dataset, verbose=0, return_dict=True)
    elif model_family == "resnext":
        # Torch path: synthetic-only (real noisy augmentation is TF-specific).
        device = torch.device(
            "mps" if torch.backends.mps.is_available()
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        train_loader = DataLoader(PatchDataset(train_syn_images, train_syn_masks, augment=True), batch_size=8, shuffle=True, num_workers=0)
        val_loader = DataLoader(PatchDataset(val_syn_images, val_syn_masks, augment=False), batch_size=8, shuffle=False, num_workers=0)

        model = MaskingResNeXt(num_classes=3, pretrained=use_pretrained).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
        best_model_state = None
        best_val_loss = float("inf")

        for epoch in range(20):
            model.train()
            running_loss = 0.0
            for imgs, masks in train_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                optimizer.zero_grad()
                preds = model(imgs)
                loss = weighted_crossentropy_torch(preds, masks, device=device)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * imgs.size(0)

            val_metrics = evaluate_torch_model(model, val_loader, device)
            scheduler.step(val_metrics["loss"])
            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                best_model_state = copy.deepcopy(model.state_dict())

        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        model_path = Path("models") / f"{model_name}.pth"
        torch.save(model.state_dict(), model_path)

        val_metrics = evaluate_torch_model(model, val_loader, device)
    else:
        raise ValueError(
            f"Unsupported model_family '{model_family}'. "
            "Choose from 'unet', 'unet_modified', or 'resnext'."
        )
    metrics = {
        "val_loss": float(val_metrics["loss"]),
        "val_accuracy": float(val_metrics["accuracy"]),
        "val_dice_loss": float(val_metrics["dice_loss"]),
        "model_path": str(model_path),
    }
    return model, metrics

def dice_loss(predicted_probs, true_masks, eps=1e-7):
    """Multi-class Dice loss averaged over images. Lower = more similar."""
    total = 0.0
    for pred, true in zip(predicted_probs, true_masks):
        h, w = min(pred.shape[0], true.shape[0]), min(pred.shape[1], true.shape[1])
        pred, true = pred[:h, :w], true[:h, :w]
        one_hot = np.eye(pred.shape[-1])[true]
        intersection = np.sum(pred * one_hot, axis=(0, 1))
        union = np.sum(pred + one_hot, axis=(0, 1))
        total += float(1 - np.mean((2 * intersection + eps) / (union + eps)))
    return total / max(1, len(predicted_probs))


def count_penalty(predicted_probs, true_masks):
    """Normalized absolute difference in number of grain instances. Lower = more similar."""
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
    """Composite loss: Dice (per-pixel overlap) + count penalty (grain consistency)."""
    dice = dice_loss(predicted_probs, true_masks)
    count = count_penalty(predicted_probs, true_masks)
    return dice_weight * dice + count_weight * count


def evaluate_model_masks(model, image_dir, patch_dir, model_family, device=None, tile_size=256):
    """Run model inference on images and return per-image predictions + true masks."""
    pairs = load_image_mask_pairs(image_dir)
    pair_dir = stage_pairs(pairs, Path(patch_dir) / "staged")
    staged_pairs = load_image_mask_pairs(str(pair_dir))

    if model_family in {"unet", "unet_modified"}:
        pred_probs_list = []
        true_masks_list = []
        for img_path, mask_path in staged_pairs:
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pred = seg.predict_image_mirror(img, model, tile_size)
            pred_probs_list.append(pred)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            true_masks_list.append(mask)
        return pred_probs_list, true_masks_list

    elif model_family == "resnext":
        img_dir, mask_dir = seg.patchify_training_data(f"{pair_dir}/", Path(patch_dir) / "patched")
        image_files = sorted(glob(img_dir + "/*.png"))
        mask_files = sorted(glob(mask_dir + "/*.png"))

        if device is None:
            device = torch.device(
                "mps" if torch.backends.mps.is_available()
                else ("cuda" if torch.cuda.is_available() else "cpu")
            )
        loader = DataLoader(
            PatchDataset(image_files, mask_files, augment=False),
            batch_size=8, shuffle=False, num_workers=0,
        )
        model.eval()
        all_preds = []
        with torch.no_grad():
            for imgs, _ in loader:
                imgs = imgs.to(device)
                preds = torch.softmax(model(imgs), dim=1).cpu().numpy()
                all_preds.append(preds)
        pred_probs = np.concatenate(all_preds, axis=0)
        pred_probs = np.transpose(pred_probs, (0, 2, 3, 1))

        true_masks = []
        for mp in mask_files:
            mask = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
            true_masks.append(mask)
        return [pred_probs[i] for i in range(len(pred_probs))], true_masks
    else:
        raise ValueError(f"Unsupported model_family '{model_family}'")


def black_box(
    theta,
    model_family="unet",
    model_weights_file=None,
    use_pretrained=True,
    dice_weight=1.0,
    count_weight=0.4,
    tag=None,
    combine_with_clean=COMBINE_WITH_CLEAN,
    augment_clean=AUGMENT_CLEAN,
    augment_noisy=AUGMENT_NOISY,
    n_synthetic_variants=N_SYNTHETIC_VARIANTS,
    n_crop_views=N_CROP_VIEWS,
    master_seed=None,
):
    """Evaluate one candidate theta and return a single scalar score."""
    if master_seed is not None:
        _derive_seeds(master_seed)
    global _noise_master_rng
    _noise_master_rng = np.random.default_rng(_noise_seed)
    _rng_cache.clear()

    theta = np.asarray(theta, dtype=float)
    theta_tag = "_".join(f"{value:.4g}" for value in theta)
    tag = tag if tag is not None else theta_tag
    workspace = reset_dir("./blackbox_workspace")

    # 1) Generate synthetic noisy image/mask pairs from the candidate theta.
    print(f"Running black box for theta={theta_tag} using model_family={model_family}")
    synthetic_folder = generate_synthetic_images_from_script(
        theta,
        input_folder=CLEAN_PATH,
        noise_reference_folder=TARGET_PATH,
        output_folder=workspace / "synthetic_noisy_images",
        seed=42,
        n=n_synthetic_variants,
    )

    theta_params = NoiseParams(
        a=float(theta[0]), b=float(theta[1]),
        sigma_r=float(theta[2]), l=float(theta[3]), k=float(theta[4]),
    )

    # 2) Train the chosen model family on synthetic patches and evaluate on real data.
    model, metrics = train_model_on_resolutions(
        synthetic_folder = synthetic_folder,
        real_noisy_folder=TARGET_PATH,
        model_name=f"{tag}",
        workspace=workspace,
        model_family=model_family,
        model_weights_file=model_weights_file,
        use_pretrained=use_pretrained,
        n_crop_views=n_crop_views,
        noise_params=theta_params,
        combine_with_clean=combine_with_clean,
        augment_clean=augment_clean,
        augment_noisy=augment_noisy,
    )

    # 3) Predict masks on PREDICT_PATH images and compare with ground truth masks.
    pred_probs, true_masks = evaluate_model_masks(
        model,
        image_dir=PREDICT_PATH,
        patch_dir=workspace / "eval_patches",
        model_family=model_family,
    )
    #mask_loss = compute_mask_loss(pred_probs, true_masks, dice_weight, count_weight)

    # 4) Final score: composite mask prediction loss.
    objective_value = float(metrics["val_loss"])

    summary = {
        "theta": theta.tolist(),
        "model_family": model_family,
        "model_weights_file": model_weights_file,
        "use_pretrained": use_pretrained,
        "objective": objective_value,
        "dice_loss": dice_loss(pred_probs, true_masks),
        "count_penalty": count_penalty(pred_probs, true_masks),
        "dice_weight": dice_weight,
        "count_weight": count_weight,
        "metrics": metrics,
    }
    print(
        "Black-box metrics: "
        f"objective={objective_value:.6f}, "
        f"val_loss={metrics['val_loss']:.6f}, "
        f"dice={dice_loss(pred_probs, true_masks):.6f}, "
        f"count_pen={count_penalty(pred_probs, true_masks):.6f}, "
        f"val_accuracy={metrics['val_accuracy']:.6f}"
    )
    return summary
