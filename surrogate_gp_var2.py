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
import re
from collections import defaultdict
from pathlib import Path
import segmenteverygrain as seg
from create_synthetic_images import (
    generate_synthetic_images as generate_synthetic_images_from_script,
    load_image_mask_pairs,
)
from synthetic_noise import NoiseParams, synthetic_noise_model_input
import tensorflow as tf
from sklearn.model_selection import train_test_split
from glob import glob
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scipy.stats import qmc

import albumentations as A
import json
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from keras.optimizers import Adam
from segmenteverygrain.resnext_model import MaskingResNeXt, weighted_crossentropy_torch


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
def add_synthetic_noise(dataset, noise_params):
    """Wrap a TF dataset so synthetic noise is applied to each image."""
    def _apply_noise(img, mask):
        gray = img[..., 0]
        noisy = tf.py_function(
            lambda x: synthetic_noise_model_input(x, noise_params, np.random.default_rng()),
            [gray],
            tf.float32,
        )
        noisy.set_shape((256, 256))
        noisy = tf.stack([noisy, noisy, noisy], axis=-1)
        return noisy, mask
    return dataset.map(_apply_noise, num_parallel_calls=tf.data.AUTOTUNE)


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

    dataset = dataset.map(seg.load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    if batch_size is not None:
        dataset = dataset.shuffle(shuffle_buffer).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset


# TensorFlow dataset builder for region-based real noisy augmentation.
def build_real_noisy_dataset(pairs, split, augment=True, batch_size=32, shuffle_buffer=1000):
    """Build a TF dataset from region-assigned real noisy image pairs.

    *pairs* is a list of dicts with keys: img_path, mask_path, x, y, split.
    Filters to *split*, then applies load_and_augment_real_noisy via py_function.

    If *batch_size* is None, returns an unbatched (element-level) dataset
    suitable for concatenation before batching.
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
        return img_np, mask_np

    dataset = tf.data.Dataset.from_tensor_slices((img_paths, mask_paths, xs, ys))
    dataset = dataset.map(_map_fn, num_parallel_calls=tf.data.AUTOTUNE)

    # One-hot encode the mask to depth 3 (background / grain / boundary).
    def _onehot(img, mask):
        mask = tf.one_hot(tf.cast(mask, tf.int32), depth=3, axis=-1)
        mask = tf.reshape(mask, (256, 256, 3))
        return img, mask

    dataset = dataset.map(_onehot, num_parallel_calls=tf.data.AUTOTUNE)

    if batch_size is not None:
        dataset = dataset.shuffle(shuffle_buffer).batch(batch_size).prefetch(tf.data.AUTOTUNE)
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
    model_name="synthetic_blackbox_var2",
    scales=(0.5, 0.75, 1.0),
    workspace="./blackbox_workspace",
    model_family="unet",
    model_weights_file="./models/seg_model.keras",
    use_pretrained=True,
    loss = "weighted_crossentropy",
    n_crop_views=8,
    noise_params=None,
    combine_with_clean=False,
    augment_clean=False,
    augment_noisy=True,
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
                for i in range(16)
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
    
        rng = np.random.default_rng(42)
        perm = rng.permutation(len(pairs_with_regions))
        half = len(perm) // 2

        real_pairs_subset = [pairs_with_regions[i] for i in perm[:half]]
        layer_pairs_subset = [pairs_with_regions[i] for i in perm[half:]]

        real_train_ds = build_real_noisy_dataset(real_pairs_subset, "train", augment=augment_clean, batch_size=None)
        real_val_ds = build_real_noisy_dataset(real_pairs_subset, "val", augment=False, batch_size=None)
        layer_train_ds = build_real_noisy_dataset(layer_pairs_subset, "train", augment=augment_clean, batch_size=None)
        layer_val_ds = build_real_noisy_dataset(layer_pairs_subset, "val", augment=False, batch_size=None)

        if noise_params is not None:
            layer_train_ds = add_synthetic_noise(layer_train_ds, noise_params)

        # Build training streams: synthetic + (optional clean) + (optional real noisy).
        streams_train = [syn_train_ds,real_train_ds,layer_train_ds]
        streams_val = [syn_val_ds,real_val_ds,layer_val_ds]
        if augment_clean:
            clean_train_ds = build_real_noisy_dataset(clean_region_pairs, "train", augment=True, batch_size=None)
            clean_val_ds = build_real_noisy_dataset(clean_region_pairs, "val", augment=False, batch_size=None)
            streams_train.append(clean_train_ds)
            streams_val.append(clean_val_ds)

        train_dataset = streams_train[0]
        for ds in streams_train[1:]:
            train_dataset = train_dataset.concatenate(ds)
        train_dataset = train_dataset.shuffle(2000).batch(32).prefetch(tf.data.AUTOTUNE)

        val_dataset = streams_val[0]
        for ds in streams_val[1:]:
            val_dataset = val_dataset.concatenate(ds)
        val_dataset = val_dataset.shuffle(1000).batch(32).prefetch(tf.data.AUTOTUNE)

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
    combine_with_clean=True,
    augment_clean=True,
    augment_noisy=True,
    n_synthetic_variants=8,
    n_crop_views=8,
):
    """Evaluate one candidate theta and return a single scalar score."""
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
        model_name=f"synthetic_blackbox_var2_{tag}",
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
    mask_loss = compute_mask_loss(pred_probs, true_masks, dice_weight, count_weight)

    # 4) Final score: composite mask prediction loss.
    objective_value = float(mask_loss)

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


# Parameter bounds for theta = [a, b, sigma_r, l, k] from synthetic_noise.py differential_evolution bounds
bounds = np.array([
    [1e-6, 0.2],    # a
    [1e-6, 0.05],   # b
    [1e-6, 0.05],   # sigma_r
    [0.3, 8.0],     # l
    [1e-6, 0.1],    # k
])
n_dim = bounds.shape[0]
lb, ub = bounds[:, 0], bounds[:, 1]

DATA_PATH = "gp_data_var2.json"
MAX_KEPT_MODELS = 5


def prune_old_models(records, keep_last=MAX_KEPT_MODELS):
    if len(records) <= keep_last:
        return
    all_paths = [Path(r["metrics"]["model_path"]) for r in records if r.get("metrics", {}).get("model_path")]
    if len(all_paths) <= keep_last:
        return
    for p in all_paths[:-keep_last]:
        if p.exists():
            print(f"  Pruning old model: {p}")
            p.unlink()

def load_gp_data(path):
    p = Path(path)
    if p.exists():
        with open(p) as f:
            data = json.load(f)
        if "records" in data:
            records = data["records"]
            X = np.array([r["theta"] for r in records])
            y = np.array([r["objective"] for r in records])
            print(f"Loaded {len(records)} records from {path}")
            return records, X, y
        else:
            # Backward compat: old format with just X, y keys
            X = np.array(data["X"])
            y = np.array(data["y"])
            print(f"Loaded {X.shape[0]} previous data points from {path} (old format)")
            return None, X, y
    return None, None, None

def save_gp_data(path, records):
    with open(path, "w") as f:
        json.dump({"records": records}, f, indent=2)
    prune_old_models(records)

def suggest_next(X_scaled, y, n_test=500, beta=1.96):
    """Fit a GP surrogate and propose the next theta to evaluate."""
    gp = GaussianProcessRegressor(kernel=C(1.0) * RBF(length_scale=np.ones(n_dim)), n_restarts_optimizer=10)
    gp.fit(X_scaled, y)

    sampler = qmc.LatinHypercube(d=n_dim, seed=42)
    X_test_unit = sampler.random(n=n_test)
    X_test = qmc.scale(X_test_unit, lb, ub)
    X_test_scaled = (X_test - lb) / (ub - lb)

    y_pred, sigma = gp.predict(X_test_scaled, return_std=True)
    # Lower objective is better, so use a lower-confidence-bound style acquisition.
    acquisition = y_pred - beta * sigma
    best_idx = np.argmin(acquisition)
    return gp, X_test[best_idx], y_pred[best_idx], sigma[best_idx]

def run_gp_loop(
    n_iterations,
    initial_theta=None,
    data_path=DATA_PATH,
    n_test=500,
    beta=1.96,
    model_family="unet",
    model_weights_file="./models/seg_model.keras",
    use_pretrained=True,
    combine_with_clean=False,
    augment_clean=True,
    augment_noisy=True,
    n_synthetic_variants=8,
    n_crop_views=8,
):
    """Template Bayesian optimization loop around the expensive black box."""
    records, X_prev, y_prev = load_gp_data(data_path)

    if X_prev is not None and len(X_prev) > 0:
        records = records if records is not None else []
        X_train = X_prev.copy()
        y_train = y_prev.copy()
        print(f"Continuing with {len(X_train)} existing data points")
    else:
        if initial_theta is None:
            theta_initial = np.array([0.0023589515117326183, 0.001712502743955444, 0.0006997093027690107, 0.7603779994083678, 0.07404317063233228])
        else:
            theta_initial = np.array(initial_theta)
        print(f"Evaluating initial theta: {theta_initial}")
        X_train = theta_initial.reshape(1, -1)
        summary = black_box(
            theta_initial,
            model_family=model_family,
            model_weights_file=model_weights_file,
            use_pretrained=use_pretrained,
            tag="init",
            combine_with_clean=combine_with_clean,
            augment_clean=augment_clean,
            augment_noisy=augment_noisy,
            n_synthetic_variants=n_synthetic_variants,
            n_crop_views=n_crop_views,
        )
        y_train = np.array([summary["objective"]])
        records = [{"iteration": 0, "tag": "init", **summary}]
        save_gp_data(data_path, records)

    for i in range(n_iterations):
        iter_num = len(records)
        iter_tag = f"iter_{iter_num:03d}"
        print(f"\n--- Iteration {i+1}/{n_iterations} (total #{iter_num}) ---")
        print(f"Current dataset size: {len(X_train)}")

        X_scaled = (X_train - lb) / (ub - lb)
        gp, theta_next, pred_mean, pred_std = suggest_next(X_scaled, y_train, n_test=n_test, beta=beta)

        print(f"GP suggests theta: {theta_next}")
        print(f"GP prediction: mean={pred_mean:.4f}, std={pred_std:.4f}")

        print("Evaluating black_box(theta_next)...")
        summary = black_box(
            theta_next,
            model_family=model_family,
            model_weights_file=model_weights_file,
            use_pretrained=use_pretrained,
            tag=iter_tag,
            combine_with_clean=combine_with_clean,
            augment_clean=augment_clean,
            augment_noisy=augment_noisy,
            n_synthetic_variants=n_synthetic_variants,
            n_crop_views=n_crop_views,
        )
        print(f"Result: f(theta) = {summary['objective']:.6f}")

        X_train = np.vstack([X_train, theta_next])
        y_train = np.append(y_train, summary["objective"])
        records.append({"iteration": iter_num, "tag": iter_tag, **summary})
        save_gp_data(data_path, records)

        best_idx = np.argmin(y_train)
        print(f"Best theta so far (iter {best_idx}): f={y_train[best_idx]:.4f}")
        print(f"Data saved to {data_path}")

    return X_train, y_train

if __name__ == "__main__":
    N_ITERATIONS = 100  # Change this to control how many searches to run
    COMBINE_WITH_CLEAN = True  # Set True to include pre-injection clean images in training
    AUGMENT_CLEAN = True  # Set True to apply Albumentations spatial augmentation to real clean images
    AUGMENT_NOISY = True  # Set True to apply Albumentations spatial augmentation to real noisy images
    N_SYNTHETIC_VARIANTS = 8  # Number of noisy variants per clean image
    N_CROP_VIEWS = 8  # Replicate each real-noisy crop region for more views per epoch

    X_final, y_final = run_gp_loop(
        n_iterations=N_ITERATIONS,
        combine_with_clean=COMBINE_WITH_CLEAN,
        augment_clean=AUGMENT_CLEAN,
        augment_noisy=AUGMENT_NOISY,
        n_synthetic_variants=N_SYNTHETIC_VARIANTS,
        n_crop_views=N_CROP_VIEWS,
    )

    print("\n=== Final Results ===")
    for i in range(len(X_final)):
        print(f"  [{i}] a={X_final[i,0]:.6f}, b={X_final[i,1]:.6f}, sigma_r={X_final[i,2]:.6f}, l={X_final[i,3]:.4f}, k={X_final[i,4]:.4f} -> f={y_final[i]:.4f}")
    best = np.argmin(y_final)
    print(f"\nBest: {X_final[best]} with f={y_final[best]:.4f}")

