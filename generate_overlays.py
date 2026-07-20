"""
Run inference on segmentation models and generate overlay visualizations.

For each image a grid figure is produced (auto-sized rows × cols):
    [Input Image] [Ground Truth] [Model 1 Overlay] [Model 2 Overlay] ...
and for each (image, model) pair a solo overlay figure.

This is the visualisation half of model_comparison.py — no metrics, no DataFrames.
"""

import numpy as np
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from surrogate_gp import evaluate_model_masks
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

MODELS = [
    ("CleanPatchesWithoutAugmentation", "./models/clean_blackbox.keras", "unet"),
    ("CleanPatchesWithoutAugmentationFullInfo", "./models/clean_only_model_noaugmentation.keras", "unet"),
    ("ClassicalAugmentationOnClean", "./clean_only_model.keras", "unet"),
    ("SyntheticValOnlyNoise0","./synthetic_blackbox_val_loss_SyntheticAugBOMinValLossOnlyNoise_0.keras","unet"),
    ("SyntheticValOnlyNoise4","./synthetic_blackbox_val_loss_SyntheticAugBOMinValLossOnlyNoise_4.keras","unet"),
]

MODELS = [("MultiresolutionBaseline_same_image_balance", "./NoPatchMultiresolutionBaselineT1.keras", "unet"),
          #("classic_aug_noisy_and_synthetic_aug_clean_HiAcc", "./synthetic_blackbox_val_loss_SyntheticAugBOMinValLossOnlyNoise_2.keras", "unet"),
          #("classic_aug_noisy_and_synthetic_aug_clean_HiDice", "./synthetic_blackbox_val_loss_SyntheticAugBOMinValLossOnlyNoise_7.keras", "unet"),
          #("no_classic_aug_noisy_and_synthetic_aug_clean_HiAcc", "./", "unet"),
          #("check","./layered_augmentation_SyntheticAugBOMinValLossOnlyNoiseNoRealNoisyAug_2.keras","unet"),
          #("cHigher","./layered_augmentation_SyntheticAugBOMinValLossOnlyNoiseNoRealNoisyAug_1.keras","unet"),
          ("classic_aug_noisy_and_classic_aug_clean", "./clean_only_model.keras", "unet"), #train_clean_only.py and surrogate_gp.py
          ("no_classic_aug_noisy_and_classic_aug_clean", "./NoRealNoisyAugClassicalOnClean.keras", "unet"),
          #("classic_aug_noisy_and_synthetic_plus_classic_aug_clean_HiAcc", "./synthetic_blackbox_val_loss_LayeredAugBOMinValLossOnlyNoise_8.keras", "unet"), #layered_augmentation_model_training.py
          ("no_classic_aug_noisy_and_synthetic_plus_classic_aug_clean_HiAcc", "./layered_augmentation_LayeredAugBOMinValLossOnlyNoiseNoRealNoisyAug_1.keras", "unet"),
]

MODELS = [("MultiresolutionBaseline_same_image_balance", "./NoPatchMultiresolutionBaselineT1.keras", "unet"), # Baseline
          ("no_classic_aug_noisy_and_synthetic_aug_clean_HiDice", "./SyntheticAugBOMinValLossOnlyNoiseNoRealNoisyAug_6.keras", "unet"),
          #("no_classic_aug_noisy_and_synthetic_aug_clean_HiDiceT2", "./SyntheticAugBOMinValLossOnlyNoiseNoRealNoisyAugT2_0.keras", "unet"),
          #("no_classic_aug_noisy_and_synthetic_aug_clean_LowValLossT2", "./SyntheticAugBOMinValLossOnlyNoiseNoRealNoisyAugT2_1.keras", "unet"),
          ("multipleNoiseT1ValLoss","./multipleNoise_3.keras","unet"),
          ("multipleNoiseT1ValAcc","./multipleNoise_1.keras","unet"),
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
    ("MultiresolutionBaseline_same_image_balance", "./NoPatchMultiresolutionBaselineT1.keras", "unet")]+[(f"multipleNoise{i}",f"./multipleNoise_{i}.keras","unet") for i in range (10)
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

IMAGE_DIRS = ["./withheldNoisyTestImages/confident_style(model_style)/"]
PATCH_DIR = "./overlay_workspace"
OUTPUT_DIR = "./overlaysOnlyNoiseValidationSyntheticSeg3"
SHOW_PLOTS = True

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
# Per-directory: run inference + generate overlays
# ═══════════════════════════════════════════════════════════════════

def process_directory(image_dir, output_dir, show=SHOW_PLOTS):
    """Load all models, run inference on image_dir, and generate overlays."""
    image_dir = Path(image_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dir_tag = image_dir.name

    print(f"\n{'='*60}")
    print(f"Processing directory: {image_dir}")
    print(f"{'='*60}")

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

    if not show:
        return

    cmap = ListedColormap(['black', 'steelblue', 'orange'])
    pairs = load_image_mask_pairs(str(image_dir))

    for i, (img_path, mask_path) in enumerate(pairs):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        true = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        img_stem = Path(img_path).stem

        # ── Grid figure: auto-sized rows × cols ────────────────────
        n_models = len(MODELS)
        total_panels = 2 + n_models
        ncols = int(np.ceil(np.sqrt(total_panels)))
        nrows = int(np.ceil(total_panels / ncols))

        fig, grid_axes = plt.subplots(
            nrows, ncols, figsize=(6 * ncols, 5 * nrows),
        )
        grid_axes = np.atleast_1d(grid_axes).ravel()

        for idx in range(nrows * ncols):
            ax = grid_axes[idx]
            if idx == 0:
                ax.imshow(img)
                ax.set_title("Input image")
            elif idx == 1:
                ax.imshow(true, cmap=cmap, vmin=0, vmax=2)
                ax.set_title("Ground truth")
            elif idx - 2 < n_models:
                model_name = MODELS[idx - 2][0]
                pred_probs = all_predictions[model_name][0][i]
                pred_label = np.argmax(pred_probs, axis=-1).astype(np.uint8)
                ax.imshow(img)
                ax.imshow(pred_label, cmap=cmap, vmin=0, vmax=2, alpha=0.4)
                ax.set_title(f"Overlay — {model_name}")
            else:
                ax.axis("off")
                continue
            ax.axis("off")

        plt.suptitle(f"{dir_tag} — Image {i}")
        plt.tight_layout()
        overview_path = output_dir / f"{img_stem}_overview.png"
        plt.savefig(overview_path, dpi=150)
        print(f"  Saved {overview_path}")
        plt.show()

        # ── Solo overlay for each model ───────────────────────────
        for j, (model_name, _, _) in enumerate(MODELS):
            pred_probs = all_predictions[model_name][0][i]
            pred_label = np.argmax(pred_probs, axis=-1).astype(np.uint8)
            fig2, ax2 = plt.subplots(1, 1, figsize=(6, 5))
            ax2.imshow(img)
            ax2.imshow(pred_label, cmap=cmap, vmin=0, vmax=2, alpha=0.4)
            ax2.set_title(f"{model_name} overlay")
            ax2.axis("off")
            plt.tight_layout()
            solo_path = output_dir / f"{img_stem}_{model_name}_overlay.png"
            plt.savefig(solo_path, dpi=150)
            print(f"  Saved {solo_path}")
            plt.close(fig2)

    print(f"  → Overlays saved to {output_dir}/")


# ═══════════════════════════════════════════════════════════════════
# Run for every directory
# ═══════════════════════════════════════════════════════════════════

for img_dir in IMAGE_DIRS:
    process_directory(img_dir, Path(OUTPUT_DIR) / Path(img_dir).name, show=SHOW_PLOTS)
