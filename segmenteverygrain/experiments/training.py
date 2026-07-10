"""Train/fine-tune U-Net models from staged experiment datasets."""

from __future__ import annotations

from pathlib import Path
import shutil

import tensorflow as tf
from keras.optimizers import Adam

import segmenteverygrain as seg

from .dataset_builder import build_training_set
from .io import ensure_dir, write_json


def train_unet_from_staged_pairs(
    *,
    staged_dir: str | Path,
    run_dir: str | Path,
    output_model: str | Path | None = None,
    pretrained_model: str | Path | None = "models/seg_model.keras",
    model_type: str = "unet",
    augmentation: bool = True,
    epochs: int = 50,
    learning_rate: float = 1e-4,
    use_reduce_lr: bool = True,
    patch_dir: str | Path | None = None,
    save_plot_path: str | Path | None = None,
) -> dict:
    """Patchify staged pairs, train/fine-tune a U-Net, and save outputs."""

    # At this point the data mixture is already decided; this function focuses only on
    # converting full images into patches and sending them into the U-Net trainer.
    run_dir = ensure_dir(run_dir)
    patch_dir = Path(patch_dir) if patch_dir else run_dir / "patches"
    if patch_dir.exists():
        shutil.rmtree(patch_dir)
    patch_dir.mkdir(parents=True, exist_ok=True)

    staged_dir_str = str(staged_dir)
    if not staged_dir_str.endswith("/"):
        staged_dir_str += "/"
    # The existing SEG training utilities expect patch folders, so we reuse them instead
    # of rewriting the low-level patching and dataset split logic.
    image_dir, mask_dir = seg.patchify_training_data(staged_dir_str, str(patch_dir))
    train_dataset, val_dataset, test_dataset = seg.create_train_val_test_data(
        image_dir,
        mask_dir,
        augmentation=augmentation,
    )

    save_plot_path = Path(save_plot_path) if save_plot_path else run_dir / "training_loss.png"
    output_model = Path(output_model) if output_model else run_dir / "model.keras"
    output_model.parent.mkdir(parents=True, exist_ok=True)
    save_plot_path.parent.mkdir(parents=True, exist_ok=True)

    if pretrained_model:
        # Fine-tuning starts from the existing segmentation model, which should need less
        # data than training a U-Net completely from scratch.
        model = seg.create_and_train_model_from_pretrained(
            str(pretrained_model),
            train_dataset,
            val_dataset,
            test_dataset,
            epochs=epochs,
            learning_rate=learning_rate,
            model_type=model_type,
            save_plot_path=str(save_plot_path),
            show_plot=False,
            use_reduce_lr=use_reduce_lr,
        )
    else:
        # This fallback is useful for ablations where we want to see how the architecture
        # behaves without borrowing weights from a pretrained model.
        model = seg.UnetModified() if model_type == "unet_modified" else seg.Unet()
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss=seg.weighted_crossentropy,
            metrics=["accuracy"],
        )
        model.fit(train_dataset, epochs=epochs, validation_data=val_dataset)
        model.evaluate(test_dataset, verbose=0)

    model.save(output_model)
    test_metrics = model.evaluate(test_dataset, verbose=0, return_dict=True)
    summary = {
        "staged_dir": str(staged_dir),
        "run_dir": str(run_dir),
        "patch_dir": str(patch_dir),
        "image_patch_dir": str(image_dir),
        "mask_patch_dir": str(mask_dir),
        "output_model": str(output_model),
        "pretrained_model": str(pretrained_model) if pretrained_model else None,
        "model_type": model_type,
        "augmentation": augmentation,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "use_reduce_lr": use_reduce_lr,
        "training_plot": str(save_plot_path),
        "test_metrics": {k: float(v) for k, v in test_metrics.items()},
    }
    write_json(run_dir / "training_summary.json", summary)
    return summary


def train_unet_experiment(
    *,
    run_dir: str | Path,
    clean_dir: str | Path | None = None,
    synthetic_dir: str | Path | None = None,
    real_noisy_dir: str | Path | None = None,
    output_model: str | Path | None = None,
    pretrained_model: str | Path | None = "models/seg_model.keras",
    model_type: str = "unet",
    augmentation: bool = True,
    epochs: int = 50,
    learning_rate: float = 1e-4,
    use_reduce_lr: bool = True,
) -> dict:
    """Build a staged training set, then train/fine-tune a U-Net."""

    # This combines the dataset-building step and the training step into one experiment.
    # Config files call this when they are ready to train a model from selected sources.
    run_dir = ensure_dir(run_dir)
    training_dir = run_dir / "training_pairs"
    manifest = build_training_set(
        output_dir=training_dir,
        clean_dir=clean_dir,
        synthetic_dir=synthetic_dir,
        real_noisy_dir=real_noisy_dir,
    )
    training = train_unet_from_staged_pairs(
        staged_dir=training_dir,
        run_dir=run_dir,
        output_model=output_model,
        pretrained_model=pretrained_model,
        model_type=model_type,
        augmentation=augmentation,
        epochs=epochs,
        learning_rate=learning_rate,
        use_reduce_lr=use_reduce_lr,
    )
    summary = {"dataset": manifest, "training": training}
    write_json(run_dir / "experiment_summary.json", summary)
    return summary


def load_keras_unet(model_path: str | Path):
    """Load a saved U-Net model for inference."""

    # The custom loss must be supplied when loading older saved models that reference it.
    # compile=False keeps loading focused on inference instead of resuming training.
    return tf.keras.models.load_model(
        model_path,
        custom_objects={"weighted_crossentropy": seg.weighted_crossentropy},
        compile=False,
    )
