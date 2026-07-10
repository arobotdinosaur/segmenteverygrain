"""File and manifest helpers for controlled training experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import shutil
from pathlib import Path
from typing import Iterable, Sequence


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


@dataclass(frozen=True)
class ImageMaskPair:
    """One image/mask pair plus its data source label."""

    # This keeps an image, its matching mask, and where it came from bundled together.
    # Later stages use this source label to know if a pair was clean, synthetic, or real noisy.
    image: str
    mask: str
    source: str
    base_name: str


def ensure_dir(path: str | Path, *, reset: bool = False) -> Path:
    """Create a directory, optionally deleting the previous contents first."""

    path = Path(path)
    if reset and path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_json(path: str | Path):
    with open(path) as f:
        return json.load(f)


def write_json(path: str | Path, data) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    return path


def normalize_optional_dir(path: str | Path | None) -> Path | None:
    """Return a Path for a meaningful directory argument, otherwise None."""

    if path is None:
        return None
    path = Path(path)
    if str(path).strip() in {"", "None", "none", "null"}:
        return None
    return path


def _base_name_for_image(stem: str) -> str:
    lowered = stem.lower()
    for suffix in ("_image", "-image", "image"):
        if lowered.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def _base_name_for_mask(stem: str) -> str:
    lowered = stem.lower()
    for suffix in ("_mask", "-mask", "mask"):
        if lowered.endswith(suffix):
            return stem[: -len(suffix)]
    return stem.replace("_mask", "").replace("mask", "")


def load_image_mask_pairs(
    folder: str | Path | None,
    *,
    source: str = "unknown",
    require_pairs: bool = False,
) -> list[ImageMaskPair]:
    """Load paired image/mask files from a folder.

    The repository convention is:

    - image files include ``image`` in the filename or simply omit ``mask``;
    - mask files include ``mask`` in the filename;
    - pairs match after stripping the ``_image`` / ``_mask`` suffixes.

    Empty or missing folders are allowed by default so experiment configs can
    explicitly leave a data source out.
    """

    # Empty folders are okay here because some experiment recipes intentionally skip
    # clean, synthetic, or real-noisy inputs.
    folder = normalize_optional_dir(folder)
    if folder is None:
        return []
    if not folder.exists():
        if require_pairs:
            raise FileNotFoundError(f"{source} directory does not exist: {folder}")
        return []

    images: dict[str, Path] = {}
    masks: dict[str, Path] = {}
    # We first separate files into image and mask buckets, then match them by base name.
    # For example, sample_image.tif and sample_mask.tif become one ImageMaskPair.
    for path in sorted(folder.iterdir()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        stem = path.stem
        if "mask" in stem.lower():
            masks[_base_name_for_mask(stem)] = path
        else:
            images[_base_name_for_image(stem)] = path

    pairs = [
        ImageMaskPair(
            image=str(images[base]),
            mask=str(masks[base]),
            source=source,
            base_name=base,
        )
        for base in sorted(images)
        if base in masks
    ]
    if require_pairs and not pairs:
        raise ValueError(f"No paired image/mask files found in {folder}")
    return pairs


def copy_pairs_to_folder(
    pairs: Sequence[ImageMaskPair],
    output_dir: str | Path,
    *,
    reset: bool = True,
) -> tuple[Path, list[dict]]:
    """Copy pairs into one folder with stable source-prefixed names."""

    output_dir = ensure_dir(output_dir, reset=reset)
    manifest = []
    counters: dict[str, int] = {}

    # Staging gives every experiment a simple folder layout, even when the inputs
    # originally came from several different directories.
    for pair in pairs:
        source = pair.source or "unknown"
        counters[source] = counters.get(source, 0) + 1
        idx = counters[source]
        prefix = f"{source}_{idx:04d}_{pair.base_name}"

        image_ext = Path(pair.image).suffix.lower() or ".png"
        mask_ext = Path(pair.mask).suffix.lower() or ".png"
        image_out = output_dir / f"{prefix}_image{image_ext}"
        mask_out = output_dir / f"{prefix}_mask{mask_ext}"
        shutil.copy2(pair.image, image_out)
        shutil.copy2(pair.mask, mask_out)
        manifest.append(
            {
                "image": str(image_out),
                "mask": str(mask_out),
                "source": source,
                "base_name": pair.base_name,
                "original_image": pair.image,
                "original_mask": pair.mask,
            }
        )

    return output_dir, manifest


def count_by_source(pairs: Iterable[ImageMaskPair]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for pair in pairs:
        counts[pair.source] = counts.get(pair.source, 0) + 1
    return counts


def pairs_to_dicts(pairs: Sequence[ImageMaskPair]) -> list[dict]:
    return [asdict(pair) for pair in pairs]
