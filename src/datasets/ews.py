from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class EWSSample:
    image_path: Path
    mask_path: Path


def list_pairs(dataset_root: str | Path, split: str) -> List[EWSSample]:
    """
    dataset_root: e.g. 'data/EWS-Dataset'
    split: 'train' | 'validation' | 'test'
    Returns a list of (image_path, mask_path) pairs.
    """
    split_dir = Path(dataset_root) / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Split dir not found: {split_dir}")

    mask_paths = sorted(split_dir.glob("*_mask.png"))
    if len(mask_paths) == 0:
        raise FileNotFoundError(f"No mask files found in: {split_dir}")

    pairs: List[EWSSample] = []
    for mp in mask_paths:
        img_name = mp.name.replace("_mask.png", ".png")
        ip = split_dir / img_name
        if not ip.exists():
            raise FileNotFoundError(f"Missing image for mask {mp.name}: expected {ip.name}")
        pairs.append(EWSSample(image_path=ip, mask_path=mp))

    return pairs


def load_image(path: str | Path) -> np.ndarray:
    """
    Returns image as float32 array in [0,1], shape (H,W,3).
    """
    img = np.array(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return img


def load_mask01(path: str | Path) -> np.ndarray:
    """
    Returns binary mask as uint8 {0,1}, shape (H,W).
    Confirmed rule from overlays:
      - Use channel 0 of the 2-channel PNG
      - Plant pixels are value 0
    """
    m = np.array(Image.open(path))
    if m.ndim == 3:
        m = m[:, :, 0]  # channel 0

    mask01 = (m == 0).astype(np.uint8)  # plant=1 where raw==0
    return mask01


def load_sample(sample: EWSSample) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convenience: returns (image, mask01)
    """
    return load_image(sample.image_path), load_mask01(sample.mask_path)


def sanity_check(dataset_root: str | Path = "data/EWS-Dataset") -> None:
    """
    Quick check counts for each split.
    """
    for split in ["train", "validation", "test"]:
        pairs = list_pairs(dataset_root, split)
        print(f"{split}: {len(pairs)} pairs")
