import random
from pathlib import Path

import numpy as np
from PIL import Image

DATA_ROOT = Path("data/EWS-Dataset")
SPLITS = ["train", "validation", "test"]
OUT_DIR = Path("results/inspect")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def overlay(image_rgb: np.ndarray, mask01: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    img = image_rgb.astype(np.float32).copy()
    m = mask01.astype(bool)
    red = np.array([255, 0, 0], dtype=np.float32)
    img[m] = (1 - alpha) * img[m] + alpha * red
    return np.clip(img, 0, 255).astype(np.uint8)

def load_rgb(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))

# We use channel 0 of the provided 2-channel PNG mask and 
# binarize with plant pixels defined as value 255. 
def load_mask01(path):
    m = np.array(Image.open(path))
    if m.ndim == 3:
        m = m[:, :, 0]

    mask01 = (m == 0).astype(np.uint8)   # <-- changed here

    # debug (optional)
    u = np.unique(m)
    if (i ==0) {
         print(" raw min/max:", int(u[0]), int(u[-1]), "has255:", bool((u==255).any()), "n_unique:", len(u))
        print(" pct==255:", float((m==255).mean()))
        print(" pct==254:", float((m==254).mean()))
        print(" pct==0  :", float((m==0).mean()), "<- plant ratio now")
    }
   
    return mask01


def list_pairs(split: str):
    split_dir = DATA_ROOT / split
    img_paths = sorted([p for p in split_dir.glob("*.png") if not p.name.endswith("_mask.png")])
    pairs = []
    for img_path in img_paths:
        mask_path = img_path.with_name(img_path.stem + "_mask.png")
        if mask_path.exists():
            pairs.append((img_path, mask_path))
    return pairs

def main():
    random.seed(0)

    for split in SPLITS:
        pairs = list_pairs(split)
        print(f"\n=== {split} ===")
        print("Pairs:", len(pairs))
        if not pairs:
            continue

        sample = random.sample(pairs, k=min(5, len(pairs)))
        for i, (img_path, mask_path) in enumerate(sample, start=1):
            img = load_rgb(img_path)
            mask01 = load_mask01(mask_path)

            print(f"\nPair {i}:")
            print(" image:", img_path.name, img.shape, img.dtype)
            raw_mask = np.array(Image.open(mask_path))
            uniq_raw = np.unique(raw_mask)
            print(" mask :", mask_path.name, raw_mask.shape, raw_mask.dtype, "unique:", uniq_raw[:20])

            ov = overlay(img, mask01)
            out_path = OUT_DIR / f"{split}_{i}_overlay.png"
            Image.fromarray(ov).save(out_path)
            print(" saved:", out_path)

if __name__ == "__main__":
    main()
