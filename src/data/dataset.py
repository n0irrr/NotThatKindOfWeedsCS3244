"""Image loading utilities and (optionally) a torch Dataset wrapper."""
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = ROOT / "data" / "processed"


def load_image(path, size=(224, 224)):
    """RGB uint8 array of shape (H, W, 3), resized to size=(H, W)."""
    with Image.open(path) as img:
        img = img.convert("RGB")
        if img.size != (size[1], size[0]):
            img = img.resize((size[1], size[0]), Image.BILINEAR)
        return np.asarray(img, dtype=np.uint8)


def load_split(split_csv, images_dir, size, classes, cache=True):
    """Load every image in a split CSV. Returns (images (n, H, W, 3), y (n,)),
    where y indexes into `classes`. Cached as .npz under data/processed/."""
    split_csv, images_dir = Path(split_csv), Path(images_dir)
    cache_path = CACHE_DIR / f"{split_csv.stem}_{size[0]}x{size[1]}.npz"
    df = pd.read_csv(split_csv)
    y = df["Species"].map({c: i for i, c in enumerate(classes)}).to_numpy()
    assert not np.isnan(y.astype(float)).any(), "unknown Species in split CSV"

    if cache and cache_path.exists():
        cached = np.load(cache_path, allow_pickle=False)
        if list(cached["filenames"]) == df["Filename"].tolist():
            return cached["images"], y

    images = np.stack([
        load_image(images_dir / f, size)
        for f in tqdm(df["Filename"], desc=f"loading {split_csv.stem}")
    ])
    if cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.savez(cache_path, images=images, filenames=df["Filename"].to_numpy(dtype=str))
    return images, y
