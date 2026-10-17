"""Image loading utilities.

load_split  loads a whole split into memory as arrays (classical models).
DeepWeedsDataset  reads one JPEG at a time, for the CNNs via torch DataLoader.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = ROOT / "data" / "processed"
SPLITS = ("train", "val", "test")


def _class_indices(df: pd.DataFrame, classes) -> np.ndarray:
    y = df["Species"].map({c: i for i, c in enumerate(classes)}).to_numpy()
    assert not np.isnan(y.astype(float)).any(), "unknown Species in split CSV"
    return y.astype(np.int64)


class DeepWeedsDataset:
    """Map-style dataset over one split CSV, usable with torch DataLoader.

    Images are decoded lazily, so memory use doesn't grow with the split size.
    Returns (transform(PIL image), class index into `classes`).
    """

    def __init__(self, split_csv, images_dir, classes, transform=None):
        df = pd.read_csv(split_csv)
        self.images_dir = Path(images_dir)
        self.filenames = df["Filename"].tolist()
        self.targets = _class_indices(df, classes)
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, i):
        with Image.open(self.images_dir / self.filenames[i]) as img:
            img = img.convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, int(self.targets[i])


def build_datasets(config) -> dict[str, DeepWeedsDataset]:
    """train/val/test datasets for a config: train gets the training transforms
    (with any augmentation), val and test get the eval transforms."""
    from src.features.augment import get_eval_transforms, get_train_transforms

    data = config["data"]
    return {
        split: DeepWeedsDataset(
            ROOT / data["splits_dir"] / f"{split}.csv", ROOT / data["images_dir"], config["classes"],
            get_train_transforms(config) if split == "train" else get_eval_transforms(config))
        for split in SPLITS
    }


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
    y = _class_indices(df, classes)

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
