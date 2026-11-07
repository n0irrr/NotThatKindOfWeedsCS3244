"""Per-channel mean/std of the training split, for normalising the CNN input.

Fitted on train only (no val/test leakage), on images resized exactly as the
model sees them (data.image_size), and saved next to the split CSVs. src.train
creates the file on first use if it is missing; re-run by hand whenever the
splits change:

    python -m src.data.stats --config experiments/configs/default.yaml
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from src.features.augment import get_eval_transforms, train_stats_path

ROOT = Path(__file__).resolve().parents[2]


def channel_stats(filenames, images_dir, transform) -> dict:
    """Mean and std per RGB channel over every pixel of transform(image)."""
    total = np.zeros(3)
    total_sq = np.zeros(3)
    n_pixels = 0
    for f in tqdm(filenames, desc="train stats"):
        with Image.open(images_dir / f) as img:
            x = transform(img.convert("RGB")).double().flatten(1)   # (3, H*W) in [0, 1]
        total += x.sum(1).numpy()
        total_sq += (x ** 2).sum(1).numpy()
        n_pixels += x.shape[1]
    mean = total / n_pixels
    std = np.sqrt(total_sq / n_pixels - mean ** 2)
    return {"mean": mean.round(6).tolist(), "std": std.round(6).tolist()}


def save_train_stats(config, root=ROOT) -> Path:
    """Compute the train split's stats at data.image_size and write them to
    <splits_dir>/train_stats_<H>x<W>.json. Returns the path."""
    data, root = config["data"], Path(root)
    train = pd.read_csv(root / data["splits_dir"] / "train.csv")
    stats = channel_stats(train["Filename"], root / data["images_dir"],
                          get_eval_transforms(config, normalise=False, root=root))
    stats.update(n_images=len(train), image_size=data["image_size"])
    out = train_stats_path(config, root)
    out.write_text(json.dumps(stats, indent=2) + "\n")
    return out


if __name__ == "__main__":
    from src.train import load_config

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="experiments/configs/default.yaml")
    out = save_train_stats(load_config(parser.parse_args().config))
    print(out.read_text())
    print(f"saved to {out}")
