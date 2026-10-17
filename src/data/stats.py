"""Per-channel mean/std of the training split, for normalising the CNN input.

Fitted on train only (no val/test leakage), on images resized exactly as the
model sees them (data.image_size), and saved next to the split CSVs. Re-run
whenever the splits or the image size change:

    python -m src.data.stats --config experiments/configs/default.yaml
"""
import argparse
import json

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from src.features.augment import get_eval_transforms, train_stats_path
from src.train import ROOT, load_config


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="experiments/configs/default.yaml")
    config = load_config(parser.parse_args().config)

    data = config["data"]
    train = pd.read_csv(ROOT / data["splits_dir"] / "train.csv")
    stats = channel_stats(train["Filename"], ROOT / data["images_dir"],
                          get_eval_transforms(config, normalise=False))
    stats.update(n_images=len(train), image_size=data["image_size"])
    out = train_stats_path(config)
    out.write_text(json.dumps(stats, indent=2) + "\n")
    print(stats)
    print(f"saved to {out}")
