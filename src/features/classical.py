"""Hand-crafted feature representations for classical ML baselines.

Per the proposal: flattened pixels, downsampled pixels, colour histograms.
"""
import numpy as np
from PIL import Image


def flattened_pixels(images, size=(32, 32)):
    """Downsample each image to size=(H, W) and flatten to [0, 1] floats."""
    out = np.empty((len(images), size[0] * size[1] * 3), dtype=np.float32)
    for i, img in enumerate(images):
        small = Image.fromarray(img).resize((size[1], size[0]), Image.BILINEAR)
        out[i] = np.asarray(small, dtype=np.float32).ravel() / 255.0
    return out


def colour_histogram(images, bins=32):
    """Per-channel HSV histograms, each L1-normalised, concatenated (n, 3*bins)."""
    out = np.empty((len(images), 3 * bins), dtype=np.float32)
    for i, img in enumerate(images):
        hsv = np.asarray(Image.fromarray(img).convert("HSV"))
        hists = [np.histogram(hsv[..., c], bins=bins, range=(0, 256))[0] for c in range(3)]
        out[i] = np.concatenate([h / h.sum() for h in hists])
    return out


FEATURES = {"flattened_pixels": flattened_pixels, "colour_histogram": colour_histogram}


def extract(images, name, **kwargs):
    if name not in FEATURES:
        raise ValueError(f"unknown feature {name!r}; choose from {list(FEATURES)}")
    if "size" in kwargs:
        kwargs["size"] = tuple(kwargs["size"])
    return FEATURES[name](images, **kwargs)
