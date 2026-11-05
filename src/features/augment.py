"""Image transforms for the CNN models.

Eval transform: resize 256 -> data.image_size, scale to [0, 1], normalise.
Train transform: the same, plus whichever augmentations are listed in
training.augmentation. Each augmentation can be switched on on its own, so
the one-factor-at-a-time runs in the proposal (4.1.2) are just config edits:

    flip       random horizontal + vertical flips
    rotate     random multiple of 90 degrees (no black corners)
    crop       random resized crop (scale 0.5-1) instead of a plain resize
    translate  shift up to 10% in each direction
    colour     brightness / contrast / saturation / hue jitter
    blur       Gaussian blur on half the images
    noise      additive Gaussian noise

Augmentation is only ever applied to the training split.
(train.py uses get_pixel_augment / get_float_augment / AugmentedImages below.)

Normalisation (data.normalise):
    train     per-channel mean/std of the training split at data.image_size,
              from <splits_dir>/train_stats_<H>x<W>.json (python -m src.data.stats).
              Computed after the resize because resizing smooths fine texture
              (256 -> 224 cuts pixel std by ~13%).
    imagenet  ImageNet statistics, for pretrained backbones
"""
import json
from pathlib import Path

import numpy as np
import torch
from torchvision.transforms import v2

ROOT = Path(__file__).resolve().parents[2]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class RandomRot90(torch.nn.Module):
    """Rotate by a random multiple of 90 degrees. Field images have no
    canonical orientation, and unlike RandomRotation this adds no corners."""

    def forward(self, img):
        return torch.rot90(img, int(torch.randint(4, ())), dims=(-2, -1))


# Applied to uint8 images, before scaling to [0, 1]
PIXEL_AUGS = {
    "flip": lambda: v2.Compose([v2.RandomHorizontalFlip(), v2.RandomVerticalFlip()]),
    "rotate": RandomRot90,
    "translate": lambda: v2.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    "colour": lambda: v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.02),
    "blur": lambda: v2.RandomApply([v2.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.5),
}
# Applied to float images in [0, 1], before normalisation
FLOAT_AUGS = {
    "noise": lambda: v2.GaussianNoise(sigma=0.03),
}
AUGMENTATIONS = {"crop", *PIXEL_AUGS, *FLOAT_AUGS}


def train_stats_path(config) -> Path:
    h, w = config["data"]["image_size"]
    return ROOT / config["data"]["splits_dir"] / f"train_stats_{h}x{w}.json"


def normalisation(config):
    """(mean, std) per channel, as selected by config["data"]["normalise"]."""
    mode = config["data"].get("normalise", "train")
    if mode == "imagenet":
        return IMAGENET_MEAN, IMAGENET_STD
    if mode != "train":
        raise ValueError(f"unknown data.normalise {mode!r}; use 'train' or 'imagenet'")
    stats_path = train_stats_path(config)
    if not stats_path.exists():
        raise FileNotFoundError(f"{stats_path} missing: run python -m src.data.stats --config <your config>")
    stats = json.loads(stats_path.read_text())
    return stats["mean"], stats["std"]


def _augmentations(config):
    augs = config.get("training", {}).get("augmentation") or []
    unknown = set(augs) - AUGMENTATIONS
    if unknown:
        raise ValueError(f"unknown augmentation(s) {sorted(unknown)}; choose from {sorted(AUGMENTATIONS)}")
    return augs


def _build(config, augs, normalise=True):
    size = tuple(config["data"]["image_size"])
    resize = (v2.RandomResizedCrop(size, scale=(0.5, 1.0), antialias=True) if "crop" in augs
              else v2.Resize(size, antialias=True))
    return v2.Compose([
        v2.ToImage(),
        resize,
        *[PIXEL_AUGS[a]() for a in augs if a in PIXEL_AUGS],
        v2.ToDtype(torch.float32, scale=True),
        *[FLOAT_AUGS[a]() for a in augs if a in FLOAT_AUGS],
        *([v2.Normalize(*normalisation(config))] if normalise else []),
    ])


def get_train_transforms(config):
    return _build(config, _augmentations(config))


def get_eval_transforms(config, normalise=True):
    """normalise=False gives the [0, 1] image the stats are computed on."""
    return _build(config, [], normalise)


# ---------------------------------------------------------------------------
# Augmentation for the cached-array training loop (src/train.py).
#
# train.py works on images that are already resized and cached as uint8 arrays,
# so it cannot use get_train_transforms (which starts from the PIL file). These
# helpers build only the random part, in the same order as _build:
#     crop -> pixel augs (uint8)   |   float augs (after scaling to [0, 1])
# They are used on the TRAIN split only; val/test batches are never touched.
# ---------------------------------------------------------------------------

def get_pixel_augment(config):
    """Per-image transform for uint8 CHW tensors, or None if no pixel augmentation."""
    augs = _augmentations(config)
    size = tuple(config["data"]["image_size"])
    ops = []
    if "crop" in augs:
        ops.append(v2.RandomResizedCrop(size, scale=(0.5, 1.0), antialias=True))
    ops += [PIXEL_AUGS[a]() for a in augs if a in PIXEL_AUGS]
    return v2.Compose(ops) if ops else None


def get_float_augment(config):
    """Transform for float [0, 1] batches (noise), or None."""
    augs = _augmentations(config)
    ops = [FLOAT_AUGS[a]() for a in augs if a in FLOAT_AUGS]
    return v2.Compose(ops) if ops else None


class AugmentedImages(torch.utils.data.Dataset):
    """uint8 NHWC array + labels -> randomly augmented uint8 NHWC image + label.

    Augmentation is applied per image, so every image in a batch gets its own random
    flip / rotation / jitter (applying a transform to a whole batch tensor would give
    all images the same one). Returns NHWC so the training loop's input code is the
    same for augmented and plain batches."""

    def __init__(self, x: np.ndarray, y: np.ndarray, pixel_aug):
        self.x, self.y, self.pixel_aug = x, y, pixel_aug

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        img = torch.from_numpy(np.ascontiguousarray(self.x[i])).permute(2, 0, 1)   # CHW uint8
        img = self.pixel_aug(img)
        return img.permute(1, 2, 0).contiguous(), int(self.y[i])