"""Every experiment config loads and has what src.train needs for its model."""
from pathlib import Path

import pytest

from src.features.augment import AUGMENTATIONS
from src.features.classical import FEATURES
from src.train import RUNNERS, build_model, load_config

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = sorted((ROOT / "experiments" / "configs").glob("*.yaml"))


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.stem)
def test_config(path):
    config = load_config(path)
    model = config["model"]
    assert model["name"] in RUNNERS

    classes = config["classes"]
    assert len(classes) == 9 and len(set(classes)) == 9

    splits_dir = ROOT / config["data"]["splits_dir"]
    for split in ("train", "val", "test"):
        assert (splits_dir / f"{split}.csv").exists(), f"missing {splits_dir}/{split}.csv"

    if model["name"] == "knn":
        if "features" not in config:   # default.yaml names knn but no features
            pytest.skip("no features section")
        assert config["features"]["name"] in FEATURES
        assert model["params"]["k_grid"] and model["params"]["weights"]
        return

    training = config["training"]
    for key in ("epochs", "batch_size", "learning_rate"):
        assert key in training
    if training.get("freeze_epochs", 0) > 0:
        assert "head_learning_rate" in training
    assert set(training.get("augmentation") or []) <= AUGMENTATIONS

    # build the network (without downloading pretrained weights)
    config["model"]["params"] = {**model["params"], "pretrained": False}
    build_model(config)
