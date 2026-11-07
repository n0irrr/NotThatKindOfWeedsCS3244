"""End-to-end smoke test of src.train on a tiny fake dataset: each model trains
for an epoch or two and writes its metrics, confusion matrix and checkpoint.
Checks the pipeline runs, not how good the scores are."""
import json

import numpy as np
import pandas as pd
import pytest
import torch
import yaml
from PIL import Image

import src.data.dataset as dataset
import src.train as train
from src.data.dataset import build_datasets
from src.features.augment import AUGMENTATIONS

PER_CLASS = {"train": 4, "val": 2, "test": 2}


@pytest.fixture
def fake_root(tmp_path, monkeypatch):
    """tmp_path laid out like the repo: data/images, data/splits/*.csv."""
    classes = train.load_config(train.DEFAULT_CONFIG)["classes"]
    rng = np.random.default_rng(0)
    (tmp_path / "data" / "images").mkdir(parents=True)
    (tmp_path / "data" / "splits").mkdir()
    for split, n in PER_CLASS.items():
        rows = []
        for label, species in enumerate(classes):
            for i in range(n):
                name = f"{split}-{label}-{i}.jpg"
                # class-dependent tint plus noise, so the images aren't all identical
                img = rng.integers(0, 80, (48, 48, 3)) + np.array([label * 20, 100, 200 - label * 20])
                Image.fromarray(img.clip(0, 255).astype(np.uint8)).save(tmp_path / "data" / "images" / name)
                rows.append({"Filename": name, "Label": label, "Species": species})
        pd.DataFrame(rows).to_csv(tmp_path / "data" / "splits" / f"{split}.csv", index=False)

    monkeypatch.setattr(train, "ROOT", tmp_path)
    monkeypatch.setattr(dataset, "CACHE_DIR", tmp_path / "data" / "processed")
    # CPU everywhere, so local runs match CI (and MPS can't crash the test run)
    monkeypatch.setattr(train, "get_device", lambda: torch.device("cpu"))
    return tmp_path


def run(root, run_name, overrides, **kwargs):
    # load in the main process: worker start-up would dominate on these tiny datasets
    training = {"num_workers": 0, **overrides.get("training", {})}
    path = root / f"{run_name}.yaml"
    path.write_text(yaml.safe_dump({"run_name": run_name, **overrides, "training": training}))
    train.main(str(path), **kwargs)
    metrics = json.loads((root / "results" / "metrics" / f"{run_name}.json").read_text())
    assert (root / "results" / "confusion_matrices" / f"{run_name}_test.png").exists()
    for split in ("val", "test"):
        assert 0.0 <= metrics["splits"][split]["model"]["macro_f1"] <= 1.0
    return metrics


@pytest.mark.parametrize("feature", [{"name": "colour_histogram", "bins": 8},
                                     {"name": "flattened_pixels", "size": [8, 8]}])
def test_knn(fake_root, feature):
    metrics = run(fake_root, "knn_smoke", {
        "features": {**feature, "load_size": [32, 32]},
        "model": {"name": "knn", "params": {"k_grid": [1, 3], "weights": ["uniform", "distance"],
                                            "metric": "euclidean", "pca": None}},
    })
    assert len(metrics["search"]) == 4


def test_simple_cnn(fake_root):
    config = {
        "data": {"image_size": [32, 32]},
        "model": {"name": "simple_cnn", "params": {"channels": [8, 16], "dropout": 0.1}},
        "training": {"epochs": 2, "batch_size": 16, "learning_rate": 1.0e-3, "patience": 2},
    }
    metrics = run(fake_root, "cnn_smoke", config)
    assert len(metrics["history"]) == 2
    assert (fake_root / "experiments" / "checkpoints" / "cnn_smoke.pt").exists()
    # reload the saved weights and evaluate again
    run(fake_root, "cnn_smoke", config, eval_only=True)


def test_resnet50_two_stage(fake_root):
    metrics = run(fake_root, "resnet_smoke", {
        "data": {"image_size": [32, 32]},
        "model": {"name": "resnet50", "params": {"pretrained": False, "dropout": 0.1}},
        "training": {"freeze_epochs": 1, "epochs": 2, "batch_size": 16, "learning_rate": 1.0e-4,
                     "head_learning_rate": 1.0e-3, "patience": 2},
    })
    assert len(metrics["history"]) == 2   # one head-only epoch, then one full epoch


def test_simple_cnn_all_augmentations(fake_root):
    metrics = run(fake_root, "cnn_aug_smoke", {
        "data": {"image_size": [32, 32]},
        "model": {"name": "simple_cnn", "params": {"channels": [8, 16], "dropout": 0.1}},
        "training": {"epochs": 1, "batch_size": 16, "learning_rate": 1.0e-3,
                     "augmentation": sorted(AUGMENTATIONS)},
    })
    assert len(metrics["history"]) == 1
    # train-only stats were created on first use, at the training image size
    stats = json.loads((fake_root / "data" / "splits" / "train_stats_32x32.json").read_text())
    assert stats["n_images"] == 9 * PER_CLASS["train"] and stats["image_size"] == [32, 32]


def test_augmentation_only_on_train(fake_root):
    config = train.load_config(train.DEFAULT_CONFIG)
    config["data"].update(image_size=[32, 32], normalise="imagenet")
    config["training"]["augmentation"] = sorted(AUGMENTATIONS)
    datasets = build_datasets(config, fake_root)
    for split in ("val", "test"):
        assert torch.equal(datasets[split][0][0], datasets[split][0][0])
    torch.manual_seed(0)
    draws = [datasets["train"][0][0] for _ in range(5)]
    assert not all(torch.equal(draws[0], d) for d in draws[1:])
