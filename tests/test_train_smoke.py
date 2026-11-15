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
    """tmp_path laid out like the repo: data/images, <data.splits_dir>/*.csv."""
    defaults = train.load_config(train.DEFAULT_CONFIG)
    classes, split_dir = defaults["classes"], tmp_path / defaults["data"]["splits_dir"]
    rng = np.random.default_rng(0)
    (tmp_path / "data" / "images").mkdir(parents=True)
    split_dir.mkdir(parents=True)
    for split, n in PER_CLASS.items():
        rows = []
        for label, species in enumerate(classes):
            for i in range(n):
                name = f"{split}-{label}-{i}.jpg"
                # class-dependent tint plus noise, so the images aren't all identical
                img = rng.integers(0, 80, (48, 48, 3)) + np.array([label * 20, 100, 200 - label * 20])
                Image.fromarray(img.clip(0, 255).astype(np.uint8)).save(tmp_path / "data" / "images" / name)
                rows.append({"Filename": name, "Label": label, "Species": species})
        pd.DataFrame(rows).to_csv(split_dir / f"{split}.csv", index=False)

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
    # --val-only runs report on val and never score test; normal runs report on test
    scored = ("val",) if kwargs.get("val_only") else ("val", "test")
    assert set(metrics["splits"]) == set(scored)
    split = scored[-1]
    assert (root / "results" / "confusion_matrices" / f"{run_name}_{split}.png").exists()
    preds = pd.read_csv(root / "results" / "predictions" / f"{run_name}_{split}.csv")
    assert len(preds) == 9 * PER_CLASS[split] and preds["confidence"].between(0, 1).all()
    for s in scored:
        assert 0.0 <= metrics["splits"][s]["model"]["macro_f1"] <= 1.0
    return metrics


FEATURES = [{"name": "colour_histogram", "bins": 8}, {"name": "flattened_pixels", "size": [8, 8]},
            {"name": "hog", "size": [32, 32]}]


@pytest.mark.parametrize("feature", FEATURES, ids=lambda f: f["name"])
def test_knn(fake_root, feature):
    metrics = run(fake_root, "knn_smoke", {
        "features": {**feature, "load_size": [32, 32]},
        "model": {"name": "knn", "params": {"k_grid": [1, 3], "weights": ["uniform", "distance"],
                                            "metric": "euclidean", "pca": None}},
    })
    assert len(metrics["search"]) == 4


@pytest.mark.parametrize("feature", FEATURES, ids=lambda f: f["name"])
def test_logreg(fake_root, feature):
    metrics = run(fake_root, "logreg_smoke", {
        "features": {**feature, "load_size": [32, 32]},
        "model": {"name": "logreg", "params": {"C_grid": [0.1, 1.0], "pca": None, "max_iter": 200}},
    })
    assert [r["C"] for r in metrics["search"]] == [0.1, 1.0]


def test_val_only_never_scores_test(fake_root, monkeypatch):
    """Tuning runs must not even load the test split."""
    loaded = []
    real_load_split = train.load_split
    monkeypatch.setattr(train, "load_split", lambda csv, *a, **k: loaded.append(csv.stem) or real_load_split(csv, *a, **k))
    run(fake_root, "knn_val_only", {
        "features": {"name": "colour_histogram", "bins": 8, "load_size": [32, 32]},
        "model": {"name": "knn", "params": {"k_grid": [1], "weights": "uniform", "metric": "euclidean", "pca": None}},
    }, val_only=True)
    assert "test" not in loaded

    config = {"data": {"image_size": [32, 32]},
              "model": {"name": "simple_cnn", "params": {"channels": [8, 16], "dropout": 0.1}},
              "training": {"epochs": 1, "batch_size": 16, "learning_rate": 1.0e-3}}
    run(fake_root, "cnn_val_only", config, val_only=True)
    # once a config is chosen, its checkpoint is scored on test with --eval-only
    run(fake_root, "cnn_val_only", config, eval_only=True)


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
    split_dir = fake_root / train.load_config(train.DEFAULT_CONFIG)["data"]["splits_dir"]
    stats = json.loads((split_dir / "train_stats_32x32.json").read_text())
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


def test_explainability_helpers(fake_root):
    """Grad-CAM, LIME and the error-gallery helpers run on trained smoke models."""
    import matplotlib
    matplotlib.use("Agg")
    from src.evaluation import explain

    cnn = {"data": {"image_size": [32, 32]},
           "model": {"name": "simple_cnn", "params": {"channels": [8, 16], "dropout": 0.1}},
           "training": {"epochs": 1, "batch_size": 16, "learning_rate": 1.0e-3}}
    run(fake_root, "cnn_xai", cnn)
    knn = {"features": {"name": "colour_histogram", "bins": 8, "load_size": [32, 32]},
           "model": {"name": "knn", "params": {"k_grid": [1, 3], "weights": "uniform",
                                               "metric": "euclidean", "pca": None}}}
    run(fake_root, "knn_xai", knn)

    preds = fake_root / "results" / "predictions" / "cnn_xai_test.csv"
    rows = explain.confident_errors(preds, k=4)
    images_dir = fake_root / "data" / "images"
    pick = pd.read_csv(preds).head(2)

    model, config = explain.load_cnn(fake_root / "cnn_xai.yaml", "cnn_xai", root=fake_root)
    assert explain.gradcam_figure(model, config, pick["Filename"], pick["true"], root=fake_root).axes
    model, config, feat = explain.fit_classical(fake_root / "knn_xai.yaml", root=fake_root)
    assert explain.lime_knn_figure(model, config, feat, pick["Filename"][:1], pick["true"][:1],
                                   root=fake_root, num_samples=20).axes
    assert explain.gallery_figure(rows, images_dir).axes
    tags = explain.tagging_template(rows, fake_root / "tags.csv")
    assert list(tags.columns) == ["Filename", "true", "pred", "confidence", "cause", "notes"]
    stats = explain.image_stats(pick["Filename"], images_dir)
    assert {"brightness", "saturation", "contrast", "sharpness"} <= set(stats.columns)
