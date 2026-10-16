"""Entry point: python -m src.train --config experiments/configs/default.yaml"""
import argparse
import itertools
import json
import time
from pathlib import Path

import yaml

from src.data.dataset import load_split
from src.evaluation.confusion import plot_confusion_matrix, top_confusions
from src.evaluation.metrics import compute_metrics, majority_class_baseline
from src.features.classical import extract

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "experiments" / "configs" / "default.yaml"


def deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(path) -> dict:
    with open(path) as f:
        config = yaml.safe_load(f)
    if Path(path).resolve() != DEFAULT_CONFIG:
        with open(DEFAULT_CONFIG) as f:
            config = deep_merge(yaml.safe_load(f), config)
    return config


def load_features(config):
    data, feat = config["data"], dict(config["features"])
    load_size = tuple(feat.pop("load_size", data["image_size"]))
    out = {}
    for split in ("train", "val", "test"):
        images, y = load_split(ROOT / data["splits_dir"] / f"{split}.csv",
                               ROOT / data["images_dir"], load_size, config["classes"])
        out[split] = (extract(images, **feat), y)
    return out


def train_knn(config, feats):
    from src.models import knn

    params = dict(config["model"]["params"])
    k_grid, weights_grid = params.pop("k_grid"), params.pop("weights")
    weights_grid = [weights_grid] if isinstance(weights_grid, str) else weights_grid
    params["seed"] = config["seed"]
    (X_tr, y_tr), (X_va, y_va) = feats["train"], feats["val"]

    # Select (k, weights) on val macro F1; test is never touched here.
    search = []
    for k, w in itertools.product(k_grid, weights_grid):
        model = knn.build({**params, "n_neighbors": k, "weights": w}).fit(X_tr, y_tr)
        f1 = compute_metrics(y_va, model.predict(X_va), config["classes"])["macro_f1"]
        search.append({"n_neighbors": k, "weights": w, "val_macro_f1": f1})
        print(f"  k={k:<3} weights={w:<8} val macro F1={f1:.4f}")
    best = max(search, key=lambda r: r["val_macro_f1"])
    best_params = {**params, "n_neighbors": best["n_neighbors"], "weights": best["weights"]}
    return knn.build(best_params).fit(X_tr, y_tr), best_params, search


def main(config_path: str) -> None:
    config = load_config(config_path)
    name, classes = config["model"]["name"], config["classes"]
    run_name = config.get("run_name", Path(config_path).stem)
    if name != "knn":
        raise NotImplementedError(f"Training not implemented yet for model: {name}")

    t0 = time.time()
    feats = load_features(config)
    print(f"features: {config['features']['name']} "
          f"train {feats['train'][0].shape}, loaded in {time.time() - t0:.1f}s")

    model, best_params, search = train_knn(config, feats)
    print(f"best: k={best_params['n_neighbors']} weights={best_params['weights']}")

    y_tr = feats["train"][1]
    results = {"run_name": run_name, "config": config, "best_params": best_params,
               "search": search, "splits": {}}
    for split in ("val", "test"):
        X, y = feats[split]
        y_pred = model.predict(X)
        results["splits"][split] = {
            "model": compute_metrics(y, y_pred, classes),
            "majority_baseline": majority_class_baseline(y, y_tr, classes),
        }
        if split == "test":
            cm_path = ROOT / "results" / "confusion_matrices" / f"{run_name}_test.png"
            plot_confusion_matrix(y, y_pred, classes, cm_path, title=f"{run_name} (test)")
            results["top_confusions_test"] = top_confusions(y, y_pred, classes)

    metrics_path = ROOT / "results" / "metrics" / f"{run_name}.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(results, indent=2))

    keys = ["macro_f1", "macro_precision", "macro_recall", "accuracy"]
    print(f"\n{'':<22}" + "".join(f"{k:>16}" for k in keys))
    for split, r in results["splits"].items():
        for who in ("model", "majority_baseline"):
            print(f"{split + ' ' + who:<22}" + "".join(f"{r[who][k]:>16.4f}" for k in keys))
    print("\ntest per-class F1:")
    for c, m in results["splits"]["test"]["model"]["per_class"].items():
        print(f"  {c:<16} F1={m['f1']:.3f}  P={m['precision']:.3f}  R={m['recall']:.3f}  n={m['support']}")
    print("top test confusions (true -> pred):")
    for t, p, n in results["top_confusions_test"]:
        print(f"  {t} -> {p}: {n}")
    print(f"\nsaved {metrics_path.relative_to(ROOT)} and confusion matrix "
          f"({time.time() - t0:.1f}s total)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="experiments/configs/default.yaml")
    main(parser.parse_args().config)
