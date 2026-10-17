"""Entry point: python -m src.train --config experiments/configs/default.yaml"""
import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np
import yaml

from src.data.dataset import load_split
from src.evaluation.confusion import plot_confusion_matrix, top_confusions
from src.evaluation.metrics import compute_metrics, majority_class_baseline
from src.features.classical import extract

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "experiments" / "configs" / "default.yaml"
SPLITS = ("train", "val", "test")


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


def load_images(config, size):
    data = config["data"]
    return {split: load_split(ROOT / data["splits_dir"] / f"{split}.csv",
                              ROOT / data["images_dir"], size, config["classes"])
            for split in SPLITS}


def load_features(config):
    feat = dict(config["features"])
    load_size = tuple(feat.pop("load_size", config["data"]["image_size"]))
    return {split: (extract(images, **feat), y)
            for split, (images, y) in load_images(config, load_size).items()}


# ---------------------------------------------------------------- KNN

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


def run_knn(config):
    t0 = time.time()
    feats = load_features(config)
    print(f"features: {config['features']['name']} "
          f"train {feats['train'][0].shape}, loaded in {time.time() - t0:.1f}s")
    model, best_params, search = train_knn(config, feats)
    print(f"best: k={best_params['n_neighbors']} weights={best_params['weights']}")
    preds = {s: (feats[s][1], model.predict(feats[s][0])) for s in ("val", "test")}
    return preds, feats["train"][1], {"best_params": best_params, "search": search}


# ---------------------------------------------------------------- CNN

def get_device():
    import torch
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def seed_everything(seed):
    import random
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_cnn(config, eval_only=False, resume=False):
    import warnings

    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    from src.models import simple_cnn

    seed_everything(config["seed"])
    device = get_device()
    tr_cfg, params = config["training"], config["model"]["params"]
    run_name = config["run_name"]

    images = load_images(config, tuple(config["data"]["image_size"]))
    y_tr = images["train"][1]

    # Per-channel normalisation stats from train only.
    x_tr = images["train"][0]
    mean = torch.tensor(x_tr.reshape(-1, 3).mean(0) / 255, dtype=torch.float32, device=device).view(1, 3, 1, 1)
    std = torch.tensor(x_tr.reshape(-1, 3).std(0) / 255, dtype=torch.float32, device=device).view(1, 3, 1, 1)

    def loader(split, shuffle):
        x, y = images[split]
        # Keep uint8 NHWC on CPU; convert per batch to save memory. Arrays from
        # the npz cache are read-only, which is fine since nothing writes to them.
        warnings.filterwarnings("ignore", message="The given NumPy array is not writable")
        ds = TensorDataset(torch.from_numpy(x), torch.from_numpy(y).long())
        g = torch.Generator().manual_seed(config["seed"])
        return DataLoader(ds, batch_size=tr_cfg["batch_size"], shuffle=shuffle, generator=g)

    def to_input(xb):
        xb = xb.to(device).permute(0, 3, 1, 2).float().div_(255)
        return (xb - mean) / std

    loaders = {s: loader(s, s == "train") for s in SPLITS}

    model = simple_cnn.build(len(config["classes"]), params).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"device: {device}, params: {n_params:,}, train images: {x_tr.shape}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=tr_cfg["learning_rate"],
                                 weight_decay=params.get("weight_decay", 0.0))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=tr_cfg["epochs"])

    @torch.no_grad()
    def predict(split):
        model.eval()
        losses, preds = [], []
        for xb, yb in loaders[split]:
            out = model(to_input(xb))
            losses.append(criterion(out, yb.to(device)).item() * len(yb))
            preds.append(out.argmax(1).cpu().numpy())
        return float(np.sum(losses) / len(loaders[split].dataset)), np.concatenate(preds)

    # {run}.pt = best weights (by val macro F1); {run}_last.pt = full state after
    # the latest epoch, for exact resumes.
    ckpt_path = ROOT / "experiments" / "checkpoints" / f"{run_name}.pt"
    last_path = ckpt_path.with_name(f"{run_name}_last.pt")
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    history, best_f1, best_epoch, patience = [], -1.0, 0, tr_cfg.get("patience", 5)
    start_epoch = 1

    if resume:
        if last_path.exists():
            state = torch.load(last_path, map_location=device)
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            history, best_f1, best_epoch = state["history"], state["best_f1"], state["best_epoch"]
            start_epoch = state["epoch"] + 1
            print(f"resuming exactly from {last_path.relative_to(ROOT)} (epoch {state['epoch']})")
        else:
            # Only best weights exist (older runs): restart after the best epoch with
            # the LR schedule fast-forwarded; Adam moments start fresh.
            prev = json.loads((ROOT / "results" / "metrics" / f"{run_name}.json").read_text())
            best_epoch = prev["best_epoch"]
            history = [h for h in prev["history"] if h["epoch"] <= best_epoch]
            best_f1 = max(h["val_macro_f1"] for h in history)
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for _ in range(best_epoch):
                    scheduler.step()
            start_epoch = best_epoch + 1
            print(f"resuming from best weights {ckpt_path.relative_to(ROOT)} (epoch {best_epoch}, "
                  f"val F1 {best_f1:.4f}); fresh optimizer state, lr {scheduler.get_last_lr()[0]:.2e}")

    for epoch in range(start_epoch, 0 if eval_only else tr_cfg["epochs"] + 1):
        t0 = time.time()
        model.train()
        loss_sum, correct = 0.0, 0
        for xb, yb in loaders["train"]:
            yb = yb.to(device)
            out = model(to_input(xb))
            loss = criterion(out, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(yb)
            correct += (out.argmax(1) == yb).sum().item()
        scheduler.step()

        val_loss, val_pred = predict("val")
        val_f1 = compute_metrics(images["val"][1], val_pred, config["classes"])["macro_f1"]
        row = {"epoch": epoch, "train_loss": loss_sum / len(y_tr), "train_acc": correct / len(y_tr),
               "val_loss": val_loss, "val_macro_f1": val_f1}
        history.append(row)
        improved = val_f1 > best_f1
        if improved:
            best_f1, best_epoch = val_f1, epoch
            torch.save(model.state_dict(), ckpt_path)
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "epoch": epoch, "history": history,
                    "best_f1": best_f1, "best_epoch": best_epoch}, last_path)
        print(f"  epoch {epoch:>2}  train loss {row['train_loss']:.3f} acc {row['train_acc']:.3f}  "
              f"val loss {val_loss:.3f} macro F1 {val_f1:.4f}{' *' if improved else ''}  "
              f"({time.time() - t0:.0f}s)")
        if epoch - best_epoch >= patience:
            print(f"  early stop: no val macro F1 gain for {patience} epochs")
            break

    if eval_only:
        print(f"eval only: loading {ckpt_path.relative_to(ROOT)}")
    else:
        print(f"best epoch {best_epoch}, val macro F1 {best_f1:.4f}")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    if history:
        plot_curves(history, ROOT / "results" / "figures" / f"{run_name}_curves.png", run_name)
    preds = {s: (images[s][1], predict(s)[1]) for s in ("val", "test")}
    return preds, y_tr, {"best_epoch": best_epoch, "n_params": n_params,
                         "device": str(device), "history": history}


def plot_curves(history, out_path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ep = [h["epoch"] for h in history]
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4))
    a.plot(ep, [h["train_loss"] for h in history], label="train")
    a.plot(ep, [h["val_loss"] for h in history], label="val")
    a.set_xlabel("epoch"); a.set_ylabel("cross-entropy loss"); a.legend()
    b.plot(ep, [h["train_acc"] for h in history], label="train accuracy")
    b.plot(ep, [h["val_macro_f1"] for h in history], label="val macro F1")
    b.set_xlabel("epoch"); b.set_ylim(0, 1); b.legend()
    fig.suptitle(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- shared

RUNNERS = {"knn": run_knn, "simple_cnn": run_cnn}


def report(config, preds, y_train, extra, elapsed):
    """Metrics JSON, test confusion matrix, console summary. Same for all models."""
    classes, run_name = config["classes"], config["run_name"]
    results = {"run_name": run_name, "config": config, **extra, "splits": {}}
    for split, (y, y_pred) in preds.items():
        results["splits"][split] = {
            "model": compute_metrics(y, y_pred, classes),
            "majority_baseline": majority_class_baseline(y, y_train, classes),
        }
    y, y_pred = preds["test"]
    plot_confusion_matrix(y, y_pred, classes,
                          ROOT / "results" / "confusion_matrices" / f"{run_name}_test.png",
                          title=f"{run_name} (test)")
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
    print(f"\nsaved {metrics_path.relative_to(ROOT)} and confusion matrix ({elapsed:.1f}s total)")


def main(config_path: str, eval_only: bool = False, resume: bool = False) -> None:
    config = load_config(config_path)
    config.setdefault("run_name", Path(config_path).stem)
    name = config["model"]["name"]
    if name not in RUNNERS:
        raise NotImplementedError(f"Training not implemented yet for model: {name}")
    t0 = time.time()
    kwargs = {k: True for k, v in [("eval_only", eval_only), ("resume", resume)] if v}
    if kwargs and name == "knn":
        raise ValueError("--eval-only / --resume need a saved model; KNN has none")
    preds, y_train, extra = RUNNERS[name](config, **kwargs)
    report(config, preds, y_train, extra, time.time() - t0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="experiments/configs/default.yaml")
    parser.add_argument("--eval-only", action="store_true",
                        help="skip training; evaluate experiments/checkpoints/{run_name}.pt")
    parser.add_argument("--resume", action="store_true",
                        help="continue training from experiments/checkpoints/{run_name}[_last].pt")
    args = parser.parse_args()
    main(args.config, args.eval_only, args.resume)
