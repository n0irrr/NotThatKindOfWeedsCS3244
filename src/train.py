"""Entry point: python -m src.train --config experiments/configs/default.yaml"""
import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
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


def load_images(config, size, splits=SPLITS):
    data = config["data"]
    return {split: load_split(ROOT / data["splits_dir"] / f"{split}.csv",
                              ROOT / data["images_dir"], size, config["classes"])
            for split in splits}


def load_features(config, splits=SPLITS):
    feat = dict(config["features"])
    load_size = tuple(feat.pop("load_size", config["data"]["image_size"]))
    return {split: (extract(images, **feat), y)
            for split, (images, y) in load_images(config, load_size, splits).items()}


def eval_splits(val_only):
    """Held-out splits a run is scored on. --val-only (tuning runs) never loads
    or scores test, so test can't influence any choice."""
    return ("val",) if val_only else ("val", "test")


# ---------------------------------------------------------------- classical (KNN, logistic regression)

def grid_search(build, params, grid, feats, classes):
    """Fit one model per grid point and keep the best by val macro F1 (test is
    never touched here). Returns (refit best model, best params, search log)."""
    (X_tr, y_tr), (X_va, y_va) = feats["train"], feats["val"]
    search = []
    for point in grid:
        model = build({**params, **point}).fit(X_tr, y_tr)
        f1 = compute_metrics(y_va, model.predict(X_va), classes)["macro_f1"]
        search.append({**point, "val_macro_f1": f1})
        print("  " + "  ".join(f"{k}={v}" for k, v in point.items()) + f"  val macro F1={f1:.4f}")
    best = max(search, key=lambda r: r["val_macro_f1"])
    best_params = {**params, **{k: best[k] for k in grid[0]}}
    return build(best_params).fit(X_tr, y_tr), best_params, search


def classical_grid(config):
    """(build function, fixed params, list of grid points) for knn or logreg."""
    params = {**config["model"]["params"], "seed": config["seed"]}
    if config["model"]["name"] == "knn":
        from src.models import knn
        k_grid, weights = params.pop("k_grid"), params.pop("weights")
        weights = [weights] if isinstance(weights, str) else weights
        return knn.build, params, [{"n_neighbors": k, "weights": w} for k, w in itertools.product(k_grid, weights)]
    from src.models import logreg
    return logreg.build, params, [{"C": c} for c in params.pop("C_grid")]


def run_classical(config, val_only=False):
    t0 = time.time()
    splits = ("train", *eval_splits(val_only))
    feats = load_features(config, splits)
    print(f"features: {config['features']['name']} "
          f"train {feats['train'][0].shape}, loaded in {time.time() - t0:.1f}s")
    build, params, grid = classical_grid(config)
    model, best_params, search = grid_search(build, params, grid, feats, config["classes"])
    print("best: " + ", ".join(f"{k}={best_params[k]}" for k in grid[0]))
    preds = {s: (feats[s][1], model.predict_proba(feats[s][0])) for s in eval_splits(val_only)}
    return preds, feats["train"][1], {"best_params": best_params, "search": search}


# ---------------------------------------------------------------- CNN

def get_device():
    import torch
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch, "xpu") and torch.xpu.is_available():   # Intel GPUs
        return torch.device("xpu")
    return torch.device("cpu")


def seed_everything(seed):
    import random
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_cnn(config, eval_only=False, resume=False, val_only=False):
    import warnings

    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader

    from src.data.dataset import build_datasets
    from src.features.augment import train_stats_path

    seed_everything(config["seed"])
    device = get_device()
    tr_cfg, params = config["training"], config["model"]["params"]
    run_name = config["run_name"]

    # Images are decoded per batch and pass through the transforms in
    # src/features/augment.py: resize, augmentation (train only), normalisation
    # (data.normalise: train-only stats, or ImageNet stats for pretrained backbones).
    if (config["data"].get("normalise", "train") == "train"
            and not train_stats_path(config, ROOT).exists()):
        from src.data.stats import save_train_stats
        print(f"saved {save_train_stats(config, ROOT).relative_to(ROOT)}")
    datasets = build_datasets(config, ROOT)
    y_tr = datasets["train"].targets
    num_workers = tr_cfg.get("num_workers", 0)

    def loader(split, shuffle):
        g = torch.Generator().manual_seed(config["seed"])
        return DataLoader(datasets[split], batch_size=tr_cfg["batch_size"], shuffle=shuffle, generator=g,
                          num_workers=num_workers, persistent_workers=num_workers > 0)

    # Optional speed-ups (both off by default): channels_last memory layout, and
    # bfloat16 autocast for the forward pass (weights and optimizer stay fp32).
    memory_format = torch.channels_last if tr_cfg.get("channels_last") else torch.contiguous_format
    amp = tr_cfg.get("amp")
    if amp not in (None, "bf16"):
        raise ValueError(f"unknown training.amp {amp!r}; use bf16 or leave it unset")

    def to_input(xb):
        return xb.to(device, memory_format=memory_format)

    def autocast():
        return torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp == "bf16")

    loaders = {s: loader(s, s == "train") for s in ("train", *eval_splits(val_only))}

    model = build_model(config).to(device, memory_format=memory_format)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"device: {device}, params: {n_params:,}, train images: {len(y_tr)}, "
          f"augmentation: {tr_cfg.get('augmentation') or 'none'}, seed: {config['seed']}"
          + (f", amp: {amp}" if amp else ""))

    criterion = nn.CrossEntropyLoss()
    # Stage "head": backbone frozen, head only at head_learning_rate (constant).
    # Stage "full": everything trainable, learning_rate with cosine decay over the
    # remaining epochs. freeze_epochs=0 (the default) means "full" throughout.
    freeze_epochs = tr_cfg.get("freeze_epochs", 0)

    def stage_for(epoch):
        return "head" if epoch <= freeze_epochs else "full"

    def make_optimizer(stage):
        if hasattr(model, "fc"):
            from src.models.transfer import set_backbone_trainable
            set_backbone_trainable(model, stage == "full")
        trainable = [p for p in model.parameters() if p.requires_grad]
        lr = tr_cfg["head_learning_rate"] if stage == "head" else tr_cfg["learning_rate"]
        opt = torch.optim.Adam(trainable, lr=lr, weight_decay=params.get("weight_decay", 0.0))
        if stage == "head":
            sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda _: 1.0)
        else:
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=tr_cfg["epochs"] - freeze_epochs)
        return opt, sched

    stage = stage_for(1)
    optimizer, scheduler = make_optimizer(stage)

    @torch.no_grad()
    def predict(split):
        """(mean loss, class probabilities (n, n_classes)) on a split."""
        model.eval()
        losses, probs = [], []
        for xb, yb in loaders[split]:
            with autocast():
                out = model(to_input(xb))
            out = out.float()
            losses.append(criterion(out, yb.to(device)).item() * len(yb))
            probs.append(out.softmax(1).cpu().numpy())
        return float(np.sum(losses) / len(loaders[split].dataset)), np.concatenate(probs)

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
            stage = state.get("stage", "full")
            optimizer, scheduler = make_optimizer(stage)
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
            stage = stage_for(best_epoch)
            optimizer, scheduler = make_optimizer(stage)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for _ in range(best_epoch - (0 if stage == "head" else freeze_epochs)):
                    scheduler.step()
            start_epoch = best_epoch + 1
            print(f"resuming from best weights {ckpt_path.relative_to(ROOT)} (epoch {best_epoch}, "
                  f"val F1 {best_f1:.4f}); fresh optimizer state, lr {scheduler.get_last_lr()[0]:.2e}")

    for epoch in range(start_epoch, 0 if eval_only else tr_cfg["epochs"] + 1):
        t0 = time.time()
        if stage_for(epoch) != stage:
            stage = stage_for(epoch)
            optimizer, scheduler = make_optimizer(stage)
            print(f"  stage -> {stage} (lr {scheduler.get_last_lr()[0]:.1e})")
        model.train()
        if stage == "head":
            model.eval()       # frozen backbone: keep BatchNorm running stats fixed
            model.fc.train()
        loss_sum, correct = 0.0, 0
        for xb, yb in loaders["train"]:
            yb = yb.to(device)
            with autocast():
                out = model(to_input(xb))
                loss = criterion(out, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(yb)
            correct += (out.argmax(1) == yb).sum().item()
        scheduler.step()

        val_loss, val_proba = predict("val")
        val_f1 = compute_metrics(datasets["val"].targets, val_proba.argmax(1), config["classes"])["macro_f1"]
        row = {"epoch": epoch, "train_loss": loss_sum / len(y_tr), "train_acc": correct / len(y_tr),
               "val_loss": val_loss, "val_macro_f1": val_f1}
        history.append(row)
        improved = val_f1 > best_f1
        if improved:
            best_f1, best_epoch = val_f1, epoch
            torch.save(model.state_dict(), ckpt_path)
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "stage": stage,
                    "epoch": epoch, "history": history,
                    "best_f1": best_f1, "best_epoch": best_epoch}, last_path)
        print(f"  epoch {epoch:>2}  train loss {row['train_loss']:.3f} acc {row['train_acc']:.3f}  "
              f"val loss {val_loss:.3f} macro F1 {val_f1:.4f}{' *' if improved else ''}  "
              f"({time.time() - t0:.0f}s)")
        if epoch - best_epoch >= patience:
            print(f"  early stop: no val macro F1 gain for {patience} epochs")
            break

    if eval_only:
        print(f"eval only: loading {ckpt_path.relative_to(ROOT)}")
        if last_path.exists():  # keep the training history in the rewritten metrics JSON
            state = torch.load(last_path, map_location="cpu")
            history, best_epoch = state["history"], state["best_epoch"]
    else:
        print(f"best epoch {best_epoch}, val macro F1 {best_f1:.4f}")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    if history:
        plot_curves(history, ROOT / "results" / "figures" / f"{run_name}_curves.png", run_name)
    preds = {s: (datasets[s].targets, predict(s)[1]) for s in eval_splits(val_only)}
    return preds, y_tr, {"best_epoch": best_epoch, "n_params": n_params,
                         "device": str(device), "history": history}


def build_model(config):
    name, params, n = config["model"]["name"], config["model"]["params"], len(config["classes"])
    if name == "simple_cnn":
        from src.models import simple_cnn
        return simple_cnn.build(n, params)
    from src.models import transfer
    return transfer.build(name, n, freeze_backbone=False, dropout=params.get("dropout", 0.3),
                          pretrained=params.get("pretrained", True))


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

CLASSICAL = ("knn", "logreg")
RUNNERS = {"knn": run_classical, "logreg": run_classical, "simple_cnn": run_cnn, "resnet50": run_cnn}


def save_predictions(config, split, y, proba):
    """Per-image true/predicted class and class probabilities, for error analysis.
    Rows are in split-CSV order (val/test loaders don't shuffle)."""
    classes = config["classes"]
    df = pd.read_csv(ROOT / config["data"]["splits_dir"] / f"{split}.csv", usecols=["Filename"])
    assert len(df) == len(y), f"{split}: {len(df)} rows in CSV but {len(y)} predictions"
    df["true"] = [classes[i] for i in y]
    df["pred"] = [classes[i] for i in proba.argmax(1)]
    df["confidence"] = proba.max(1).round(4)
    for i, c in enumerate(classes):
        df[f"p_{c}"] = proba[:, i].round(4)
    path = ROOT / "results" / "predictions" / f"{config['run_name']}_{split}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def report(config, preds, y_train, extra, elapsed):
    """Metrics JSON, confusion matrix, per-image predictions and console summary,
    the same for every model. preds maps split -> (true labels, class probabilities).
    Final runs report on test; --val-only tuning runs on val (test never loaded)."""
    classes, run_name = config["classes"], config["run_name"]
    results = {"run_name": run_name, "config": config, **extra, "splits": {}}
    for split, (y, proba) in preds.items():
        results["splits"][split] = {
            "model": compute_metrics(y, proba.argmax(1), classes),
            "majority_baseline": majority_class_baseline(y, y_train, classes),
        }
    split = "test" if "test" in preds else "val"
    y, proba = preds[split]
    y_pred = proba.argmax(1)
    cm = plot_confusion_matrix(y, y_pred, classes,
                               ROOT / "results" / "confusion_matrices" / f"{run_name}_{split}.png",
                               title=f"{run_name} ({split})")
    results[f"top_confusions_{split}"] = top_confusions(y, y_pred, classes)
    # raw counts (rows = true class), so repeated runs can be pooled later
    results[f"confusion_matrix_{split}"] = cm.tolist()
    pred_path = save_predictions(config, split, y, proba)

    metrics_path = ROOT / "results" / "metrics" / f"{run_name}.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(results, indent=2))

    keys = ["macro_f1", "macro_precision", "macro_recall", "accuracy"]
    print(f"\n{'':<22}" + "".join(f"{k:>16}" for k in keys))
    for s, r in results["splits"].items():
        for who in ("model", "majority_baseline"):
            print(f"{s + ' ' + who:<22}" + "".join(f"{r[who][k]:>16.4f}" for k in keys))
    print(f"\n{split} per-class F1:")
    for c, m in results["splits"][split]["model"]["per_class"].items():
        print(f"  {c:<16} F1={m['f1']:.3f}  P={m['precision']:.3f}  R={m['recall']:.3f}  n={m['support']}")
    print(f"top {split} confusions (true -> pred):")
    for t, p, n in results[f"top_confusions_{split}"]:
        print(f"  {t} -> {p}: {n}")
    print(f"\nsaved {metrics_path.relative_to(ROOT)}, {pred_path.relative_to(ROOT)} and confusion matrix "
          f"({elapsed:.1f}s total)")


def main(config_path: str, eval_only: bool = False, resume: bool = False, seed=None,
         val_only: bool = False) -> None:
    config = load_config(config_path)
    config.setdefault("run_name", Path(config_path).stem)
    if seed is not None:
        # repeated runs of one config: same split, different init / shuffling / augmentation
        config["seed"] = seed
        config["run_name"] = f"{config['run_name']}_s{seed}"
    name = config["model"]["name"]
    if name not in RUNNERS:
        raise NotImplementedError(f"Training not implemented yet for model: {name}")
    if (eval_only or resume) and name in CLASSICAL:
        raise ValueError(f"--eval-only / --resume need a saved model; {name} has none")
    t0 = time.time()
    kwargs = {k: True for k, v in [("eval_only", eval_only), ("resume", resume), ("val_only", val_only)] if v}
    preds, y_train, extra = RUNNERS[name](config, **kwargs)
    report(config, preds, y_train, extra, time.time() - t0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="experiments/configs/default.yaml")
    parser.add_argument("--eval-only", action="store_true",
                        help="skip training; evaluate experiments/checkpoints/{run_name}.pt")
    parser.add_argument("--resume", action="store_true",
                        help="continue training from experiments/checkpoints/{run_name}[_last].pt")
    parser.add_argument("--seed", type=int,
                        help="override the config seed and save as {run_name}_s{seed} "
                             "(combine runs with python -m src.evaluation.aggregate)")
    parser.add_argument("--val-only", action="store_true",
                        help="tuning run: score val only and never load test. Score test later, once the "
                             "config is chosen, with --eval-only (CNNs) or a normal run (KNN/logreg)")
    args = parser.parse_args()
    main(args.config, args.eval_only, args.resume, args.seed, args.val_only)
