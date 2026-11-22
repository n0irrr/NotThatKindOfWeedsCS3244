"""Test-time augmentation (TTA) and seed ensembles for trained CNN runs.

TTA: each image is scored in the 8 orientations that the training augmentation
(flip + rotate) already shows the model, i.e. the 4 multiples of 90 degrees,
each with and without a left-right flip, and the 8 softmax outputs are averaged.
Ensemble: the softmax outputs of one config trained with different seeds
(--seed N runs, checkpoints <run_name>_s<N>.pt) are averaged.

    python -m src.evaluation.tta experiments/configs/resnet50_aug_geo_crop_30ep_bf16_kaggle0.yaml --seeds 1 2 3

scores every combination (each seed alone, with and without TTA; the ensemble,
with and without TTA) on val, the default, and writes
results/metrics/<run_name>_tta_val.json. Whether to use TTA or the ensemble is
decided on val; --split test is only for scoring a final choice once.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from src.evaluation.metrics import compute_metrics

ROOT = Path(__file__).resolve().parents[2]
N_VIEWS = 8
KEYS = ("accuracy", "macro_f1", "macro_precision", "macro_recall")


def dihedral_view(x, i):
    """View i (0-7) of an image batch (..., H, W): rotate by 90 * (i % 4) degrees,
    then flip left-right if i >= 4. View 0 is the image itself."""
    import torch

    x = torch.rot90(x, i % 4, dims=(-2, -1))
    return torch.flip(x, dims=(-1,)) if i >= 4 else x


def view_probs(model, loader, device, amp=None, channels_last=False):
    """Softmax outputs of `model` for every view: array (8, n_images, n_classes)."""
    import torch

    memory_format = torch.channels_last if channels_last else torch.contiguous_format
    model.eval()
    out = [[] for _ in range(N_VIEWS)]
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device, memory_format=memory_format)
            for i in range(N_VIEWS):
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp == "bf16"):
                    logits = model(dihedral_view(xb, i).contiguous(memory_format=memory_format))
                out[i].append(logits.float().softmax(1).cpu().numpy())
    return np.stack([np.concatenate(v) for v in out])


def combos(probs_by_seed: dict) -> dict:
    """Class probabilities (n, n_classes) for each way of combining the runs.
    probs_by_seed: {seed: (8, n, n_classes) from view_probs}."""
    out = {}
    for seed, p in probs_by_seed.items():
        out[f"s{seed}"] = p[0]
        out[f"s{seed} + TTA"] = p.mean(0)
    if len(probs_by_seed) > 1:
        stacked = np.stack(list(probs_by_seed.values()))       # (seeds, 8, n, C)
        out["ensemble"] = stacked[:, 0].mean(0)
        out["ensemble + TTA"] = stacked.mean((0, 1))
    return out


def evaluate(config_path, seeds, split="val", batch_size=64, root=ROOT):
    config, y, probs_by_seed = score_views(config_path, seeds, split, batch_size, root)
    run_name = config.get("run_name", Path(config_path).stem)
    results = {name: compute_metrics(y, p.argmax(1), config["classes"])
               for name, p in combos(probs_by_seed).items()}
    for tta in ("", " + TTA"):   # single runs: mean +/- std over seeds
        vals = {k: [results[f"s{s}{tta}"][k] for s in seeds] for k in KEYS}
        results[f"single{tta} (mean)"] = {k: float(np.mean(v)) for k, v in vals.items()}
        results[f"single{tta} (std)"] = {k: float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
                                         for k, v in vals.items()}
    return run_name, results


def score_views(config_path, seeds, split="val", batch_size=64, root=ROOT):
    """Labels and the 8-view softmax outputs of each seed's checkpoint on one split.
    Returns (config, y, {seed: (8, n, n_classes)})."""
    import torch
    from torch.utils.data import DataLoader

    from src.data.dataset import DeepWeedsDataset
    from src.evaluation.explain import load_cnn
    from src.features.augment import get_eval_transforms
    from src.train import get_device, load_config

    config = load_config(config_path)
    run_name = config.get("run_name", Path(config_path).stem)
    data, tr_cfg = config["data"], config["training"]
    dataset = DeepWeedsDataset(Path(root) / data["splits_dir"] / f"{split}.csv", Path(root) / data["images_dir"],
                               config["classes"], get_eval_transforms(config, root=root))
    workers = tr_cfg.get("num_workers", 0)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=workers, persistent_workers=workers > 0)
    device = get_device()
    memory_format = torch.channels_last if tr_cfg.get("channels_last") else torch.contiguous_format

    probs_by_seed = {}
    for seed in seeds:
        model, _ = load_cnn(config_path, f"{run_name}_s{seed}", root=root)
        model = model.to(device, memory_format=memory_format)
        probs_by_seed[seed] = view_probs(model, loader, device, tr_cfg.get("amp"), tr_cfg.get("channels_last"))
        print(f"  scored {run_name}_s{seed} on {split} ({len(dataset)} images x {N_VIEWS} views)")

    return config, np.asarray(dataset.targets), probs_by_seed


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config")
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    args = parser.parse_args()

    run_name, results = evaluate(args.config, args.seeds, args.split)
    print(f"\n{args.split}: " + "  ".join(f"{k:>15}" for k in KEYS))
    for name, m in results.items():
        print(f"{name:<22}" + "  ".join(f"{m[k]:>15.4f}" for k in KEYS))
    out = ROOT / "results" / "metrics" / f"{run_name}_tta_{args.split}.json"
    out.write_text(json.dumps({"run_name": run_name, "split": args.split, "seeds": args.seeds,
                               "results": results}, indent=2))
    print(f"\nsaved {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
