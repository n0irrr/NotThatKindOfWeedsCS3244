"""Error analysis for a trained CNN: per-image predictions, errors by capture session / date,
and Grad-CAM overlays for misclassified images of chosen species.

    python -m src.evaluation.error_analysis --config experiments/configs/resnet50_aug_geo_30ep.yaml --seed 0

Needs experiments/checkpoints/{run_name}.pt (from src.train) and the images.
Outputs under results/error_analysis/: {run}_predictions.csv, {run}_sessions.csv, {run}_gradcam_*.png.
The CSV has no site column, so the capture session and the date in the filename are the proxies.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.data.dataset import build_datasets
from src.train import ROOT, build_model, get_device, load_config

FOCUS = ["Parthenium", "Snake weed", "Chinee apple"]


def predict(model, dataset, device, batch_size=64):
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    model.eval()
    probs = []
    with torch.no_grad():
        for xb, _ in loader:
            probs.append(torch.softmax(model(xb.to(device)), 1).cpu())
    return torch.cat(probs).numpy()


def cam_layer(model):
    """Last conv feature map: ResNet50 layer4 output, or the final ReLU of the simple CNN's last block."""
    return model.layer4[-1] if hasattr(model, "fc") else model[3][5]


def grad_cam(model, layer, x, target):
    """x: (1, 3, H, W) on device. Returns an (H, W) map in [0, 1] for class `target`."""
    store = {}
    def capture(module, inp, out):  # tensor hook: module backward hooks clash with in-place ReLU
        store["act"] = out
        out.register_hook(lambda g: store.update(grad=g))
    h1 = layer.register_forward_hook(capture)
    model.eval()
    model.zero_grad()
    model(x)[0, target].backward()
    h1.remove()
    w = store["grad"].mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((w * store["act"]).sum(1, keepdim=True))
    cam = torch.nn.functional.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    cam = cam - cam.min()
    return (cam / cam.max().clamp(min=1e-8)).detach().cpu().numpy()


def session_table(df):
    g = df.groupby("session").agg(n=("correct", "size"), errors=("correct", lambda c: int((~c).sum())))
    g["error_rate"] = g["errors"] / g["n"]
    return g.sort_values("errors", ascending=False)


def main(config_path, seed, split, n_cam):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    config = load_config(config_path)
    run = config.setdefault("run_name", Path(config_path).stem)
    if seed is not None:
        config["seed"], run = seed, f"{run}_s{seed}"
    classes, device = config["classes"], get_device()
    out_dir = ROOT / "results" / "error_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(config).to(device)
    model.load_state_dict(torch.load(ROOT / "experiments" / "checkpoints" / f"{run}.pt", map_location=device))
    ds = build_datasets(config, ROOT)[split]
    probs = predict(model, ds, device)

    df = pd.read_csv(ROOT / config["data"]["splits_dir"] / f"{split}.csv")
    df["true"], df["pred"] = df["Species"], [classes[i] for i in probs.argmax(1)]
    df["conf"] = probs.max(1)
    df["correct"] = df["true"] == df["pred"]
    df["date"] = df["Filename"].str[:8]
    df[["Filename", "true", "pred", "conf", "correct", "session", "date"]].to_csv(out_dir / f"{run}_{split}_predictions.csv", index=False)
    print(f"{run} {split}: accuracy {df.correct.mean():.3f}, {int((~df.correct).sum())} errors")

    # --- is each focus species' error tied to particular sessions / days?
    for sp in FOCUS:
        d = df[df.true == sp]
        errs = d[~d.correct]
        sess = session_table(d)
        top3 = sess.errors.head(3).sum() / max(len(errs), 1)
        top3_imgs = sess.loc[sess.errors.head(3).index, "n"].sum() / len(d)
        print(f"\n{sp}: {len(errs)}/{len(d)} wrong; predicted as:")
        print(errs.pred.value_counts().head(4).to_string())
        print(f"  {len(sess)} sessions, {errs.session.nunique()} with an error; the 3 worst sessions hold "
              f"{top3:.0%} of errors but {top3_imgs:.0%} of images")
        print(f"  worst days (error rate): " + ", ".join(
            f"{k} {v:.0%}" for k, v in (d.groupby('date').correct.apply(lambda c: 1 - c.mean())
                                         .sort_values(ascending=False).head(3)).items()))
        sess.to_csv(out_dir / f"{run}_{split}_sessions_{sp.replace(' ', '_')}.csv")
    print("\noverall error rate by session, worst 5:")
    print(session_table(df).sort_values("error_rate", ascending=False).query("n >= 5").head(5).to_string())

    # --- Grad-CAM for the most confident misclassifications of each focus species
    layer = cam_layer(model)
    from src.features.augment import get_eval_transforms
    show = get_eval_transforms(config, normalise=False, root=ROOT)
    from PIL import Image
    for sp in FOCUS:
        errs = df[(df.true == sp) & ~df.correct].sort_values("conf", ascending=False).head(n_cam)
        if errs.empty:
            continue
        fig, axes = plt.subplots(2, len(errs), figsize=(2.6 * len(errs), 5.4), squeeze=False)
        for j, (i, row) in enumerate(errs.iterrows()):
            x = ds[i][0].unsqueeze(0).to(device)
            cam = grad_cam(model, layer, x, classes.index(row.pred))
            with Image.open(ROOT / config["data"]["images_dir"] / row.Filename) as im:
                rgb = show(im.convert("RGB")).permute(1, 2, 0).numpy()
            axes[0, j].imshow(rgb); axes[1, j].imshow(rgb); axes[1, j].imshow(cam, cmap="jet", alpha=0.45)
            axes[0, j].set_title(f"{row.pred}\n{row.conf:.2f} · s{row.session}", fontsize=8)
            for a in axes[:, j]:
                a.axis("off")
        fig.suptitle(f"{run}: true {sp}, Grad-CAM for the predicted (wrong) class", fontsize=10)
        fig.tight_layout()
        path = out_dir / f"{run}_gradcam_{sp.replace(' ', '_')}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"saved {path.relative_to(ROOT)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--n-cam", type=int, default=6)
    a = ap.parse_args()
    main(a.config, a.seed, a.split, a.n_cam)
