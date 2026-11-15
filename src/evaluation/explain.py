"""Explainability and error-analysis helpers for notebooks/05_error_analysis.ipynb.

    load_cnn            trained simple_cnn / resnet50 from its checkpoint
    gradcam_figure      image | Grad-CAM for the predicted class | Grad-CAM for the true class
    lime_knn_figure     LIME superpixels that push a KNN prediction (classical features)
    confident_errors    the most confident mistakes from a results/predictions CSV
    gallery_figure      grid of images titled true -> predicted (confidence)
    tagging_template    CSV for the team to tag each mistake's likely cause
    image_stats         brightness / saturation / sharpness per image, to test the
                        "lighting" and "blur" explanations against correct predictions

Everything runs on CPU: a few dozen images, not a training loop.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
CAUSES = ["look-alike species", "occlusion / blur", "lighting / shadow", "possible label noise", "other"]


# ---------------------------------------------------------------- CNNs: Grad-CAM

def load_cnn(config_path, run_name, root=ROOT):
    """(model in eval mode, config) for a trained run, weights from
    experiments/checkpoints/<run_name>.pt (best epoch by val macro F1)."""
    import torch

    from src.train import build_model, load_config

    config = load_config(config_path)
    config["run_name"] = run_name
    config["model"]["params"] = {**config["model"]["params"], "pretrained": False}  # weights come from the checkpoint
    model = build_model(config)
    state = torch.load(Path(root) / "experiments" / "checkpoints" / f"{run_name}.pt", map_location="cpu")
    model.load_state_dict(state)
    return model.eval(), config


def gradcam_layer(model, config):
    """Last convolutional stage: ResNet's layer4, or the simple CNN's last block."""
    if config["model"]["name"] == "resnet50":
        return model.layer4[-1]
    return model[len(config["model"]["params"].get("channels", [32, 64, 128, 256])) - 1]


def gradcam_figure(model, config, filenames, true_labels, images_dir=None, root=ROOT):
    """One row per image: the image, Grad-CAM for the predicted class and for the
    true class. On a mistake, the two maps show what drove the wrong answer and
    where (if anywhere) the model saw evidence of the right one."""
    import matplotlib.pyplot as plt
    import torch
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    from src.features.augment import get_eval_transforms

    classes = config["classes"]
    images_dir = Path(images_dir or Path(root) / config["data"]["images_dir"])
    to_input = get_eval_transforms(config, root=root)
    to_rgb = get_eval_transforms(config, normalise=False, root=root)
    fig, axes = plt.subplots(len(filenames), 3, figsize=(7.5, 2.6 * len(filenames)), squeeze=False)
    with GradCAM(model=model, target_layers=[gradcam_layer(model, config)]) as cam:
        for row, (f, true) in enumerate(zip(filenames, true_labels)):
            with Image.open(images_dir / f) as img:
                img = img.convert("RGB")
            x = to_input(img).unsqueeze(0)
            rgb = to_rgb(img).permute(1, 2, 0).numpy()
            with torch.no_grad():
                probs = model(x).softmax(1)[0]
            pred = int(probs.argmax())
            true_idx = classes.index(true)
            axes[row, 0].imshow(rgb)
            axes[row, 0].set_title(f"{f}\ntrue {true}", fontsize=8)
            for col, (cls, label) in enumerate([(pred, "pred"), (true_idx, "true")], start=1):
                heat = cam(input_tensor=x, targets=[ClassifierOutputTarget(cls)])[0]
                axes[row, col].imshow(show_cam_on_image(rgb, heat, use_rgb=True))
                axes[row, col].set_title(f"CAM for {label}: {classes[cls]} ({probs[cls]:.2f})", fontsize=8)
    for ax in axes.flat:
        ax.axis("off")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------- KNN: LIME

def fit_classical(config_path, root=ROOT):
    """Refit a classical run (knn / logreg) with the hyperparameters it picked on
    val (from results/metrics/<run_name>.json), on train only. Returns
    (model, config, feature kwargs)."""
    from src.train import classical_grid, load_config, load_features

    config = load_config(config_path)
    run_name = config.get("run_name", Path(config_path).stem)
    best = json.loads((Path(root) / "results" / "metrics" / f"{run_name}.json").read_text())["best_params"]
    build, params, grid = classical_grid(config)
    X_tr, y_tr = load_features(config, ("train",))["train"]
    model = build({**params, **{k: best[k] for k in grid[0]}}).fit(X_tr, y_tr)
    feat = {k: v for k, v in config["features"].items() if k != "load_size"}
    return model, config, feat


def lime_knn_figure(model, config, feat, filenames, true_labels, images_dir=None, root=ROOT,
                    num_samples=1000, num_features=6, seed=0):
    """LIME on full-size images: superpixels are switched off at random and the
    classical pipeline (resize -> features -> model) is re-run. Green regions
    support the predicted class. Shows whether e.g. a colour-histogram KNN
    keys on the plant or on soil / sky / shadow."""
    import matplotlib.pyplot as plt
    from lime import lime_image
    from skimage.segmentation import mark_boundaries

    from src.features.classical import extract

    classes = config["classes"]
    images_dir = Path(images_dir or Path(root) / config["data"]["images_dir"])
    load_size = tuple(config["features"].get("load_size", config["data"]["image_size"]))

    def classifier_fn(batch):
        small = np.stack([np.asarray(Image.fromarray(im.astype(np.uint8)).resize(load_size[::-1], Image.BILINEAR))
                          for im in batch])
        return model.predict_proba(extract(small, **feat))

    explainer = lime_image.LimeImageExplainer(random_state=seed)
    fig, axes = plt.subplots(len(filenames), 2, figsize=(5.5, 2.7 * len(filenames)), squeeze=False)
    for row, (f, true) in enumerate(zip(filenames, true_labels)):
        with Image.open(images_dir / f) as img:
            arr = np.asarray(img.convert("RGB"))
        exp = explainer.explain_instance(arr, classifier_fn, top_labels=1, hide_color=0,
                                         num_samples=num_samples, random_seed=seed)
        pred = exp.top_labels[0]
        overlay, mask = exp.get_image_and_mask(pred, positive_only=False, num_features=num_features,
                                               hide_rest=False)
        axes[row, 0].imshow(arr)
        axes[row, 0].set_title(f"{f}\ntrue {true}", fontsize=8)
        axes[row, 1].imshow(mark_boundaries(overlay / 255.0, mask))
        axes[row, 1].set_title(f"LIME for pred: {classes[pred]}\n(green supports, red opposes)", fontsize=8)
    for ax in axes.flat:
        ax.axis("off")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------- mistakes

def confident_errors(pred_csv, k=24, true=None, pred=None):
    """The k most confident mistakes in a results/predictions CSV, optionally
    restricted to one true and/or predicted class."""
    df = pd.read_csv(pred_csv)
    err = df[df["true"] != df["pred"]]
    if true is not None:
        err = err[err["true"] == true]
    if pred is not None:
        err = err[err["pred"] == pred]
    return err.sort_values("confidence", ascending=False).head(k).reset_index(drop=True)


def gallery_figure(rows, images_dir, ncols=6):
    """Grid of images titled 'true -> predicted (confidence)'."""
    import matplotlib.pyplot as plt

    nrows = max(1, int(np.ceil(len(rows) / ncols)))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.2 * ncols, 2.5 * nrows), squeeze=False)
    for ax, (_, r) in zip(axes.flat, rows.iterrows()):
        with Image.open(Path(images_dir) / r["Filename"]) as img:
            ax.imshow(img.convert("RGB"))
        ax.set_title(f"{r['true']}\n-> {r['pred']} ({r['confidence']:.2f})", fontsize=7)
    for ax in axes.flat:
        ax.axis("off")
    fig.tight_layout()
    return fig


def tagging_template(rows, out_csv):
    """CSV with an empty 'cause' column for the team to fill in (one of CAUSES)."""
    out = rows[["Filename", "true", "pred", "confidence"]].assign(cause="", notes="")
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    return out


def image_stats(filenames, images_dir):
    """Per-image brightness (mean V), saturation (mean S), contrast (std of grey)
    and sharpness (variance of the Laplacian of grey; low = blurry)."""
    from scipy.ndimage import laplace

    rows = []
    for f in filenames:
        with Image.open(Path(images_dir) / f) as img:
            rgb = img.convert("RGB")
            hsv = np.asarray(rgb.convert("HSV"), dtype=np.float32) / 255.0
            grey = np.asarray(rgb.convert("L"), dtype=np.float32) / 255.0
        rows.append({"Filename": f, "brightness": hsv[..., 2].mean(), "saturation": hsv[..., 1].mean(),
                     "contrast": grey.std(), "sharpness": laplace(grey).var()})
    return pd.DataFrame(rows)
