"""The Kaggle fold-0 split in data/splits_kaggle0: no image in two splits, every
image used once, labels from labels.csv, and the converter keeps Kaggle's rows."""
from pathlib import Path

import pandas as pd
import pytest

from src.data.kaggle_split import convert_fold

ROOT = Path(__file__).resolve().parents[1]
SPLIT_DIR = ROOT / "data" / "splits_kaggle0"
SIZES = {"train": 10501, "val": 3501, "test": 3507}   # rows in Kaggle's *_subset0.csv


def load_splits():
    return {n: pd.read_csv(SPLIT_DIR / f"{n}.csv") for n in SIZES}


def test_committed_kaggle_split():
    """Runs everywhere, CI included: the split CSVs are committed."""
    splits = load_splits()
    assert {n: len(d) for n, d in splits.items()} == SIZES
    files = {n: set(d["Filename"]) for n, d in splits.items()}
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        assert not files[a] & files[b], f"images shared between {a} and {b}"
    assert len(set.union(*files.values())) == sum(SIZES.values())
    for name, df in splits.items():
        assert df["Species"].nunique() == 9, f"{name} is missing a class"


@pytest.mark.skipif(not (ROOT / "data" / "labels.csv").exists(),
                    reason="data/labels.csv is gitignored (download it, see README)")
def test_kaggle_split_matches_labels_csv():
    labels = pd.read_csv(ROOT / "data" / "labels.csv").set_index("Filename")
    splits = load_splits()
    assert set.union(*(set(d["Filename"]) for d in splits.values())) == set(labels.index)
    for name, df in splits.items():
        official = labels.loc[df["Filename"]]
        assert (df["Label"].to_numpy() == official["Label"].to_numpy()).all(), name
        assert (df["Species"].to_numpy() == official["Species"].to_numpy()).all(), name


def test_convert_fold_keeps_rows_and_uses_official_labels(tmp_path):
    labels = pd.DataFrame({"Filename": ["a.jpg", "b.jpg", "c.jpg"], "Label": [0, 1, 8],
                           "Species": ["Chinee apple", "Lantana", "Negative"]})
    labels.to_csv(tmp_path / "labels.csv", index=False)
    kaggle = tmp_path / "kaggle"
    kaggle.mkdir()
    pd.DataFrame({"Filename": ["c.jpg", "b.jpg"], "Label": [8, 0]}).to_csv(kaggle / "train_subset0.csv", index=False)
    pd.DataFrame({"Filename": ["a.jpg"], "Label": [0]}).to_csv(kaggle / "val_subset0.csv", index=False)
    pd.DataFrame({"Filename": [], "Label": []}).to_csv(kaggle / "test_subset0.csv", index=False)

    out = convert_fold(kaggle, 0, tmp_path / "labels.csv", tmp_path / "out")

    train = pd.read_csv(tmp_path / "out" / "train.csv")
    assert train["Filename"].tolist() == ["c.jpg", "b.jpg"]          # Kaggle's rows, Kaggle's order
    assert train["Label"].tolist() == [8, 1]                         # b.jpg: labels.csv wins over the fold file
    assert train["Species"].tolist() == ["Negative", "Lantana"]
    assert len(out["val"]) == 1


def test_only_the_kaggle_split_is_committed():
    """data/ holds one split, the one every result uses."""
    folders = sorted(p.name for p in (ROOT / "data").glob("splits*") if p.is_dir())
    assert folders == [SPLIT_DIR.name], f"unexpected split folders: {folders}"
