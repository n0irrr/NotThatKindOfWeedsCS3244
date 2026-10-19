"""Split logic: sessions never cross splits, every class is in every split,
the per-session cap holds, and seeds are reproducible. Runs on a synthetic
labels table and on the split CSVs committed in data/splits*."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.split import assign_sessions, stratified_split

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ["Chinee apple", "Lantana", "Parkinsonia", "Parthenium", "Prickly acacia",
           "Rubber vine", "Siam weed", "Snake weed", "Negative"]
FRACS = (0.6, 0.2, 0.2)
CAP = 0.25


@pytest.fixture(scope="module")
def labels():
    """~30 sessions per class of 1-10 frames each. Frames in a session are 5s
    apart, sessions are 10 min apart, so assign_sessions should recover them."""
    rng = np.random.default_rng(0)
    rows, t, true_session = [], pd.Timestamp("2016-09-28 08:00:00"), 0
    for label, species in enumerate(CLASSES):
        for _ in range(30):
            for _ in range(rng.integers(1, 11)):
                rows.append({"Filename": f"{t:%Y%m%d-%H%M%S}-{len(rows)}.jpg", "Label": label,
                             "Species": species, "true_session": true_session})
                t += pd.Timedelta(seconds=5)
            t += pd.Timedelta(minutes=10)
            true_session += 1
    # shuffle so the split can't rely on file order
    return pd.DataFrame(rows).sample(frac=1, random_state=0).reset_index(drop=True)


def check_disjoint(splits, by):
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        shared = set(splits[a][by]) & set(splits[b][by])
        assert not shared, f"{by} shared between {a} and {b}: {sorted(shared)[:5]}"


def check_all_classes(splits):
    for name, df in splits.items():
        missing = set(CLASSES) - set(df["Species"])
        assert not missing, f"{name} is missing classes {missing}"


def check_session_cap(splits):
    """No val/test session is more than CAP of that class's val/test target."""
    totals = pd.concat(splits.values())["Species"].value_counts()
    for name, frac in [("val", FRACS[1]), ("test", FRACS[2])]:
        sizes = splits[name].groupby(["Species", "session"]).size()
        for (species, session), n in sizes.items():
            limit = CAP * frac * totals[species]
            assert n <= limit, f"{name} session {session} is {n} {species} images (limit {limit:.1f})"


def test_assign_sessions_recovers_capture_sessions(labels):
    sessions = assign_sessions(labels)
    # same partition as the true sessions, whatever the ids
    assert pd.crosstab(sessions, labels["true_session"]).astype(bool).sum(axis=1).eq(1).all()
    assert sessions.nunique() == labels["true_session"].nunique()


def test_grouped_split(labels):
    splits = stratified_split(labels, seed=42, fracs=FRACS, max_session_share=CAP)
    assert sum(len(df) for df in splits.values()) == len(labels)
    check_disjoint(splits, "Filename")
    check_disjoint(splits, "session")
    check_all_classes(splits)
    check_session_cap(splits)
    for name, frac in zip(("train", "val", "test"), FRACS):
        assert len(splits[name]) / len(labels) == pytest.approx(frac, abs=0.05)


def test_grouped_split_is_reproducible(labels):
    a = stratified_split(labels, seed=42)
    b = stratified_split(labels, seed=42)
    c = stratified_split(labels, seed=7)
    for name in a:
        assert a[name]["Filename"].tolist() == b[name]["Filename"].tolist()
    assert any(set(a[name]["Filename"]) != set(c[name]["Filename"]) for name in a), \
        "different seeds gave the same split"


def test_random_split(labels):
    splits = stratified_split(labels, seed=42, fracs=FRACS, group=False)
    assert sum(len(df) for df in splits.values()) == len(labels)
    check_disjoint(splits, "Filename")
    check_all_classes(splits)
    overall = labels["Species"].value_counts(normalize=True)
    for df in splits.values():
        props = df["Species"].value_counts(normalize=True)
        assert (props - overall).abs().max() < 0.02


@pytest.mark.parametrize("splits_dir", ["splits", "splits_random"])
def test_committed_splits(splits_dir):
    splits = {name: pd.read_csv(ROOT / "data" / splits_dir / f"{name}.csv")
              for name in ("train", "val", "test")}
    check_disjoint(splits, "Filename")
    check_all_classes(splits)
    if splits_dir == "splits":
        check_disjoint(splits, "session")
        check_session_cap(splits)
