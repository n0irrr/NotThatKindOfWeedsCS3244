"""Create stratified 60/20/20 train/val/test splits with a fixed seed.

Splits are stratified on Species and grouped by capture session, so that
near-duplicate consecutive frames never land in more than one split (see
notebooks/01_eda.ipynb, section 6). No resampling is done here: val/test keep
the natural class distribution, and class imbalance is handled on train only.

A few sessions are huge (one Parkinsonia session is 262 images, more than a
whole 20% share of that class), so a session may only go to val/test if it
makes up at most `max_session_share` of that class's val/test target. Bigger
sessions always go to train. Without this cap a class's whole val or test set
can be a single photo shoot.

The public release has no per-image site label, so leave-one-site-out is not
possible; the session-grouped split is our stand-in. `group=False` gives a
plain random stratified split (the paper's protocol) for comparison.

    python -m src.data.split               # session-grouped -> data/splits/
    python -m src.data.split --no-group    # random          -> data/splits_random/
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
LABELS_CSV = ROOT / "data" / "labels.csv"
SPLITS_DIR = ROOT / "data" / "splits"
RANDOM_SPLITS_DIR = ROOT / "data" / "splits_random"


def assign_sessions(labels: pd.DataFrame, gap_s: int = 60) -> pd.Series:
    """Session id per row: a new session starts when the gap to the previous
    capture (in time order) is >= gap_s seconds. Same definition as the EDA."""
    ts = pd.to_datetime(labels["Filename"].str[:15], format="%Y%m%d-%H%M%S")
    order = ts.sort_values(kind="stable")
    new_session = order.diff().dt.total_seconds().fillna(gap_s) >= gap_s
    return new_session.cumsum().sub(1).reindex(labels.index).rename("session")


def _grouped_split(labels: pd.DataFrame, seed: int, fracs,
                   max_session_share: float, jitter: float = 0.5):
    """Greedily assign whole sessions to train/val/test, largest first.

    Each session goes to the split furthest below its per-class target count.
    The order is largest-first with random jitter, so sizes balance out while
    different seeds still give different splits. (sklearn's
    StratifiedGroupKFold was used before, but its assignment is deterministic,
    so every seed produced the same split.)
    """
    counts = pd.crosstab(labels["session"], labels["Species"])   # sessions x classes
    n = counts.to_numpy(float)
    targets = np.outer(fracs, n.sum(axis=0))                     # splits x classes
    filled = np.zeros_like(targets)
    rng = np.random.default_rng(seed)
    order = np.argsort(-(np.log(n.sum(axis=1)) + jitter * rng.gumbel(size=len(n))))

    split_idx = np.empty(len(n), dtype=int)
    for i in order:
        # train (0) always allowed; val/test only if this session is a small part of it
        allowed = [0] + [s for s in (1, 2) if np.all(n[i] <= max_session_share * targets[s])]
        deficit = ((targets - filled) / targets * n[i]).sum(axis=1)
        split_idx[i] = max(allowed, key=lambda s: deficit[s])
        filled[split_idx[i]] += n[i]

    which = labels["session"].map(pd.Series(split_idx, index=counts.index))
    return [labels[which == s] for s in range(3)]


def stratified_split(labels: pd.DataFrame, seed: int = 42,
                     fracs=(0.6, 0.2, 0.2),
                     group: bool = True,
                     max_session_share: float = 0.25) -> dict[str, pd.DataFrame]:
    train_frac, val_frac, test_frac = fracs
    assert abs(sum(fracs) - 1.0) < 1e-9
    relative_val_frac = val_frac / (train_frac + val_frac)

    if group:
        if "session" not in labels:
            labels = labels.assign(session=assign_sessions(labels))
        train, val, test = _grouped_split(labels, seed, fracs, max_session_share)
    else:
        # first split off the test set
        train_val, test = train_test_split(
            labels,
            test_size = test_frac,
            stratify = labels["Species"],
            random_state = seed,
        )
        # split train_val into train and val
        train, val = train_test_split(
            train_val,
            test_size = relative_val_frac,
            stratify = train_val["Species"],
            random_state = seed,
        )

    return {"train" : train, "val" : val, "test" : test}


def save_splits(splits: dict[str, pd.DataFrame], out_dir: Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in splits.items():
        df.sort_index().to_csv(out_dir / f"{name}.csv", index=False)


def summarise(splits: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Per-class counts and proportions for each split, plus (for held-out
    splits) how many sessions each class spans and its largest session's share."""
    counts = pd.DataFrame({k: v["Species"].value_counts() for k, v in splits.items()})
    props = counts / counts.sum()
    out = {"count": counts, "prop": props.round(3)}
    if "session" in splits["val"]:
        held = {k: v for k, v in splits.items() if k != "train"}
        out["sessions"] = pd.DataFrame(
            {k: v.groupby("Species")["session"].nunique() for k, v in held.items()})
        out["largest_session"] = pd.DataFrame(
            {k: v.groupby("Species")["session"].agg(lambda s: s.value_counts().iloc[0] / len(s))
             for k, v in held.items()}).round(2)
    return pd.concat(out, axis=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-group", action="store_true",
                        help="random stratified split (paper protocol) instead of session-grouped")
    args = parser.parse_args()

    labels = pd.read_csv(LABELS_CSV)
    labels = labels.assign(session=assign_sessions(labels))
    splits = stratified_split(labels, seed=args.seed, group=not args.no_group)

    sizes = {k: len(v) for k, v in splits.items()}
    print("sizes:", sizes, "total:", sum(sizes.values()))
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)
    print(summarise(splits))

    # sanity checks: no image shared between splits, and (grouped) no session either
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        assert not set(splits[a]["Filename"]) & set(splits[b]["Filename"])
        if not args.no_group:
            assert not set(splits[a]["session"]) & set(splits[b]["session"])
    assert sum(sizes.values()) == len(labels)

    out_dir = RANDOM_SPLITS_DIR if args.no_group else SPLITS_DIR
    save_splits(splits, out_dir)
    print(f"saved to {out_dir}")
