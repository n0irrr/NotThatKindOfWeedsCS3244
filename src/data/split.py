"""Create stratified 60/20/20 train/val/test splits with a fixed seed.

Splits are stratified on Species and grouped by capture session, so that
near-duplicate consecutive frames never land in more than one split (see
notebooks/01_eda.ipynb, section 6). No resampling is done here: val/test keep
the natural class distribution, and class imbalance is handled on train only.

Also exposes a leave-one-site-out split for testing cross-site generalisation
(an A-grade goal in the proposal).
"""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

ROOT = Path(__file__).resolve().parents[2]
LABELS_CSV = ROOT / "data" / "labels.csv"
SPLITS_DIR = ROOT / "data" / "splits"


def assign_sessions(labels: pd.DataFrame, gap_s: int = 60) -> pd.Series:
    """Session id per row: a new session starts when the gap to the previous
    capture (in time order) is >= gap_s seconds. Same definition as the EDA."""
    ts = pd.to_datetime(labels["Filename"].str[:15], format="%Y%m%d-%H%M%S")
    order = ts.sort_values(kind="stable")
    new_session = order.diff().dt.total_seconds().fillna(gap_s) >= gap_s
    return new_session.cumsum().sub(1).reindex(labels.index).rename("session")


def _group_fold(df: pd.DataFrame, n_splits: int, seed: int):
    """Hold out one fold of a StratifiedGroupKFold; returns (rest, held_out)."""
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    rest_idx, held_idx = next(sgkf.split(df, df["Species"], groups=df["session"]))
    return df.iloc[rest_idx], df.iloc[held_idx]


def stratified_split(labels: pd.DataFrame, seed: int = 42,
                     fracs=(0.6, 0.2, 0.2),
                     group: bool = True) -> dict[str, pd.DataFrame]:
    train_frac, val_frac, test_frac = fracs
    assert abs(sum(fracs) - 1.0) < 1e-9
    relative_val_frac = val_frac / (train_frac + val_frac)

    if group:
        # Fold-based, so fractions must be 1/k: 0.2 of all -> 5 folds,
        # then 0.2 / 0.8 = 0.25 of the remainder -> 4 folds.
        if "session" not in labels:
            labels = labels.assign(session=assign_sessions(labels))
        train_val, test = _group_fold(labels, round(1 / test_frac), seed)
        train, val = _group_fold(train_val, round(1 / relative_val_frac), seed)
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


def leave_one_site_out(labels: pd.DataFrame, held_out_site: str) -> dict[str, pd.DataFrame]:
    raise NotImplementedError


def save_splits(splits: dict[str, pd.DataFrame], out_dir: Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in splits.items():
        df.sort_index().to_csv(out_dir / f"{name}.csv", index=False)


def summarise(splits: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Per-class counts and proportions for each split."""
    counts = pd.DataFrame({k: v["Species"].value_counts() for k, v in splits.items()})
    props = counts / counts.sum()
    return pd.concat({"count": counts, "prop": props.round(3)}, axis=1)


if __name__ == "__main__":
    labels = pd.read_csv(LABELS_CSV)
    splits = stratified_split(labels)

    sizes = {k: len(v) for k, v in splits.items()}
    print("sizes:", sizes, "total:", sum(sizes.values()))
    print(summarise(splits))

    # sanity checks: no image or session shared between splits
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        assert not set(splits[a]["Filename"]) & set(splits[b]["Filename"])
        assert not set(splits[a]["session"]) & set(splits[b]["session"])
    assert sum(sizes.values()) == len(labels)

    save_splits(splits, SPLITS_DIR)
    print(f"saved to {SPLITS_DIR}")
