"""Create stratified 60/20/20 train/val/test splits with a fixed seed.

Also exposes a leave-one-site-out split for testing cross-site generalisation
(an A-grade goal in the proposal).
"""
from pathlib import Path
from sklearn import train_test_split
import pandas as pd


def stratified_split(labels: pd.DataFrame, seed: int = 42,
                     fracs=(0.6, 0.2, 0.2)) -> dict[str, pd.DataFrame]:
    train_frac, val_frac, test_frac = fracs
    assert abs(sum(fracs) - 1.0) < 1e-9

    # first split off the test set
    train_val, test = train_test_split(
        labels,
        test_size = test_frac,
        stratify = labels["Species"],
        random_state = seed,
    )
    # split train_val into train and val
    relative_val_fra = val_frac / (train_frac + val_frac)

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
    raise NotImplementedError
