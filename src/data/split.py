"""Create stratified 60/20/20 train/val/test splits with a fixed seed.

Also exposes a leave-one-site-out split for testing cross-site generalisation
(an A-grade goal in the proposal).
"""
from pathlib import Path

import pandas as pd


def stratified_split(labels: pd.DataFrame, seed: int = 42,
                     fracs=(0.6, 0.2, 0.2)) -> dict[str, pd.DataFrame]:
    raise NotImplementedError


def leave_one_site_out(labels: pd.DataFrame, held_out_site: str) -> dict[str, pd.DataFrame]:
    raise NotImplementedError


def save_splits(splits: dict[str, pd.DataFrame], out_dir: Path) -> None:
    raise NotImplementedError
