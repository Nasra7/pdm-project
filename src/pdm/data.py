"""Data loading, labeling, and leakage-safe splitting.

The single most important idea in this module is `split_by_engine`: we split
at the ENGINE level, never the row level, so that no engine appears in both
train and test. A naive random row split would leak each engine's trajectory
across the split and produce optimistic, meaningless scores.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def load_raw(path=None, subset=None) -> pd.DataFrame:
    """Load a raw C-MAPSS training file into a DataFrame.

    The files are whitespace-separated with no header and trailing spaces that
    create phantom columns; `sep=r"\\s+"` with explicit names handles this.

    Args:
        path: explicit path to a train_*.txt file (overrides `subset`).
        subset: e.g. "FD001" or "FD003"; resolved against the data directory.
                Lets the same code load different C-MAPSS subsets, which is
                what enables the single-mode vs two-mode comparison.
    """
    if path is None:
        if subset is not None:
            path = config.DATA_DIR / f"train_{subset}.txt"
        else:
            path = config.RAW_TRAIN_FILE
    df = pd.read_csv(path, sep=r"\s+", header=None, names=config.COL_NAMES)
    return df


def add_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Add RUL (remaining useful life) and the binary `fail_soon` label.

    Because every training engine runs to failure, RUL for any row is simply
    (that engine's max cycle) - (current cycle).
    """
    df = df.copy()
    max_cycle = df.groupby("unit")["cycle"].transform("max")
    df["RUL"] = max_cycle - df["cycle"]
    df["fail_soon"] = (df["RUL"] <= config.FAIL_WINDOW).astype(int)
    return df


def split_by_engine(df: pd.DataFrame, test_size=None, seed=None):
    """Split into train/test by ENGINE id to prevent data leakage.

    Returns (train_df, test_df). Entire engines are assigned to one side only.
    """
    test_size = test_size if test_size is not None else config.TEST_SIZE
    seed = seed if seed is not None else config.RANDOM_SEED

    unit_ids = df["unit"].unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(unit_ids)

    n_train = int(len(unit_ids) * (1 - test_size))
    train_units, test_units = unit_ids[:n_train], unit_ids[n_train:]

    train_df = df[df["unit"].isin(train_units)].reset_index(drop=True)
    test_df = df[df["unit"].isin(test_units)].reset_index(drop=True)
    return train_df, test_df


def get_xy(df: pd.DataFrame):
    """Split a labeled DataFrame into (X, y) using the configured features."""
    feature_cols = config.get_feature_cols()
    return df[feature_cols], df["fail_soon"]
