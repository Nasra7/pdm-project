"""Tests for the data module.

The leakage test is the important one: it encodes the single most critical
correctness property of the whole project — no engine may appear in both
train and test.
"""
import numpy as np
import pandas as pd
import pytest

from pdm import data, config


@pytest.fixture
def toy_df():
    """Three engines with known lifetimes."""
    rows = []
    for unit, lifetime in [(1, 10), (2, 20), (3, 15)]:
        for cycle in range(1, lifetime + 1):
            row = {"unit": unit, "cycle": cycle}
            for c in config.OP_SETTING_COLS + config.SENSOR_COLS:
                row[c] = np.random.rand()
            rows.append(row)
    return pd.DataFrame(rows)


def test_rul_counts_down_to_zero(toy_df):
    labeled = data.add_labels(toy_df)
    # Engine 1 has lifetime 10, so its first row's RUL must be 9, last must be 0.
    eng1 = labeled[labeled["unit"] == 1].sort_values("cycle")
    assert eng1["RUL"].iloc[0] == 9
    assert eng1["RUL"].iloc[-1] == 0


def test_fail_soon_label_matches_window(toy_df):
    labeled = data.add_labels(toy_df)
    # Every row with RUL <= FAIL_WINDOW must be labeled 1.
    assert (labeled.loc[labeled["RUL"] <= config.FAIL_WINDOW, "fail_soon"] == 1).all()
    assert (labeled.loc[labeled["RUL"] > config.FAIL_WINDOW, "fail_soon"] == 0).all()


def test_no_engine_leakage_across_split(toy_df):
    """THE critical test: train and test engines must be disjoint."""
    labeled = data.add_labels(toy_df)
    train_df, test_df = data.split_by_engine(labeled, test_size=0.34, seed=0)
    train_units = set(train_df["unit"])
    test_units = set(test_df["unit"])
    assert train_units.isdisjoint(test_units)


def test_split_covers_all_engines(toy_df):
    labeled = data.add_labels(toy_df)
    train_df, test_df = data.split_by_engine(labeled, test_size=0.34, seed=0)
    all_units = set(train_df["unit"]) | set(test_df["unit"])
    assert all_units == set(labeled["unit"])
