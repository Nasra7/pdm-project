"""Tests for the drift-detection module.

The two properties that matter: no drift is reported when distributions match
(no false alarms), and drift IS reported when a feature is clearly shifted.
"""
import numpy as np
import pandas as pd

from pdm import drift


def _frame(means, n=1500, sd=10.0, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({f: rng.normal(m, sd, n) for f, m in means.items()})


def test_psi_near_zero_for_identical_distributions():
    rng = np.random.default_rng(1)
    a = rng.normal(0, 1, 3000)
    b = rng.normal(0, 1, 3000)
    assert drift.population_stability_index(a, b) < 0.1


def test_psi_large_for_shifted_distribution():
    rng = np.random.default_rng(2)
    a = rng.normal(0, 1, 3000)
    b = rng.normal(3, 1, 3000)     # big mean shift
    assert drift.population_stability_index(a, b) >= 0.2


def test_ks_detects_shift():
    rng = np.random.default_rng(3)
    a = rng.normal(0, 1, 2000)
    b = rng.normal(1.5, 1, 2000)
    stat, p = drift.ks_test(a, b)
    assert p < 0.05                 # shift is statistically significant


def test_report_flags_only_drifted_features():
    feats = ["f0", "f1", "f2"]
    ref = _frame({"f0": 100, "f1": 200, "f2": 300}, seed=10)
    # f1 shifted a lot; f0 and f2 unchanged
    prod = _frame({"f0": 100, "f1": 260, "f2": 300}, seed=11)
    report = drift.drift_report(ref, prod, feats)
    drifted = set(report.loc[report["drifted"], "feature"])
    assert "f1" in drifted
    assert "f0" not in drifted and "f2" not in drifted


def test_summarize_reports_no_drift_on_control():
    feats = ["f0", "f1"]
    ref = _frame({"f0": 50, "f1": 75}, seed=20)
    prod = _frame({"f0": 50, "f1": 75}, seed=21)   # same distribution
    summary = drift.summarize(drift.drift_report(ref, prod, feats))
    assert summary["any_drift"] is False
