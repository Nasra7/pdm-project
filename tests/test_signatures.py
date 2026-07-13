"""Tests for the failure-signatures module.

The key properties: signatures are built only from failing rows, one row per
engine; and choose_k returns an honest verdict rather than always claiming
structure.
"""
import numpy as np
import pandas as pd

from pdm import signatures


def _fake_inputs(n_engines=8, n_features=5, seed=0):
    """Build a fake test_df + shap_values with a clear 2-group structure."""
    rng = np.random.default_rng(seed)
    rows, shap_rows = [], []
    for unit in range(1, n_engines + 1):
        group = unit % 2  # two groups
        for cycle in range(5):
            row = {"unit": unit, "fail_soon": 1}  # all failing for test
            base = np.zeros(n_features)
            # group 0 driven by feature 0, group 1 by feature 1
            base[group] = 5.0 + rng.normal(0, 0.1)
            # include feature columns too, so all-engines builder can slice them
            for i in range(n_features):
                row[f"f{i}"] = base[i]
            rows.append(row)
            shap_rows.append(base)
    test_df = pd.DataFrame(rows)
    shap_values = np.array(shap_rows)
    feature_cols = [f"f{i}" for i in range(n_features)]
    return shap_values, test_df, feature_cols


def test_signature_matrix_one_row_per_engine():
    shap_values, test_df, feature_cols = _fake_inputs(n_engines=8)
    sig = signatures.build_signature_matrix(shap_values, test_df, feature_cols)
    assert sig.shape[0] == 8            # one signature per engine
    assert list(sig.columns) == feature_cols


def test_signature_ignores_non_failing_rows():
    shap_values, test_df, feature_cols = _fake_inputs(n_engines=4)
    # Flip half the rows to non-failing; they must be excluded.
    test_df.loc[: len(test_df) // 2, "fail_soon"] = 0
    sig = signatures.build_signature_matrix(shap_values, test_df, feature_cols)
    # only engines that still have a failing row should appear
    remaining = test_df.loc[test_df["fail_soon"] == 1, "unit"].nunique()
    assert sig.shape[0] == remaining


def test_choose_k_reports_strong_structure():
    """With clean 2-group data, verdict should not be 'no_structure'."""
    shap_values, test_df, feature_cols = _fake_inputs(n_engines=12)
    sig = signatures.build_signature_matrix(shap_values, test_df, feature_cols)
    best_k, scores, verdict = signatures.choose_k(sig)
    assert best_k == 2
    assert verdict in ("strong", "weak")   # real structure detected


def test_cluster_and_characterize_align():
    shap_values, test_df, feature_cols = _fake_inputs(n_engines=8)
    sig = signatures.build_signature_matrix(shap_values, test_df, feature_cols)
    labels = signatures.cluster(sig, k=2)
    assert len(labels) == len(sig)
    char = signatures.characterize(sig, labels)
    assert set(char.keys()) == set(labels.unique())


class _FakeExplainer:
    """Minimal stand-in: returns a fixed SHAP array for any X passed."""
    def __init__(self, shap_array):
        self._shap = shap_array

    def shap_values(self, X):
        return self._shap[: len(X)]


def test_all_engines_uses_more_data_than_test_only():
    """The all-engines builder should yield one signature per failing engine
    across the whole frame, i.e. at least as many as a test-only subset."""
    shap_values, df, feature_cols = _fake_inputs(n_engines=10)
    explainer = _FakeExplainer(shap_values)
    # df here plays the role of the full dataset
    sig_all = signatures.build_signatures_all_engines(
        model=None, explainer=explainer, df=df, feature_cols=feature_cols
    )
    assert sig_all.shape[0] == df["unit"].nunique()
    assert list(sig_all.columns) == feature_cols
