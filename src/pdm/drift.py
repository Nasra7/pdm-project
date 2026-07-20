"""Data-drift detection for the predictive-maintenance model.

Two complementary detectors, implemented from scratch to show the mechanics:

  * PSI (Population Stability Index) -- how MUCH a feature's distribution moved.
    Interpretable magnitude with industry-standard thresholds.
  * KS (Kolmogorov-Smirnov) -- WHETHER the move is statistically real (p-value).

Rule of thumb (PSI):  <0.1 stable | 0.1-0.2 moderate | >=0.2 significant.

Why data drift matters: a model doesn't crash when the world changes -- it
silently gets worse. Monitoring the INPUT distribution catches this early,
before ground-truth labels (failures) are even available.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

# Industry-standard PSI thresholds
PSI_STABLE = 0.10
PSI_SIGNIFICANT = 0.20


def population_stability_index(reference, production, bins=10):
    """PSI between a reference and production sample of ONE feature.

    Bins are defined on the REFERENCE distribution (quantile edges), then both
    samples are histogrammed into those fixed bins. PSI sums, over bins:
        (prod_frac - ref_frac) * ln(prod_frac / ref_frac)

    A small epsilon avoids division-by-zero / log(0) in empty bins.
    """
    reference = np.asarray(reference, dtype=float)
    production = np.asarray(production, dtype=float)

    # Quantile-based bin edges from the reference (robust to skew).
    edges = np.quantile(reference, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf  # catch out-of-range production values

    ref_counts, _ = np.histogram(reference, bins=edges)
    prod_counts, _ = np.histogram(production, bins=edges)

    eps = 1e-6
    ref_frac = ref_counts / max(ref_counts.sum(), 1) + eps
    prod_frac = prod_counts / max(prod_counts.sum(), 1) + eps

    return float(np.sum((prod_frac - ref_frac) * np.log(prod_frac / ref_frac)))


def ks_test(reference, production):
    """Two-sample KS test for ONE feature. Returns (statistic, p_value).

    Small p (<0.05) => the two samples likely come from different distributions.
    """
    stat, p = stats.ks_2samp(np.asarray(reference), np.asarray(production))
    return float(stat), float(p)


def psi_label(psi):
    if psi < PSI_STABLE:
        return "stable"
    if psi < PSI_SIGNIFICANT:
        return "moderate"
    return "significant"


def drift_report(reference_df, production_df, feature_cols, bins=10):
    """Per-feature drift table combining PSI and KS.

    Returns a DataFrame (one row per feature) with psi, psi_label, ks_stat,
    ks_pvalue, and a `drifted` flag (PSI significant OR KS p<0.05).
    """
    rows = []
    for col in feature_cols:
        ref = reference_df[col].values
        prod = production_df[col].values
        psi = population_stability_index(ref, prod, bins=bins)
        ks_stat, ks_p = ks_test(ref, prod)
        rows.append({
            "feature": col,
            "psi": round(psi, 4),
            "psi_label": psi_label(psi),
            "ks_stat": round(ks_stat, 4),
            "ks_pvalue": round(ks_p, 6),
            "drifted": bool(psi >= PSI_SIGNIFICANT or ks_p < 0.05),
        })
    report = pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)
    return report


def summarize(report):
    """One-line summary of a drift report."""
    n = len(report)
    n_drift = int(report["drifted"].sum())
    return {
        "n_features": n,
        "n_drifted": n_drift,
        "drift_fraction": round(n_drift / max(n, 1), 3),
        "max_psi": float(report["psi"].max()),
        "any_drift": n_drift > 0,
    }


# --------------------------------------------------------------------------- #
# Evidently integration (the production-grade tool, alongside the hand-built    #
# core above). Kept optional so the core has no hard dependency on Evidently.   #
# --------------------------------------------------------------------------- #
def evidently_report(reference_df, production_df, feature_cols, html_path=None):
    """Run Evidently's DataDriftPreset and return a compact summary dict.

    Mirrors the hand-built detector so results can be compared. Optionally
    saves Evidently's rich interactive HTML report to `html_path` -- the kind
    of artifact a real monitoring stack would surface to a team.

    Requires `evidently` (pip install evidently). API targets Evidently 0.7.x.
    """
    from evidently import Report, Dataset, DataDefinition
    from evidently.presets import DataDriftPreset

    data_def = DataDefinition(numerical_columns=list(feature_cols))
    ref_ds = Dataset.from_pandas(reference_df[feature_cols], data_definition=data_def)
    cur_ds = Dataset.from_pandas(production_df[feature_cols], data_definition=data_def)

    report = Report([DataDriftPreset()])
    result = report.run(reference_data=ref_ds, current_data=cur_ds)

    if html_path is not None:
        result.save_html(str(html_path))

    # Pull the dataset-level "drifted columns" metric out of the result dict.
    d = result.dict()
    drifted_count, drift_share = None, None
    per_column = {}
    for m in d.get("metrics", []):
        name = m.get("metric_name", "")
        val = m.get("value", {})
        if name.startswith("DriftedColumnsCount"):
            if isinstance(val, dict):
                drifted_count = val.get("count")
                drift_share = val.get("share")
        elif name.startswith("ValueDrift(column="):
            col = name.split("column=")[1].split(",")[0]
            per_column[col] = val  # K-S p-value for that column

    return {
        "drifted_count": drifted_count,
        "drift_share": drift_share,
        "per_column_pvalues": per_column,
        "html_saved": str(html_path) if html_path else None,
    }
