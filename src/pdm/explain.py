"""SHAP-based explainability — the core value proposition of this project.

Provides both a GLOBAL view (which sensors drive predictions overall) and a
LOCAL view (why one specific engine was flagged). The local waterfall is what
turns a black-box score into something a maintenance engineer can act on.
"""
from __future__ import annotations

import numpy as np
import shap

from . import config


def build_explainer(model):
    """TreeExplainer is exact and fast for tree ensembles like XGBoost."""
    return shap.TreeExplainer(model)


def global_summary(explainer, X, show=True):
    """Global feature-importance beeswarm plot."""
    shap_values = explainer.shap_values(X)
    if show:
        shap.summary_plot(shap_values, X, feature_names=config.get_feature_cols())
    return shap_values


def explain_one(explainer, X, idx, show=True):
    """Local waterfall explanation for a single row (one engine at one cycle)."""
    shap_values = explainer.shap_values(X)
    explanation = shap.Explanation(
        values=shap_values[idx],
        base_values=explainer.expected_value,
        data=X.iloc[idx],
        feature_names=config.get_feature_cols(),
    )
    if show:
        shap.plots.waterfall(explanation)
    return explanation


def highest_confidence_failure(model, X):
    """Index of the row the model most confidently predicts as 'fail soon'."""
    probs = model.predict_proba(X)[:, 1]
    return int(np.argmax(probs))
