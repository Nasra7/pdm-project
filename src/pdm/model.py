"""Model training, persistence, and the imbalance correction.

`scale_pos_weight` is set from the observed negative/positive ratio so the
model is penalized appropriately for missing rare failures rather than
defaulting to a lazy "always healthy" prediction.
"""
from __future__ import annotations

import joblib
from xgboost import XGBClassifier

from . import config


def compute_scale_pos_weight(y) -> float:
    """Ratio of negatives to positives — tells XGBoost to weight the rare class."""
    pos = int((y == 1).sum())
    neg = int((y == 0).sum())
    return neg / max(pos, 1)


def train(X_train, y_train) -> XGBClassifier:
    """Fit an XGBoost classifier with imbalance correction."""
    params = dict(config.XGB_PARAMS)
    params["scale_pos_weight"] = compute_scale_pos_weight(y_train)

    model = XGBClassifier(**params)
    model.fit(X_train, y_train)
    return model


def save(model, path=None):
    path = path or (config.MODEL_DIR / "xgb_pdm.joblib")
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path


def load(path=None) -> XGBClassifier:
    path = path or (config.MODEL_DIR / "xgb_pdm.joblib")
    return joblib.load(path)
