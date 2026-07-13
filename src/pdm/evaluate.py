"""Evaluation focused on the metrics that matter for imbalanced PdM.

Accuracy is deliberately reported but de-emphasized: with ~85% healthy rows,
a trivial "always healthy" model scores ~85% accuracy while catching zero
failures. Recall and precision on the FAILURE class are the real signal.
"""
from __future__ import annotations

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)


def evaluate(model, X_test, y_test) -> dict:
    """Return a dict of metrics; prints a human-readable report as a side effect."""
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(
        y_test, y_pred, target_names=["Healthy", "Fail Soon"], output_dict=True
    )

    print(f"Accuracy (misleading on its own): {acc:.3f}")
    print("\nConfusion matrix [rows=actual, cols=predicted]:")
    print("               pred:Healthy  pred:FailSoon")
    print(f"  act:Healthy   {cm[0,0]:>10}   {cm[0,1]:>11}")
    print(f"  act:FailSoon  {cm[1,0]:>10}   {cm[1,1]:>11}")
    print("\nKey metrics on the FAILURE class:")
    fs = report["Fail Soon"]
    print(f"  recall    = {fs['recall']:.3f}  (fraction of real failures caught)")
    print(f"  precision = {fs['precision']:.3f}  (fraction of alarms that were real)")
    print(f"  f1        = {fs['f1-score']:.3f}")

    return {"accuracy": acc, "confusion_matrix": cm, "report": report}
