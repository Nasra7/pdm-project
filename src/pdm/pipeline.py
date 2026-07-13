"""End-to-end training pipeline: load -> label -> split -> train -> evaluate -> save.

Run with:  python -m pdm.pipeline
This single reproducible entry point is what a reviewer runs to regenerate
your entire result from raw data.
"""
from __future__ import annotations

from . import data, model, evaluate


def run():
    print("1. Loading raw data...")
    df = data.load_raw()

    print("2. Adding labels (RUL + fail_soon)...")
    df = data.add_labels(df)

    print("3. Splitting by engine (leakage-safe)...")
    train_df, test_df = data.split_by_engine(df)
    print(f"   train engines={train_df['unit'].nunique()}, "
          f"test engines={test_df['unit'].nunique()}")

    X_train, y_train = data.get_xy(train_df)
    X_test, y_test = data.get_xy(test_df)

    print("4. Training XGBoost (with imbalance correction)...")
    clf = model.train(X_train, y_train)

    print("5. Evaluating...\n")
    metrics = evaluate.evaluate(clf, X_test, y_test)

    print("\n6. Saving model...")
    path = model.save(clf)
    print(f"   saved -> {path}")

    return clf, metrics


if __name__ == "__main__":
    run()
