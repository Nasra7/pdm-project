"""Failure-signature discovery: do engines fail in distinct ways?

This module builds on the trained classifier and its SHAP explanations to ask a
question the base project can't: not just *whether* an engine will fail, but
*how* it fails. The approach:

    1. For each failing engine, average its SHAP vectors over the failure window
       into a single "signature" -- a fingerprint of which sensors drove its
       failure.
    2. Reduce those signatures to 2D with PCA (linear + interpretable, on
       purpose -- UMAP axes aren't interpretable, and interpretability is the
       whole point here).
    3. Cluster the signatures (KMeans), choosing k with silhouette score rather
       than guessing.
    4. Characterize each cluster: which sensors define each failure mode.

Validation idea baked into the design: run this on FD001 (ground-truth ONE
failure mode) and FD003 (ground-truth TWO failure modes). A method that
recovers ~1 cluster on FD001 and ~2 on FD003 is demonstrably finding real
structure, not inventing it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from . import config


def build_signature_matrix(shap_values, test_df, feature_cols):
    """One averaged SHAP signature per failing engine.

    Note: this works on ANY (shap_values, df) pair that are row-aligned — the
    `test_df` name is historical. `build_signatures_all_engines` below uses it
    over the whole dataset to get a larger, more stable sample for clustering.

    Args:
        shap_values: array (n_rows, n_features) of SHAP values, row-aligned
                     with `test_df`.
        test_df: DataFrame with 'unit' and 'fail_soon', index-aligned with
                 shap_values.
        feature_cols: list of feature names (columns of the signature matrix).

    Returns:
        DataFrame indexed by engine id, columns = feature_cols. Each row is the
        mean SHAP contribution across that engine's failing-window rows.
    """
    test_reset = test_df.reset_index(drop=True)
    fail_mask = test_reset["fail_soon"].values == 1

    signatures, engine_ids = [], []
    for unit in sorted(test_reset.loc[fail_mask, "unit"].unique()):
        unit_mask = (test_reset["unit"].values == unit) & fail_mask
        signatures.append(shap_values[unit_mask].mean(axis=0))
        engine_ids.append(unit)

    return pd.DataFrame(signatures, index=engine_ids, columns=feature_cols)


def build_signatures_all_engines(model, explainer, df, feature_cols):
    """Signatures over EVERY engine in `df`, not just a held-out test split.

    Why this is legitimate (no leakage): these signatures are used for
    *unsupervised* characterization of the model's explanations, not to measure
    predictive performance. We are not scoring the model on this data — we are
    describing how it explains failures — so using all engines is fair and
    simply gives clustering a larger, more stable sample (~5x the engines).

    Args:
        model: a fitted classifier (used only to align features).
        explainer: a SHAP explainer built from `model`.
        df: a labeled DataFrame (with 'unit' and 'fail_soon') for one subset.
        feature_cols: model feature columns.

    Returns:
        signature DataFrame (one row per failing engine in the whole subset).
    """
    X_all = df[feature_cols]
    shap_all = explainer.shap_values(X_all)
    return build_signature_matrix(shap_all, df, feature_cols)


def reduce_2d(sig_matrix, seed=None):
    """PCA to 2D for visualization. Returns (coords, fitted_pca, scaler).

    Signatures are standardized first so no single high-variance feature
    dominates the projection purely by scale.
    """
    seed = seed if seed is not None else config.RANDOM_SEED
    scaler = StandardScaler()
    scaled = scaler.fit_transform(sig_matrix.values)
    pca = PCA(n_components=2, random_state=seed)
    coords = pca.fit_transform(scaled)
    return coords, pca, scaler


def choose_k(sig_matrix, k_range=range(2, 7), seed=None,
             strong_threshold=0.50, margin=0.03):
    """Pick a number of clusters by silhouette score -- honestly.

    Silhouette has two well-known pitfalls we guard against here:
      * It requires k>=2, so it can never *directly* say "one cluster".
      * On small/weakly-structured data the scores are noisy and nearly flat,
        and naively taking argmax invents structure that isn't there.

    So we return a verdict, not just a number:
      - "strong"     : best score clears `strong_threshold` -> trust best_k.
      - "weak"        : best score is positive but modest -> best_k is tentative.
      - "no_structure": scores are flat/low across all k -> the signatures are
                        effectively one blob; report ~1 failure mode.

    Returns (best_k, scores_dict, verdict).

    Rule of thumb for silhouette: >0.5 strong, 0.25-0.5 weak, <0.25 negligible.
    """
    seed = seed if seed is not None else config.RANDOM_SEED
    scaler = StandardScaler()
    scaled = scaler.fit_transform(sig_matrix.values)

    scores = {}
    for k in k_range:
        if k >= len(sig_matrix):  # can't have more clusters than samples
            continue
        km = KMeans(n_clusters=k, random_state=seed, n_init=10)
        labels = km.fit_predict(scaled)
        scores[k] = silhouette_score(scaled, labels)

    if not scores:
        return None, scores, "no_structure"

    best_k = max(scores, key=scores.get)
    best_score = scores[best_k]
    spread = max(scores.values()) - min(scores.values())

    if best_score >= strong_threshold:
        verdict = "strong"
    elif best_score >= 0.25 and spread >= margin:
        verdict = "weak"
    else:
        # low and/or flat -> no real cluster separation
        verdict = "no_structure"

    return best_k, scores, verdict


def cluster(sig_matrix, k, seed=None):
    """KMeans on standardized signatures. Returns labels aligned to sig_matrix."""
    seed = seed if seed is not None else config.RANDOM_SEED
    scaler = StandardScaler()
    scaled = scaler.fit_transform(sig_matrix.values)
    km = KMeans(n_clusters=k, random_state=seed, n_init=10)
    labels = km.fit_predict(scaled)
    return pd.Series(labels, index=sig_matrix.index, name="cluster")


def characterize(sig_matrix, labels, top_n=4):
    """For each cluster, the features with the largest mean SHAP contribution.

    Returns a dict: cluster_id -> DataFrame of top features and their mean
    signed SHAP value (positive = pushed toward failure). This is what turns a
    cluster id into a physically meaningful "failure signature".
    """
    out = {}
    for c in sorted(labels.unique()):
        members = sig_matrix.loc[labels[labels == c].index]
        mean_sig = members.mean(axis=0)
        top = mean_sig.reindex(mean_sig.abs().sort_values(ascending=False).index)
        out[c] = pd.DataFrame({
            "mean_shap": top.head(top_n).round(3),
        })
        out[c].index.name = f"cluster_{c} (n={len(members)})"
    return out
