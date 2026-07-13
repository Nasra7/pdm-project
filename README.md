# Explainable Predictive Maintenance on Turbofan Engines

Predicting engine failures **before they happen** — and explaining *why* each
prediction was made — using XGBoost and SHAP on NASA's C-MAPSS turbofan
degradation dataset.

> **The problem:** In industrial settings, an unexpected equipment failure is
> expensive and dangerous. A model that flags "this engine will fail soon" is
> only useful if a human engineer can trust and act on it. This project builds
> that model *and* opens the black box so every prediction is explainable.

---

## Results

| Metric (on failure class) | Score | What it means |
|---|---|---|
| **Recall** | ~0.91 | Catches ~91% of engines that were genuinely about to fail |
| **Precision** | ~0.77 | ~77% of raised alarms were real failures |
| **F1** | ~0.83 | Balanced score across the two |

Accuracy (~0.95) is reported but deliberately **not** the headline: with ~85%
healthy samples, a trivial "always healthy" model scores ~85% accuracy while
catching zero failures. On imbalanced maintenance data, **recall on the failure
class is what matters** — missing a failure is far costlier than a false alarm.

---

## Why this project is structured the way it is

This repo is built to demonstrate **production/MLOps practices**, not just a
notebook that happens to work:

- **Leakage-safe splitting.** Data is split *by engine*, never by row, so no
  engine appears in both train and test. A naive random split would leak each
  engine's trajectory and inflate scores meaninglessly. This property is
  enforced by an automated test.
- **Config-driven.** Every tunable parameter lives in `config.py` — no magic
  numbers scattered across scripts.
- **Modular package.** Load / label / split / train / evaluate / explain are
  separate, importable, testable modules.
- **Tested.** The critical correctness properties (no leakage, correct labels)
  are covered by `pytest`.
- **Reproducible.** One command regenerates the entire result from raw data.

## Explainability (the core value)

The project produces two kinds of SHAP explanation:

- **Global** — which sensors drive failure predictions across all engines.
- **Local** — a per-engine waterfall showing exactly why *one specific engine*
  was flagged. This is what lets a maintenance engineer act:
  *"Engine 34 is flagged because sensors 11 and 14 show late-stage degradation."*

### Failure-signature discovery (going beyond standard SHAP)

The standout analysis (`notebooks/02_failure_signatures.ipynb`,
`src/pdm/signatures.py`): instead of stopping at prediction, I ask *how* engines
fail. Each failing engine's SHAP vector is a "signature" of which sensors drove
its failure. Clustering those signatures probes whether the model's reasoning
contains distinct **failure modes** — discovered purely from explanation
structure, with no failure-mode labels.

**Validation against known ground truth.** C-MAPSS subsets have *known*
failure-mode counts — FD001 has one, FD003 has two. Running the same method on
both tests whether it recovers that structure. The honest result: silhouette
scores were "weak" for both subsets (exact cluster counts are unreliable at this
sample size), **but the silhouette profiles discriminated correctly in
direction** — FD001 showed a monotonic decline from k=2 (single-mode
signature), while FD003 showed an interior peak at k=3 with a higher best score
(multi-mode signature). Forcing k=2 on FD003 produced two physically distinct
signatures (sensor_11/sensor_4 vs sensor_12/sensor_9/sensor_14).

**Conclusion:** the method distinguishes single-mode from multi-mode datasets by
their explanation structure, though it cannot pin exact mode counts at this
scale — a bounded finding validated against ground truth rather than tuned for a
clean number. The `choose_k` logic reports an honest verdict
(`strong` / `weak` / `no_structure`) rather than always claiming clusters exist,
since silhouette is noisy at small N.

---

## Project structure

```
pdm-project/
├── src/pdm/
│   ├── config.py       # all parameters & schema in one place
│   ├── data.py         # loading, labeling, leakage-safe split
│   ├── model.py        # training, imbalance correction, persistence
│   ├── evaluate.py     # imbalance-aware metrics
│   ├── explain.py      # SHAP global + local explanations
│   ├── signatures.py   # failure-signature discovery (clustering SHAP vectors)
│   └── pipeline.py     # end-to-end entry point
├── tests/
│   ├── test_data.py        # leakage & labeling correctness tests
│   └── test_signatures.py  # signature-building & honest-k tests
├── notebooks/
│   ├── 01_exploration_and_shap.ipynb   # EDA, model, global+local SHAP
│   └── 02_failure_signatures.ipynb     # failure-mode discovery + validation
├── requirements.txt
└── pyproject.toml
```

## Quickstart

```bash
# 1. Install
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# 2. Add data
# Download C-MAPSS FD001 and place train_FD001.txt in data/

# 3. Run the full pipeline
cd src && python -m pdm.pipeline

# 4. Run tests
pytest
```

## Dataset

NASA C-MAPSS Turbofan Engine Degradation Simulation (subset FD001): 100 engines
run from healthy operation to failure, with 21 sensor readings per cycle. The
prediction target is a binary label: *will this engine fail within 30 cycles?*

## Getting the data

This repo does not include the raw data (it belongs to NASA and is excluded via
`.gitignore`). To run the pipeline, download the C-MAPSS Turbofan Engine
Degradation dataset and place the required files in `data/`:

- `data/train_FD001.txt`
- `data/train_FD003.txt`

The dataset is available from NASA's Prognostics Data Repository (search "NASA
C-MAPSS Turbofan Engine Degradation") or its Kaggle mirror (search "NASA
Turbofan Engine Degradation Simulation").

## Possible extensions

- Reframe as regression (predict exact RUL) or survival analysis (time-to-event)
- Add a Streamlit demo for interactive per-engine explanations
- Hyperparameter tuning with time-aware cross-validation
- Extend to FD002–FD004 (multiple operating conditions)

---

*Built as a hands-on study of explainable ML for predictive maintenance.
Results reflect clean simulated data; real-world sensor data is noisier and
scores are typically lower — the techniques transfer, the easy numbers don't.*
