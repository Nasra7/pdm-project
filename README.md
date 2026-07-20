# Explainable Predictive Maintenance on Turbofan Engines

![tests](https://github.com/Nasra7/turbofan-pdm/actions/workflows/tests.yml/badge.svg)

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
`src/pdm/signatures.py`): instead of stopping at prediction, we ask *how* engines
fail. Each failing engine's SHAP vector is a "signature" of which sensors drove
its failure. Clustering those signatures reveals distinct **failure modes** —
discovered purely from explanation structure, with no failure-mode labels.

**Built-in validation.** C-MAPSS subsets have *known* failure-mode counts —
FD001 has one, FD003 has two. Running the same method on both tests whether it
recovers ground truth: little structure in FD001 (one mode), stronger 2-cluster
structure in FD003 (two modes). A method that matches the known counts is
demonstrably finding real physics, not inventing patterns. The `choose_k` logic
reports an honest verdict (`strong` / `weak` / `no_structure`) rather than
always claiming clusters exist — silhouette scores are noisy at small N, and
pretending otherwise would be the easy mistake.

---

## Data drift monitoring (operating the model, not just building it)

A deployed model doesn't crash when the world changes — it **silently gets
worse**. The monitoring layer (`src/pdm/drift.py`,
`notebooks/03_drift_monitoring.ipynb`) watches the *input* distribution and warns
when incoming data stops resembling the training data, before failure labels are
even available.

- **Hand-built detectors** (PSI + KS from scratch) to show the mechanics, plus
  **Evidently** (the industry-standard tool) run alongside for comparison.
- **Validated against a controlled experiment:** FD001-vs-FD001 (no drift —
  control) vs FD001-vs-FD002 (real drift, since FD002 runs under different
  operating conditions). The detector correctly stays quiet on the control and
  fires on the treatment.
- **Tied to real impact:** on the drifted data, failure-class **recall collapsed
  from 0.96 to 0.17** (F1 from 0.83 to 0.28) — the model missed ~83% of the
  failures it caught in-distribution, while throwing no errors. This proves the
  drift alarm predicts severe, concrete performance loss, not just a statistical
  curiosity. (Which metric breaks — recall here, precision elsewhere — depends on
  how the shift hits the model, so F1 is tracked rather than a single metric.)
  - **A practical lesson from the control:** the KS test flagged trivial
  differences as "significant" on large samples, while PSI correctly stayed low —
  so PSI's magnitude threshold is the more trustworthy signal in production.

## Continuous integration

A GitHub Actions workflow (`.github/workflows/tests.yml`) runs the full `pytest`
suite on every push — the badge at the top reflects its status.

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
│   ├── drift.py        # data-drift monitoring (hand-built PSI/KS + Evidently)
│   └── pipeline.py     # end-to-end entry point
├── tests/
│   ├── test_data.py        # leakage & labeling correctness tests
│   ├── test_signatures.py  # signature-building & honest-k tests
│   └── test_drift.py       # drift-detection correctness tests
├── notebooks/
│   ├── 01_exploration_and_shap.ipynb   # EDA, model, global+local SHAP
│   ├── 02_failure_signatures.ipynb     # failure-mode discovery + validation
│   └── 03_drift_monitoring.ipynb       # data-drift detection + degradation proof
├── .github/workflows/
│   └── tests.yml       # CI: runs pytest on every push
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

## Possible extensions

- Reframe as regression (predict exact RUL) or survival analysis (time-to-event)
- Add a Streamlit demo for interactive per-engine explanations
- Hyperparameter tuning with time-aware cross-validation
- Extend to FD002–FD004 (multiple operating conditions)

---

*Built as a hands-on study of explainable ML for predictive maintenance.
Results reflect clean simulated data; real-world sensor data is noisier and
scores are typically lower — the techniques transfer, the easy numbers don't.*
