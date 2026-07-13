"""Central configuration for the predictive maintenance pipeline.

Keeping all tunable parameters in one place (rather than scattered as magic
numbers across scripts) is a basic MLOps hygiene practice: it makes runs
reproducible, makes experiments easy to track, and means a reviewer can see
every choice that shapes the model in a single file.
"""
from pathlib import Path

# --- Paths ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
RAW_TRAIN_FILE = DATA_DIR / "train_FD001.txt"

# --- Data schema ---
OP_SETTING_COLS = [f"op_setting_{i}" for i in range(1, 4)]
SENSOR_COLS = [f"sensor_{i}" for i in range(1, 22)]
COL_NAMES = ["unit", "cycle"] + OP_SETTING_COLS + SENSOR_COLS

# Sensors with zero / near-zero variance in FD001 (no predictive signal).
# Confirmed empirically via standard-deviation inspection, not assumed.
DROP_SENSORS = [
    "sensor_1", "sensor_5", "sensor_6", "sensor_10",
    "sensor_16", "sensor_18", "sensor_19",
]

# --- Problem framing ---
# Binary label: will the engine fail within FAIL_WINDOW cycles?
FAIL_WINDOW = 30

# --- Split ---
TEST_SIZE = 0.2          # fraction of ENGINES (not rows) held out
RANDOM_SEED = 42

# --- Model hyperparameters ---
XGB_PARAMS = {
    "n_estimators": 200,
    "max_depth": 4,
    "learning_rate": 0.1,
    "eval_metric": "logloss",
    "random_state": RANDOM_SEED,
    # scale_pos_weight is computed at train time from the actual class balance
}


def get_feature_cols() -> list[str]:
    """Return the final model feature list (surviving sensors + op settings)."""
    surviving = [c for c in SENSOR_COLS if c not in DROP_SENSORS]
    return surviving + OP_SETTING_COLS
