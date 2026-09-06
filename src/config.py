"""
Centralized configuration.

Everything here can be overridden with an environment variable of the same
name (e.g. `FPL_API_URL=https://... python main.py`), so the pipeline can be
pointed at a mock API in tests/CI without touching code.
"""
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _env(name: str, default):
    return os.environ.get(name, default)


# --- API ------------------------------------------------------------------
FPL_API_URL = _env("FPL_API_URL", "https://fantasy.premierleague.com/api/bootstrap-static/")
API_TIMEOUT_SECONDS = float(_env("API_TIMEOUT_SECONDS", 15))
API_MAX_RETRIES = int(_env("API_MAX_RETRIES", 4))
API_BACKOFF_FACTOR = float(_env("API_BACKOFF_FACTOR", 1.5))

# --- Directories ------------------------------------------------------------
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
HISTORY_DATA_DIR = os.path.join(BASE_DIR, "data", "history")  # timestamped snapshots
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
FIGURES_DIR = os.path.join(BASE_DIR, "outputs", "figures")
MODEL_DIR = os.path.join(BASE_DIR, "models")
LOG_DIR = os.path.join(BASE_DIR, "logs")

# --- Raw data files (always "latest") --------------------------------------
PLAYERS_RAW_PATH = os.path.join(RAW_DATA_DIR, "players_raw.csv")
TEAMS_RAW_PATH = os.path.join(RAW_DATA_DIR, "teams_raw.csv")
POSITIONS_RAW_PATH = os.path.join(RAW_DATA_DIR, "positions_raw.csv")

# --- Processed data ----------------------------------------------------------
PLAYERS_CLEANED_PATH = os.path.join(PROCESSED_DATA_DIR, "players_cleaned.csv")

# --- Figures ------------------------------------------------------------------
TOP_SCORERS_FIG_PATH = os.path.join(FIGURES_DIR, "top_goalscorers.png")
XG_VS_ACTUAL_FIG_PATH = os.path.join(FIGURES_DIR, "xg_vs_actual_goals.png")
FEATURE_IMPORTANCE_FIG_PATH = os.path.join(FIGURES_DIR, "feature_importance.png")
VALUE_PICKS_FIG_PATH = os.path.join(FIGURES_DIR, "value_picks.png")

# --- Model persistence --------------------------------------------------------
MODEL_PATH = os.path.join(MODEL_DIR, "points_model.joblib")
MODEL_METADATA_PATH = os.path.join(MODEL_DIR, "metadata.json")
RANDOM_STATE = int(_env("RANDOM_STATE", 42))
N_CV_FOLDS = int(_env("N_CV_FOLDS", 5))
HYPERPARAM_SEARCH_ITER = int(_env("HYPERPARAM_SEARCH_ITER", 20))

# --- Machine learning ----------------------------------------------------------
ML_TARGET = "total_points"

# Raw numeric signals pulled straight from the FPL API / cleaned dataset.
ML_BASE_FEATURES = [
    "minutes",
    "goals_scored",
    "assists",
    "clean_sheets",
    "expected_goals",
    "expected_assists",
    "market_value_m",
]

# Engineered signals added in src/features.py (per-90 rates + value metrics).
ML_ENGINEERED_FEATURES = [
    "goals_per_90",
    "assists_per_90",
    "xg_per_90",
    "xa_per_90",
    "points_per_million",
    "minutes_share",
]

# Position is one-hot encoded at train time (see src/model.py) rather than
# listed here, since its columns depend on which positions are present.
ML_FEATURES = ML_BASE_FEATURES + ML_ENGINEERED_FEATURES

# Minimum minutes played to be considered for modeling/rankings — filters out
# small-sample noise (a player with 12 minutes and 1 goal isn't a signal).
MIN_MINUTES_THRESHOLD = int(_env("MIN_MINUTES_THRESHOLD", 90))

LOG_LEVEL = _env("LOG_LEVEL", "INFO")
