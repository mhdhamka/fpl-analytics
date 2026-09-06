"""Inference helper: loads the persisted model once, predicts many times.

Both app.py and api/main.py import this instead of duplicating load/predict
logic, and neither of them trains a model at request time anymore.
"""
import json
import os

import joblib
import pandas as pd

from src.config import MODEL_METADATA_PATH, MODEL_PATH
from src.logging_config import get_logger
from src.model import ALL_MODEL_FEATURES

logger = get_logger(__name__)


class ModelNotTrainedError(RuntimeError):
    """Raised when prediction is requested before a model has been trained."""


def model_is_available() -> bool:
    return os.path.exists(MODEL_PATH)


def load_model():
    if not model_is_available():
        raise ModelNotTrainedError(
            f"No trained model found at {MODEL_PATH}. Run `python main.py` "
            "or `python -m src.model` first."
        )
    return joblib.load(MODEL_PATH)


def load_metadata() -> dict:
    if not os.path.exists(MODEL_METADATA_PATH):
        return {}
    with open(MODEL_METADATA_PATH) as f:
        return json.load(f)


def predict_points(model, player_rows: pd.DataFrame) -> pd.Series:
    """Predict total_points for one or more players.

    `player_rows` must contain all of ALL_MODEL_FEATURES as columns.
    """
    missing = [c for c in ALL_MODEL_FEATURES if c not in player_rows.columns]
    if missing:
        raise ValueError(f"Missing required feature columns for prediction: {missing}")
    return pd.Series(model.predict(player_rows[ALL_MODEL_FEATURES]), index=player_rows.index)
