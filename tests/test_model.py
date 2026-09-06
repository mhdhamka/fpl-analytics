import json
import os

import numpy as np
import pandas as pd
import pytest

import src.model as model_module
from src.model import ALL_MODEL_FEATURES, load_feature_importances, train_points_model


def _make_synthetic_dataset(n=200, seed=0):
    rng = np.random.default_rng(seed)
    minutes = rng.integers(90, 3400, n)
    goals = rng.poisson(minutes / 900)
    assists = rng.poisson(minutes / 1200)
    df = pd.DataFrame(
        {
            "minutes": minutes,
            "goals_scored": goals,
            "assists": assists,
            "clean_sheets": rng.integers(0, 15, n),
            "expected_goals": goals + rng.normal(0, 0.5, n),
            "expected_assists": assists + rng.normal(0, 0.5, n),
            "market_value_m": rng.uniform(4, 14, n),
            "goals_per_90": goals / (minutes / 90),
            "assists_per_90": assists / (minutes / 90),
            "xg_per_90": (goals + rng.normal(0, 0.5, n)) / (minutes / 90),
            "xa_per_90": (assists + rng.normal(0, 0.5, n)) / (minutes / 90),
            "points_per_million": rng.uniform(1, 20, n),
            "minutes_share": np.clip(minutes / (38 * 90), 0, 1),
            "position": rng.choice(["Forward", "Midfielder", "Defender", "Goalkeeper"], n),
        }
    )
    # Total points roughly driven by minutes + goals + assists, plus noise —
    # enough signal for a model to learn something above baseline.
    df["total_points"] = (
        minutes / 90 * 2 + goals * 5 + assists * 3 + rng.normal(0, 3, n)
    ).clip(min=0)
    return df


@pytest.fixture(autouse=True)
def _redirect_model_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(model_module, "MODEL_DIR", str(tmp_path))
    monkeypatch.setattr(model_module, "MODEL_PATH", str(tmp_path / "points_model.joblib"))
    monkeypatch.setattr(model_module, "MODEL_METADATA_PATH", str(tmp_path / "metadata.json"))


def test_train_points_model_persists_a_working_model(tmp_path):
    df = _make_synthetic_dataset()
    df_path = tmp_path / "players_cleaned.csv"
    df.to_csv(df_path, index=False)

    best_model = train_points_model(df_path=str(df_path), n_iter=3)

    assert best_model is not None
    assert os.path.exists(model_module.MODEL_PATH)
    assert os.path.exists(model_module.MODEL_METADATA_PATH)

    with open(model_module.MODEL_METADATA_PATH) as f:
        metadata = json.load(f)
    assert "test_r2" in metadata
    assert metadata["features"] == ALL_MODEL_FEATURES

    # Model should generalize better than a naive "predict the mean" baseline.
    assert metadata["test_r2"] > 0


def test_train_points_model_returns_none_when_data_missing(tmp_path):
    result = train_points_model(df_path=str(tmp_path / "does_not_exist.csv"), n_iter=2)
    assert result is None


def test_load_feature_importances_returns_all_features(tmp_path):
    df = _make_synthetic_dataset()
    df_path = tmp_path / "players_cleaned.csv"
    df.to_csv(df_path, index=False)

    best_model = train_points_model(df_path=str(df_path), n_iter=3)
    importances = load_feature_importances(best_model)

    assert set(importances["feature"]) >= {"minutes", "goals_scored"}
    assert (importances["importance"] >= 0).all()
    assert abs(importances["importance"].sum() - 1.0) < 1e-6
