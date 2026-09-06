import os

from src import config


def test_paths_are_absolute():
    for path_attr in [
        "RAW_DATA_DIR", "PROCESSED_DATA_DIR", "FIGURES_DIR", "MODEL_DIR",
        "PLAYERS_RAW_PATH", "PLAYERS_CLEANED_PATH", "MODEL_PATH",
    ]:
        assert os.path.isabs(getattr(config, path_attr))


def test_ml_features_has_no_duplicates_and_matches_base_plus_engineered():
    assert len(config.ML_FEATURES) == len(set(config.ML_FEATURES))
    assert config.ML_FEATURES == config.ML_BASE_FEATURES + config.ML_ENGINEERED_FEATURES


def test_env_override(monkeypatch):
    monkeypatch.setenv("MIN_MINUTES_THRESHOLD", "500")
    import importlib

    reloaded = importlib.reload(config)
    assert reloaded.MIN_MINUTES_THRESHOLD == 500
    # reload back to defaults for any subsequent tests importing config
    monkeypatch.delenv("MIN_MINUTES_THRESHOLD", raising=False)
    importlib.reload(config)
