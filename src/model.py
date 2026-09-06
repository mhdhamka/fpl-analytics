import json
import os
from datetime import UTC, datetime

import joblib
import pandas as pd
from scipy.stats import randint, uniform
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import (
    KFold,
    RandomizedSearchCV,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

try:
    from xgboost import XGBRegressor
    _XGBOOST_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only when xgboost isn't installed
    _XGBOOST_AVAILABLE = False

from src.config import (
    HYPERPARAM_SEARCH_ITER,
    MIN_MINUTES_THRESHOLD,
    ML_FEATURES,
    ML_TARGET,
    MODEL_DIR,
    MODEL_METADATA_PATH,
    MODEL_PATH,
    N_CV_FOLDS,
    PLAYERS_CLEANED_PATH,
    RANDOM_STATE,
)
from src.logging_config import get_logger

logger = get_logger(__name__)

CATEGORICAL_FEATURES = ["position"]
ALL_MODEL_FEATURES = ML_FEATURES + CATEGORICAL_FEATURES


def _build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("position_ohe", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ],
        remainder="passthrough",
    )


def _candidate_pipelines() -> dict:
    """Return {name: (pipeline, param_distributions)} for the search."""
    rf_pipeline = Pipeline(
        [
            ("preprocess", _build_preprocessor()),
            ("model", RandomForestRegressor(random_state=RANDOM_STATE)),
        ]
    )
    rf_params = {
        "model__n_estimators": randint(100, 500),
        "model__max_depth": randint(3, 20),
        "model__min_samples_leaf": randint(1, 10),
        "model__max_features": uniform(0.3, 0.7),
    }

    candidates = {"Random Forest Regressor": (rf_pipeline, rf_params)}

    if _XGBOOST_AVAILABLE:
        xgb_pipeline = Pipeline(
            [
                ("preprocess", _build_preprocessor()),
                (
                    "model",
                    XGBRegressor(random_state=RANDOM_STATE, objective="reg:squarederror"),
                ),
            ]
        )
        xgb_params = {
            "model__n_estimators": randint(100, 500),
            "model__max_depth": randint(2, 10),
            "model__learning_rate": uniform(0.01, 0.29),
            "model__subsample": uniform(0.6, 0.4),
            "model__colsample_bytree": uniform(0.6, 0.4),
        }
        candidates["XGBoost Regressor"] = (xgb_pipeline, xgb_params)
    else:
        logger.warning("xgboost not installed — benchmarking Random Forest only.")

    return candidates


def _load_training_frame(df_path: str) -> pd.DataFrame:
    df = pd.read_csv(df_path)
    before = len(df)
    df = df[df["minutes"] >= MIN_MINUTES_THRESHOLD].copy()
    logger.info(
        "Filtered to players with >= %d minutes: %d -> %d rows.",
        MIN_MINUTES_THRESHOLD, before, len(df),
    )
    return df


def train_points_model(
    df_path: str = PLAYERS_CLEANED_PATH,
    n_iter: int = HYPERPARAM_SEARCH_ITER,
    persist: bool = True,
):
    """Tunes and benchmarks Random Forest vs XGBoost, persists the winner.

    Returns the fitted best pipeline (preprocessing + model in one object),
    so callers can call `.predict(df[ALL_MODEL_FEATURES])` directly.
    """
    if not os.path.exists(df_path):
        logger.error("Processed data not found at %s. Run `python -m src.clean` first.", df_path)
        return None

    df = _load_training_frame(df_path)
    ml_df = df[ALL_MODEL_FEATURES + [ML_TARGET]].dropna()

    X = ml_df[ALL_MODEL_FEATURES]
    y = ml_df[ML_TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    kf = KFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    logger.info("=== Model benchmark: %d-fold CV + randomized hyperparameter search ===", N_CV_FOLDS)

    results = {}
    for name, (pipeline, param_dist) in _candidate_pipelines().items():
        logger.info("Tuning %s (%d candidate configs)...", name, n_iter)
        search = RandomizedSearchCV(
            pipeline,
            param_distributions=param_dist,
            n_iter=n_iter,
            cv=kf,
            scoring="r2",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        search.fit(X_train, y_train)

        best_estimator = search.best_estimator_
        cv_r2 = search.best_score_

        predictions = best_estimator.predict(X_test)
        test_r2 = r2_score(y_test, predictions)
        test_mae = mean_absolute_error(y_test, predictions)
        test_rmse = mean_squared_error(y_test, predictions) ** 0.5

        logger.info(
            "%s -> best CV R2: %.3f | hold-out R2: %.3f | MAE: %.2f | RMSE: %.2f",
            name, cv_r2, test_r2, test_mae, test_rmse,
        )

        results[name] = {
            "estimator": best_estimator,
            "cv_r2": cv_r2,
            "test_r2": test_r2,
            "test_mae": test_mae,
            "test_rmse": test_rmse,
            "best_params": search.best_params_,
        }

    best_name = max(results, key=lambda k: results[k]["cv_r2"])
    best = results[best_name]
    logger.info("Best model: %s (CV R2: %.3f)", best_name, best["cv_r2"])

    if persist:
        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump(best["estimator"], MODEL_PATH)

        metadata = {
            "model_name": best_name,
            "trained_at_utc": datetime.now(UTC).isoformat(),
            "cv_r2": best["cv_r2"],
            "test_r2": best["test_r2"],
            "test_mae": best["test_mae"],
            "test_rmse": best["test_rmse"],
            "best_params": best["best_params"],
            "features": ALL_MODEL_FEATURES,
            "target": ML_TARGET,
            "n_training_rows": len(ml_df),
            "min_minutes_threshold": MIN_MINUTES_THRESHOLD,
            "all_candidates": {
                k: {
                    "cv_r2": v["cv_r2"],
                    "test_r2": v["test_r2"],
                    "test_mae": v["test_mae"],
                    "test_rmse": v["test_rmse"],
                }
                for k, v in results.items()
            },
        }
        with open(MODEL_METADATA_PATH, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info("Model persisted -> %s (metadata -> %s)", MODEL_PATH, MODEL_METADATA_PATH)

    return best["estimator"]


def load_feature_importances(estimator=None) -> pd.DataFrame:
    """Extract feature importances from a fitted pipeline, expanding the
    one-hot encoded position columns back into readable names.
    """
    if estimator is None:
        estimator = joblib.load(MODEL_PATH)

    preprocessor = estimator.named_steps["preprocess"]
    model = estimator.named_steps["model"]

    ohe = preprocessor.named_transformers_["position_ohe"]
    position_names = list(ohe.get_feature_names_out(CATEGORICAL_FEATURES))
    remainder_names = [f for f in ALL_MODEL_FEATURES if f not in CATEGORICAL_FEATURES]
    feature_names = position_names + remainder_names

    importances = pd.DataFrame(
        {"feature": feature_names, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)
    return importances


if __name__ == "__main__":
    train_points_model()
