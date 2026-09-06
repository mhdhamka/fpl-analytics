import os

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from xgboost import XGBRegressor

from src.config import ML_FEATURES, ML_TARGET, PLAYERS_CLEANED_PATH


def train_points_model(df_path=PLAYERS_CLEANED_PATH):
    """Trains and benchmarks multiple regression models (Random Forest vs XGBoost)

    using 5-Fold Cross-Validation to predict player total points.
    """
    if not os.path.exists(df_path):
        print("Processed data not found! Please run src/clean.py first.")
        return

    df = pd.read_csv(df_path)

    # Define features and target variable using config variables
    features = ML_FEATURES
    target = ML_TARGET

    # Filter and clean data for modeling
    ml_df = df[features + [target]].dropna()

    X = ml_df[features]
    y = ml_df[target]

    # Initialize models for benchmarking
    models = {
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=100, random_state=42
        ),
        "XGBoost Regressor": XGBRegressor(
            n_estimators=100, learning_rate=0.1, random_state=42
        ),
    }

    # Setup 5-Fold Cross-Validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    print("=== Advanced Machine Learning Model Benchmark ===")
    print(f"Target Variable: {target}")
    print(
        "Evaluation Strategy: 5-Fold Cross-Validation (R-squared & MSE)\n"
    )

    best_model_name = None
    best_score = -float("inf")
    trained_models = {}

    # Train-test split for final hold-out evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    for name, model in models.items():
        print(f"--- Training {name} ---")

        # Cross-validation R2 scores
        cv_r2_scores = cross_val_score(
            model, X, y, cv=kf, scoring="r2"
        )
        mean_cv_r2 = cv_r2_scores.mean()

        # Fit on training set for hold-out metrics
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        test_mse = mean_squared_error(y_test, predictions)
        test_r2 = r2_score(y_test, predictions)

        print(f"Mean CV R2 Score: {mean_cv_r2:.2f}")
        print(f"Hold-out Test R2: {test_r2:.2f}")
        print(f"Hold-out Test MSE: {test_mse:.2f}\n")

        trained_models[name] = model

        # Track the best performing model based on CV score
        if mean_cv_r2 > best_score:
            best_score = mean_cv_r2
            best_model_name = name

    print(
        f"Best Performing Model: {best_model_name} (CV R2: {best_score:.2f})"
    )

    # Display Feature Importance breakdown using the best model
    best_model = trained_models[best_model_name]
    importances = pd.DataFrame(
        {"Feature": features, "Importance": best_model.feature_importances_}
    ).sort_values(by="Importance", ascending=False)

    print("\nTop Feature Importances (from best model):")
    print(importances.to_string(index=False))

    return best_model


if __name__ == "__main__":
    train_points_model()