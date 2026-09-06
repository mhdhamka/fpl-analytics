import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

def train_points_model(df_path="data/processed/players_cleaned.csv"):
    """Trains a machine learning model to predict player total points using performance metrics."""
    if not os.path.exists(df_path):
        print("Processed data not found! Please run src/clean.py first.")
        return

    df = pd.read_csv(df_path)
    
    # Define features and target variable
    features = [
        "minutes", "goals_scored", "assists", 
        "clean_sheets", "expected_goals", "expected_assists", "market_value_m"
    ]
    target = "total_points"
    
    # Filter and clean data for modeling
    ml_df = df[features + [target]].dropna()
    
    X = ml_df[features]
    y = ml_df[target]
    
    # Train-test split (80% training, 20% testing)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize and train Random Forest Regressor
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Predict and evaluate
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    print("=== Machine Learning Model Evaluation ===")
    print(f"Model: Random Forest Regressor")
    print(f"Target Variable: {target}")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"R-squared (R2) Score: {r2:.2f}")
    
    # Display Feature Importance breakdown
    importances = pd.DataFrame({
        "Feature": features,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False)
    
    print("\nFeature Importances:")
    print(importances.to_string(index=False))
    
    return model

if __name__ == "__main__":
    train_points_model()