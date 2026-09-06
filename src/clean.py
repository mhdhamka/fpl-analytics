import os
import pandas as pd
from src.config import (
    PLAYERS_RAW_PATH, 
    TEAMS_RAW_PATH, 
    POSITIONS_RAW_PATH, 
    PLAYERS_CLEANED_PATH,
    PROCESSED_DATA_DIR
)

def clean_data():
    """Loads raw FPL data, cleans it, performs feature engineering, and saves it."""
    # Load raw data from CSVs using config paths
    try:
        players_df = pd.read_csv(PLAYERS_RAW_PATH)
        teams_df = pd.read_csv(TEAMS_RAW_PATH)
        positions_df = pd.read_csv(POSITIONS_RAW_PATH)
    except FileNotFoundError as e:
        print("Raw data files not found! Please run src/ingest.py first.")
        raise e

    # Map team IDs to actual team names (brackets placed *after* set_index)
    team_mapping = teams_df.set_index("id")["name"].to_dict()
    players_df["team_name"] = players_df["team"].map(team_mapping)

    # Map position IDs to readable names (Forward, Midfielder, etc.)
    position_mapping = positions_df.set_index("id")["singular_name"].to_dict()
    players_df["position"] = players_df["element_type"].map(position_mapping)

    # Select relevant columns for our analytics pipeline
    selected_columns = [
        "first_name", "second_name", "team_name", "position", 
        "now_cost", "minutes", "goals_scored", "assists", 
        "clean_sheets", "expected_goals", "expected_assists", "total_points"
    ]
    df_cleaned = players_df[selected_columns].copy()

    # Feature Engineering: Scale down FPL price (e.g., 100 = £10.0m)
    df_cleaned["market_value_m"] = df_cleaned["now_cost"] / 10.0

    # Ensure numeric columns are properly formatted and handle any nulls
    numeric_cols = ["minutes", "goals_scored", "assists", "clean_sheets", "expected_goals", "expected_assists", "total_points"]
    for col in numeric_cols:
        df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors="coerce").fillna(0)

    # Save the processed dataset using config paths
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    output_path = PLAYERS_CLEANED_PATH
    df_cleaned.to_csv(output_path, index=False)
    
    print(f"Cleaned dataset successfully saved to {output_path}!")
    return df_cleaned

if __name__ == "__main__":
    clean_data()