import os
import pandas as pd

def clean_data():
    """Loads raw FPL data, cleans it, performs feature engineering, and saves it."""
    # Load raw data from CSVs
    try:
        players_df = pd.read_csv("data/raw/players_raw.csv")
        teams_df = pd.read_csv("data/raw/teams_raw.csv")
        positions_df = pd.read_csv("data/raw/positions_raw.csv")
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

    # Save the processed dataset
    os.makedirs("data/processed", exist_ok=True)
    output_path = "data/processed/players_cleaned.csv"
    df_cleaned.to_csv(output_path, index=False)
    
    print(f"Cleaned dataset successfully saved to {output_path}!")
    return df_cleaned

if __name__ == "__main__":
    clean_data()