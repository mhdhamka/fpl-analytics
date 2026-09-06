import os

# Base Directory of the project
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# API Configuration
FPL_API_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"

# Directory Paths
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
FIGURES_DIR = os.path.join(BASE_DIR, "outputs", "figures")

# Raw Data File Paths
PLAYERS_RAW_PATH = os.path.join(RAW_DATA_DIR, "players_raw.csv")
TEAMS_RAW_PATH = os.path.join(RAW_DATA_DIR, "teams_raw.csv")
POSITIONS_RAW_PATH = os.path.join(RAW_DATA_DIR, "positions_raw.csv")

# Processed Data File Paths
PLAYERS_CLEANED_PATH = os.path.join(PROCESSED_DATA_DIR, "players_cleaned.csv")

# Output Figure File Paths
TOP_SCORERS_FIG_PATH = os.path.join(FIGURES_DIR, "top_goalscorers.png")
XG_VS_ACTUAL_FIG_PATH = os.path.join(FIGURES_DIR, "xg_vs_actual_goals.png")

# Machine Learning Configuration
ML_TARGET = "total_points"
ML_FEATURES = [
    "minutes", 
    "goals_scored", 
    "assists", 
    "clean_sheets", 
    "expected_goals", 
    "expected_assists", 
    "market_value_m"
]