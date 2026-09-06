import os
import requests
import pandas as pd
from src.config import (
    FPL_API_URL, 
    RAW_DATA_DIR, 
    PLAYERS_RAW_PATH, 
    TEAMS_RAW_PATH, 
    POSITIONS_RAW_PATH
)

def fetch_fpl_data():
    """Fetches static data from the Fantasy Premier League API."""
    url = FPL_API_URL
    response = requests.get(url)
    
    if response.status_code == 200:
        print("Successfully connected to the Premier League API!")
        return response.json()
    else:
        raise Exception(f"Failed to fetch data: {response.status_code}")

def save_raw_data():
    """Fetches and saves raw API data into the data/raw directory."""
    data = fetch_fpl_data()
    
    # Ensure directory exists using config path
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    
    # Save elements (players), teams, and positions as raw CSVs using config paths
    pd.DataFrame(data["elements"]).to_csv(PLAYERS_RAW_PATH, index=False)
    pd.DataFrame(data["teams"]).to_csv(TEAMS_RAW_PATH, index=False)
    pd.DataFrame(data["element_types"]).to_csv(POSITIONS_RAW_PATH, index=False)
    
    print("Raw data successfully saved to data/raw/!")

if __name__ == "__main__":
    save_raw_data()