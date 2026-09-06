import os
import requests
import pandas as pd

def fetch_fpl_data():
    """Fetches static data from the Fantasy Premier League API."""
    url = "https://fantasy.premierleague.com/api/bootstrap-static/"
    response = requests.get(url)
    
    if response.status_code == 200:
        print("Successfully connected to the Premier League API!")
        return response.json()
    else:
        raise Exception(f"Failed to fetch data: {response.status_code}")

def save_raw_data():
    """Fetches and saves raw API data into the data/raw directory."""
    data = fetch_fpl_data()
    
    # Ensure directory exists
    os.makedirs("data/raw", exist_ok=True)
    
    # Save elements (players), teams, and positions as raw CSVs for backup
    pd.DataFrame(data["elements"]).to_csv("data/raw/players_raw.csv", index=False)
    pd.DataFrame(data["teams"]).to_csv("data/raw/teams_raw.csv", index=False)
    pd.DataFrame(data["element_types"]).to_csv("data/raw/positions_raw.csv", index=False)
    
    print("Raw data successfully saved to data/raw/!")

if __name__ == "__main__":
    save_raw_data()