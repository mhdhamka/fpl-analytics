"""Generates a synthetic FPL bootstrap-static-shaped dataset for local dev,
offline testing, and CI — none of which should depend on hitting the live
API. Run this once, then `python -m src.clean` / `python -m src.model` work
exactly as they would on real data.
"""
import os
import random

import numpy as np
import pandas as pd

from src.config import PLAYERS_RAW_PATH, POSITIONS_RAW_PATH, RAW_DATA_DIR, TEAMS_RAW_PATH

random.seed(7)
np.random.seed(7)

TEAMS = [
    "Arsenal", "Aston Villa", "Bournemouth", "Brentford", "Brighton",
    "Chelsea", "Crystal Palace", "Everton", "Fulham", "Liverpool",
    "Man City", "Man Utd", "Newcastle", "Nott'm Forest", "Spurs",
    "West Ham", "Wolves", "Burnley", "Leeds", "Sunderland",
]

POSITIONS = [
    (1, "Goalkeeper"),
    (2, "Defender"),
    (3, "Midfielder"),
    (4, "Forward"),
]

FIRST_NAMES = ["James", "Mo", "Kevin", "Bruno", "Erling", "Marcus", "Declan",
               "Cole", "Bukayo", "Phil", "Jack", "Alexander", "Martin", "Ivan"]
LAST_NAMES = ["Smith", "Salah", "De Bruyne", "Fernandes", "Haaland", "Rashford",
              "Rice", "Palmer", "Saka", "Foden", "Grealish", "Isak", "Odegaard", "Toney"]


def _generate_players(n=400):
    rows = []
    for i in range(n):
        pos_id, pos_name = random.choice(POSITIONS)
        team_id = random.randint(1, len(TEAMS))

        if pos_name == "Goalkeeper":
            minutes = np.random.choice([0, 900, 2000, 3400], p=[0.3, 0.2, 0.2, 0.3])
        else:
            minutes = int(np.clip(np.random.normal(1500, 900), 0, 3420))

        nineties = max(minutes, 1) / 90.0
        goal_rate = {"Goalkeeper": 0.0, "Defender": 0.04, "Midfielder": 0.15, "Forward": 0.35}[pos_name]
        assist_rate = {"Goalkeeper": 0.0, "Defender": 0.04, "Midfielder": 0.18, "Forward": 0.12}[pos_name]

        goals = int(np.random.poisson(goal_rate * nineties))
        assists = int(np.random.poisson(assist_rate * nineties))
        expected_goals = round(max(0, goals + np.random.normal(0, 1.5)), 2)
        expected_assists = round(max(0, assists + np.random.normal(0, 1.0)), 2)
        clean_sheets = int(np.random.poisson(0.3 * nineties)) if pos_name in ("Goalkeeper", "Defender") else 0

        now_cost = int(np.clip(np.random.normal(65, 20), 39, 145))  # tenths of a million

        base_points = minutes / 90 * 2  # appearance points
        points = int(max(0, base_points + goals * 5 + assists * 3 + clean_sheets * 4
                          + np.random.normal(0, 8)))

        rows.append(
            {
                "first_name": random.choice(FIRST_NAMES),
                "second_name": f"{random.choice(LAST_NAMES)}{i}",
                "team": team_id,
                "element_type": pos_id,
                "now_cost": now_cost,
                "minutes": minutes,
                "goals_scored": goals,
                "assists": assists,
                "clean_sheets": clean_sheets,
                "expected_goals": expected_goals,
                "expected_assists": expected_assists,
                "total_points": points,
            }
        )
    return pd.DataFrame(rows)


def generate_and_save():
    os.makedirs(RAW_DATA_DIR, exist_ok=True)

    players_df = _generate_players()
    teams_df = pd.DataFrame({"id": range(1, len(TEAMS) + 1), "name": TEAMS})
    positions_df = pd.DataFrame(POSITIONS, columns=["id", "singular_name"])

    players_df.to_csv(PLAYERS_RAW_PATH, index=False)
    teams_df.to_csv(TEAMS_RAW_PATH, index=False)
    positions_df.to_csv(POSITIONS_RAW_PATH, index=False)

    print(f"Sample data written: {len(players_df)} players, {len(teams_df)} teams.")


if __name__ == "__main__":
    generate_and_save()
