import pandas as pd
import pytest


@pytest.fixture
def raw_players_df():
    return pd.DataFrame(
        {
            "first_name": ["Alan", "Beth", None],
            "second_name": ["Alpha", "Beta", "Gamma"],
            "team": [1, 2, 1],
            "element_type": [3, 4, 1],
            "now_cost": [70, 95, 45],
            "minutes": [1800, 2500, 0],
            "goals_scored": [5, 15, 0],
            "assists": [4, 6, 0],
            "clean_sheets": [0, 0, 3],
            "expected_goals": [4.5, 13.2, 0.0],
            "expected_assists": [3.8, 5.1, 0.0],
            "total_points": [120, 200, 15],
        }
    )


@pytest.fixture
def teams_df():
    return pd.DataFrame({"id": [1, 2], "name": ["Alphaville FC", "Betatown United"]})


@pytest.fixture
def positions_df():
    return pd.DataFrame(
        {"id": [1, 2, 3, 4], "singular_name": ["Goalkeeper", "Defender", "Midfielder", "Forward"]}
    )


@pytest.fixture
def cleaned_players_df(raw_players_df, teams_df, positions_df):
    from src.features import add_engineered_features

    df = raw_players_df.copy()
    df["team_name"] = df["team"].map(teams_df.set_index("id")["name"])
    df["position"] = df["element_type"].map(positions_df.set_index("id")["singular_name"])
    df["market_value_m"] = df["now_cost"] / 10.0
    return add_engineered_features(df)
