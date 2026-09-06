import pandas as pd
import pytest

import src.clean as clean_module
from src.clean import DataValidationError, clean_data


@pytest.fixture(autouse=True)
def _redirect_output_paths(tmp_path, monkeypatch):
    """clean_data() writes its output to config-defined paths; redirect those
    to a temp dir for every test in this file so tests never touch real
    project data.
    """
    monkeypatch.setattr(clean_module, "PROCESSED_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(clean_module, "PLAYERS_CLEANED_PATH", str(tmp_path / "players_cleaned.csv"))


def test_clean_data_merges_team_and_position_names(tmp_path, raw_players_df, teams_df, positions_df):
    players_path = tmp_path / "players_raw.csv"
    teams_path = tmp_path / "teams_raw.csv"
    positions_path = tmp_path / "positions_raw.csv"
    raw_players_df.to_csv(players_path, index=False)
    teams_df.to_csv(teams_path, index=False)
    positions_df.to_csv(positions_path, index=False)

    result = clean_data(str(players_path), str(teams_path), str(positions_path))

    assert set(result["team_name"]) == {"Alphaville FC", "Betatown United"}
    assert set(result["position"]) <= {"Goalkeeper", "Defender", "Midfielder", "Forward"}
    assert "goals_per_90" in result.columns
    assert "points_per_million" in result.columns


def test_clean_data_raises_on_missing_required_column(tmp_path, raw_players_df, teams_df, positions_df):
    bad_players = raw_players_df.drop(columns=["total_points"])
    players_path = tmp_path / "players_raw.csv"
    teams_path = tmp_path / "teams_raw.csv"
    positions_path = tmp_path / "positions_raw.csv"
    bad_players.to_csv(players_path, index=False)
    teams_df.to_csv(teams_path, index=False)
    positions_df.to_csv(positions_path, index=False)

    with pytest.raises((DataValidationError, KeyError)):
        clean_data(str(players_path), str(teams_path), str(positions_path))


def test_clean_data_raises_helpful_error_when_raw_files_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        clean_data(
            str(tmp_path / "missing_players.csv"),
            str(tmp_path / "missing_teams.csv"),
            str(tmp_path / "missing_positions.csv"),
        )
