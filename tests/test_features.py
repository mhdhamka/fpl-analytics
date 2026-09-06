from src.features import add_engineered_features, flag_low_sample


def test_per_90_rates_are_scaled_correctly(cleaned_players_df):
    row = cleaned_players_df.iloc[0]
    expected_goals_per_90 = row["goals_scored"] / (row["minutes"] / 90.0)
    assert row["goals_per_90"] == expected_goals_per_90


def test_zero_minutes_does_not_produce_nan_or_inf(cleaned_players_df):
    # third fixture row has 0 minutes
    row = cleaned_players_df.iloc[2]
    assert row["goals_per_90"] == 0
    assert row["minutes_share"] == 0
    
    # Check only numeric/rate columns for NaNs to ignore intentionally empty string fields (like first_name)
    numeric_cols = ["goals_per_90", "assists_per_90", "xg_per_90", "xa_per_90", "points_per_million", "minutes_share"]
    assert not row[numeric_cols].isna().any()


def test_points_per_million_uses_total_points_over_value(cleaned_players_df):
    row = cleaned_players_df.iloc[1]
    assert row["points_per_million"] == row["total_points"] / row["market_value_m"]


def test_minutes_share_is_capped_at_one():
    import pandas as pd

    df = pd.DataFrame(
        {
            "minutes": [10000],
            "goals_scored": [0],
            "assists": [0],
            "expected_goals": [0],
            "expected_assists": [0],
            "total_points": [0],
            "market_value_m": [5.0],
        }
    )
    result = add_engineered_features(df)
    assert result["minutes_share"].iloc[0] == 1.0


def test_flag_low_sample_respects_threshold(cleaned_players_df):
    mask = flag_low_sample(cleaned_players_df, threshold=1000)
    # rows with minutes 1800, 2500, 0 -> only the 0-minute row is "low sample"
    assert list(mask) == [False, False, True]