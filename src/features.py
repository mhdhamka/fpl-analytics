"""Feature engineering, kept separate from cleaning so it can be unit tested
and reused by both the training pipeline and live inference.
"""
import numpy as np
import pandas as pd

from src.config import MIN_MINUTES_THRESHOLD


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add per-90 rate stats and value metrics to a cleaned players frame.

    Per-90 stats matter because raw totals conflate playing time with
    quality — a sub who scores 2 goals in 200 minutes is a very different
    signal from a starter who scores 2 in 2000. Dividing by minutes/90 lets
    the model (and the humans reading the dashboard) compare like-for-like.
    """
    df = df.copy()
    # Avoid divide-by-zero: treat sub-90-minute samples as 90 for rate calcs,
    # rate columns are still meaningless at that end, which MIN_MINUTES_THRESHOLD
    # downstream is meant to filter out of modeling (but we don't want NaNs here).
    minutes_safe = df["minutes"].clip(lower=1)
    nineties = minutes_safe / 90.0

    df["goals_per_90"] = df["goals_scored"] / nineties
    df["assists_per_90"] = df["assists"] / nineties
    df["xg_per_90"] = df["expected_goals"] / nineties
    df["xa_per_90"] = df["expected_assists"] / nineties

    # Fantasy-value metric: points squeezed out of every million in price.
    df["points_per_million"] = np.where(
        df["market_value_m"] > 0, df["total_points"] / df["market_value_m"], 0.0
    )

    # Share of a (roughly) 38-game, 90-minute season actually played — a
    # rough nailed-on-starter proxy without needing gameweek-by-gameweek data.
    max_possible_minutes = 38 * 90
    df["minutes_share"] = (df["minutes"] / max_possible_minutes).clip(upper=1.0)

    return df


def flag_low_sample(df: pd.DataFrame, threshold: int = MIN_MINUTES_THRESHOLD) -> pd.Series:
    """Boolean mask of rows with too little playing time to trust rate stats."""
    return df["minutes"] < threshold
