"""Cleans and validates raw FPL data, then hands off to features.py for
feature engineering.
"""
import os

import pandas as pd

try:
    import pandera.pandas as pa
    from pandera.pandas import Column, DataFrameSchema
    _PANDERA_AVAILABLE = True
except ImportError:  # pandera is an optional dependency; degrade gracefully
    _PANDERA_AVAILABLE = False

from src.config import (
    PLAYERS_CLEANED_PATH,
    PLAYERS_RAW_PATH,
    POSITIONS_RAW_PATH,
    PROCESSED_DATA_DIR,
    TEAMS_RAW_PATH,
)
from src.features import add_engineered_features
from src.logging_config import get_logger

logger = get_logger(__name__)


class DataValidationError(RuntimeError):
    """Raised when incoming FPL data fails a schema check — usually a sign
    the API shape has changed upstream and downstream code needs review.
    """


# Minimal, dependency-free schema description used both as a fallback when
# pandera isn't installed and as the single source of truth pandera schemas
# below are built from — a hardcoded API shape change only needs updating here.
_RAW_REQUIRED = {
    "second_name": {"nullable": False},
    "team": {"nullable": False},
    "element_type": {"nullable": False},
    "now_cost": {"nullable": False, "min": 0},
    "minutes": {"nullable": False, "min": 0},
    "goals_scored": {"nullable": False, "min": 0},
    "total_points": {"nullable": False},
}
_CLEANED_REQUIRED = {
    "team_name": {"nullable": False},
    "position": {"nullable": False},
    "market_value_m": {"nullable": False, "min": 0},
    "goals_per_90": {"nullable": False, "min": 0},
    "points_per_million": {"nullable": False},
}


def _manual_validate(df: pd.DataFrame, required: dict, label: str) -> pd.DataFrame:
    """Dependency-free fallback validator, used when pandera isn't installed."""
    errors = []
    for col, rules in required.items():
        if col not in df.columns:
            errors.append(f"missing required column '{col}'")
            continue
        if not rules.get("nullable", True) and df[col].isna().any():
            errors.append(f"column '{col}' has null values but is required")
        if "min" in rules and (df[col].dropna() < rules["min"]).any():
            errors.append(f"column '{col}' has values below minimum {rules['min']}")
    if errors:
        raise DataValidationError(f"{label} failed validation: {'; '.join(errors)}")
    return df


if _PANDERA_AVAILABLE:
    def _pandera_schema(required: dict) -> "DataFrameSchema":
        columns = {}
        for col, rules in required.items():
            checks = [pa.Check.ge(rules["min"])] if "min" in rules else None
            columns[col] = Column(nullable=rules.get("nullable", True), checks=checks)
        return DataFrameSchema(columns, strict=False, coerce=False)

    _RAW_SCHEMA = _pandera_schema(_RAW_REQUIRED)
    _CLEANED_SCHEMA = _pandera_schema(_CLEANED_REQUIRED)

    def _validate(df: pd.DataFrame, required: dict, label: str) -> pd.DataFrame:
        schema = _RAW_SCHEMA if required is _RAW_REQUIRED else _CLEANED_SCHEMA
        try:
            return schema.validate(df)
        except pa.errors.SchemaError as exc:
            raise DataValidationError(f"{label} failed schema validation: {exc}") from exc
else:
    def _validate(df: pd.DataFrame, required: dict, label: str) -> pd.DataFrame:
        logger.warning("pandera not installed — using lightweight fallback validation for %s.", label)
        return _manual_validate(df, required, label)


def clean_data(
    players_path: str = PLAYERS_RAW_PATH,
    teams_path: str = TEAMS_RAW_PATH,
    positions_path: str = POSITIONS_RAW_PATH,
) -> pd.DataFrame:
    """Load raw FPL CSVs, validate, merge, engineer features, and persist."""
    try:
        players_df = pd.read_csv(players_path)
        teams_df = pd.read_csv(teams_path)
        positions_df = pd.read_csv(positions_path)
    except FileNotFoundError as e:
        logger.error("Raw data files not found. Run `python -m src.ingest` first.")
        raise e

    players_df = _validate(players_df, _RAW_REQUIRED, "players_raw")

    # Map team/position IDs to readable names.
    team_mapping = teams_df.set_index("id")["name"].to_dict()
    players_df["team_name"] = players_df["team"].map(team_mapping)

    position_mapping = positions_df.set_index("id")["singular_name"].to_dict()
    players_df["position"] = players_df["element_type"].map(position_mapping)

    unmapped_teams = players_df["team_name"].isna().sum()
    unmapped_positions = players_df["position"].isna().sum()
    if unmapped_teams or unmapped_positions:
        logger.warning(
            "%d players missing team_name, %d missing position after mapping "
            "(check for new/renamed team or position IDs upstream).",
            unmapped_teams, unmapped_positions,
        )

    selected_columns = [
        "first_name", "second_name", "team_name", "position",
        "now_cost", "minutes", "goals_scored", "assists",
        "clean_sheets", "expected_goals", "expected_assists", "total_points",
    ]
    df_cleaned = players_df[selected_columns].copy()
    df_cleaned["market_value_m"] = df_cleaned["now_cost"] / 10.0

    numeric_cols = [
        "minutes", "goals_scored", "assists", "clean_sheets",
        "expected_goals", "expected_assists", "total_points",
    ]
    for col in numeric_cols:
        df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors="coerce").fillna(0)

    df_cleaned = add_engineered_features(df_cleaned)
    df_cleaned = _validate(df_cleaned, _CLEANED_REQUIRED, "players_cleaned")

    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    df_cleaned.to_csv(PLAYERS_CLEANED_PATH, index=False)
    logger.info("Cleaned dataset saved: %d players -> %s", len(df_cleaned), PLAYERS_CLEANED_PATH)
    return df_cleaned


if __name__ == "__main__":
    clean_data()
