"""Fault-tolerant ingestion of FPL bootstrap-static data.

Improvements over the original version:
  * Retries with exponential backoff on transient network/HTTP failures.
  * Explicit timeout (a hung connection no longer stalls the whole pipeline).
  * Every fetch is saved as a *timestamped snapshot* under data/history/, in
    addition to the "latest" CSVs the rest of the pipeline reads — this is
    what actually gives us gameweek-over-gameweek history to build form
    features from later, and makes runs reproducible/auditable.
"""
import json
import os
import time
from datetime import datetime, timezone

import pandas as pd
import requests
from requests.exceptions import RequestException

from src.config import (
    API_BACKOFF_FACTOR,
    API_MAX_RETRIES,
    API_TIMEOUT_SECONDS,
    FPL_API_URL,
    HISTORY_DATA_DIR,
    PLAYERS_RAW_PATH,
    POSITIONS_RAW_PATH,
    RAW_DATA_DIR,
    TEAMS_RAW_PATH,
)
from src.logging_config import get_logger

logger = get_logger(__name__)


class IngestionError(RuntimeError):
    """Raised when the FPL API cannot be reached after all retries."""


def fetch_fpl_data(
    url: str = FPL_API_URL,
    max_retries: int = API_MAX_RETRIES,
    backoff_factor: float = API_BACKOFF_FACTOR,
    timeout: float = API_TIMEOUT_SECONDS,
) -> dict:
    """Fetch bootstrap-static JSON, retrying transient failures with backoff.

    Raises IngestionError if every attempt fails.
    """
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            logger.info("Fetching FPL data (attempt %d/%d): %s", attempt, max_retries, url)
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            logger.info("Successfully fetched FPL bootstrap-static data.")
            return response.json()
        except RequestException as exc:
            last_error = exc
            wait = backoff_factor ** attempt
            logger.warning(
                "Fetch attempt %d/%d failed (%s). Retrying in %.1fs...",
                attempt, max_retries, exc, wait,
            )
            if attempt < max_retries:
                time.sleep(wait)

    raise IngestionError(
        f"Failed to fetch FPL data after {max_retries} attempts: {last_error}"
    )


def _snapshot_dir_for_now() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = os.path.join(HISTORY_DATA_DIR, ts)
    os.makedirs(path, exist_ok=True)
    return path


def save_raw_data(data: dict | None = None) -> dict:
    """Persist bootstrap-static data as both a timestamped snapshot and the
    'latest' CSVs the rest of the pipeline reads.

    Returns a dict describing what was written (useful for logging/tests).
    """
    if data is None:
        data = fetch_fpl_data()

    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    players_df = pd.DataFrame(data["elements"])
    teams_df = pd.DataFrame(data["teams"])
    positions_df = pd.DataFrame(data["element_types"])

    # "Latest" copies — what clean.py reads by default.
    players_df.to_csv(PLAYERS_RAW_PATH, index=False)
    teams_df.to_csv(TEAMS_RAW_PATH, index=False)
    positions_df.to_csv(POSITIONS_RAW_PATH, index=False)

    # Versioned snapshot for history/audit/form-feature building.
    snapshot_dir = _snapshot_dir_for_now()
    players_df.to_csv(os.path.join(snapshot_dir, "players_raw.csv"), index=False)
    teams_df.to_csv(os.path.join(snapshot_dir, "teams_raw.csv"), index=False)
    positions_df.to_csv(os.path.join(snapshot_dir, "positions_raw.csv"), index=False)
    with open(os.path.join(snapshot_dir, "meta.json"), "w") as f:
        json.dump(
            {
                "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
                "n_players": len(players_df),
                "n_teams": len(teams_df),
                "n_positions": len(positions_df),
            },
            f,
            indent=2,
        )

    logger.info(
        "Raw data saved: %d players, %d teams, %d positions -> %s (+ snapshot %s)",
        len(players_df), len(teams_df), len(positions_df), RAW_DATA_DIR, snapshot_dir,
    )
    return {
        "players": len(players_df),
        "teams": len(teams_df),
        "positions": len(positions_df),
        "snapshot_dir": snapshot_dir,
    }


if __name__ == "__main__":
    save_raw_data()
