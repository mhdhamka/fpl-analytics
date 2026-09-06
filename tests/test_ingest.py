import json
import os

import pytest
import requests

import src.ingest as ingest_module
from src.ingest import IngestionError, fetch_fpl_data, save_raw_data


class _FakeResponse:
    def __init__(self, json_data, status_code=200):
        self._json_data = json_data
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(f"status {self.status_code}")

    def json(self):
        return self._json_data


def _sample_payload():
    return {
        "elements": [{"first_name": "A", "second_name": "B", "team": 1, "element_type": 3,
                      "now_cost": 70, "minutes": 900, "goals_scored": 2, "assists": 1,
                      "clean_sheets": 0, "expected_goals": 1.5, "expected_assists": 0.9,
                      "total_points": 60}],
        "teams": [{"id": 1, "name": "Testville"}],
        "element_types": [{"id": 3, "singular_name": "Midfielder"}],
    }


def test_fetch_fpl_data_succeeds_on_first_try(monkeypatch):
    monkeypatch.setattr(requests, "get", lambda *a, **k: _FakeResponse(_sample_payload()))
    data = fetch_fpl_data(max_retries=3, backoff_factor=0, timeout=1)
    assert "elements" in data


def test_fetch_fpl_data_retries_then_succeeds(monkeypatch):
    calls = {"n": 0}

    def flaky_get(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] < 3:
            raise requests.exceptions.ConnectionError("boom")
        return _FakeResponse(_sample_payload())

    monkeypatch.setattr(requests, "get", flaky_get)
    monkeypatch.setattr(ingest_module.time, "sleep", lambda *_: None)  # skip real waiting

    data = fetch_fpl_data(max_retries=5, backoff_factor=0, timeout=1)
    assert calls["n"] == 3
    assert "elements" in data


def test_fetch_fpl_data_raises_ingestion_error_after_exhausting_retries(monkeypatch):
    monkeypatch.setattr(requests, "get", lambda *a, **k: (_ for _ in ()).throw(requests.exceptions.Timeout("timeout")))
    monkeypatch.setattr(ingest_module.time, "sleep", lambda *_: None)

    with pytest.raises(IngestionError):
        fetch_fpl_data(max_retries=2, backoff_factor=0, timeout=1)


def test_save_raw_data_writes_latest_and_snapshot_files(tmp_path, monkeypatch):
    monkeypatch.setattr(ingest_module, "RAW_DATA_DIR", str(tmp_path / "raw"))
    monkeypatch.setattr(ingest_module, "PLAYERS_RAW_PATH", str(tmp_path / "raw" / "players_raw.csv"))
    monkeypatch.setattr(ingest_module, "TEAMS_RAW_PATH", str(tmp_path / "raw" / "teams_raw.csv"))
    monkeypatch.setattr(ingest_module, "POSITIONS_RAW_PATH", str(tmp_path / "raw" / "positions_raw.csv"))
    monkeypatch.setattr(ingest_module, "HISTORY_DATA_DIR", str(tmp_path / "history"))

    summary = save_raw_data(data=_sample_payload())

    assert os.path.exists(ingest_module.PLAYERS_RAW_PATH)
    assert os.path.exists(summary["snapshot_dir"])
    assert os.path.exists(os.path.join(summary["snapshot_dir"], "meta.json"))

    with open(os.path.join(summary["snapshot_dir"], "meta.json")) as f:
        meta = json.load(f)
    assert meta["n_players"] == 1
