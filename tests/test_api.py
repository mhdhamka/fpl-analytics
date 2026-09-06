import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_health_endpoint_returns_ok():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_rejects_missing_fields():
    response = client.post("/predict", json={"minutes": 1000})
    assert response.status_code == 422  # pydantic validation error


def test_predict_returns_404_if_no_model_trained(monkeypatch):
    import api.main as api_module
    from src.predict import ModelNotTrainedError

    monkeypatch.setattr(api_module, "_get_model", lambda: (_ for _ in ()).throw(ModelNotTrainedError("no model")))

    payload = {
        "minutes": 1000, "goals_scored": 5, "assists": 3, "clean_sheets": 0,
        "expected_goals": 4.5, "expected_assists": 2.5, "market_value_m": 7.0,
        "goals_per_90": 0.45, "assists_per_90": 0.27, "xg_per_90": 0.4,
        "xa_per_90": 0.22, "points_per_million": 15.0, "minutes_share": 0.6,
        "position": "Midfielder",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 404
