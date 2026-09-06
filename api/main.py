"""FastAPI service exposing the trained model and cleaned dataset.

Run with: uvicorn api.main:app --reload
Docs at:  http://localhost:8000/docs

This exists so the ML model can be consumed by anything (a mobile app, a
cron job, another team's service) without going through the Streamlit UI —
and so Streamlit itself can eventually call this instead of importing
src.model directly.
"""
import os

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.config import PLAYERS_CLEANED_PATH
from src.logging_config import get_logger
from src.model import ALL_MODEL_FEATURES, load_feature_importances
from src.predict import ModelNotTrainedError, load_metadata, load_model, model_is_available

logger = get_logger(__name__)

app = FastAPI(
    title="PL Analytics API",
    description="Serves Premier League player data and fantasy-points predictions.",
    version="2.0.0",
)

_model = None  # loaded lazily on first request, cached after that


def _get_model():
    global _model
    if _model is None:
        _model = load_model()
    return _model


class PlayerFeatures(BaseModel):
    minutes: float = Field(..., ge=0)
    goals_scored: float = Field(..., ge=0)
    assists: float = Field(..., ge=0)
    clean_sheets: float = Field(..., ge=0)
    expected_goals: float = Field(..., ge=0)
    expected_assists: float = Field(..., ge=0)
    market_value_m: float = Field(..., gt=0)
    goals_per_90: float = Field(..., ge=0)
    assists_per_90: float = Field(..., ge=0)
    xg_per_90: float = Field(..., ge=0)
    xa_per_90: float = Field(..., ge=0)
    points_per_million: float
    minutes_share: float = Field(..., ge=0, le=1)
    position: str


class PredictionResponse(BaseModel):
    predicted_total_points: float
    model_name: str


@app.get("/health")
def health():
    return {"status": "ok", "model_available": model_is_available()}


@app.get("/model/metadata")
def model_metadata():
    metadata = load_metadata()
    if not metadata:
        raise HTTPException(status_code=404, detail="No trained model metadata found.")
    return metadata


@app.get("/model/feature-importance")
def feature_importance():
    try:
        model = _get_model()
    except ModelNotTrainedError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    df = load_feature_importances(model)
    return df.to_dict(orient="records")


@app.post("/predict", response_model=PredictionResponse)
def predict(features: PlayerFeatures):
    try:
        model = _get_model()
    except ModelNotTrainedError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    row = pd.DataFrame([features.model_dump()])
    missing = [c for c in ALL_MODEL_FEATURES if c not in row.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing features: {missing}")

    prediction = float(model.predict(row[ALL_MODEL_FEATURES])[0])
    metadata = load_metadata()
    return PredictionResponse(
        predicted_total_points=round(prediction, 2),
        model_name=metadata.get("model_name", "unknown"),
    )


@app.get("/players")
def list_players(team: str | None = None, position: str | None = None, limit: int = 50):
    if not os.path.exists(PLAYERS_CLEANED_PATH):
        raise HTTPException(status_code=404, detail="No processed player data found. Run the pipeline first.")

    df = pd.read_csv(PLAYERS_CLEANED_PATH)
    if team:
        df = df[df["team_name"].str.lower() == team.lower()]
    if position:
        df = df[df["position"].str.lower() == position.lower()]

    df["full_name"] = df["first_name"].fillna("") + " " + df["second_name"]
    return df.head(limit).to_dict(orient="records")


@app.get("/players/{full_name}")
def get_player(full_name: str):
    if not os.path.exists(PLAYERS_CLEANED_PATH):
        raise HTTPException(status_code=404, detail="No processed player data found. Run the pipeline first.")

    df = pd.read_csv(PLAYERS_CLEANED_PATH)
    df["full_name"] = df["first_name"].fillna("") + " " + df["second_name"]
    match = df[df["full_name"].str.lower() == full_name.lower()]
    if match.empty:
        raise HTTPException(status_code=404, detail=f"Player '{full_name}' not found.")
    return match.iloc[0].to_dict()