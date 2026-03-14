import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src import DEFAULT_MODEL_PATH


class PopularityPredictionIn(BaseModel):
    """Модель входного запроса для предсказания популярности трека."""

    danceability: float = Field(ge=0.0, le=1.0)
    energy: float = Field(ge=0.0, le=1.0)
    valence: float = Field(ge=0.0, le=1.0)
    tempo: float = Field(gt=0.0, le=400.0)
    track_genre: str = Field(default="__other__", min_length=1)


class PopularityPredictionOut(BaseModel):
    """Модель ответа API с предсказанной популярностью."""

    predicted_popularity: float
    model_path: str


def _build_popularity_features(payload: PopularityPredictionIn) -> pd.DataFrame:
    genre = payload.track_genre.strip().lower() or "__other__"
    return pd.DataFrame(
        {
            "track_genre": [genre],
            "danceability": [payload.danceability],
            "energy": [payload.energy],
            "valence": [payload.valence],
            "tempo": [payload.tempo],
            "tempo_log": [float(np.log1p(payload.tempo))],
            "energy_x_danceability": [payload.energy * payload.danceability],
            "mood_index": [0.6 * payload.valence + 0.4 * payload.energy],
        }
    )


app = FastAPI(
    title="Spotify Popularity Predictor",
    description="Educational API that predicts track popularity by numeric features.",
    version="0.1.0",
)


@app.on_event("startup")
def startup() -> None:
    """Загружает модель предсказания популярности в память."""
    model_path = Path(os.environ.get("POPULARITY_MODEL_PATH", str(DEFAULT_MODEL_PATH)))
    app.state.model_path = model_path

    try:
        app.state.popularity_model = joblib.load(model_path)
    except FileNotFoundError as exc:
        raise RuntimeError(
            "Unable to load popularity model. Run `uv run dvc repro train_popularity_model` "
            "to build models/popylarity_model.joblib."
        ) from exc
    except Exception as exc:
        raise RuntimeError(
            "Unable to load popularity model. Run `uv run dvc repro train_popularity_model` "
            "to build models/popylarity_model.joblib."
        ) from exc


@app.get("/health")
def health() -> dict:
    """Возвращает статус доступности сервиса и путь к загруженной модели."""
    return {
        "status": "ok",
        "model_path": str(app.state.model_path),
    }


@app.post("/predict-popularity", response_model=PopularityPredictionOut)
def predict_popularity(payload: PopularityPredictionIn) -> PopularityPredictionOut:
    """Предсказывает популярность трека на базе подготовленных фичей."""
    features = _build_popularity_features(payload)
    try:
        prediction = float(app.state.popularity_model.predict(features)[0])
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Model inference failed.") from exc

    bounded_prediction = float(np.clip(prediction, 0.0, 100.0))
    return PopularityPredictionOut(
        predicted_popularity=bounded_prediction,
        model_path=str(app.state.model_path),
    )
