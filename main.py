import re
import threading
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

MODEL_DIR = Path(__file__).parent / "models" / "sarimax"

# Stations are discovered from the model files in models/sarimax/ (named
# sarimax_<STATION>.pkl, as written by train_sarimax.py), so a newly trained model is
# served without code changes. A bundle is loaded on first use, because each can be
# very large (~1 GB+) and loading all of them at startup is wasteful.
_STATION_FILE = re.compile(r"^sarimax_([A-Za-z0-9_]+)\.pkl$")


def available_stations() -> list[str]:
    return sorted(m.group(1) for p in MODEL_DIR.glob("sarimax_*.pkl") if (m := _STATION_FILE.match(p.name)))

_bundles: Dict[str, dict] = {}
_load_lock = threading.Lock()

app = FastAPI(title="Cauvery River Level Forecast API")


def get_bundle(station: str) -> dict:
    """Return the cached model bundle for a station, loading it if needed."""
    stations = available_stations()
    if not stations:
        raise HTTPException(
            status_code=404,
            detail="No trained models found in models/sarimax/. "
                   "Train one first with: python train_sarimax.py",
        )
    if station not in stations:
        raise HTTPException(
            status_code=404,
            detail=f"No model for station '{station}'. Available: {stations}. "
                   "Train it with: python train_sarimax.py",
        )
    with _load_lock:
        if station not in _bundles:
            path = MODEL_DIR / f"sarimax_{station}.pkl"
            try:
                _bundles[station] = joblib.load(path)
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to load model for {station}: {e}")
        return _bundles[station]


class ForecastRequest(BaseModel):
    # Which monitoring station to forecast (see GET /stations)
    station: str
    # Exogenous features for the hour being forecast, e.g. {"lag_1h": 12.4, ...}
    features: Dict[str, float]
    # Optional: last observed water level and that hour's features. When both are
    # given, the SARIMAX state is updated (append, refit=False) before forecasting.
    previous_actual_level: float | None = None
    previous_features: Dict[str, float] | None = None


@app.get("/stations")
def list_stations():
    """List stations that have a trained model on disk."""
    stations = available_stations()
    response = {"stations": stations}
    if not stations:
        response["message"] = "No trained models found. Train one with: python train_sarimax.py"
    return response


@app.get("/features/{station}")
def station_features(station: str):
    """Return the ordered feature names a station's model expects."""
    bundle = get_bundle(station)
    return {"station": station, "features": list(bundle["features"])}


@app.post("/predict")
def predict_next_hour(request: ForecastRequest):
    bundle = get_bundle(request.station)

    model = bundle["model"]
    scaler = bundle["scaler"]
    required_features = list(bundle["features"])

    # 1. All features used in training must be present
    missing = [c for c in required_features if c not in request.features]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing required exogenous features: {missing}")

    # 2. Optional state update. Both previous fields must come together.
    has_level = request.previous_actual_level is not None
    has_prev = request.previous_features is not None
    if has_level != has_prev:
        raise HTTPException(
            status_code=400,
            detail="Provide both previous_actual_level and previous_features, or neither.",
        )
    if has_level:
        missing_prev = [c for c in required_features if c not in request.previous_features]
        if missing_prev:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required features in previous_features: {missing_prev}",
            )
        prev_df = pd.DataFrame([request.previous_features])[required_features]
        # append() returns a NEW results object. It is used for this request only
        # and is NOT stored: every call starts from the trained state, and
        # concurrent requests stay independent.
        model = model.append(
            endog=np.asarray([request.previous_actual_level]),
            exog=scaler.transform(prev_df),
            refit=False,
        )

    # 3. Scale current features in the training column order
    input_df = pd.DataFrame([request.features])[required_features]
    X_scaled = scaler.transform(input_df)

    # 4. One-step-ahead forecast
    forecast = model.get_forecast(steps=1, exog=X_scaled)
    predicted_level = float(np.asarray(forecast.predicted_mean).reshape(-1)[0])

    return {
        "station": request.station,
        "target": bundle.get("target", "Water Level"),
        "predicted_level_1h": round(predicted_level, 4),
    }
