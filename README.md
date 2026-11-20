# Cauvery River Level Forecast

One-hour-ahead water-level forecasting for Cauvery basin CWC gauging stations. Each station
gets its own **SARIMAX** model with exogenous features (lagged levels, rainfall, calendar
features). This repository contains the **training script** and a **FastAPI service** that
serves the models you train. Trained models are not included; you train the ones you need.

## Project structure

```
cauvery-water-level-forecast/
├── main.py              # FastAPI service (POST /predict, GET /stations, GET /features/{station})
├── train_sarimax.py     # Trains a per-station SARIMAX model
├── requirements.txt     # Python dependencies
├── data/                # Put the dataset here (not in git)
│   └── cauvery_model_dataset.csv
├── models/sarimax/      # Trained models are written here (not in git)
│   └── sarimax_<STATION_ID>.pkl
└── results/sarimax/     # Training predictions and metrics (created by training, not in git)
```

## Workflow

1. [Set up the environment](#1-setup)
2. [Add the dataset](#2-add-the-dataset)
3. [Train a model](#3-train-a-model)
4. [Run the API](#4-run-the-api)
5. [Get a forecast](#5-get-a-forecast)

### 1. Setup

Requires Python 3.10+.

```powershell
python -m venv .venv
.venv\Scripts\activate          # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Add the dataset

Training needs the prepared hourly dataset, expected at `data/cauvery_model_dataset.csv`. It has one row per station and hour, with the columns `station_id`,
`timestamp`, `split`, `target_level_1h` and the feature columns (`lag_*`, `delta_*`,
`rain_*`, `doy_*`, `hour_*`). It is large (~880 MB), so it is not stored in git.

Get it from **https://github.com/lekhar04/cauvery-water-level-forecast/releases/tag/v1.0** and place it in `data/`, or pass
`--data <path>` to the training script.

### 3. Train a model

```powershell
python train_sarimax.py
```

An interactive menu lists every station with its number of usable data points and whether a
model already exists. Enter one or more numbers (comma-separated), or `all` for every
untrained station. **Stations that already have a model are skipped**, so nothing is
overwritten unless you pass `--force`.

Non-interactive options:

```powershell
python train_sarimax.py --list                                 # show the station table only
python train_sarimax.py --station-id CWC_T_NARASIPUR           # one or more station IDs
python train_sarimax.py --all                                  # every station without a model
python train_sarimax.py --station-id CWC_T_NARASIPUR --force   # retrain an existing model
python train_sarimax.py --data path/to/file.parquet            # use a different dataset
python train_sarimax.py --model-dir DIR --results-dir DIR    # save somewhere other than models/ and results/
```

What it does:

- **Target:** `target_level_1h`, the water level one hour ahead.
- **Model:** `SARIMAX(1,1,1)x(1,0,1,24)` on standardised exogenous features.
- **Data used:** the first 10,000 chronological observations of a station, split 70/15/15
  into train / validation / test. Stations with fewer than 100 observations can't be trained.
- **Features:** only features with enough coverage in the training split are used, so the
  feature list can differ per station.
- **Evaluation:** one forecast per validation/test row, then MAE, RMSE and R² per split.
  Training a station with 10,000 rows can take a long time.
- **Output:** the model is saved to `models/sarimax/sarimax_<STATION_ID>.pkl`, and
  `predictions_*.csv` / `metrics_*.csv` to `results/sarimax/`. The file is written only when
  training finishes, so an interrupted run leaves no partial model.

Each saved model is a `joblib` dictionary with the keys `model` (fitted SARIMAX results),
`scaler` (fitted scikit-learn scaler), `features` (ordered feature names), `target`,
`order` and `seasonal_order`.

### 4. Run the API

Run from the project root:

```powershell
uvicorn main:app --reload
```

Interactive docs: <http://127.0.0.1:8000/docs>

The API serves every `sarimax_<STATION_ID>.pkl` found in `models/sarimax/`. A model is loaded
the first time it is used. A newly trained station appears without a restart, but restart the server after retraining (`--force`) a model that was already loaded, because loaded models are cached.

| Endpoint | Purpose |
|----------|---------|
| `GET /stations` | `{"stations": [...]}`: stations that have a trained model (with a hint if there are none) |
| `GET /features/{station}` | Feature names (in order) that station's model expects |
| `POST /predict` | One-hour-ahead forecast |

### 5. Get a forecast

First check which features your trained model needs: `GET /features/<STATION_ID>`. Then:

```json
{
  "station": "CWC_T_NARASIPUR",
  "features": { "lag_1h": -0.15, "lag_3h": -0.15, "...": 0.0 },
  "previous_actual_level": -0.15,
  "previous_features": { "lag_1h": -0.15, "lag_3h": -0.16, "...": 0.0 }
}
```

- `station`: **required**. A station that has a trained model.
- `features`: **required**. Must contain every name returned by `GET /features/{station}`.
  Extra names are ignored.
- `previous_actual_level` + `previous_features`: *optional*, and must be sent together.
  The model state is updated with the last observed hour (`append`, `refit=False`) before
  forecasting. The update applies to that request only; every call starts from the trained
  state.

Response:

```json
{ "station": "CWC_T_NARASIPUR", "target": "target_level_1h", "predicted_level_1h": -0.1499 }
```

Example call:

```powershell
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d "@input.json"
```

Errors: `404` unknown station or no trained model, `400` missing features or only one of the
two `previous_*` fields, `500` the model file could not be loaded.

## Notes and limitations

- Forecast horizon is fixed at 1 hour ahead.
- Saved models can be large (hundreds of MB to GB) and are cached in memory after first use;
  run the API with a single worker.
- Models are loaded with `joblib`/pickle: only load model files you trained yourself or
  trust, and use the same `statsmodels` / `scikit-learn` versions that trained them.
