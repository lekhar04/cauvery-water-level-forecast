"""Train per-station SARIMAX models for the Cauvery river-level forecast.

Each model forecasts the water level one hour ahead from lagged levels, rainfall
and calendar features. A trained model is saved as a joblib bundle
(model, scaler, features, ...) at  models/sarimax/sarimax_<STATION>.pkl,
which is what main.py (the API) serves.

Usage
-----
    python train_sarimax.py                      # interactive menu
    python train_sarimax.py --list               # show stations, data counts, status
    python train_sarimax.py --station-id CWC_T_NARASIPUR [CWC_UPPER_ANICUT ...]
    python train_sarimax.py --all                # train every station without a model
    python train_sarimax.py --station-id X --force   # retrain even if a model exists

Stations that already have a model are skipped unless --force is given.
"""

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
# Defaults are anchored to this file's folder (like main.py), so they are the same
# wherever the script is launched from.
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = str(BASE_DIR / "data" / "cauvery_model_dataset.csv")  # .csv or .parquet
DEFAULT_MODEL_DIR = str(BASE_DIR / "models" / "sarimax")
DEFAULT_RESULT_DIR = str(BASE_DIR / "results" / "sarimax")

TARGET = "target_level_1h"

EXOG_FEATURES = [
    "lag_1h", "lag_3h", "lag_6h", "lag_12h", "lag_24h", "lag_48h",
    "delta_1h", "delta_6h", "delta_24h",
    "rain_cum_1h", "rain_cum_3h", "rain_cum_6h", "rain_cum_12h",
    "rain_cum_24h", "rain_cum_72h", "rain_cum_168h",
    "rain_wet_hours_24h", "rain_hours_since_last", "rain_covered",
    "doy_sin", "doy_cos", "hour_sin", "hour_cos",
]

ORDER = (1, 1, 1)
SEASONAL_ORDER = (1, 0, 1, 24)  # hourly data, daily seasonality

MAX_OBSERVATIONS = 10_000  # first N chronological observations per station
MIN_OBSERVATIONS = 100     # fewer than this -> station cannot be trained
TRAIN_FRAC, VAL_FRAC = 0.70, 0.85  # chronological 70 / 15 / 15 split
MIN_FEATURE_COVERAGE = 0.80


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def read_dataset(path, columns):
    """Read only the needed columns of a CSV or parquet dataset."""
    path = Path(path)
    if not path.exists():
        sys.exit(f"Dataset not found: {path}\nUse --data to point to cauvery_model_dataset.csv/.parquet")
    wanted = set(columns)
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq
        available = pq.ParquetFile(path).schema.names
        return pd.read_parquet(path, columns=[c for c in available if c in wanted])
    return pd.read_csv(path, usecols=lambda c: c in wanted)


def load_data(path, include_features=True):
    """Load and clean the dataset (valid timestamp, target and split)."""
    columns = ["station_id", "timestamp", TARGET, "split"]
    if include_features:
        columns += EXOG_FEATURES

    print("Loading dataset...")
    df = read_dataset(path, columns)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp", "station_id", TARGET, "split"])
    df["split"] = df["split"].astype(str).str.strip().str.lower()
    df["split"] = df["split"].replace({"validation": "val", "valid": "val"})
    print("Dataset shape:", df.shape)
    return df


def count_observations(df):
    """Usable observations per station (unique timestamps with a numeric target)."""
    d = df.assign(**{TARGET: pd.to_numeric(df[TARGET], errors="coerce")}).dropna(subset=[TARGET])
    return d.groupby(d["station_id"].astype(str))["timestamp"].nunique().sort_index()


# --------------------------------------------------------------------------- #
# Model files
# --------------------------------------------------------------------------- #
def model_path(model_dir, station_id):
    return Path(model_dir) / f"sarimax_{str(station_id).replace('/', '_')}.pkl"


def model_exists(model_dir, station_id):
    return model_path(model_dir, station_id).exists()


# --------------------------------------------------------------------------- #
# Station preparation
# --------------------------------------------------------------------------- #
def prepare_station(df, station_id):
    """Return chronological train / val / test frames and the usable feature list."""
    station = df[df["station_id"].astype(str) == str(station_id)].copy()
    if station.empty:
        raise ValueError(f"Station not found: {station_id}")

    # Chronological order; the dataset's own split column is ignored
    station = station.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")
    station[TARGET] = pd.to_numeric(station[TARGET], errors="coerce")
    station = station.dropna(subset=[TARGET]).head(MAX_OBSERVATIONS).copy()

    if len(station) < MIN_OBSERVATIONS:
        raise ValueError(f"Insufficient observations for {station_id}: {len(station)}")

    print("\nStation:", station_id)
    print("Usable observations:", len(station))
    print("Date range:", station["timestamp"].min(), "to", station["timestamp"].max())

    # Chronological split; the last observation before each boundary is purged
    n = len(station)
    train_end = int(n * TRAIN_FRAC)
    val_end = int(n * VAL_FRAC)
    train = station.iloc[:train_end - 1].copy()
    val = station.iloc[train_end:val_end - 1].copy()
    test = station.iloc[val_end:].copy()

    if train.empty or val.empty or test.empty:
        raise ValueError("One or more splits are empty.")
    print(f"Split sizes - train: {len(train)}, val: {len(val)}, test: {len(test)}")

    # Feature selection uses training data only
    features = []
    for col in EXOG_FEATURES:
        if col not in train.columns:
            continue
        train[col] = pd.to_numeric(train[col], errors="coerce")
        if train[col].notna().mean() >= MIN_FEATURE_COVERAGE and train[col].nunique(dropna=True) > 1:
            features.append(col)
    if not features:
        raise ValueError("No usable predictors found in training data.")

    for frame in (train, val, test):
        for col in features:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")

    # Impute with training medians only
    medians = train[features].median()
    features = [c for c in features if pd.notna(medians[c])]
    if not features:
        raise ValueError("No predictors have valid training medians.")
    medians = medians[features]
    for frame in (train, val, test):
        frame[features] = frame[features].fillna(medians)

    # Drop features that are constant after imputation
    features = [c for c in features if train[c].nunique() > 1]
    if not features:
        raise ValueError("All predictors are constant after preparation.")
    print("Features:", features)

    for frame in (train, val, test):
        frame.sort_values("timestamp", inplace=True)
        frame.reset_index(drop=True, inplace=True)

    return train, val, test, features


# --------------------------------------------------------------------------- #
# Training and evaluation
# --------------------------------------------------------------------------- #
def train_model(train, val, test, features):
    """Fit SARIMAX on train, then run sequential one-step-ahead forecasts on val and test."""
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train[features])
    X_val = scaler.transform(val[features])
    X_test = scaler.transform(test[features])
    y_train = train[TARGET].astype(float).to_numpy()

    print(f"\nTraining SARIMAX {ORDER}x{SEASONAL_ORDER} on {len(train)} rows...")
    model = SARIMAX(
        endog=y_train,
        exog=X_train,
        order=ORDER,
        seasonal_order=SEASONAL_ORDER,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fitted = model.fit(disp=False, maxiter=200)
    print(fitted.summary())
    if not fitted.mle_retvals.get("converged", False):
        print("\nWARNING: SARIMAX did not converge. Treat its predictions as experimental.")

    predictions, actuals, timestamps, split_names = [], [], [], []
    current = fitted

    # After each forecast the model state is updated with the true value (refit=False)
    for split_name, part, X_part in (("val", val, X_val), ("test", test, X_test)):
        print(f"\nPredicting {split_name} data...")
        times = part["timestamp"].to_numpy()
        for i in range(len(part)):
            x_row = X_part[i:i + 1]
            forecast = current.get_forecast(steps=1, exog=x_row)
            predictions.append(float(np.asarray(forecast.predicted_mean).reshape(-1)[0]))
            actual = float(part[TARGET].iloc[i])
            actuals.append(actual)
            timestamps.append(times[i])
            split_names.append(split_name)
            current = current.append(endog=np.asarray([actual]), exog=x_row, refit=False)
            if (i + 1) % 1000 == 0:
                print(f"  {i + 1}/{len(part)} {split_name} observations")

    results = pd.DataFrame({
        "timestamp": timestamps,
        "split": split_names,
        "actual": actuals,
        "predicted": predictions,
    })
    return fitted, scaler, results


def evaluate(results):
    """MAE / RMSE / R2 for the validation and test splits."""
    rows = []
    print("\nMODEL EVALUATION")
    for split_name in ("val", "test"):
        part = results[results["split"] == split_name]
        mae = mean_absolute_error(part["actual"], part["predicted"])
        rmse = float(np.sqrt(mean_squared_error(part["actual"], part["predicted"])))
        r2 = r2_score(part["actual"], part["predicted"])
        rows.append({"split": split_name, "MAE": mae, "RMSE": rmse, "R2": r2})
        print(f"{split_name.upper():5s} MAE {mae:.4f}  RMSE {rmse:.4f}  R2 {r2:.4f}")
    return pd.DataFrame(rows)


def save_outputs(model, scaler, results, metrics, station_id, features, model_dir, result_dir):
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    Path(result_dir).mkdir(parents=True, exist_ok=True)
    name = str(station_id).replace("/", "_")

    joblib.dump(
        {
            "model": model,
            "scaler": scaler,
            "features": features,
            "target": TARGET,
            "order": ORDER,
            "seasonal_order": SEASONAL_ORDER,
        },
        model_path(model_dir, station_id),
    )
    results.to_csv(Path(result_dir) / f"predictions_{name}.csv", index=False)
    metrics.to_csv(Path(result_dir) / f"metrics_{name}.csv", index=False)
    print(f"Saved model to {model_path(model_dir, station_id)}")


def train_station(df, station_id, model_dir, result_dir):
    train, val, test, features = prepare_station(df, station_id)
    model, scaler, results = train_model(train, val, test, features)
    metrics = evaluate(results)
    save_outputs(model, scaler, results, metrics, station_id, features, model_dir, result_dir)


# --------------------------------------------------------------------------- #
# Station selection
# --------------------------------------------------------------------------- #
def print_station_table(counts, model_dir):
    print(f"\n{'#':>3}  {'Station':<28}{'Data points':>12}  Status")
    for i, (station, n) in enumerate(counts.items(), start=1):
        if model_exists(model_dir, station):
            status = "model exists"
        elif n < MIN_OBSERVATIONS:
            status = f"too few observations (<{MIN_OBSERVATIONS})"
        else:
            status = "not trained"
        print(f"{i:>3}  {station:<28}{n:>12,}  {status}")
    print(f"\n(Training uses at most the first {MAX_OBSERVATIONS:,} observations of a station.)")


def choose_interactively(counts, model_dir, force):
    print_station_table(counts, model_dir)
    stations = list(counts.index)
    while True:
        try:
            raw = input("\nEnter station number(s) to train (comma-separated), 'all' for every "
                        "untrained station, or 'q' to quit: ").strip().lower()
        except EOFError:
            return []
        if raw in ("q", "quit", ""):
            return []
        if raw == "all":
            return [s for s in stations if force or not model_exists(model_dir, s)]
        try:
            picks = [stations[int(p) - 1] for p in raw.split(",") if int(p) >= 1]
            if picks:
                return picks
        except (ValueError, IndexError):
            pass
        print("Invalid selection.")


def parse_args():
    p = argparse.ArgumentParser(description="Train per-station SARIMAX water-level models.")
    p.add_argument("--station-id", nargs="+", metavar="ID", help="station(s) to train")
    p.add_argument("--all", action="store_true", help="train every station that has no model yet")
    p.add_argument("--list", action="store_true", help="list stations, data counts and model status, then exit")
    p.add_argument("--force", action="store_true", help="retrain even if a model already exists")
    p.add_argument("--data", default=DEFAULT_DATA_PATH, help="dataset path (default: data/cauvery_model_dataset.csv)")
    p.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    p.add_argument("--results-dir", default=DEFAULT_RESULT_DIR)
    return p.parse_args()


def main():
    args = parse_args()

    # Light read (4 columns) for the station table and the menu
    counts = count_observations(load_data(args.data, include_features=False))

    if args.list:
        print_station_table(counts, args.model_dir)
        return

    if args.station_id:
        selected = args.station_id
        unknown = [s for s in selected if s not in counts.index]
        if unknown:
            sys.exit(f"Unknown station(s): {unknown}. Run with --list to see valid IDs.")
    elif args.all:
        selected = [s for s in counts.index if args.force or not model_exists(args.model_dir, s)]
    elif sys.stdin.isatty():
        selected = choose_interactively(counts, args.model_dir, args.force)
    else:
        sys.exit("No station selected. Use --station-id, --all or --list.")

    # Never overwrite an existing model unless --force
    todo = []
    for s in selected:
        if model_exists(args.model_dir, s) and not args.force:
            print(f"Skipping {s}: model already exists (use --force to retrain).")
        elif counts[s] < MIN_OBSERVATIONS:
            print(f"Skipping {s}: only {counts[s]} usable observations.")
        else:
            todo.append(s)

    if not todo:
        print("Nothing to train.")
        return

    df = load_data(args.data)  # full load, now including feature columns
    failed = []
    for s in todo:
        try:
            train_station(df, s, args.model_dir, args.results_dir)
        except Exception as e:  # keep going so one bad station doesn't stop a batch
            print(f"\nFAILED {s}: {e}")
            failed.append(s)

    print(f"\nDone. Trained {len(todo) - len(failed)}/{len(todo)} station(s).")
    if failed:
        print("Failed:", failed)
        sys.exit(1)


if __name__ == "__main__":
    main()
