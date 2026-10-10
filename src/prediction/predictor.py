import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import hopsworks
import joblib
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.config.config import (
    FEATURE_GROUP_NAME,
    FEATURE_GROUP_VERSION,
    HOPSWORKS_API_KEY,
    HOPSWORKS_HOST,
    HOPSWORKS_PROJECT_NAME,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
)

REGISTRY_NAMES = {
    "random_forest": "aqi_random_forest",
    "gradient_boosting": "aqi_gradient_boosting",
    "lightgbm": "aqi_lightgbm",
    "decision_tree": "aqi_decision_tree",
    "sklearn_gradient_boosting": "aqi_sklearn_gradient_boosting",
}
LABELS = {1: "Good", 2: "Fair", 3: "Moderate", 4: "Poor", 5: "Very Poor"}
LAGS = (1, 3, 6, 12, 24, 48)
ROLL_WINDOWS = (3, 6, 12, 24)
EXCLUDE_COLS = {"datetime", "timestamp", "aqi"}
METRIC_KEYS = ("f1_score", "accuracy", "precision", "recall")
HORIZON_DAYS = 3
HISTORY_HOURS = 72
MAX_FORECAST_HOURS = 240
HOUR = pd.Timedelta(hours=1)


def utc_now():
    return datetime.now(timezone.utc).replace(tzinfo=None)


def connect():
    print("Connecting to Hopsworks...")
    project = hopsworks.login(
        host=HOPSWORKS_HOST,
        api_key_value=HOPSWORKS_API_KEY,
        project=HOPSWORKS_PROJECT_NAME,
    )
    return project.get_model_registry(), project.get_feature_store()


def _norm_metrics(raw, version):
    raw = raw or {}
    out = {k: float(raw.get(k, 0) or 0) for k in METRIC_KEYS}
    out["version"] = version
    return out


def load_models(mr):
    print("\nLoading classification models...")
    models, metrics = {}, {}
    for name, registry_name in REGISTRY_NAMES.items():
        try:
            model_path = Path(MODELS_DIR) / f"{name}.joblib"
            metrics_path = Path(MODELS_DIR) / f"{name}_metrics.json"
            if model_path.exists() and metrics_path.exists():
                models[name] = joblib.load(model_path)
                metrics[name] = _norm_metrics(json.loads(metrics_path.read_text()), "local")
            else:
                versions = mr.get_models(registry_name)
                if not versions:
                    print(f"   No versions found for {registry_name}")
                    continue
                latest = sorted(versions, key=lambda m: m.version, reverse=True)[0]
                path = Path(latest.download()) / "model.joblib"
                models[name] = joblib.load(path)
                metrics[name] = _norm_metrics(latest.training_metrics, latest.version)
            m = metrics[name]
            print(f"   {name:28} v{m['version']} | F1: {m['f1_score']:.4f} | Acc: {m['accuracy']:.4f}")
        except Exception as e:
            print(f"   Error loading {registry_name}: {e}")

    if not models:
        raise RuntimeError("No models loaded successfully")
    best = max(models, key=lambda n: metrics[n]["f1_score"])
    print(f"\nBest model: {best} (F1: {metrics[best]['f1_score']:.4f})")
    return models, metrics, best


def load_features(fs):
    print("\nFetching features...")
    fg = fs.get_feature_group(name=FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)
    try:
        df = fg.read(online=True)
        if df is None or df.empty:
            raise ValueError("online store returned no rows")
    except Exception as e:
        print(f"   Online read failed ({e}); falling back to offline store")
        df = fg.read()

    df["datetime"] = pd.to_datetime(df["datetime"])
    if df["datetime"].dt.tz is not None:
        df["datetime"] = df["datetime"].dt.tz_convert("UTC").dt.tz_localize(None)
    df["datetime"] = df["datetime"].dt.floor("60min")
    df = (
        df.sort_values("datetime")
        .drop_duplicates("datetime", keep="last")
        .reset_index(drop=True)
    )
    print(f"   Loaded {len(df)} records | latest: {df['datetime'].max()}")
    return df


def prepare_history(df, hours=HISTORY_HOURS):
    """Last 7 days reindexed to a gap-free hourly grid so lag lookups are exact."""
    last = df["datetime"].max()
    recent = df[df["datetime"] >= last - timedelta(days=7)].set_index("datetime")
    recent = recent.asfreq(pd.offsets.Hour()).ffill().bfill()
    if len(recent) < 48:
        raise ValueError(f"Need at least 48 hourly rows of history, got {len(recent)}")
    return recent.tail(hours).reset_index()


def get_feature_cols(model, history):
    names = getattr(model, "feature_names_in_", None)
    if names is None and hasattr(model, "booster_"):
        try:
            names = model.booster_.feature_name()
        except Exception:
            names = None
    if names is None:
        names = [c for c in history.columns if c not in EXCLUDE_COLS]
    return list(names)


def forecast(history, model, end_dt):
    """Recursive hourly forecast from the last history row up to end_dt (inclusive)."""
    history = history.copy().reset_index(drop=True)
    feature_cols = get_feature_cols(model, history)
    missing = [c for c in feature_cols if c not in history.columns]
    if missing:
        print(f"   WARNING: features missing from history, filled with 0: {missing}")

    aqi_values = history["aqi"].astype(float).tolist()
    last_dt = history["datetime"].iloc[-1]
    n_hours = int((end_dt - last_dt) / HOUR)
    if n_hours < 1 or n_hours > MAX_FORECAST_HOURS:
        raise ValueError(f"Unreasonable forecast horizon: {n_hours}h (last data: {last_dt})")
    print(f"   Last data: {last_dt} | forecasting {n_hours}h to {end_dt}")

    out = []
    for h in range(1, n_hours + 1):
        dt = last_dt + timedelta(hours=h)
        row = history.iloc[-24].copy()  # same hour yesterday as diurnal baseline
        idx = row.index

        def put(col, val):
            if col in idx:
                row[col] = val

        put("datetime", dt)
        put("hour_sin", np.sin(2 * np.pi * dt.hour / 24))
        put("hour_cos", np.cos(2 * np.pi * dt.hour / 24))
        put("day_of_week_sin", np.sin(2 * np.pi * dt.weekday() / 7))
        put("day_of_week_cos", np.cos(2 * np.pi * dt.weekday() / 7))
        put("month_sin", np.sin(2 * np.pi * dt.month / 12))
        put("month_cos", np.cos(2 * np.pi * dt.month / 12))
        put("is_weekend", 1 if dt.weekday() >= 5 else 0)

        for lag in LAGS:
            if lag <= len(aqi_values):
                put(f"aqi_lag_{lag}h", aqi_values[-lag])
        for w in ROLL_WINDOWS:
            if len(aqi_values) >= w:
                put(f"aqi_rolling_mean_{w}h", float(np.mean(aqi_values[-w:])))
        for w in (6, 24):
            if len(aqi_values) >= w:
                put(f"aqi_rolling_std_{w}h", float(np.std(aqi_values[-w:])))
        if len(aqi_values) >= 24:
            put("aqi_change_24h", float(aqi_values[-1] - aqi_values[-24]))

        if "pm2_5" in history.columns:
            pm = pd.to_numeric(history["pm2_5"], errors="coerce")
            put("pm2_5_rolling_mean_6h", float(pm.tail(6).mean()))
            put("pm2_5_rolling_mean_24h", float(pm.tail(24).mean()))
            if "wind_speed" in idx:
                put("pm2_5_x_wind_speed", float(pd.to_numeric(row["pm2_5"], errors="coerce"))
                    * float(pd.to_numeric(row["wind_speed"], errors="coerce")))

        X = (
            pd.DataFrame([row])
            .reindex(columns=feature_cols)
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0.0)
            .astype(float)
        )
        pred = int(round(float(np.ravel(model.predict(X))[0])))
        pred = max(1, min(5, pred))

        aqi_values.append(float(pred))
        put("aqi", pred)
        out.append({"datetime": dt, "aqi": pred})
        history = pd.concat([history, pd.DataFrame([row])], ignore_index=True)

    return pd.DataFrame(out)


def summarize(df_hourly, target_dates):
    now_str = utc_now().strftime("%Y-%m-%d %H:%M:%S")
    rows = []
    for d in target_dates:
        day = df_hourly[df_hourly["datetime"].dt.date == d]
        if day.empty:
            continue
        avg = max(1, min(5, int(round(day["aqi"].mean()))))
        rows.append({
            "date": d.strftime("%Y-%m-%d"),
            "day_name": d.strftime("%A"),
            "average_aqi": avg,
            "min_aqi": int(day["aqi"].min()),
            "max_aqi": int(day["aqi"].max()),
            "category": LABELS.get(avg, "Unknown"),
            "warning": get_warning(avg),
            "timestamp": now_str,
        })
        print(f"   {d.strftime('%A, %B %d')}: {avg} ({LABELS.get(avg)})")
    return rows


def get_warning(aqi):
    if aqi >= 5:
        return "HAZARDOUS! Avoid all outdoor exertion."
    if aqi >= 4:
        return "POOR! Limit outdoor exposure, especially for sensitive groups."
    if aqi >= 3:
        return "MODERATE! Outdoor activities may affect sensitive individuals."
    return None


def model_comparison(metrics, best):
    rows = [
        {
            "model": name.replace("_", " ").title(),
            "f1_score": round(m["f1_score"], 4),
            "accuracy": round(m["accuracy"], 4),
            "precision": round(m["precision"], 4),
            "recall": round(m["recall"], 4),
            "version": m["version"],
            "is_best": name == best,
        }
        for name, m in metrics.items()
    ]
    return sorted(rows, key=lambda r: r["f1_score"], reverse=True)


def run():
    print("=" * 70)
    print(f"AQI PREDICTION PIPELINE | {utc_now():%Y-%m-%d %H:%M:%S} UTC")
    print("=" * 70)

    mr, fs = connect()
    models, metrics, best = load_models(mr)
    history = prepare_history(load_features(fs))

    last_dt = history["datetime"].iloc[-1]
    today = max(utc_now().date(), last_dt.date())
    target_dates = [today + timedelta(days=i) for i in range(1, HORIZON_DAYS + 1)]
    end_dt = pd.Timestamp(target_dates[-1]) + timedelta(hours=23)

    print("\nGenerating forecast...")
    df_hourly = forecast(history, models[best], end_dt)
    predictions = summarize(df_hourly, target_dates)
    if len(predictions) != HORIZON_DAYS:
        raise RuntimeError(f"Expected {HORIZON_DAYS} daily predictions, got {len(predictions)}")

    out_dir = Path(PROCESSED_DATA_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(predictions).to_csv(out_dir / "latest_predictions.csv", index=False)
    pd.DataFrame(model_comparison(metrics, best)).to_csv(out_dir / "model_comparison.csv", index=False)
    print(f"\nSaved outputs to {out_dir}")


def main():
    try:
        run()
    except Exception as e:
        print(f"\nERROR in prediction pipeline: {e}")
        raise


if __name__ == "__main__":
    main()
