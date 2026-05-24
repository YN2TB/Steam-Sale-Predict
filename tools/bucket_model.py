"""
tools/bucket_model.py
Bucketed model training and prediction to reduce per-game model storage.
"""

from __future__ import annotations

import math
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.api import SVAR
from xgboost import XGBRegressor

from config.settings import MODELS_DIR
from harness.tool_harness import tool_harness
from tools.decoupled_model import FEATURE_COLUMNS, build_time_series_features, _advance_row_features


BUCKETS_YEARS = [1, 2, 3, 5, math.inf]
FREQ_BINS = [(0, 3), (3, 8), (8, math.inf)]


def _history_years(game_df: pd.DataFrame) -> float:
    dates = pd.to_datetime(game_df["date"], errors="coerce").dropna()
    if dates.empty:
        return 0.0
    span_days = (dates.max() - dates.min()).days
    return span_days / 365.25


def _sales_per_year(game_df: pd.DataFrame) -> float:
    df = game_df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    if df.empty:
        return 0.0

    sales_dates = df.loc[df["sales_percentage"] > 0, "date"].dt.date.unique()
    if len(sales_dates) == 0:
        return 0.0

    years = max(_history_years(df), 1e-6)
    return len(sales_dates) / years


def _bucket_label(years: float) -> str | None:
    if years < 1:
        return None
    for upper in BUCKETS_YEARS:
        if years <= upper:
            if math.isinf(upper):
                return "5y_plus"
            return f"{int(upper - 1)}y_{int(upper)}y" if upper > 1 else "1y_2y"
    return None


def _freq_label(freq: float) -> str:
    for low, high in FREQ_BINS:
        if low <= freq < high:
            return f"f_{int(low)}_{'inf' if math.isinf(high) else int(high)}"
    return "f_unknown"


def assign_bucket(game_df: pd.DataFrame) -> str | None:
    years = _history_years(game_df)
    length_bucket = _bucket_label(years)
    if length_bucket is None:
        return None
    freq = _sales_per_year(game_df)
    freq_bucket = _freq_label(freq)
    return f"{length_bucket}__{freq_bucket}"


def _fit_svar(endog: pd.DataFrame) -> SVAR | None:
    if len(endog) < 3:
        return None
    if (endog.nunique(dropna=True) <= 1).any():
        return None
    try:
        return SVAR(endog, svar_type="A", A=np.eye(endog.shape[1])).fit(maxlags=1)
    except Exception:
        return None


def _bucket_model_path(bucket: str) -> Path:
    return MODELS_DIR / "buckets" / f"{bucket}_sale_window_xgb.pkl"


@tool_harness("bucket_train")
def train_bucket_models(df: pd.DataFrame) -> dict:
    """
    Train a classifier per bucket (>= 1 year history). Skips shorter games.
    """
    MODELS_DIR.mkdir(exist_ok=True)
    (MODELS_DIR / "buckets").mkdir(exist_ok=True)

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    buckets: dict[str, dict[str, list[pd.DataFrame]]] = {}

    for app_id, group in df.groupby("app_id"):
        group = group.sort_values("date")
        bucket = assign_bucket(group)
        if bucket is None:
            continue
        buckets.setdefault(bucket, {}).setdefault("games", []).append(group)

    trained = []
    for bucket, payload in buckets.items():
        games = payload["games"]
        all_rows = []
        all_targets_days = []
        all_targets_discount = []
        all_targets_duration = []
        for game_df in games:
            feats = build_time_series_features(game_df)["data"]
            if feats.empty:
                continue
            feats = feats.dropna(subset=["days_to_next_sale"])
            if feats.empty:
                continue
            all_rows.append(feats[FEATURE_COLUMNS])
            all_targets_days.append(feats["days_to_next_sale"].astype(float))
            all_targets_discount.append(feats["target_discount_pct"].astype(float))
            all_targets_duration.append(feats["target_duration_days"].astype(float))

        if not all_rows:
            continue

        X = pd.concat(all_rows, axis=0)
        y_days = pd.concat(all_targets_days, axis=0)
        y_discount = pd.concat(all_targets_discount, axis=0)
        y_duration = pd.concat(all_targets_duration, axis=0)
        if len(y_days) < 20:
            continue

        days_model = XGBRegressor(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            n_jobs=-1,
            objective="reg:squarederror",
            verbosity=0,
        )
        days_model.fit(X, y_days)

        discount_model = XGBRegressor(
            n_estimators=400,
            max_depth=7,
            learning_rate=0.08,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            objective="reg:squarederror",
            verbosity=0,
        )
        discount_model.fit(X, y_discount)

        duration_model = XGBRegressor(
            n_estimators=400,
            max_depth=7,
            learning_rate=0.08,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            objective="reg:squarederror",
            verbosity=0,
        )
        duration_model.fit(X, y_duration)

        out_path = _bucket_model_path(bucket)
        with open(out_path, "wb") as f:
            pickle.dump(
                {
                    "format": "bucket_regression_v2",
                    "days_model": days_model,
                    "discount_model": discount_model,
                    "duration_model": duration_model,
                    "bucket": bucket,
                    "n_games": len(games),
                    "n_rows": int(len(y_days)),
                },
                f,
            )
        trained.append(bucket)

    return {"buckets_trained": trained, "count": len(trained)}


@tool_harness("bucket_predict")
def predict_bucketed(
    game_df: pd.DataFrame,
    app_id: str,
    threshold: float = 0.5,
) -> dict:
    """
    Predict sale start offset, duration, and discount using bucketed models.
    """
    bucket = assign_bucket(game_df)
    if bucket is None:
        raise ValueError("Game history < 1 year; bucketed model skipped")

    path = _bucket_model_path(bucket)
    if not path.exists():
        raise FileNotFoundError(f"Bucket model not found: {bucket}")

    with open(path, "rb") as f:
        payload = pickle.load(f)
    days_model = payload.get("days_model")
    discount_model = payload.get("discount_model")
    duration_model = payload.get("duration_model")

    if days_model is None:
        raise ValueError(f"Bucket model {bucket} uses old format; retrain with train_bucket_models()")

    features = build_time_series_features(game_df)["data"]
    if features.empty:
        raise ValueError("No features for prediction")

    X_latest = np.asarray(features[FEATURE_COLUMNS].iloc[-1], dtype=float).reshape(1, -1)

    days_pred = max(1, round(float(days_model.predict(X_latest)[0])))
    predicted_duration = max(1.0, round(float(duration_model.predict(X_latest)[0]))) if duration_model else 7.0
    xgb_pred = float(discount_model.predict(X_latest)[0]) if discount_model else float(features["target_discount_pct"].mean())

    endog = features[["discount_percentage", "price"]].dropna()
    svar_res = _fit_svar(endog)
    if svar_res is not None and len(endog) >= svar_res.k_ar:
        forecast = svar_res.forecast(endog.values[-svar_res.k_ar:], steps=1)
        predicted_discount = 0.5 * (xgb_pred + float(forecast[0, 0]))
    else:
        predicted_discount = xgb_pred
    predicted_discount = max(0.0, min(100.0, predicted_discount))

    last_data_date = pd.Timestamp(features["date"].iloc[-1])
    anchor = max(last_data_date, pd.Timestamp.today().normalize())
    predicted_start_date = (anchor + pd.Timedelta(days=days_pred)).strftime("%Y-%m-%d")

    return {
        "app_id": app_id,
        "bucket": bucket,
        "sale_predicted": True,
        "probability": round(max(0.0, 1.0 - days_pred / 365.0), 4),
        "predicted_start_offset_days": days_pred,
        "predicted_start_date": predicted_start_date,
        "predicted_duration_days": int(predicted_duration),
        "predicted_discount_pct": round(predicted_discount, 2),
    }


@tool_harness("bucket_predict_n")
def predict_bucketed_n(
    game_df: pd.DataFrame,
    app_id: str,
    n: int = 5,
    max_search_days: int = 365 * 3,
    min_cooldown_days: int = 30,
) -> list[dict]:
    """
    Predict the next n sale events using the bucketed model, rolling forward.
    """
    bucket = assign_bucket(game_df)
    if bucket is None:
        raise ValueError("Game history < 1 year; bucketed model skipped")

    path = _bucket_model_path(bucket)
    if not path.exists():
        raise FileNotFoundError(f"Bucket model not found: {bucket}")

    with open(path, "rb") as f:
        payload = pickle.load(f)
    days_model = payload.get("days_model")
    discount_model = payload.get("discount_model")
    duration_model = payload.get("duration_model")

    if days_model is None:
        raise ValueError(f"Bucket model {bucket} uses old format; retrain with train_bucket_models()")

    features = build_time_series_features(game_df)["data"]
    if features.empty:
        raise ValueError("No features for prediction")

    today = pd.Timestamp.today().normalize()
    last_row = features.iloc[-1].copy()
    anchor = max(pd.Timestamp(last_row["date"]), today)

    hist_durations = features.loc[features["target_duration_days"] > 0, "target_duration_days"].tolist()
    avg_duration = float(np.mean(hist_durations)) if hist_durations else 7.0

    endog = features[["discount_percentage", "price"]].dropna()
    svar_res = _fit_svar(endog)

    results: list[dict] = []
    days_searched = 0

    while len(results) < n and days_searched < max_search_days:
        X_row = np.asarray(last_row[FEATURE_COLUMNS], dtype=float).reshape(1, -1)

        days_pred = max(1, round(float(days_model.predict(X_row)[0])))
        dur_pred = max(1, round(float(duration_model.predict(X_row)[0]))) if duration_model else int(avg_duration)
        xgb_disc = float(discount_model.predict(X_row)[0]) if discount_model else 0.0

        if svar_res is not None and len(endog) >= svar_res.k_ar:
            forecast = svar_res.forecast(endog.values[-svar_res.k_ar:], steps=1)
            predicted_discount = max(0.0, min(100.0, 0.5 * (xgb_disc + float(forecast[0, 0]))))
        else:
            predicted_discount = max(0.0, min(100.0, xgb_disc))

        sale_start = anchor + pd.Timedelta(days=days_pred)
        sale_end = sale_start + pd.Timedelta(days=dur_pred - 1)

        results.append({
            "sale_number": len(results) + 1,
            "app_id": app_id,
            "bucket": bucket,
            "probability": round(max(0.0, 1.0 - days_pred / 365.0), 4),
            "predicted_start_offset_days": days_pred,
            "predicted_start_date": sale_start.strftime("%Y-%m-%d"),
            "predicted_end_date": sale_end.strftime("%Y-%m-%d"),
            "predicted_duration_days": dur_pred,
            "predicted_discount_pct": round(predicted_discount, 2),
            "days_from_now": (sale_start - today).days,
        })

        cooldown = max(1, min_cooldown_days)
        next_anchor = sale_end + pd.Timedelta(days=cooldown)
        days_searched += (next_anchor - anchor).days
        anchor = next_anchor
        hist_durations.append(float(dur_pred))
        avg_duration = float(np.mean(hist_durations))
        last_row = _advance_row_features(last_row, anchor, float(cooldown), avg_duration)

    return results
