"""
tools/decoupled_model.py
Two-stage time-series pipeline:
- Stage 1: Sale start-day classifier (t+1..t+7)
- Stage 2: Sale duration + discount regressors (conditional on sale)
"""

from __future__ import annotations

import math
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.tsa.api import SVAR
from xgboost import XGBRegressor

from harness.tool_harness import tool_harness
from config.settings import MODELS_DIR, MIN_SAMPLES_TO_TRAIN

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

FEATURE_COLUMNS = [
    "month_sin",
    "month_cos",
    "dow_sin",
    "dow_cos",
    "is_summer_sale",
    "is_autumn_sale",
    "is_winter_sale",
    "price_lag_1",
    "price_lag_7",
    "price_lag_14",
    "price_momentum_7",
    "rolling_price_mean_7",
    "rolling_price_std_7",
    "rolling_price_mean_30",
    "rolling_price_std_30",
    "days_since_last_sale",
    "avg_sale_duration",
]

TARGET_COLUMNS = ["days_to_next_sale", "target_duration_days", "target_discount_pct"]


def _sin_cos(series: pd.Series, period: float) -> tuple[pd.Series, pd.Series]:
    return (
        np.sin(2 * math.pi * series / period),
        np.cos(2 * math.pi * series / period),
    )


def _prepare_daily_series(game_df: pd.DataFrame) -> pd.DataFrame:
    df = game_df.copy()
    df["date"] = pd.to_datetime(df["date"])

    # Prefer Steam records so other shops don't dilute Steam sale patterns.
    # Fall back to all shops only when there is no Steam data at all.
    if "shop_name" in df.columns:
        steam = df[df["shop_name"].str.lower() == "steam"]
        if not steam.empty:
            df = steam

    df = df.sort_values("date").reset_index(drop=True)

    app_id = str(df["app_id"].iloc[0]) if "app_id" in df.columns else "unknown"
    game_title = df["game_title"].iloc[0] if "game_title" in df.columns else "unknown"

    # Aggregate to daily level (Steam already filtered above, so this handles
    # multiple rows on the same day due to data duplicates)
    daily = df.groupby("date", as_index=False).agg(
        price=("price", "min"),
        regular_price=("regular_price", "max"),
        discount_percentage=("sales_percentage", "max"),
    )

    # Reindex to full daily range
    full_range = pd.date_range(daily["date"].min(), daily["date"].max(), freq="D")
    daily = daily.set_index("date").reindex(full_range)
    daily.index.name = "date"

    # ffill so gap days in changelog-style data inherit the last known discount
    # (a 14-day sale recorded on day 1 + day 15 should not appear as 1-day).
    daily["discount_percentage"] = daily["discount_percentage"].ffill().fillna(0)
    daily["regular_price"] = daily["regular_price"].ffill().bfill()
    daily["price"] = daily["price"].ffill()
    daily["price"] = daily["price"].fillna(daily["regular_price"])

    daily["app_id"] = app_id
    daily["game_title"] = game_title
    daily["is_on_sale"] = (daily["discount_percentage"] > 0).astype(int)

    return daily.reset_index()


def _build_time_series_features_df(game_df: pd.DataFrame) -> pd.DataFrame:
    daily = _prepare_daily_series(game_df)

    # Calendar features
    daily["month_sin"], daily["month_cos"] = _sin_cos(daily["date"].dt.month, 12)
    daily["dow_sin"], daily["dow_cos"] = _sin_cos(daily["date"].dt.dayofweek, 7)

    daily["is_summer_sale"] = daily["date"].dt.month.isin([6, 7]).fillna(False).astype(int)
    daily["is_autumn_sale"] = daily["date"].dt.month.isin([11]).fillna(False).astype(int)
    daily["is_winter_sale"] = daily["date"].dt.month.isin([12]).fillna(False).astype(int)

    # Price lags and momentum
    daily["price_lag_1"] = daily["price"].shift(1)
    daily["price_lag_7"] = daily["price"].shift(7)
    daily["price_lag_14"] = daily["price"].shift(14)

    denom = daily["price_lag_7"].replace(0, np.nan)
    daily["price_momentum_7"] = (daily["price"] - daily["price_lag_7"]) / denom

    # Rolling stats
    daily["rolling_price_mean_7"] = daily["price"].rolling(7, min_periods=1).mean()
    daily["rolling_price_std_7"] = daily["price"].rolling(7, min_periods=1).std()
    daily["rolling_price_mean_30"] = daily["price"].rolling(30, min_periods=1).mean()
    daily["rolling_price_std_30"] = daily["price"].rolling(30, min_periods=1).std()

    # Cooldown tracking — shift(1) so on-sale rows look at the PREVIOUS sale date,
    # not the current row's own date (which would give days_since_last_sale = 0 always).
    last_sale_date = daily["date"].where(daily["is_on_sale"] == 1).shift(1).ffill()
    daily["days_since_last_sale"] = (daily["date"] - last_sale_date).dt.days
    fallback_days = (daily["date"] - daily["date"].min()).dt.days
    daily["days_since_last_sale"] = daily["days_since_last_sale"].fillna(fallback_days)

    # Sale duration and running average (no look-ahead)
    sale_block = (daily["is_on_sale"] != daily["is_on_sale"].shift(1)).cumsum()
    sale_duration = daily.groupby(sale_block)["is_on_sale"].transform("size")
    sale_duration = sale_duration.where(daily["is_on_sale"] == 1, 0)

    avg_sale_duration = (
        sale_duration.where(daily["is_on_sale"] == 1).expanding().mean().shift(1)
    )
    daily["avg_sale_duration"] = avg_sale_duration.ffill().fillna(0)

    # Targets: next sale start offset (1..7), duration, discount
    sale_start_flag = (daily["is_on_sale"] == 1) & (
        daily["is_on_sale"].shift(1, fill_value=0) == 0
    )
    daily["sale_block_id"] = sale_start_flag.cumsum()

    sale_periods = (
        daily.loc[daily["is_on_sale"] == 1]
        .groupby("sale_block_id")
        .agg(
            sale_duration_days=("is_on_sale", "size"),
            sale_max_discount=("discount_percentage", "max"),
        )
        .reset_index()
    )

    sale_starts = daily.loc[sale_start_flag, ["date", "sale_block_id"]].rename(
        columns={"date": "sale_start_date"}
    )
    sale_meta = sale_starts.merge(sale_periods, on="sale_block_id", how="left")
    sale_meta = sale_meta.sort_values("sale_start_date")

    lookup = daily[["date"]].copy()
    lookup["_lookup_date"] = lookup["date"] + pd.Timedelta(days=1)
    next_sales = pd.merge_asof(
        lookup,
        sale_meta,
        left_on="_lookup_date",
        right_on="sale_start_date",
        direction="forward",
    )

    # Actual days to the next sale start — no cap, no binning.
    # Rows at the end of the series with no future sale will have NaN here;
    # they are dropped in training but kept for prediction (last-row inference).
    daily["days_to_next_sale"] = (next_sales["sale_start_date"] - daily["date"]).dt.days
    daily["target_duration_days"] = next_sales["sale_duration_days"].fillna(0)
    daily["target_discount_pct"] = next_sales["sale_max_discount"].fillna(0)

    # Final cleanup
    keep_cols = [
        "date",
        "app_id",
        "game_title",
        "price",
        "regular_price",
        "discount_percentage",
        "is_on_sale",
        *FEATURE_COLUMNS,
        *TARGET_COLUMNS,
    ]
    daily = daily[keep_cols].replace([np.inf, -np.inf], np.nan)
    # Drop NaN only on feature columns; keep rows with NaN days_to_next_sale
    # (end-of-series rows) so they can still be used for prediction.
    daily = daily.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)

    return daily


@tool_harness("two_stage_features")
def build_time_series_features(game_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build features for the two-stage pipeline using historical data only.
    """
    return _build_time_series_features_df(game_df)


def _fit_svar(endog: pd.DataFrame) -> SVAR | None:
    if len(endog) < 3:
        return None
    if (endog.nunique(dropna=True) <= 1).any():
        return None
    try:
        return SVAR(endog, svar_type="A", A=np.eye(endog.shape[1])).fit(maxlags=1)
    except Exception:
        return None


def _make_xgb_regressor() -> XGBRegressor:
    return XGBRegressor(
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


@tool_harness("two_stage_train")
def train_two_stage_models(
    game_df: pd.DataFrame,
    app_id: str,
    n_splits: int = 3,
) -> dict:
    """
    Train Stage 1 (sale start classifier) and Stage 2 (duration/discount regressors).
    """
    features = _build_time_series_features_df(game_df)

    # Drop rows with no known future sale (end-of-series rows with NaN days_to_next_sale)
    features = features.dropna(subset=["days_to_next_sale"])
    if len(features) < max(MIN_SAMPLES_TO_TRAIN, 30):
        raise ValueError(f"Not enough samples: {len(features)}")

    X = features[FEATURE_COLUMNS]
    y_days = features["days_to_next_sale"].astype(float)
    y_duration = features["target_duration_days"].astype(float)
    y_discount = features["target_discount_pct"].astype(float)

    endog_cols = ["discount_percentage", "price"]
    endog = features[endog_cols].copy().reset_index(drop=True)

    n_splits = min(n_splits, max(2, len(X) - 1))
    tscv = TimeSeriesSplit(n_splits=n_splits)

    days_mae_scores: list[float] = []
    disc_mae_scores: list[float] = []
    dur_mae_scores: list[float] = []

    for train_idx, val_idx in tscv.split(X):
        X_tr = np.asarray(X.iloc[train_idx], dtype=float)
        X_va = np.asarray(X.iloc[val_idx], dtype=float)

        xgb_days = _make_xgb_regressor()
        xgb_days.fit(X_tr, np.asarray(y_days.iloc[train_idx], dtype=float))
        days_mae_scores.append(mean_absolute_error(
            np.asarray(y_days.iloc[val_idx], dtype=float),
            xgb_days.predict(X_va),
        ))

        xgb_disc = _make_xgb_regressor()
        xgb_disc.fit(X_tr, np.asarray(y_discount.iloc[train_idx], dtype=float))
        disc_mae_scores.append(mean_absolute_error(
            np.asarray(y_discount.iloc[val_idx], dtype=float),
            xgb_disc.predict(X_va),
        ))

        xgb_dur = _make_xgb_regressor()
        xgb_dur.fit(X_tr, np.asarray(y_duration.iloc[train_idx], dtype=float))
        dur_mae_scores.append(mean_absolute_error(
            np.asarray(y_duration.iloc[val_idx], dtype=float),
            xgb_dur.predict(X_va),
        ))

    # Final models on all data
    X_np = np.asarray(X, dtype=float)
    final_days_xgb = _make_xgb_regressor()
    final_days_xgb.fit(X_np, np.asarray(y_days, dtype=float))

    final_discount_xgb = _make_xgb_regressor()
    final_discount_xgb.fit(X_np, np.asarray(y_discount, dtype=float))

    final_duration_xgb = _make_xgb_regressor()
    final_duration_xgb.fit(X_np, np.asarray(y_duration, dtype=float))

    final_svar = _fit_svar(endog)

    MODELS_DIR.mkdir(exist_ok=True)
    safe_id = str(app_id).replace("/", "_")

    model_path = MODELS_DIR / f"{safe_id}_sale_window_xgb.pkl"

    with open(model_path, "wb") as f:
        pickle.dump(
            {
                "format": "regression_v2",
                "days_xgb": final_days_xgb,
                "discount_xgb": final_discount_xgb,
                "duration_xgb": final_duration_xgb,
                "discount_svar": final_svar,
                "endog_cols": endog_cols,
            },
            f,
        )

    days_mae = float(np.mean(days_mae_scores)) if days_mae_scores else 0.0
    discount_mae = float(np.mean(disc_mae_scores)) if disc_mae_scores else 0.0
    duration_mae = float(np.mean(dur_mae_scores)) if dur_mae_scores else 0.0

    return {
        "app_id": app_id,
        "n_samples": int(len(features)),
        "days_mae": round(days_mae, 1),
        "discount_mae": round(discount_mae, 3),
        "duration_mae": round(duration_mae, 3),
        # kept for interface compat with main.py score display
        "window_f1": 0.0,
        "window_pr_auc": 0.0,
        "model_paths": {"model": str(model_path)},
    }


def _load_sale_model(app_id: str) -> dict:
    safe_id = str(app_id).replace("/", "_")
    model_path = MODELS_DIR / f"{safe_id}_sale_window_xgb.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not trained for app_id={app_id}. Run train first.")
    with open(model_path, "rb") as f:
        payload = pickle.load(f)
    if not isinstance(payload, dict) or payload.get("format") != "regression_v2":
        raise ValueError(
            f"Stale model for app_id={app_id} (old classifier format). "
            "Delete the .pkl file and retrain."
        )
    return payload


def _predict_discount(payload: dict, X: np.ndarray, features: pd.DataFrame) -> float:
    discount_xgb = payload.get("discount_xgb")
    discount_svar = payload.get("discount_svar")
    endog_cols = payload.get("endog_cols", ["discount_percentage", "price"])

    xgb_pred = (
        float(discount_xgb.predict(X)[0])
        if discount_xgb is not None
        else float(features["target_discount_pct"].mean())
    )
    svar_pred = None
    if discount_svar is not None:
        endog = features[endog_cols].dropna().values
        if len(endog) >= discount_svar.k_ar:
            try:
                forecast = discount_svar.forecast(endog[-discount_svar.k_ar:], steps=1)
                svar_pred = float(forecast[0, 0])
            except Exception:
                pass
    disc = 0.5 * (xgb_pred + svar_pred) if svar_pred is not None else xgb_pred
    return max(0.0, min(100.0, disc))


@tool_harness("two_stage_predict")
def predict_sale_event(
    game_features_dataframe: pd.DataFrame,
    app_id: str,
    threshold: float = 0.5,  # noqa: ARG001 — kept for interface compat
) -> dict:
    """Predict the next sale: days away, duration, and discount."""
    if set(FEATURE_COLUMNS).issubset(game_features_dataframe.columns):
        features = game_features_dataframe.copy()
    else:
        features = _build_time_series_features_df(game_features_dataframe)

    if features.empty:
        raise ValueError("No features available for prediction")

    payload = _load_sale_model(app_id)
    latest = features.iloc[-1]
    X = np.asarray(latest[FEATURE_COLUMNS], dtype=float).reshape(1, -1)

    days_pred = max(1, round(float(payload["days_xgb"].predict(X)[0])))
    dur_pred = max(1, round(float(payload["duration_xgb"].predict(X)[0])))
    disc_pred = _predict_discount(payload, X, features)

    last_data_date = pd.Timestamp(latest["date"])
    anchor = max(last_data_date, pd.Timestamp.today().normalize())
    sale_start = anchor + pd.Timedelta(days=days_pred)

    return {
        "sale_predicted": True,
        "probability": round(max(0.0, 1.0 - days_pred / 365.0), 4),
        "predicted_start_offset_days": days_pred,
        "predicted_start_date": sale_start.strftime("%Y-%m-%d"),
        "predicted_duration_days": int(dur_pred),
        "predicted_discount_pct": round(disc_pred, 2),
    }


@tool_harness("two_stage_predict_batch")
def predict_sale_event_batch(
    features_df: pd.DataFrame,
    app_id: str,
    threshold: float = 0.5,  # noqa: ARG001 — kept for interface compat
) -> list[dict]:
    """Batch predict days-to-next-sale, duration, and discount for each row."""
    if not set(FEATURE_COLUMNS).issubset(features_df.columns):
        raise ValueError("features_df must contain engineered features")

    payload = _load_sale_model(app_id)
    X_all = np.asarray(features_df[FEATURE_COLUMNS].values, dtype=float)
    days_preds = payload["days_xgb"].predict(X_all)
    dur_preds = payload["duration_xgb"].predict(X_all)
    disc_preds = payload["discount_xgb"].predict(X_all)

    results: list[dict] = []
    for days, dur, disc in zip(days_preds, dur_preds, disc_preds):
        days_int = max(1, round(float(days)))
        dur_int = max(1, round(float(dur)))
        disc_f = max(0.0, min(100.0, float(disc)))
        results.append({
            "sale_predicted": True,
            "probability": round(max(0.0, 1.0 - days_int / 365.0), 4),
            "predicted_start_offset_days": days_int,
            "predicted_duration_days": dur_int,
            "predicted_discount_pct": round(disc_f, 2),
        })
    return results


def _advance_row_features(row: pd.Series, new_date: pd.Timestamp, days_since_last_sale: float, avg_duration: float) -> pd.Series:
    """Return a copy of row with calendar and cooldown features updated to new_date."""
    row = row.copy()
    row["month_sin"] = math.sin(2 * math.pi * new_date.month / 12)
    row["month_cos"] = math.cos(2 * math.pi * new_date.month / 12)
    row["dow_sin"] = math.sin(2 * math.pi * new_date.dayofweek / 7)
    row["dow_cos"] = math.cos(2 * math.pi * new_date.dayofweek / 7)
    row["is_summer_sale"] = int(new_date.month in [6, 7])
    row["is_autumn_sale"] = int(new_date.month == 11)
    row["is_winter_sale"] = int(new_date.month == 12)
    row["days_since_last_sale"] = days_since_last_sale
    row["avg_sale_duration"] = avg_duration
    return row


@tool_harness("two_stage_predict_n")
def predict_next_n_sales(
    game_df: pd.DataFrame,
    app_id: str,
    n: int = 5,
    threshold: float = 0.5,  # noqa: ARG001 — kept for interface compat
    max_search_days: int = 365 * 3,
    min_cooldown_days: int = 30,
) -> list[dict]:
    """
    Predict the next n sale events by rolling forward through time.

    The regressor predicts days-to-next-sale directly (no 7-day window cap).
    After each sale ends, the anchor advances by the predicted duration plus
    the minimum cooldown period before querying for the next sale.
    """
    features = _build_time_series_features_df(game_df)
    if features.empty:
        raise ValueError("No features available for prediction")

    payload = _load_sale_model(app_id)
    days_xgb = payload["days_xgb"]
    duration_xgb = payload.get("duration_xgb")

    today = pd.Timestamp.today().normalize()
    last_row = features.iloc[-1].copy()
    anchor = max(pd.Timestamp(last_row["date"]), today)

    hist_durations = features.loc[features["target_duration_days"] > 0, "target_duration_days"].tolist()
    avg_duration = float(np.mean(hist_durations)) if hist_durations else 7.0

    results: list[dict] = []
    days_searched = 0

    while len(results) < n and days_searched < max_search_days:
        X_row = np.asarray(last_row[FEATURE_COLUMNS], dtype=float).reshape(1, -1)

        days_pred = max(1, round(float(days_xgb.predict(X_row)[0])))
        dur_pred = max(1, round(float(duration_xgb.predict(X_row)[0]))) if duration_xgb else int(avg_duration)
        disc_pred = _predict_discount(payload, X_row, features)

        sale_start = anchor + pd.Timedelta(days=days_pred)
        sale_end = sale_start + pd.Timedelta(days=dur_pred - 1)

        results.append({
            "sale_number": len(results) + 1,
            "probability": round(max(0.0, 1.0 - days_pred / 365.0), 4),
            "predicted_start_date": sale_start.strftime("%Y-%m-%d"),
            "predicted_end_date": sale_end.strftime("%Y-%m-%d"),
            "predicted_start_offset_days": days_pred,
            "predicted_duration_days": dur_pred,
            "predicted_discount_pct": round(disc_pred, 2),
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
