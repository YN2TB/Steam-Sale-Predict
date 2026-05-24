"""
tools/model_tool.py
Train model và dự đoán ngày sale + % giảm giá tiếp theo.

Model: GradientBoostingRegressor (không cần deep learning,
       đủ mạnh cho tabular time-series với ít data)
"""

# import json
import pickle
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import average_precision_score, f1_score, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

from harness.tool_harness import tool_harness
from tools.feature_tool import FEATURE_COLS

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)


@tool_harness("model_train")
def train(features_df: pd.DataFrame, app_id: str) -> dict:
    """
    Train 2 models cho 1 game: dự đoán gap_days và discount.

    Args:
        features_df: output từ feature_tool.build_features()
        app_id:      để đặt tên file model

    Returns:
        {app_id, gap_mae, discount_mae, gap_r2, discount_r2, n_samples}
    """
    if len(features_df) < 5:
        raise ValueError(
            f"Không đủ data để train: {len(features_df)} samples (cần ≥ 5)"
        )

    X = features_df[FEATURE_COLS].values
    y_gap = features_df["next_gap_days"].values
    y_discount = features_df["next_discount"].values

    # TimeSeriesSplit — không shuffle, giữ thứ tự thời gian
    tscv = TimeSeriesSplit(n_splits=min(3, len(features_df) // 2))
    gap_maes, disc_maes = [], []
    window_f1s, window_pr_aucs = [], []

    for train_idx, val_idx in tscv.split(X):
        for y, maes in [(y_gap, gap_maes), (y_discount, disc_maes)]:
            m = GradientBoostingRegressor(
                n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42
            )
            m.fit(X[train_idx], y[train_idx])
            pred = m.predict(X[val_idx])
            maes.append(mean_absolute_error(y[val_idx], pred))

            if y is y_gap:
                y_window = y[val_idx] <= 7
                pred_window = pred <= 7
                window_f1s.append(f1_score(y_window, pred_window))
                try:
                    window_pr_aucs.append(
                        average_precision_score(y_window, -pred)
                    )
                except ValueError:
                    window_pr_aucs.append(np.nan)

    # Train final models trên toàn bộ data
    model_gap = GradientBoostingRegressor(
        n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42
    )
    model_gap.fit(X, y_gap)

    model_disc = GradientBoostingRegressor(
        n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42
    )
    model_disc.fit(X, y_discount)

    # Lưu models
    safe_id = str(app_id).replace("/", "_")
    pickle.dump(model_gap, open(MODELS_DIR / f"{safe_id}_gap.pkl", "wb"))
    pickle.dump(model_disc, open(MODELS_DIR / f"{safe_id}_disc.pkl", "wb"))

    window_pr_auc = (
        float(np.nanmean(window_pr_aucs)) if window_pr_aucs else float("nan")
    )

    return {
        "app_id": app_id,
        "n_samples": len(features_df),
        "window_f1": round(float(np.mean(window_f1s)), 3) if window_f1s else 0.0,
        "window_pr_auc": round(window_pr_auc, 3) if not np.isnan(window_pr_auc) else 0.0,
        "gap_mae_days": round(float(np.mean(gap_maes)), 1),
        "discount_mae": round(float(np.mean(disc_maes)), 1),
        "gap_r2": round(r2_score(y_gap, model_gap.predict(X)), 3),
        "discount_r2": round(r2_score(y_discount, model_disc.predict(X)), 3),
    }


@tool_harness("model_predict")
def predict_next_sales(
    features_df: pd.DataFrame,
    app_id: str,
    n: int = 3,
) -> list[dict]:
    """
    Dự đoán n lần sale tiếp theo của 1 game.

    Args:
        features_df: output từ feature_tool.build_features()
        app_id:      ID của game
        n:           số lần sale muốn dự đoán

    Returns:
        list of {predicted_date, predicted_discount, confidence}
    """
    safe_id = str(app_id).replace("/", "_")
    gap_path = MODELS_DIR / f"{safe_id}_gap.pkl"
    disc_path = MODELS_DIR / f"{safe_id}_disc.pkl"

    if not gap_path.exists() or not disc_path.exists():
        raise FileNotFoundError(
            f"Model chưa được train cho app_id={app_id}. Chạy model_tool.train() trước."
        )

    model_gap = pickle.load(open(gap_path, "rb"))
    model_disc = pickle.load(open(disc_path, "rb"))

    # Dùng row cuối cùng làm base để predict rolling
    last_row = features_df.iloc[-1].copy()
    last_date = pd.Timestamp(features_df["date"].iloc[-1])
    predictions = []

    for i in range(n):
        X = last_row[FEATURE_COLS].values.reshape(1, -1)

        gap_days = max(1, round(float(model_gap.predict(X)[0])))
        discount = round(float(model_disc.predict(X)[0]), 1)
        discount = max(0.0, min(100.0, discount))  # clamp 0–100%

        next_date = last_date + timedelta(days=gap_days)

        predictions.append(
            {
                "sale_number": i + 1,
                "predicted_date": next_date.strftime("%Y-%m-%d"),
                "window_start": next_date.strftime("%Y-%m-%d"),
                "window_end": (next_date + timedelta(days=6)).strftime("%Y-%m-%d"),
                "predicted_discount": discount,
                "days_from_now": (next_date - datetime.now()).days,
            }
        )

        # Rolling: dùng prediction này làm input cho lần tiếp theo
        last_row["prev_discount"] = discount
        last_row["prev_gap_days"] = gap_days
        last_row["month_sin"] = np.sin(2 * np.pi * next_date.month / 12)
        last_row["month_cos"] = np.cos(2 * np.pi * next_date.month / 12)
        last_row["is_year_end"] = int(next_date.month >= 11)
        last_date = next_date

    return predictions


@tool_harness("model_predict_features")
def predict_on_features(features_df: pd.DataFrame, app_id: str) -> list[dict]:
    """
    Predict gap_days and discount for each feature row.

    Args:
        features_df: output from feature_tool.build_features()
        app_id:      ID of the game

    Returns:
        list of {predicted_gap_days, predicted_discount}
    """
    safe_id = str(app_id).replace("/", "_")
    gap_path = MODELS_DIR / f"{safe_id}_gap.pkl"
    disc_path = MODELS_DIR / f"{safe_id}_disc.pkl"

    if not gap_path.exists() or not disc_path.exists():
        raise FileNotFoundError(
            f"Model chưa được train cho app_id={app_id}. Chạy model_tool.train() trước."
        )

    model_gap = pickle.load(open(gap_path, "rb"))
    model_disc = pickle.load(open(disc_path, "rb"))

    X = features_df[FEATURE_COLS].values
    gap_preds = model_gap.predict(X)
    disc_preds = model_disc.predict(X)

    results = []
    for gap, disc in zip(gap_preds, disc_preds):
        gap_days = max(1, round(float(gap)))
        discount = round(float(disc), 1)
        discount = max(0.0, min(100.0, discount))
        results.append(
            {"predicted_gap_days": gap_days, "predicted_discount": discount}
        )

    return results
