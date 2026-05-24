"""
tools/report_tool.py
Save predictions to CSV for easy review.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from config.settings import RESULTS_DIR
from harness.tool_harness import tool_harness


@tool_harness("report_save_predictions")
def save_predictions(predictions: list[dict], filename: str = "sale_predictions.csv") -> dict:
    """
    Save predictions to CSV, sorted by game_title, shop_name, and date.

    Dates are formatted as DD/MM/YYYY.
    """
    if not predictions:
        raise ValueError("No predictions to save")

    df = pd.DataFrame(predictions)
    if "date" not in df.columns:
        raise ValueError("Missing 'date' in predictions")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if "window_end" in df.columns:
        df["window_end"] = pd.to_datetime(df["window_end"], errors="coerce")

    sort_cols = [c for c in ["game_title", "shop_name", "date"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols)

    df["date"] = df["date"].dt.strftime("%d/%m/%Y")
    if "window_end" in df.columns:
        df["window_end"] = df["window_end"].dt.strftime("%d/%m/%Y")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / filename
    df.to_csv(out_path, index=False)

    return {"path": str(out_path), "rows": int(len(df))}
