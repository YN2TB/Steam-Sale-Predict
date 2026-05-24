"""
tools/data_tool.py
Đọc và validate CSV Steam sales data.

CSV columns:
    game_title, app_id, price, date, regular_price,
    sales_percentage, shop_id, shop_name, currency
    (is_historical_low bị drop ngay khi đọc)
"""

import argparse
import os

import sys
from pathlib import Path
# Thêm thư mục parent vào sys.path để import từ các module ngoài
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from config.settings import DATA_PROC_DIR
from harness.tool_harness import tool_harness

DATA_DIR = Path(os.getenv("LOCAL_DATA_DIR", "./data/raw/"))

# Cột bắt buộc phải có
REQUIRED_COLS = {
    "game_title",
    "app_id",
    "price",
    "date",
    "regular_price",
    "sales_percentage",
    "shop_id",
    "shop_name",
    "currency",
}

# Drop ngay — tránh data leakage
LEAKAGE_COLS = ["is_historical_low"]


@tool_harness("data_load", max_retries=1)
def load_sales_data(filename: str | None = None) -> pd.DataFrame:
    """
    Đọc CSV sales data, drop leakage columns, validate.

    Args:
        filename: tên file CSV (mặc định: tự tìm file đầu tiên trong data/raw/)

    Returns:
        DataFrame đã clean với đúng dtypes
    """
    if filename:
        path = DATA_DIR / filename
        if not path.exists():
            alt_path = DATA_PROC_DIR / filename
            if alt_path.exists():
                path = alt_path
    else:
        files = list(DATA_DIR.glob("*.csv"))
        if not files:
            files = list(DATA_PROC_DIR.glob("*.csv"))
        if not files:
            raise FileNotFoundError(
                f"Không tìm thấy CSV trong {DATA_DIR.resolve()} hoặc {DATA_PROC_DIR.resolve()}"
            )
        path = files[0]

    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy: {path.resolve()}")

    df = pd.read_csv(path, dtype={"app_id": str})

    # Drop leakage columns nếu có
    leaked = [c for c in LEAKAGE_COLS if c in df.columns]
    if leaked:
        df = df.drop(columns=leaked)

    # Validate required columns
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"CSV thiếu columns: {missing}")

    # Fix dtypes
    df["date"] = pd.to_datetime(df["date"])
    df["app_id"] = df["app_id"].astype(str)
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["regular_price"] = pd.to_numeric(df["regular_price"], errors="coerce")
    df["sales_percentage"] = pd.to_numeric(df["sales_percentage"], errors="coerce")

    return df


@tool_harness("data_clean")
def clean_sales_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean data for modeling (USD only, drop required-null rows).

    Args:
        df: DataFrame from load_sales_data()

    Returns:
        Cleaned DataFrame
    """
    df = df[df["currency"] == "USD"]
    return df.dropna(subset=["date", "app_id", "game_title"]).reset_index(drop=True)


@tool_harness("data_save")
def save_sales_data(df: pd.DataFrame, filename: str = "TS_history_cleaned.csv") -> dict:
    """
    Save cleaned sales data to data/processed.

    Args:
        df: cleaned DataFrame from load_sales_data()
        filename: output file name

    Returns:
        {path, rows}
    """
    DATA_PROC_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DATA_PROC_DIR / filename
    print(out_path)
    df.to_csv(out_path, index=False)
    return {"path": str(out_path), "rows": int(len(df))}


@tool_harness("data_get_game", max_retries=1)
def get_game_history(df: pd.DataFrame, app_id: str) -> pd.DataFrame:
    """
    Lấy lịch sử sale của 1 game, sắp xếp theo ngày tăng dần.

    Args:
        df:     DataFrame từ load_sales_data()
        app_id: app ID của game

    Returns:
        DataFrame lịch sử sale của game đó
    """
    game_df = df[df["app_id"] == str(app_id)].sort_values("date").reset_index(drop=True)
    if game_df.empty:
        raise ValueError(f"Không tìm thấy app_id={app_id} trong data")
    return game_df


@tool_harness("data_list_games", max_retries=1)
def list_games(df: pd.DataFrame) -> list[dict]:
    """
    Liệt kê tất cả games trong data kèm số lần sale.
    """
    summary = (
        df.groupby(["app_id", "game_title"])
        .agg(
            sale_count=("date", "count"),
            first_sale=("date", "min"),
            last_sale=("date", "max"),
            avg_discount=("sales_percentage", "mean"),
        )
        .reset_index()
        .sort_values("sale_count", ascending=False)
    )
    summary["avg_discount"] = summary["avg_discount"].round(1)
    return summary.to_dict(orient="records")


# ── Test ─────────────────────────────────────────────────────────
def main() -> None:
    p = argparse.ArgumentParser(description="Data tool")
    p.add_argument("--csv", default="TS_price_history.csv")
    p.add_argument("--save", action="store_true", help="Save cleaned CSV")
    p.add_argument("--out", default="TS_history_cleaned.csv")
    args = p.parse_args()

    print(f"📂 DATA_DIR = {DATA_DIR.resolve()}\n")

    r = load_sales_data(filename=args.csv)
    if r["status"] == "ok":
        df = clean_sales_data(r["data"])["data"]
        df = df.sort_values(["app_id", "shop_name", "date"]).reset_index(drop=True)
        print(f"✅ Loaded {len(df):,} rows, {df['app_id'].nunique()} games")
        print(f"   Columns: {list(df.columns)}")
        print(f"   Date range: {df['date'].min().date()} → {df['date'].max().date()}")
        print(f"\n   Sample:\n{df.head(3).to_string()}\n")

        if args.save:
            saved = save_sales_data(df, filename=args.out)["data"]
            print(f"💾 Saved: {saved['path']} ({saved['rows']} rows)")

        r2 = list_games(df)
        if r2["status"] == "ok":
            games = r2["data"]
            print("📋 Top 5 games by sale count:")
            for g in games[:5]:
                print(
                    f"   [{g['app_id']}] {g['game_title']}: "
                    f"{g['sale_count']} sales, avg {g['avg_discount']}% off"
                )
    else:
        print(f"❌ {r['error']}")


if __name__ == "__main__":
    main()
