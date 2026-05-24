"""
tools/feature_tool.py
Feature engineering cho Steam sale prediction.

Không look-ahead bias — tất cả features chỉ dùng data QUÁ KHỨ.
"""

import argparse
import math
import sys
from pathlib import Path

# Thêm thư mục parent vào sys.path để import từ các module ngoài
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from config.settings import DATA_PROC_DIR, DATA_RAW_DIR
from harness.tool_harness import tool_harness


_GAME_INFO_CACHE: pd.DataFrame | None = None


def _coerce_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.astype(int)
    return (
        series.astype(str)
        .str.lower()
        .map({"true": 1, "false": 0, "1": 1, "0": 0, "yes": 1, "no": 0})
        .fillna(0)
        .astype(int)
    )


def _load_game_info() -> pd.DataFrame:
    global _GAME_INFO_CACHE
    if _GAME_INFO_CACHE is not None:
        return _GAME_INFO_CACHE.copy()

    path = DATA_RAW_DIR / "TS_game_info.csv"
    if not path.exists():
        _GAME_INFO_CACHE = pd.DataFrame(columns=["app_id"])
        return _GAME_INFO_CACHE.copy()

    info = pd.read_csv(path, dtype={"app_id": str})
    keep_cols = [
        "app_id",
        "mature",
        "early_access",
        "release_date",
        "first_price_record",
        "latest_price_record",
        "historical_low",
        "tags",
    ]
    info = info[[c for c in keep_cols if c in info.columns]].copy()

    if "release_date" in info.columns:
        info["release_date"] = pd.to_datetime(info["release_date"], errors="coerce")
    if "first_price_record" in info.columns:
        info["first_price_record"] = pd.to_datetime(
            info["first_price_record"], errors="coerce"
        )
    if "latest_price_record" in info.columns:
        info["latest_price_record"] = pd.to_datetime(
            info["latest_price_record"], errors="coerce"
        )

    if "mature" in info.columns:
        info["mature_flag"] = _coerce_bool(info["mature"])
    if "early_access" in info.columns:
        info["early_access_flag"] = _coerce_bool(info["early_access"])
    if "tags" in info.columns:
        info["tag_count"] = info["tags"].fillna("").apply(
            lambda x: len([t for t in str(x).split(",") if t.strip()])
        )

    if {"first_price_record", "latest_price_record"}.issubset(info.columns):
        info["days_tracked"] = (
            info["latest_price_record"] - info["first_price_record"]
        ).dt.days

    drop_cols = [c for c in ["mature", "early_access", "tags"] if c in info.columns]
    if drop_cols:
        info = info.drop(columns=drop_cols)

    # Game info CSV stores app_id as float (e.g. 752590.0); price history
    # stores it as int string ("752590"). Strip the trailing .0 so the merge works.
    info["app_id"] = (
        info["app_id"]
        .apply(lambda x: str(int(float(x))) if pd.notna(x) and str(x) != "nan" else None)
        .astype(str)
    )
    info = info.drop_duplicates(subset=["app_id"])
    _GAME_INFO_CACHE = info
    return info.copy()

@tool_harness("feature_engineer")
def build_features(game_df: pd.DataFrame) -> pd.DataFrame:
    """
    Tạo features từ lịch sử sale của 1 game.

    Features:
        - Cyclical: month_sin/cos, dayofweek_sin/cos, dayofyear_sin/cos
        - Lag: prev_discount, prev_gap_days, prev_price
        - Rolling: avg_discount_3, avg_gap_3
        - Calendar: year, quarter, is_weekend, is_year_end (tháng 11-12)

    Targets:
        - next_gap_days:    số ngày đến lần sale tiếp theo
        - next_discount:    % giảm giá lần sale tiếp theo

    Args:
        game_df: DataFrame từ data_tool.get_game_history() — đã sort theo date

    Returns:
        DataFrame với features + targets, đã drop NaN
    """
    sort_cols = [c for c in ["app_id", "shop_id", "date"] if c in game_df.columns]
    df = game_df.copy().sort_values(sort_cols).reset_index(drop=True)

    # ensure date/app_id
    df["date"] = pd.to_datetime(df["date"])
    df["app_id"] = df["app_id"].astype(str)

    # merge game info (static metadata)
    game_info = _load_game_info()
    if not game_info.empty:
        df = df.merge(game_info, on="app_id", how="left")

    # re-sort after merge (merge can scramble row order)
    sort_cols = [c for c in ["app_id", "shop_id", "date"] if c in df.columns]
    df = df.sort_values(sort_cols).reset_index(drop=True)

    # ── Cyclical encoding ────────────────────────────────────────
    def sin_cos(series: pd.Series, period: float):
        return np.sin(2 * math.pi * series / period), np.cos(2 * math.pi * series / period)

    df["month_sin"], df["month_cos"] = sin_cos(df["date"].dt.month, 12)
    df["dow_sin"], df["dow_cos"] = sin_cos(df["date"].dt.dayofweek, 7)
    df["dayofyear_sin"], df["dayofyear_cos"] = sin_cos(df["date"].dt.dayofyear, 365)

    # basic calendar
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year
    df["quarter"] = df["date"].dt.quarter
    df["is_weekend"] = (df["date"].dt.dayofweek >= 5).astype(int)
    df["is_year_end"] = (df["date"].dt.month >= 11).astype(int)
    df["is_holiday_season"] = df["month"].isin([11, 12]).astype(int)

    # price / discount basics
    df["discount_amount"] = df.get("regular_price", df.get("price", 0)) - df["price"]
    df["is_on_sale"] = (df["sales_percentage"] > 0).astype(int)

    # group by shop if available to compute shop-level stats
    group_cols = ["app_id", "shop_id"] if "shop_id" in df.columns else ["app_id"]
    grp = df.groupby(group_cols)

    # historical price stats per shop
    if grp is not None:
        df["price_min"] = grp["price"].transform("min")
        df["price_max"] = grp["price"].transform("max")
        df["price_mean"] = grp["price"].transform("mean")
        df["price_median"] = grp["price"].transform("median")
        df["price_std"] = grp["price"].transform("std").fillna(0)
        df["price_q25"] = grp["price"].transform(lambda x: x.quantile(0.25))
        df["price_q75"] = grp["price"].transform(lambda x: x.quantile(0.75))
    else:
        df["price_min"] = df["price"].min()
        df["price_max"] = df["price"].max()
        df["price_mean"] = df["price"].mean()
        df["price_median"] = df["price"].median()
        df["price_std"] = float(df["price"].std()) if len(df["price"]) else 0.0
        df["price_q25"] = df["price"].quantile(0.25)
        df["price_q75"] = df["price"].quantile(0.75)

    df["price_position"] = ((df["price"] - df["price_min"]) / (df["price_max"] - df["price_min"] + 0.01)).clip(0, 1)

    today = pd.Timestamp.today().normalize()

    # days_since_last_sale: days between current row's date and the most recent
    # PREVIOUS row (within the same shop) where sales_percentage > 0.
    # Shift by 1 within the group before ffill so on-sale rows look backward.
    df["last_sale_date"] = df["date"].where(df["sales_percentage"] > 0)
    df["last_sale_date"] = df.groupby(group_cols)["last_sale_date"].shift(1)
    df["last_sale_date"] = df.groupby(group_cols)["last_sale_date"].ffill()
    fallback_days = (df["date"] - df.groupby(group_cols)["date"].transform("min")).dt.days
    df["days_since_last_sale"] = (df["date"] - df["last_sale_date"]).dt.days.fillna(fallback_days)

    # Days since the game was released on Steam (static per game, anchored to today)
    if "release_date" in df.columns:
        df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
        df["days_since_release"] = (today - df["release_date"]).dt.days
    else:
        df["days_since_release"] = 0

    first_record_date = df.groupby("app_id")["date"].transform("min")
    df["days_since_first_record"] = (df["date"] - first_record_date).dt.days

    for col in [
        "days_since_release",
        "historical_low",
        "mature_flag",
        "early_access_flag",
        "tag_count",
        "days_since_first_record",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # seasonal one-hot — cast to int so XGB sees 0/1, not True/False booleans
    df["season"] = df["month"].map({12: "Winter", 1: "Winter", 2: "Winter", 3: "Spring", 4: "Spring", 5: "Spring", 6: "Summer", 7: "Summer", 8: "Summer", 9: "Fall", 10: "Fall", 11: "Fall"})
    season_dummies = pd.get_dummies(df["season"], prefix="season").astype(int).fillna(0)
    df = pd.concat([df, season_dummies], axis=1)

    sort_cols = [c for c in ["app_id", "shop_id", "date"] if c in df.columns]
    df = df.sort_values(sort_cols).reset_index(drop=True)

    # lagged price and rolling stats
    df["lagged_price_1"] = df["price"].shift(1).fillna(df["price"])
    df["lagged_price_7"] = df["price"].shift(7).fillna(df["price"])
    df["price_rolling_mean_7d"] = df["price"].rolling(7, min_periods=1).mean().fillna(0)
    df["price_rolling_std_7d"] = df["price"].rolling(7, min_periods=1).std().fillna(0)
    df["price_rolling_mean_30d"] = df["price"].rolling(30, min_periods=1).mean().fillna(0)
    df["price_rolling_std_30d"] = df["price"].rolling(30, min_periods=1).std().fillna(0)

    # days since historical low
    low_dates = (
        df.loc[
            df["price"].eq(df.groupby(group_cols)["price"].transform("min")),
            group_cols + ["date"],
        ]
        .groupby(group_cols)["date"]
        .min()
    )
    df = df.join(low_dates.rename("historical_low_date"), on=group_cols)
    df["days_since_historical_low"] = (
        (df["date"] - df["historical_low_date"]).dt.days.fillna(0).astype(int)
    )

    # momentum and change
    df["price_change"] = df["price"].diff()
    df["price_pct_change"] = df["price"].pct_change().fillna(0)

    # previous features used by baseline
    df["prev_discount"] = df["sales_percentage"].shift(1)
    df["prev_price"] = df["price"].shift(1)
    df["gap_days"] = df["date"].diff().dt.days
    df["prev_gap_days"] = df["gap_days"].shift(1)

    # rolling discount/gap features
    df["avg_discount_3"] = df["sales_percentage"].shift(1).rolling(3, min_periods=1).mean()
    df["avg_gap_3"] = df["gap_days"].shift(1).rolling(3, min_periods=1).mean()
    df["max_discount_5"] = df["sales_percentage"].shift(1).rolling(5, min_periods=1).max()

    # sale-history lag features per app/shop
    sale_history = df[df["sales_percentage"] > 0][
        group_cols + ["date", "sales_percentage"]
    ].copy()
    if not sale_history.empty:
        sale_history = sale_history.sort_values(group_cols + ["date"])
        sale_history["prev_sale_date"] = sale_history.groupby(group_cols)["date"].shift(1)
        sale_history["inter_sale_days"] = (
            sale_history["date"] - sale_history["prev_sale_date"]
        ).dt.days
        sale_history["sale_pct_lag1"] = sale_history.groupby(group_cols)["sales_percentage"].shift(1)
        sale_history["sale_pct_lag2"] = sale_history.groupby(group_cols)["sales_percentage"].shift(2)
        sale_history["sale_pct_lag3"] = sale_history.groupby(group_cols)["sales_percentage"].shift(3)

        avg_inter = sale_history.groupby(group_cols)["inter_sale_days"].mean().rename("avg_inter_sale_days")
        std_inter = sale_history.groupby(group_cols)["inter_sale_days"].std().rename("std_inter_sale_days")

        merged_rows = []
        sale_groups = {k: v for k, v in sale_history.groupby(group_cols, sort=False)}
        for key, grp_all in df.groupby(group_cols, sort=False):
            grp_sale = sale_groups.get(key)
            grp_all = grp_all.sort_values("date").drop(columns=["last_sale_date"], errors="ignore")
            grp_all["_orig_idx"] = grp_all.index
            if grp_sale is None or grp_sale.empty:
                grp_all["last_sale_pct"] = np.nan
                grp_all["inter_sale_days"] = np.nan
                grp_all["sale_pct_lag1"] = np.nan
                grp_all["sale_pct_lag2"] = np.nan
                grp_all["sale_pct_lag3"] = np.nan
                merged_rows.append(grp_all.set_index("_orig_idx"))
                continue

            grp_sale = grp_sale.rename(
                columns={"date": "last_sale_date", "sales_percentage": "last_sale_pct"}
            )
            merged = pd.merge_asof(
                grp_all,
                grp_sale[[
                    "last_sale_date",
                    "last_sale_pct",
                    "inter_sale_days",
                    "sale_pct_lag1",
                    "sale_pct_lag2",
                    "sale_pct_lag3",
                ]].sort_values("last_sale_date"),
                left_on="date",
                right_on="last_sale_date",
                direction="backward",
            )
            merged_rows.append(merged.set_index("_orig_idx"))

        df = pd.concat(merged_rows, axis=0).sort_index().reset_index(drop=True)
        df = df.merge(avg_inter.reset_index(), on=group_cols, how="left")
        df = df.merge(std_inter.reset_index(), on=group_cols, how="left")

    # Steam seasonality features
    steam_sales = {
        "winter_sale": (12, 22),
        "spring_sale": (3, 14),
        "summer_sale": (6, 27),
        "autumn_sale": (11, 26),
        "lunar_new_year": (1, 23),
    }

    def _days_to_next_event(date: pd.Timestamp, month: int, day: int) -> int:
        target = pd.Timestamp(date.year, month, day)
        if target <= date:
            target = pd.Timestamp(date.year + 1, month, day)
        return int((target - date).days)

    for name, (m, d) in steam_sales.items():
        df[f"days_to_{name}"] = df["date"].apply(lambda dt: _days_to_next_event(dt, m, d))

    steam_dist_cols = [f"days_to_{name}" for name in steam_sales]
    df["days_to_nearest_steam_sale"] = df[steam_dist_cols].min(axis=1)

    doy = df["date"].dt.dayofyear
    df["sin_doy"] = np.sin(2 * np.pi * doy / 365)
    df["cos_doy"] = np.cos(2 * np.pi * doy / 365)
    df["sin_doy2"] = np.sin(4 * np.pi * doy / 365)
    df["cos_doy2"] = np.cos(4 * np.pi * doy / 365)

    # targets
    df["next_gap_days"] = df["gap_days"].shift(-1)
    df["next_discount"] = df["sales_percentage"].shift(-1)

    # keep important columns (including regular_price, shop_id, shop_name)
    feature_cols = [
        "month_sin",
        "month_cos",
        "dow_sin",
        "dow_cos",
        "dayofyear_sin",
        "dayofyear_cos",
        "month",
        "year",
        "quarter",
        "is_year_end",
        "is_weekend",
        "is_holiday_season",
        "price_min",
        "price_max",
        "price_mean",
        "price_median",
        "price_std",
        "price_q25",
        "price_q75",
        "price_position",
        "days_since_last_sale",
        "days_since_release",
        "historical_low",
        "mature_flag",
        "early_access_flag",
        "tag_count",
        "days_since_first_record",
        "season_Winter",
        "season_Spring",
        "season_Summer",
        "season_Fall",
        "lagged_price_1",
        "lagged_price_7",
        "price_rolling_mean_7d",
        "price_rolling_std_7d",
        "price_rolling_mean_30d",
        "price_rolling_std_30d",
        "days_since_historical_low",
        "discount_amount",
        "is_on_sale",
        "price_change",
        "price_pct_change",
        "last_sale_pct",
        "inter_sale_days",
        "avg_inter_sale_days",
        "std_inter_sale_days",
        "sale_pct_lag1",
        "sale_pct_lag2",
        "sale_pct_lag3",
        "days_to_nearest_steam_sale",
        "days_to_winter_sale",
        "days_to_spring_sale",
        "days_to_summer_sale",
        "days_to_autumn_sale",
        "days_to_lunar_new_year",
        "sin_doy",
        "cos_doy",
        "sin_doy2",
        "cos_doy2",
        "prev_discount",
        "prev_price",
        "prev_gap_days",
        "avg_discount_3",
        "avg_gap_3",
        "max_discount_5",
        "next_gap_days",
        "next_discount",
        "date",
        "app_id",
        "game_title",
        "shop_id",
        "shop_name",
        "regular_price",
        "price",
        "sales_percentage",
        "currency",
    ]

    # ensure columns exist
    feature_cols = [c for c in feature_cols if c in df.columns]

    df = df.replace([np.inf, -np.inf], np.nan)
    fill_zero_cols = [
        "price_std",
        "days_since_release",
        "game_age_days",
        "days_tracked",
        "historical_low",
        "mature_flag",
        "early_access_flag",
        "tag_count",
        "days_since_first_record",
        "days_since_historical_low",
        "price_change",
        "price_pct_change",
        "last_sale_pct",
        "inter_sale_days",
        "avg_inter_sale_days",
        "std_inter_sale_days",
        "sale_pct_lag1",
        "sale_pct_lag2",
        "sale_pct_lag3",
    ]
    for col in fill_zero_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    if "regular_price" in df.columns:
        df["regular_price"] = df["regular_price"].fillna(df.get("price"))

    df = df[feature_cols].dropna().reset_index(drop=True)
    df = df.replace([np.inf, -np.inf], 0)
    return df


@tool_harness("feature_save")
def save_features(df: pd.DataFrame, filename: str = "TS_history_fe.csv") -> dict:
    """
    Save engineered features to data/processed.

    Args:
        df: features DataFrame from build_features()
        filename: output file name

    Returns:
        {path, rows}
    """
    DATA_PROC_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DATA_PROC_DIR / filename
    df.to_csv(out_path, index=False)
    return {"path": str(out_path), "rows": int(len(df))}


FEATURE_COLS = [
    "month_sin",
    "month_cos",
    "dow_sin",
    "dow_cos",
    "dayofyear_sin",
    "dayofyear_cos",
    "month",
    "year",
    "quarter",
    "is_weekend",
    "is_year_end",
    "is_holiday_season",
    "price_min",
    "price_max",
    "price_mean",
    "price_median",
    "price_std",
    "price_q25",
    "price_q75",
    "price_position",
    "days_since_last_sale",
    "days_since_release",
    "historical_low",
    "mature_flag",
    "early_access_flag",
    "tag_count",
    "days_since_first_record",
    "season_Winter",
    "season_Spring",
    "season_Summer",
    "season_Fall",
    "lagged_price_1",
    "lagged_price_7",
    "price_rolling_mean_7d",
    "price_rolling_std_7d",
    "price_rolling_mean_30d",
    "price_rolling_std_30d",
    "days_since_historical_low",
    "discount_amount",
    "is_on_sale",
    "price_change",
    "price_pct_change",
    "last_sale_pct",
    "inter_sale_days",
    "avg_inter_sale_days",
    "std_inter_sale_days",
    "sale_pct_lag1",
    "sale_pct_lag2",
    "sale_pct_lag3",
    "days_to_nearest_steam_sale",
    "days_to_winter_sale",
    "days_to_spring_sale",
    "days_to_summer_sale",
    "days_to_autumn_sale",
    "days_to_lunar_new_year",
    "sin_doy",
    "cos_doy",
    "sin_doy2",
    "cos_doy2",
    "prev_discount",
    "prev_price",
    "prev_gap_days",
    "avg_discount_3",
    "avg_gap_3",
    "max_discount_5",
    "regular_price",
]

TARGET_COLS = ["next_gap_days", "next_discount"]


def main() -> None:
    p = argparse.ArgumentParser(description="Feature tool")
    p.add_argument("--csv", default="TS_history_cleaned.csv")
    p.add_argument("--app-id")
    p.add_argument("--all", action="store_true", help="Engineer all games")
    p.add_argument("--save", action="store_true", help="Save features CSV")
    p.add_argument("--out", default="TS_history_fe.csv")
    args = p.parse_args()

    from tools.data_tool import load_sales_data, get_game_history

    df = load_sales_data(args.csv)["data"]
    if args.all:
        all_feats = []
        for app_id in df["app_id"].astype(str).unique():
            try:
                game = get_game_history(df, app_id)["data"]
                feats = build_features(game)["data"]
                if not feats.empty:
                    all_feats.append(feats)
            except Exception:
                continue
        feats = pd.concat(all_feats, ignore_index=True) if all_feats else pd.DataFrame()
    else:
        if not args.app_id:
            raise SystemExit("--app-id is required unless --all is set")
        game = get_game_history(df, args.app_id)["data"]
        feats = build_features(game)["data"]
    print(f"✅ Features rows: {len(feats):,}")

    if args.save:
        saved = save_features(feats, filename=args.out)["data"]
        print(f"💾 Saved: {saved['path']} ({saved['rows']} rows)")


if __name__ == "__main__":
    main()
