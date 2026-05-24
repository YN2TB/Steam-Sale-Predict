"""main.py — Steam Sales Window entrypoint"""

import argparse
import pandas as pd
from dotenv import load_dotenv

load_dotenv()


def run_single(
    app_id: str,
    n: int,
    csv_file: str | None,
    compare: bool,
    baseline_only: bool,
    bucketed: bool,
):
    from tools.data_tool import load_sales_data, get_game_history
    from tools.decoupled_model import train_two_stage_models, predict_next_n_sales
    from tools.bucket_model import predict_bucketed, train_bucket_models
    from tools.feature_tool import build_features
    from tools.model_tool import train, predict_next_sales

    df = load_sales_data(csv_file)["data"]
    game = get_game_history(df, app_id)["data"]
    title = game["game_title"].iloc[1]
    feat = build_features(game)["data"]
    # print(feat.columns.tolist())

    if bucketed:
        steam_df = df[df["shop_name"].str.lower() == "steam"]
        train_bucket_models(steam_df)
        game_steam = game[game["shop_name"].str.lower() == "steam"]
        if game_steam.empty:
            print("⚠️  No Steam data for this game; bucketed model requires Steam history")
            return
        bucket_pred = predict_bucketed(game_steam, app_id)["data"]
        print(f"\n{'=' * 55}")
        print(f"  {title} (app_id={app_id})")
        print(f"{'=' * 55}")
        print("\nBucketed two-stage:")
        print(f"  Bucket: {bucket_pred['bucket']}")
        print(f"  Sale predicted: {bucket_pred['sale_predicted']}")
        print(f"  Probability: {bucket_pred['probability']}")
        if bucket_pred["sale_predicted"]:
            print(
                f"  Start date: {bucket_pred['predicted_start_date']} "
                f"(+{bucket_pred['predicted_start_offset_days']} days)"
            )
            print(
                f"  Duration: {bucket_pred['predicted_duration_days']} days"
            )
            print(f"  Predicted discount: {bucket_pred['predicted_discount_pct']}%")
        else:
            print("  No sale start predicted in next 7 days")
        return

    baseline_stats = None
    baseline_preds = None
    if compare or baseline_only:
        baseline_stats = train(feat, app_id)["data"]
        baseline_preds = predict_next_sales(feat, app_id, n=n)["data"]

    two_stage_stats = None
    two_stage_preds = None
    if compare or not baseline_only:
        two_stage_stats = train_two_stage_models(game, app_id)["data"]
        two_stage_preds = predict_next_n_sales(game, app_id, n=n)["data"]

    print(f"\n{'=' * 55}")
    print(f"  {title} (app_id={app_id})")
    print(f"{'=' * 55}")
    def _row(num, date, duration_str, discount, days_from_now, prob=None):
        prob_str = f"  prob={prob}" if prob is not None else ""
        print(f"  #{num}: {date}  |  {duration_str}  |  {discount:.1f}% off  ({days_from_now:+d} days{prob_str})")

    if baseline_preds is not None and baseline_stats is not None:
        print("\nBaseline predictions:")
        for p in baseline_preds:
            _row(p["sale_number"], p["predicted_date"], "~7d window", p["predicted_discount"], p["days_from_now"])
        print(f"\n  F1={baseline_stats['window_f1']}  PR-AUC={baseline_stats['window_pr_auc']}  Discount MAE={baseline_stats['discount_mae']}")

    if two_stage_stats is not None:
        print("\nTwo-stage predictions:")
        if two_stage_preds:
            for p in two_stage_preds:
                _row(
                    p["sale_number"],
                    p["predicted_start_date"],
                    f"{p['predicted_duration_days']}d",
                    p["predicted_discount_pct"],
                    p["days_from_now"],
                    p["probability"],
                )
        else:
            print("  No upcoming sales predicted within search window")
        print(f"\n  Days MAE={two_stage_stats['days_mae']}d  Discount MAE={two_stage_stats['discount_mae']}  Duration MAE={two_stage_stats['duration_mae']}")


def run_all(n: int, csv_file: str | None):
    from tools.data_tool import load_sales_data, list_games, clean_sales_data, get_game_history
    from tools.bucket_model import predict_bucketed_n, train_bucket_models
    from tools.report_tool import save_predictions

    df = clean_sales_data(load_sales_data(csv_file)["data"])["data"]
    games = list_games(df)["data"]
    rows = []

    steam_df = df[df["shop_name"].str.lower() == "steam"]
    train_bucket_models(steam_df)

    for g in games:
        try:
            game = get_game_history(df, g["app_id"])["data"]
            game_steam = game[game["shop_name"].str.lower() == "steam"]
            if game_steam.empty:
                print(f"  ⚠️ {g['game_title']}: no Steam data, skipping")
                continue

            preds = predict_bucketed_n(game_steam, g["app_id"], n=n)["data"]

            print(f"  ✅ {g['game_title']}")
            for pred in preds:
                start_date = pd.Timestamp(pred["predicted_start_date"])
                duration_days = max(1, int(pred["predicted_duration_days"]))
                window_end = start_date + pd.Timedelta(days=duration_days - 1)
                print(
                    f"    #{pred['sale_number']}: {start_date.strftime('%Y-%m-%d')} "
                    f"(+{pred['predicted_start_offset_days']} days) | "
                    f"{duration_days}d | "
                    f"{pred['predicted_discount_pct']}% off (prob={pred['probability']})"
                )
                rows.append(
                    {
                        "game_title": g["game_title"],
                        "app_id": g["app_id"],
                        "shop_name": "Steam",
                        "date": start_date.strftime("%Y-%m-%d"),
                        "window_end": window_end.strftime("%Y-%m-%d"),
                        "duration_days": duration_days,
                        "predicted_discount": pred["predicted_discount_pct"],
                        "probability": pred["probability"],
                        "model": "bucketed_two_stage",
                    }
                )
        except Exception as e:
            print(f"  ❌ {g['game_title']}: {e}")

    if rows:
        saved = save_predictions(rows, filename="sale_predictions_all.csv")["data"]
        print(f"\nSaved predictions: {saved['path']} ({saved['rows']} rows)")


def main():
    p = argparse.ArgumentParser(description="Steam Sales Window Predictor")
    p.add_argument("--app-id", help="App ID của game cụ thể")
    p.add_argument("--n", type=int, default=5, help="Số lần sale dự đoán")
    p.add_argument("--all", action="store_true", help="Dự đoán tất cả games")
    p.add_argument(
        "--compare",
        action="store_true",
        help="Chạy song song baseline và two-stage để so sánh",
    )
    p.add_argument(
        "--baseline",
        action="store_true",
        help="Chỉ chạy baseline (mặc định: two-stage)",
    )
    p.add_argument(
        "--bucketed",
        action="store_true",
        help="Dùng bucketed two-stage (>= 1 năm data)",
    )
    p.add_argument(
        "csv_file",
        nargs="?",
        default="TS_price_history.csv",
        help="Tên file CSV (mặc định: TS_price_history.csv)",
    )
    args = p.parse_args()

    print("""
╔══════════════════════════════════════════╗
║       Steam Sales Window Predictor      ║
╚══════════════════════════════════════════╝""")

    if args.all:
        run_all(args.n, args.csv_file)
    elif args.app_id:
        run_single(
            args.app_id,
            args.n,
            args.csv_file,
            args.compare,
            args.baseline,
            args.bucketed,
        )
    else:
        print("⚠️  Dùng --app-id <app_id> hoặc --all")
        print("   VD: python main.py --app-id <app_id> --n 3")


if __name__ == "__main__":
    main()
