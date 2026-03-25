"""Data preparation pipeline.

Loads raw CSVs, computes forward returns, creates rolling window splits,
saves processed data and statistics. Resumable via progress.json.

Usage: python scripts/data_preparation.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.data_loader import load_config, load_raw_data, build_dataset, create_splits
from utils.progress import check_progress, update_progress


def save_data_stats(df: pd.DataFrame, splits: dict, config: dict) -> None:
    """Save dataset statistics for the report."""
    stats = {
        "total_article_ticker_pairs": len(df),
        "unique_articles": df["article_id"].nunique(),
        "unique_tickers": df["ticker"].nunique(),
        "date_range": [str(df["date"].min().date()), str(df["date"].max().date())],
        "horizons": config["horizons"],
        "horizon_availability": {},
        "windows": {},
    }

    for h in config["horizons"]:
        col = f"r_{h}d"
        stats["horizon_availability"][col] = int(df[col].notna().sum())

    for name, split in splits.items():
        stats["windows"][name] = {
            "train": len(split["train"]),
            "val": len(split["val"]),
            "test": len(split["test"]),
        }

    Path("results/metrics").mkdir(parents=True, exist_ok=True)
    Path("results/metrics/data_stats.json").write_text(json.dumps(stats, indent=2))
    print(f"Data stats saved. Total pairs: {stats['total_article_ticker_pairs']}, "
          f"Unique articles: {stats['unique_articles']}, Tickers: {stats['unique_tickers']}")


def main():
    if check_progress("data_preparation") == "done":
        print("Data preparation already completed. Skipping.")
        return

    update_progress("data_preparation", status="in_progress")

    try:
        config = load_config()

        # Set seed
        np.random.seed(config["seed"])

        print("Loading raw data...")
        news, price = load_raw_data()
        print(f"News: {len(news)} articles, Price: {len(price)} rows")

        print("Building dataset (computing returns)...")
        df = build_dataset(news, price, config)
        print(f"Built {len(df)} article-ticker pairs")

        print("Creating rolling window splits...")
        splits = create_splits(df, config)

        # Save
        Path("data/processed").mkdir(parents=True, exist_ok=True)
        df.to_parquet("data/processed/full_dataset.parquet", index=False)

        for name, split in splits.items():
            for part in ["train", "val", "test"]:
                split[part].to_parquet(f"data/processed/{name}_{part}.parquet", index=False)

        save_data_stats(df, splits, config)
        update_progress("data_preparation", status="done")
        print("Data preparation complete.")

    except Exception as e:
        update_progress("data_preparation", status="error", error=str(e))
        raise


if __name__ == "__main__":
    main()
