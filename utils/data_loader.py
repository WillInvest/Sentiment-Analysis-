"""Data loading, return computation, and split creation.

Key responsibilities:
- Load news.csv and price.csv
- Expand multi-ticker articles into (article, ticker) pairs
- Compute forward-looking returns at 5 horizons using trading days only
- Shift non-trading-day publication dates to next trading day
- Create expanding rolling window splits per configs/experiment.yaml
"""

import re

import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from utils.text_processing import clean_text


def load_config() -> dict:
    return yaml.safe_load(Path("configs/experiment.yaml").read_text())


def load_raw_data(data_dir: str = "data/raw") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load news.csv and price.csv."""
    news = pd.read_csv(f"{data_dir}/news.csv")
    price = pd.read_csv(f"{data_dir}/price.csv")
    # Normalize price Date column (capital D in actual file) to lowercase
    price.rename(columns={"Date": "date"}, inplace=True)
    return news, price


def get_trading_days(price_df: pd.DataFrame, ticker: str = "SPX") -> pd.DatetimeIndex:
    """Get sorted trading days from price data for a reference ticker."""
    ref = price_df[price_df["ticker"] == ticker].copy()
    ref["date"] = pd.to_datetime(ref["date"])
    return ref["date"].sort_values().reset_index(drop=True)


def shift_to_next_trading_day(date: pd.Timestamp, trading_days: pd.DatetimeIndex) -> pd.Timestamp | None:
    """If date is not a trading day, shift to next trading day."""
    future = trading_days[trading_days >= date]
    if len(future) == 0:
        return None
    return future[0]


def compute_forward_returns(
    price_df: pd.DataFrame,
    ticker: str,
    base_date: pd.Timestamp,
    horizons: list[int],
    trading_days_for_ticker: pd.DatetimeIndex,
) -> dict[str, float | None]:
    """Compute forward returns from base_date for given horizons.

    Returns dict like {'r_1d': 0.012, 'r_3d': 0.034, ...}.
    Values are None if horizon extends beyond available data.
    """
    # Find index of base_date in trading days
    idx_arr = trading_days_for_ticker.get_indexer([base_date])
    if idx_arr[0] == -1:
        return {f"r_{h}d": None for h in horizons}
    base_idx = idx_arr[0]

    ticker_prices = price_df[price_df["ticker"] == ticker].set_index("date")["close"]
    p_t = ticker_prices.get(base_date)
    if p_t is None or p_t == 0:
        return {f"r_{h}d": None for h in horizons}

    results = {}
    for h in horizons:
        target_idx = base_idx + h
        if target_idx >= len(trading_days_for_ticker):
            results[f"r_{h}d"] = None
        else:
            target_date = trading_days_for_ticker[target_idx]
            p_target = ticker_prices.get(target_date)
            if p_target is None:
                results[f"r_{h}d"] = None
            else:
                results[f"r_{h}d"] = (p_target - p_t) / p_t
    return results


def build_dataset(news_df: pd.DataFrame, price_df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Build full dataset: expand multi-ticker, compute returns, clean text.

    Returns a DataFrame with columns:
    - article_id, ticker, date, text, cleaned_text
    - r_1d, r_3d, r_5d, r_10d, r_30d
    """
    horizons = config["horizons"]
    price_df = price_df.copy()
    price_df["date"] = pd.to_datetime(price_df["date"])

    # Get all trading days per ticker
    all_trading_days = {}
    for t in price_df["ticker"].unique():
        days = pd.DatetimeIndex(price_df[price_df["ticker"] == t]["date"].sort_values().values)
        all_trading_days[t] = days

    # Reference trading days (SPX) for date shifting
    spx_trading_days = all_trading_days.get("SPX", pd.DatetimeIndex([]))

    rows = []
    for idx, row in news_df.iterrows():
        # Handle multi-ticker: split on comma, semicolon, or space
        tickers_raw = str(row.get("tickers", ""))
        tickers = [t.strip() for t in re.split(r"[,;\s]+", tickers_raw) if t.strip()]

        # news.csv uses publication_datetime for the date column
        pub_date = pd.to_datetime(
            row.get("publication_datetime", row.get("date", row.get("pub_date", row.get("publication_date"))))
        )
        if pd.isna(pub_date):
            continue

        # Shift to next trading day if needed
        shifted_date = shift_to_next_trading_day(pub_date, spx_trading_days)
        if shifted_date is None:
            continue

        # news.csv has title and body columns; concatenate for richer text
        title = str(row.get("title", ""))
        body = str(row.get("body", row.get("text", row.get("article", row.get("headline", "")))))
        text = f"{title} {body}".strip() if title else body
        cleaned = clean_text(text)

        for ticker in tickers:
            if ticker not in all_trading_days:
                continue

            ticker_days = all_trading_days[ticker]
            # Shift for this specific ticker's trading days
            ticker_shifted = shift_to_next_trading_day(pub_date, ticker_days)
            if ticker_shifted is None:
                continue

            returns = compute_forward_returns(price_df, ticker, ticker_shifted, horizons, ticker_days)

            # Include if at least the shortest horizon is available
            if returns.get("r_1d") is None:
                continue

            rows.append({
                "article_id": idx,
                "ticker": ticker,
                "date": ticker_shifted,
                "original_pub_date": pub_date,
                "text": text,
                "cleaned_text": cleaned,
                **returns,
            })

    df = pd.DataFrame(rows)
    return df


def create_splits(df: pd.DataFrame, config: dict) -> dict:
    """Create expanding rolling window splits.

    Returns dict of {window_name: {'train': df, 'val': df, 'test': df}}.
    """
    splits = {}
    for window in config["windows"]:
        name = window["name"]
        train = df[(df["date"] >= window["train_start"]) & (df["date"] <= window["train_end"])]
        val = df[(df["date"] >= window["val_start"]) & (df["date"] <= window["val_end"])]
        test = df[(df["date"] >= window["test_start"]) & (df["date"] <= window["test_end"])]
        splits[name] = {"train": train, "val": val, "test": test}
    return splits
