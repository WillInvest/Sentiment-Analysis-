"""Data exploration plots for the report.

Generates:
  - results/figures/eda_token_length.png    — token length distribution with 512 cutoff
  - results/figures/eda_articles_per_month.png — coverage over time
  - results/figures/eda_top_tickers.png     — top 20 tickers by article count

Usage:
    python scripts/data_exploration.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DATASET = ROOT / "data/processed/full_dataset.parquet"
FIG_DIR = ROOT / "results/figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("Loading processed dataset…")
    df = pd.read_parquet(DATASET)
    print(f"  {len(df):,} rows")

    text_col = "body" if "body" in df.columns else "text"

    # ---- 1. Token length distribution ----
    print("Tokenizing a sample to measure token lengths…")
    tok = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    sample = df.sample(min(5000, len(df)), random_state=42)[text_col].fillna("").tolist()
    lens = np.array([len(tok.encode(t, add_special_tokens=True, truncation=False)) for t in sample])
    print(f"  median: {int(np.median(lens))}  mean: {int(lens.mean())}  max: {lens.max()}")
    print(f"  fraction over 512: {(lens > 512).mean():.1%}")

    fig, ax = plt.subplots(figsize=(9, 4.2))
    clipped = np.clip(lens, 0, 2500)
    ax.hist(clipped, bins=60, color="#4C72B0", edgecolor="white")
    ax.axvline(512, color="red", ls="--", lw=1.5, label="FinBERT 512-token limit")
    over = (lens > 512).mean() * 100
    ax.set_title(f"FinBERT token length per article (n={len(lens)} sample)\n"
                 f"median={int(np.median(lens))}, "
                 f"max={int(lens.max())}, "
                 f"{over:.0f}% exceed 512 tokens")
    ax.set_xlabel("number of tokens")
    ax.set_ylabel("number of articles")
    ax.legend()
    plt.tight_layout()
    out = FIG_DIR / "eda_token_length.png"
    fig.savefig(out, dpi=140)
    print(f"  wrote {out}")
    plt.close(fig)

    # ---- 2. Articles per month ----
    print("Plotting articles per month…")
    by_month = df.set_index("date").resample("ME").size()
    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.bar(by_month.index, by_month.values, width=20, color="#55A868", edgecolor="white")
    ax.set_title(f"Articles per month (total {len(df):,} rows)")
    ax.set_xlabel("month")
    ax.set_ylabel("number of articles")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out = FIG_DIR / "eda_articles_per_month.png"
    fig.savefig(out, dpi=140)
    print(f"  wrote {out}")
    plt.close(fig)

    # ---- 3. Top tickers ----
    print("Plotting top 20 tickers…")
    counts = df["ticker"].value_counts().head(20)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(counts.index, counts.values, color="#C44E52", edgecolor="white")
    ax.set_title(f"Top 20 tickers by article count "
                 f"(out of {df['ticker'].nunique()} total tickers)")
    ax.set_xlabel("ticker")
    ax.set_ylabel("articles")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out = FIG_DIR / "eda_top_tickers.png"
    fig.savefig(out, dpi=140)
    print(f"  wrote {out}")
    plt.close(fig)

    print("\nSummary stats for the report:")
    print(f"  total rows:          {len(df):,}")
    print(f"  unique tickers:      {df['ticker'].nunique()}")
    print(f"  date range:          {df['date'].min().date()} → {df['date'].max().date()}")
    print(f"  median token length: {int(np.median(lens))}")
    print(f"  pct over 512 tokens: {(lens > 512).mean():.1%}")
    print(f"  top ticker:          {counts.index[0]} ({counts.iloc[0]:,} articles)")


if __name__ == "__main__":
    main()
