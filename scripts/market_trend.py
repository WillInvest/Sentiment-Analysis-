"""Sentiment vs. market trend analysis.

For each model:
  1. Aggregate per-article sentiment to a monthly average.
  2. Compare month t sentiment to month t+1 SPX return.
  3. Fit a linear regression and report slope, intercept, R², correlation.

Models compared:
  - PRETRAINED (zero-shot FinBERT): confidence-weighted (pos − neg), all 48 months.
  - FINE-TUNED (regressor):         predicted r_1d, only 18 out-of-sample months
                                    (the union of all 3 windows' test sets).

Outputs:
  results/metrics/market_trend.json
  results/metrics/market_trend_panel.csv
  results/figures/market_trend_scatter.png
  results/figures/market_trend_timeseries.png

Usage:
    python scripts/market_trend.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DATASET = ROOT / "data/processed/full_dataset.parquet"
CHUNKS = ROOT / "results/predictions/finbert_chunks.parquet"
PRICE = ROOT / "data/raw/price.csv"
PRED_DIR = ROOT / "results/predictions"
OUT_JSON = ROOT / "results/metrics/market_trend.json"
OUT_PANEL = ROOT / "results/metrics/market_trend_panel.csv"
OUT_SCATTER = ROOT / "results/figures/market_trend_scatter.png"
OUT_TIMESERIES = ROOT / "results/figures/market_trend_timeseries.png"


def confidence_weighted_pos_minus_neg(chunks: pd.DataFrame) -> pd.DataFrame:
    """Per-article confidence-weighted (pos − neg). Returns DataFrame[article_id, score]."""
    sums = chunks.groupby("article_id", sort=False)["confidence"].transform("sum")
    chunks = chunks.copy()
    chunks["w"] = chunks["confidence"] / sums.replace(0, np.nan)
    chunks["w"] = chunks["w"].fillna(1.0 / chunks.groupby("article_id")["confidence"].transform("size"))
    chunks["weighted_diff"] = (chunks["pos"] - chunks["neg"]) * chunks["w"]
    out = chunks.groupby("article_id", sort=False)["weighted_diff"].sum().reset_index()
    out.columns = ["article_id", "pretrained_score"]
    return out


def monthly_aggregate(df: pd.DataFrame, score_col: str) -> pd.Series:
    """Mean score per calendar month, indexed by month-end timestamp."""
    s = df.set_index("date")[score_col]
    return s.resample("ME").mean()


def spx_monthly_return(price_path: Path) -> pd.Series:
    px = pd.read_csv(price_path, parse_dates=["Date"])
    spx = px[px["ticker"] == "SPX"].sort_values("Date").set_index("Date")["close"]
    monthly_close = spx.resample("ME").last()
    return monthly_close.pct_change().dropna()


def linear_fit(x: np.ndarray, y: np.ndarray) -> dict:
    """Slope, intercept, R², Pearson correlation."""
    if len(x) < 3:
        return {"slope": None, "intercept": None, "r2": None, "corr": None, "n": int(len(x))}
    a, b = np.polyfit(x, y, 1)
    yhat = a * x + b
    ss_res = float(((y - yhat) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    corr = float(np.corrcoef(x, y)[0, 1])
    return {"slope": float(a), "intercept": float(b), "r2": float(r2),
            "corr": corr, "n": int(len(x))}


def main():
    print("Loading dataset and chunks…")
    ds = pd.read_parquet(DATASET)[["article_id", "date"]].drop_duplicates("article_id")
    chunks = pd.read_parquet(CHUNKS)

    print("Computing per-article pretrained sentiment…")
    art_sent = confidence_weighted_pos_minus_neg(chunks)
    pre_articles = ds.merge(art_sent, on="article_id", how="inner")
    pre_monthly = monthly_aggregate(pre_articles, "pretrained_score").rename("pretrained")
    print(f"  pretrained months: {len(pre_monthly)}  "
          f"({pre_monthly.index.min().date()} → {pre_monthly.index.max().date()})")

    print("Loading fine-tuned test predictions…")
    reg_frames = []
    for w in ("w1", "w2", "w3"):
        f = PRED_DIR / f"regressor_{w}_test.parquet"
        df = pd.read_parquet(f)[["article_id", "date", "pred"]]
        df["window"] = w
        reg_frames.append(df)
    reg = pd.concat(reg_frames, ignore_index=True)
    reg_articles = reg.groupby(["article_id", "date"], as_index=False)["pred"].mean()
    ft_monthly = monthly_aggregate(
        reg_articles.rename(columns={"pred": "ft_score"}), "ft_score"
    ).rename("finetuned")
    print(f"  fine-tuned months: {len(ft_monthly)}  "
          f"({ft_monthly.index.min().date()} → {ft_monthly.index.max().date()})")

    print("Loading SPX monthly returns…")
    spx_ret = spx_monthly_return(PRICE).rename("spx_return")
    spx_next = spx_ret.shift(-1).rename("spx_next_return")
    print(f"  SPX months: {len(spx_ret)}")

    panel = pd.concat([pre_monthly, ft_monthly, spx_next], axis=1)
    panel.index.name = "month_end"

    pre_panel = panel[["pretrained", "spx_next_return"]].dropna()
    overlap = panel[["pretrained", "finetuned", "spx_next_return"]].dropna()

    print(f"  pretrained vs SPX(t+1) usable months: {len(pre_panel)}")
    print(f"  overlap (head-to-head):               {len(overlap)}")

    fit_pre_full = linear_fit(pre_panel["pretrained"].to_numpy(),
                              pre_panel["spx_next_return"].to_numpy())
    fit_pre_oos = linear_fit(overlap["pretrained"].to_numpy(),
                             overlap["spx_next_return"].to_numpy())
    fit_ft_oos = linear_fit(overlap["finetuned"].to_numpy(),
                            overlap["spx_next_return"].to_numpy())

    summary = {
        "pretrained_full": {
            **fit_pre_full,
            "months": [str(pre_panel.index.min().date()), str(pre_panel.index.max().date())],
        },
        "head_to_head_pretrained": {
            **fit_pre_oos,
            "months": [str(overlap.index.min().date()), str(overlap.index.max().date())],
        },
        "head_to_head_finetuned": {
            **fit_ft_oos,
            "months": [str(overlap.index.min().date()), str(overlap.index.max().date())],
        },
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(summary, indent=2))
    print(f"  wrote {OUT_JSON}")

    # Scatter plot — 3 panels
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    def scatter_panel(ax, x, y, fit, title, xlabel):
        ax.scatter(x, y, s=42, alpha=0.7, edgecolor="k", linewidth=0.4, color="#4C72B0")
        if fit["slope"] is not None and len(x) >= 2:
            xx = np.linspace(x.min(), x.max(), 50)
            ax.plot(xx, fit["slope"] * xx + fit["intercept"], "r-", lw=1.5)
        ax.axhline(0, color="gray", lw=0.5)
        ax.axvline(0, color="gray", lw=0.5)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("SPX return next month")
        if fit["corr"] is not None:
            sub = f"n={fit['n']}  corr={fit['corr']:.3f}  R²={fit['r2']:.3f}"
        else:
            sub = f"n={fit['n']}"
        ax.set_title(f"{title}\n{sub}")

    scatter_panel(axes[0],
                  pre_panel["pretrained"].to_numpy(),
                  pre_panel["spx_next_return"].to_numpy(),
                  fit_pre_full,
                  "Pretrained (full 4 years)",
                  "monthly mean (pos − neg)")
    scatter_panel(axes[1],
                  overlap["pretrained"].to_numpy(),
                  overlap["spx_next_return"].to_numpy(),
                  fit_pre_oos,
                  "Pretrained (18-month OOS window)",
                  "monthly mean (pos − neg)")
    scatter_panel(axes[2],
                  overlap["finetuned"].to_numpy(),
                  overlap["spx_next_return"].to_numpy(),
                  fit_ft_oos,
                  "Fine-tuned regressor (same 18 months)",
                  "monthly mean predicted r_1d")
    plt.tight_layout()
    OUT_SCATTER.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_SCATTER, dpi=140)
    print(f"  wrote {OUT_SCATTER}")

    # Time series with twin axis
    fig, ax1 = plt.subplots(figsize=(13, 4.5))
    ax2 = ax1.twinx()

    months = panel.index
    ax1.plot(months, panel["pretrained"], color="#4C72B0", marker="o", ms=4,
             label="pretrained sentiment", lw=1.4)
    ft_months = panel["finetuned"].dropna().index
    ax1.plot(ft_months, panel.loc[ft_months, "finetuned"],
             color="#55A868", marker="s", ms=4,
             label="fine-tuned sentiment (OOS)", lw=1.4)
    ax2.plot(months, panel["spx_next_return"], color="#C44E52", marker="x", ms=5,
             label="SPX next-month return", lw=1.2, alpha=0.8)

    ax1.axhline(0, color="gray", lw=0.4)
    ax2.axhline(0, color="gray", lw=0.4, ls="--")
    ax1.set_xlabel("month")
    ax1.set_ylabel("monthly sentiment", color="#4C72B0")
    ax2.set_ylabel("SPX return (next month)", color="#C44E52")
    ax1.tick_params(axis="y", labelcolor="#4C72B0")
    ax2.tick_params(axis="y", labelcolor="#C44E52")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", framealpha=0.9)

    plt.title("Monthly sentiment vs. next-month SPX return")
    plt.tight_layout()
    fig.savefig(OUT_TIMESERIES, dpi=140)
    print(f"  wrote {OUT_TIMESERIES}")

    panel.to_csv(OUT_PANEL)
    print(f"  wrote {OUT_PANEL}")

    print("\nResults summary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
