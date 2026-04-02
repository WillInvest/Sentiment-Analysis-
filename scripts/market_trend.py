"""Monthly sentiment aggregation and SPX market trend analysis.

Aggregates sentiment per month, compares with SPX t+1 return via
scatter plots, OLS regression, and correlation analysis.

Usage: python scripts/market_trend.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.data_loader import load_config
from utils.progress import check_progress, update_progress


def compute_spx_monthly_returns(price_df: pd.DataFrame) -> pd.DataFrame:
    """Compute SPX monthly returns."""
    spx = price_df[price_df["ticker"] == "SPX"].copy()
    spx["date"] = pd.to_datetime(spx["date"])
    spx = spx.sort_values("date")

    # Get last trading day close per month
    spx["month"] = spx["date"].dt.to_period("M")
    monthly = spx.groupby("month")["close"].last().reset_index()
    monthly["spx_return"] = monthly["close"].pct_change()
    monthly = monthly.dropna()

    return monthly


def aggregate_monthly_sentiment(
    full_df: pd.DataFrame,
    sentiment_df: pd.DataFrame,
    model_name: str,
) -> pd.DataFrame:
    """Average sentiment scores per month."""
    merged = full_df.merge(sentiment_df, on="article_id", how="inner")
    merged["month"] = pd.to_datetime(merged["date"]).dt.to_period("M")

    monthly = merged.groupby("month")["sentiment_score"].mean().reset_index()
    monthly.columns = ["month", f"sentiment_{model_name}"]
    return monthly


def aggregate_monthly_finetuned(
    pred_df: pd.DataFrame,
    encoder: str,
    approach: str,
    horizon: str,
    full_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Aggregate fine-tuned predictions per month, normalized to [-1, 1]."""
    subset = pred_df[
        (pred_df["encoder"] == encoder) &
        (pred_df["approach"] == approach) &
        (pred_df["horizon"] == horizon)
    ].copy()

    if subset.empty:
        return pd.DataFrame()

    # Min-max normalize predictions to [-1, 1]
    pmin, pmax = subset["prediction"].min(), subset["prediction"].max()
    if pmax - pmin > 0:
        subset["normalized"] = 2 * (subset["prediction"] - pmin) / (pmax - pmin) - 1
    else:
        subset["normalized"] = 0.0

    # Need date info — merge with test data
    if full_df is None:
        full_df = pd.read_parquet("data/processed/full_dataset.parquet")
    subset = subset.merge(full_df[["article_id", "date"]].drop_duplicates(), on="article_id", how="left")
    subset["month"] = pd.to_datetime(subset["date"]).dt.to_period("M")

    monthly = subset.groupby("month")["normalized"].mean().reset_index()
    col_name = f"sentiment_{encoder}_{approach}_{horizon}"
    monthly.columns = ["month", col_name]
    return monthly


def run_analysis(monthly_sentiment: pd.DataFrame, spx_monthly: pd.DataFrame, col_name: str, label: str) -> dict:
    """Run OLS regression and correlation between sentiment_t and SPX_return_{t+1}."""
    # Align: sentiment month t → SPX return month t+1
    merged = monthly_sentiment.merge(spx_monthly, on="month", how="inner")

    # Shift: compare sentiment_t with SPX return at t+1
    merged = merged.sort_values("month")
    merged["spx_return_next"] = merged["spx_return"].shift(-1)
    merged = merged.dropna(subset=["spx_return_next", col_name])

    if len(merged) < 5:
        return {"label": label, "n_months": len(merged), "error": "insufficient data"}

    X = merged[col_name].values
    y = merged["spx_return_next"].values

    # OLS
    X_ols = sm.add_constant(X)
    ols_result = sm.OLS(y, X_ols).fit()

    # Correlations
    pearson_r, pearson_p = stats.pearsonr(X, y)
    spearman_r, spearman_p = stats.spearmanr(X, y)

    result = {
        "label": label,
        "n_months": len(merged),
        "ols_beta": float(ols_result.params[1]),
        "ols_alpha": float(ols_result.params[0]),
        "ols_r2": float(ols_result.rsquared),
        "ols_p_value": float(ols_result.pvalues[1]),
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "spearman_r": float(spearman_r),
        "spearman_p": float(spearman_p),
    }

    # Save scatter plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(X, y, alpha=0.7)
    ax.plot(X, ols_result.predict(X_ols), color="red", linewidth=2)
    ax.set_xlabel(f"Monthly Sentiment ({label})")
    ax.set_ylabel("SPX Return (t+1)")
    ax.set_title(f"Sentiment vs SPX Next-Month Return\n{label} (R²={result['ols_r2']:.4f}, p={result['ols_p_value']:.4f})")
    fig.tight_layout()
    fig.savefig(f"results/figures/scatter_{label.replace(' ', '_').lower()}.png", dpi=150)
    plt.close(fig)

    return result


def main():
    if check_progress("market_trend") == "done":
        print("Market trend analysis already completed. Skipping.")
        return

    update_progress("market_trend", status="in_progress")

    try:
        config = load_config()
        Path("results/figures").mkdir(parents=True, exist_ok=True)

        # Load price data for SPX returns
        price_df = pd.read_csv("data/raw/price.csv")
        price_df.rename(columns={"Date": "date"}, inplace=True)
        price_df["date"] = pd.to_datetime(price_df["date"])
        spx_monthly = compute_spx_monthly_returns(price_df)

        full_df = pd.read_parquet("data/processed/full_dataset.parquet")

        all_results = []

        # ─── Pretrained models ───
        zero_shot_models = {
            name: cfg for name, cfg in config["models"].items()
            if "zero_shot" in cfg["roles"]
        }

        for model_name in zero_shot_models:
            sent_path = Path(f"results/predictions/{model_name}_sentiment.parquet")
            if not sent_path.exists():
                continue

            sentiment_df = pd.read_parquet(sent_path)
            monthly = aggregate_monthly_sentiment(full_df, sentiment_df, model_name)
            col_name = f"sentiment_{model_name}"
            result = run_analysis(monthly, spx_monthly, col_name, f"Pretrained {model_name}")
            all_results.append(result)

        # ─── Fine-tuned models (evaluate all horizons for monthly analysis) ───
        pred_path = Path("results/predictions/finetune_test_predictions.parquet")
        if pred_path.exists():
            pred_df = pd.read_parquet(pred_path)

            finetuned_horizons = [f"r_{h}d" for h in config["horizons"]]

            for encoder in pred_df["encoder"].unique():
                for approach in pred_df["approach"].unique():
                    for horizon in finetuned_horizons:
                        # Skip if this horizon doesn't exist for this encoder/approach combo
                        subset_check = pred_df[
                            (pred_df["encoder"] == encoder) &
                            (pred_df["approach"] == approach) &
                            (pred_df["horizon"] == horizon)
                        ]
                        if subset_check.empty:
                            continue

                        monthly = aggregate_monthly_finetuned(pred_df, encoder, approach, horizon, full_df=full_df)
                        if monthly.empty:
                            continue
                        col_name = f"sentiment_{encoder}_{approach}_{horizon}"
                        label = f"Finetuned {encoder} ({approach}, {horizon})"
                        result = run_analysis(monthly, spx_monthly, col_name, label)
                        all_results.append(result)

        # ─── Time series overlay plot ───
        _plot_time_series_overlay(config, spx_monthly, full_df)

        # Save results
        Path("results/metrics/market_trend.json").write_text(json.dumps(all_results, indent=2))
        print(f"Market trend analysis complete. {len(all_results)} model comparisons saved.")

        update_progress("market_trend", status="done")

    except Exception as e:
        update_progress("market_trend", status="error", error=str(e))
        raise


def _plot_time_series_overlay(config, spx_monthly, full_df):
    """Plot monthly sentiment and SPX return time series overlay."""
    zero_shot_models = {
        name: cfg for name, cfg in config["models"].items()
        if "zero_shot" in cfg["roles"]
    }

    fig, axes = plt.subplots(len(zero_shot_models), 1, figsize=(14, 4 * len(zero_shot_models)), sharex=True)
    if len(zero_shot_models) == 1:
        axes = [axes]

    for ax, model_name in zip(axes, zero_shot_models):
        sent_path = Path(f"results/predictions/{model_name}_sentiment.parquet")
        if not sent_path.exists():
            continue

        sentiment_df = pd.read_parquet(sent_path)
        monthly = aggregate_monthly_sentiment(full_df, sentiment_df, model_name)
        col = f"sentiment_{model_name}"

        merged = monthly.merge(spx_monthly, on="month", how="inner").sort_values("month")
        merged["month_dt"] = merged["month"].dt.to_timestamp()

        ax.plot(merged["month_dt"], merged[col], "b-o", label="Sentiment", markersize=4)
        ax2 = ax.twinx()
        ax2.plot(merged["month_dt"], merged["spx_return"].shift(-1), "r-s", label="SPX Return (t+1)", markersize=4, alpha=0.7)

        ax.set_ylabel("Sentiment", color="blue")
        ax2.set_ylabel("SPX Return (t+1)", color="red")
        ax.set_title(f"{model_name}")
        ax.legend(loc="upper left")
        ax2.legend(loc="upper right")

    fig.suptitle("Monthly Sentiment vs SPX Next-Month Return", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig("results/figures/time_series_overlay.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
