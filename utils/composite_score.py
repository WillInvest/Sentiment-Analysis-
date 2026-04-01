"""Composite score for evaluating overall pipeline quality.

Combines stock-level and market-level metrics into a single [0, 1] number.
Used by the autonomous agent to decide whether an experiment improved things.
"""

import json
from pathlib import Path

import pandas as pd

_R2_CEILING = 0.05

_WEIGHTS = {
    "stock_r2": 0.25,
    "stock_acc": 0.15,
    "stock_f1": 0.15,
    "market_r2": 0.30,
    "market_sig": 0.15,
}


def composite_score(eval_results: dict, market_trend: list) -> float:
    """Compute composite score in [0, 1]. Higher is better."""
    aggregated = pd.DataFrame(eval_results["aggregated"])
    trend_df = pd.DataFrame(market_trend)

    ft = aggregated[aggregated["model_type"] == "finetuned"]

    stock_r2 = min(1.0, max(0.0, ft["r2_mean"].max()) / _R2_CEILING)
    stock_acc = ft["learned_accuracy_mean"].max()
    stock_f1 = ft["learned_f1_mean"].max()

    market_r2 = trend_df["ols_r2"].max()
    best_idx = trend_df["ols_r2"].idxmax()
    best_p = trend_df.loc[best_idx, "pearson_p"]
    market_sig = max(0.0, 1.0 - best_p)

    score = (
        _WEIGHTS["stock_r2"] * stock_r2
        + _WEIGHTS["stock_acc"] * stock_acc
        + _WEIGHTS["stock_f1"] * stock_f1
        + _WEIGHTS["market_r2"] * market_r2
        + _WEIGHTS["market_sig"] * market_sig
    )
    return round(score, 4)


def compute_from_files(
    eval_path: str = "results/metrics/evaluation_results.json",
    trend_path: str = "results/metrics/market_trend.json",
) -> float:
    """Compute composite score from result files on disk."""
    eval_results = json.loads(Path(eval_path).read_text())
    market_trend = json.loads(Path(trend_path).read_text())
    return composite_score(eval_results, market_trend)


def print_score() -> None:
    """Print composite score and component breakdown."""
    eval_results = json.loads(Path("results/metrics/evaluation_results.json").read_text())
    market_trend = json.loads(Path("results/metrics/market_trend.json").read_text())

    aggregated = pd.DataFrame(eval_results["aggregated"])
    trend_df = pd.DataFrame(market_trend)
    ft = aggregated[aggregated["model_type"] == "finetuned"]

    raw_r2 = ft["r2_mean"].max()
    raw_acc = ft["learned_accuracy_mean"].max()
    raw_f1 = ft["learned_f1_mean"].max()
    raw_mkt_r2 = trend_df["ols_r2"].max()
    best_p = trend_df.loc[trend_df["ols_r2"].idxmax(), "pearson_p"]

    score = composite_score(eval_results, market_trend)

    print("=" * 50)
    print("COMPOSITE SCORE BREAKDOWN")
    print("=" * 50)
    print(f"  Stock R²:      {raw_r2:.4f} (norm: {min(1.0, max(0, raw_r2) / _R2_CEILING):.3f}) × {_WEIGHTS['stock_r2']}")
    print(f"  Stock Acc:     {raw_acc:.4f} × {_WEIGHTS['stock_acc']}")
    print(f"  Stock F1:      {raw_f1:.4f} × {_WEIGHTS['stock_f1']}")
    print(f"  Market R²:     {raw_mkt_r2:.4f} × {_WEIGHTS['market_r2']}")
    print(f"  Market Sig:    {1 - best_p:.4f} (p={best_p:.4f}) × {_WEIGHTS['market_sig']}")
    print(f"  {'─' * 40}")
    print(f"  COMPOSITE:     {score:.4f}")
    print("=" * 50)
