"""Compute all evaluation metrics.

Evaluates both pretrained sentiment and fine-tuned predictions:
- Regression: R², MSE per model/horizon/window
- Binary: accuracy, F1 with fixed and learned thresholds
- Comparison: separate vs. single approach

Saves results to results/metrics/evaluation_results.json

Usage: python scripts/evaluate.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.data_loader import load_config
from utils.metrics import compute_regression_metrics, compute_binary_metrics, sweep_threshold
from utils.progress import check_progress, update_progress


def evaluate_pretrained(config: dict) -> list[dict]:
    """Evaluate pretrained zero-shot sentiment models."""
    results = []
    horizons = config["horizons"]

    zero_shot_models = {
        name: cfg for name, cfg in config["models"].items()
        if "zero_shot" in cfg["roles"]
    }

    for model_name in zero_shot_models:
        sent_path = Path(f"results/predictions/{model_name}_sentiment.parquet")
        if not sent_path.exists():
            print(f"  Skipping {model_name}: no sentiment file")
            continue

        sentiment_df = pd.read_parquet(sent_path)

        for window in config["windows"]:
            wname = window["name"]
            test_df = pd.read_parquet(f"data/processed/{wname}_test.parquet")
            val_df = pd.read_parquet(f"data/processed/{wname}_val.parquet")

            # Merge sentiment scores with test data
            test_merged = test_df.merge(sentiment_df, on="article_id", how="inner")
            val_merged = val_df.merge(sentiment_df, on="article_id", how="inner")

            for horizon in horizons:
                h_col = f"r_{horizon}d"
                test_returns = test_merged[h_col].values.astype(float)
                test_scores = test_merged["sentiment_score"].values.astype(float)
                val_returns = val_merged[h_col].values.astype(float)
                val_scores = val_merged["sentiment_score"].values.astype(float)

                # Binary with fixed threshold
                fixed_metrics = compute_binary_metrics(test_returns, test_scores, threshold=0.0)

                # Binary with learned threshold (sweep on val)
                best_thresh, _ = sweep_threshold(
                    val_returns, val_scores, step=config["threshold_sweep"]["step"]
                )
                learned_metrics = compute_binary_metrics(test_returns, test_scores, threshold=best_thresh)

                results.append({
                    "model": model_name,
                    "model_type": "pretrained",
                    "approach": "zero_shot",
                    "horizon": h_col,
                    "window": wname,
                    "fixed_accuracy": fixed_metrics["accuracy"],
                    "fixed_f1": fixed_metrics["f1"],
                    "learned_accuracy": learned_metrics["accuracy"],
                    "learned_f1": learned_metrics["f1"],
                    "learned_threshold": best_thresh,
                    "n_samples": fixed_metrics["n_samples"],
                })

    return results


def evaluate_finetuned(config: dict) -> list[dict]:
    """Evaluate fine-tuned model predictions."""
    results = []
    pred_path = Path("results/predictions/finetune_test_predictions.parquet")
    if not pred_path.exists():
        print("  No fine-tuned predictions found.")
        return results

    pred_df = pd.read_parquet(pred_path)

    for (encoder, approach, horizon, window), group in pred_df.groupby(
        ["encoder", "approach", "horizon", "window"]
    ):
        actual = group["actual"].values.astype(float)
        predicted = group["prediction"].values.astype(float)

        # Regression metrics
        reg_metrics = compute_regression_metrics(actual, predicted)

        # Binary with fixed threshold
        fixed_metrics = compute_binary_metrics(actual, predicted, threshold=0.0)

        # Binary with learned threshold — load val predictions saved by finetune.py
        val_pred_path = Path("results/predictions/finetune_val_predictions.parquet")
        best_thresh = 0.0
        if val_pred_path.exists():
            val_pred_df = pd.read_parquet(val_pred_path)
            val_subset = val_pred_df[
                (val_pred_df["encoder"] == encoder) &
                (val_pred_df["approach"] == approach) &
                (val_pred_df["horizon"] == horizon) &
                (val_pred_df["window"] == window)
            ]
            if len(val_subset) > 0:
                val_actual = val_subset["actual"].values.astype(float)
                val_preds = val_subset["prediction"].values.astype(float)
                best_thresh, _ = sweep_threshold(val_actual, val_preds, step=config["threshold_sweep"]["step"])

        learned_metrics = compute_binary_metrics(actual, predicted, threshold=best_thresh)

        results.append({
            "model": encoder,
            "model_type": "finetuned",
            "approach": approach,
            "horizon": horizon,
            "window": window,
            "r2": reg_metrics["r2"],
            "mse": reg_metrics["mse"],
            "fixed_accuracy": fixed_metrics["accuracy"],
            "fixed_f1": fixed_metrics["f1"],
            "learned_accuracy": learned_metrics["accuracy"],
            "learned_f1": learned_metrics["f1"],
            "learned_threshold": best_thresh,
            "n_samples": reg_metrics["n_samples"],
        })

    return results


def aggregate_across_windows(results: list[dict]) -> list[dict]:
    """Compute mean ± std of metrics across rolling windows."""
    df = pd.DataFrame(results)
    if df.empty:
        return []

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    group_cols = ["model", "model_type", "approach", "horizon"]
    group_cols = [c for c in group_cols if c in df.columns]

    aggregated = []
    for keys, group in df.groupby(group_cols):
        row = dict(zip(group_cols, keys if isinstance(keys, tuple) else [keys]))
        for col in numeric_cols:
            if col == "n_samples":
                row[f"{col}_total"] = int(group[col].sum())
            else:
                row[f"{col}_mean"] = float(group[col].mean())
                row[f"{col}_std"] = float(group[col].std())
        aggregated.append(row)

    return aggregated


def main():
    if check_progress("evaluate") == "done":
        print("Evaluation already completed. Skipping.")
        return

    update_progress("evaluate", status="in_progress")

    try:
        config = load_config()

        print("Evaluating pretrained models...")
        pretrained_results = evaluate_pretrained(config)

        print("Evaluating fine-tuned models...")
        finetuned_results = evaluate_finetuned(config)

        all_results = pretrained_results + finetuned_results

        # Aggregate across windows
        aggregated = aggregate_across_windows(all_results)

        # Save
        Path("results/metrics").mkdir(parents=True, exist_ok=True)
        output = {
            "per_window": all_results,
            "aggregated": aggregated,
        }
        Path("results/metrics/evaluation_results.json").write_text(
            json.dumps(output, indent=2)
        )

        print(f"Saved {len(all_results)} per-window results, {len(aggregated)} aggregated results")
        update_progress("evaluate", status="done")

    except Exception as e:
        update_progress("evaluate", status="error", error=str(e))
        raise


if __name__ == "__main__":
    main()
