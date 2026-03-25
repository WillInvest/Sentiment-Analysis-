"""Evaluation metrics and threshold sweep utilities."""

import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, f1_score


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute R² and MSE, handling NaN values."""
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if mask.sum() < 2:
        return {"r2": float("nan"), "mse": float("nan"), "n_samples": 0}
    y_t, y_p = y_true[mask], y_pred[mask]
    return {
        "r2": float(r2_score(y_t, y_p)),
        "mse": float(mean_squared_error(y_t, y_p)),
        "n_samples": int(mask.sum()),
    }


def compute_binary_metrics(y_true_returns: np.ndarray, predictions: np.ndarray, threshold: float = 0.0) -> dict:
    """Convert predictions to binary (above/below threshold) and compute metrics.

    y_true_returns: actual returns (positive = up)
    predictions: either sentiment scores or predicted returns
    threshold: cutoff for predictions
    """
    mask = ~(np.isnan(y_true_returns) | np.isnan(predictions))
    if mask.sum() < 2:
        return {"accuracy": float("nan"), "f1": float("nan"), "n_samples": 0}

    y_true_binary = (y_true_returns[mask] > 0).astype(int)
    y_pred_binary = (predictions[mask] > threshold).astype(int)

    return {
        "accuracy": float(accuracy_score(y_true_binary, y_pred_binary)),
        "f1": float(f1_score(y_true_binary, y_pred_binary, average="macro")),
        "n_samples": int(mask.sum()),
    }


def sweep_threshold(
    y_true_returns: np.ndarray,
    predictions: np.ndarray,
    step: float = 0.001,
) -> tuple[float, dict]:
    """Sweep thresholds on predictions to maximize macro F1.

    Returns (best_threshold, best_metrics_dict).
    """
    mask = ~(np.isnan(y_true_returns) | np.isnan(predictions))
    if mask.sum() < 2:
        return 0.0, {"accuracy": float("nan"), "f1": float("nan"), "n_samples": 0}

    preds = predictions[mask]
    min_pred, max_pred = float(preds.min()), float(preds.max())

    best_threshold = 0.0
    best_f1 = -1.0
    best_metrics = {}

    thresholds = np.arange(min_pred, max_pred + step, step)
    for t in thresholds:
        metrics = compute_binary_metrics(y_true_returns, predictions, threshold=t)
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_threshold = float(t)
            best_metrics = metrics

    best_metrics["threshold"] = best_threshold
    return best_threshold, best_metrics
