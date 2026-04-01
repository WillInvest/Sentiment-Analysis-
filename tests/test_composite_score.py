import pytest
from utils.composite_score import composite_score


def test_baseline_score():
    """Composite score for current baseline should be ~0.44."""
    eval_results = {
        "aggregated": [
            {"model_type": "finetuned", "r2_mean": 0.0039,
             "learned_accuracy_mean": 0.5617, "learned_f1_mean": 0.538},
            {"model_type": "pretrained", "r2_mean": -0.01,
             "learned_accuracy_mean": 0.50, "learned_f1_mean": 0.49},
        ]
    }
    market_trend = [
        {"ols_r2": 0.3764, "pearson_p": 0.0447},
        {"ols_r2": 0.0234, "pearson_p": 0.31},
    ]
    score = composite_score(eval_results, market_trend)
    assert 0.40 < score < 0.50, f"Expected ~0.44, got {score}"


def test_negative_r2_clamped_to_zero():
    """Negative R² should be clamped to 0, not drag score down."""
    eval_results = {
        "aggregated": [
            {"model_type": "finetuned", "r2_mean": -0.05,
             "learned_accuracy_mean": 0.50, "learned_f1_mean": 0.45},
        ]
    }
    market_trend = [{"ols_r2": 0.10, "pearson_p": 0.20}]
    score = composite_score(eval_results, market_trend)
    assert score >= 0.0


def test_perfect_score_near_one():
    """A hypothetically perfect model should score near 1.0."""
    eval_results = {
        "aggregated": [
            {"model_type": "finetuned", "r2_mean": 0.05,
             "learned_accuracy_mean": 0.95, "learned_f1_mean": 0.95},
        ]
    }
    market_trend = [{"ols_r2": 0.90, "pearson_p": 0.001}]
    score = composite_score(eval_results, market_trend)
    assert 0.90 < score <= 1.0, f"Expected near 1.0, got {score}"


def test_score_increases_with_better_metrics():
    """Improving any metric should increase the score."""
    base = {
        "aggregated": [
            {"model_type": "finetuned", "r2_mean": 0.004,
             "learned_accuracy_mean": 0.56, "learned_f1_mean": 0.54},
        ]
    }
    trend = [{"ols_r2": 0.38, "pearson_p": 0.045}]
    base_score = composite_score(base, trend)

    improved = {
        "aggregated": [
            {"model_type": "finetuned", "r2_mean": 0.01,
             "learned_accuracy_mean": 0.58, "learned_f1_mean": 0.56},
        ]
    }
    improved_score = composite_score(improved, trend)
    assert improved_score > base_score
