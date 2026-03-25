#!/bin/bash
# Run the full pipeline. Re-runnable — skips completed stages.
set -e

echo "=== Data Preparation ==="
python scripts/data_preparation.py

echo "=== Extract Embeddings ==="
python scripts/extract_embeddings.py

echo "=== Pretrained Sentiment ==="
python scripts/pretrained_sentiment.py

echo "=== Fine-tuning ==="
python scripts/finetune.py

echo "=== Evaluation ==="
python scripts/evaluate.py

echo "=== Market Trend Analysis ==="
python scripts/market_trend.py

echo "=== Done! Open notebooks/results_and_analysis.ipynb for plots ==="
python -c "from utils.progress import print_progress_summary; print_progress_summary()"
