# Financial News Sentiment Analysis

## Project Overview
NLP sentiment analysis on WSJ financial news (2017-2021) to predict S&P500 stock returns.
See full spec: `docs/superpowers/specs/2026-03-24-financial-sentiment-analysis-design.md`
See full plan: `docs/superpowers/plans/2026-03-24-financial-sentiment-analysis.md`

## Quick Start
```bash
conda activate sentiment
python -c "from utils.progress import print_progress_summary; print_progress_summary()"
```

## How to Check Progress
1. Read `results/progress.json` — machine-readable state of every pipeline stage
2. Run `python -c "from utils.progress import print_progress_summary; print_progress_summary()"` for a summary
3. Check `docs/superpowers/plans/2026-03-24-financial-sentiment-analysis.md` for the full task list with checkboxes

## Pipeline Order
1. `scripts/data_preparation.py` — load CSVs, compute returns, create splits
2. `scripts/extract_embeddings.py` — extract [CLS] embeddings from frozen encoders
3. `scripts/pretrained_sentiment.py` — zero-shot sentiment (FinBERT, RoBERTa, Llama)
4. `scripts/finetune.py` — train FC heads on cached embeddings
5. `scripts/evaluate.py` — compute all metrics
6. `scripts/market_trend.py` — monthly aggregation + SPX analysis
7. `notebooks/results_and_analysis.ipynb` — load results, generate plots

## Resumability
- All scripts check `results/progress.json` before starting and skip completed work.
- Fine-tuning saves checkpoints to `results/checkpoints/`.
- If training crashes, re-run the same script — it picks up where it left off.
- Embeddings are cached in `results/embeddings/` — extracted once, reused everywhere.

## Data
Place `news.csv` and `price.csv` in `data/raw/` before running.

## Key Conventions
- Seed: 42 for all random operations
- Horizons: 1d, 3d, 5d, 10d, 30d (trading days, forward-looking from t+1)
- Windows: 3 expanding rolling windows (see configs/experiment.yaml)
- Models: BERT-base (finetune only), FinBERT (both), RoBERTa (both), Llama-3.2-1B (zero-shot only)
