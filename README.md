# Financial News Sentiment Analysis for S&P 500 Return Prediction

This is a class project applying FinBERT-based sentiment analysis to Wall Street Journal financial news (2017–2020) to predict next-day returns of S&P 500 stocks. The full write-up is in `docs/report.pdf`.

The central challenge the project tackles is FinBERT's 512-token context window: about 63% of WSJ articles are longer than that, so a naive head-only approach loses the body of most of the corpus. The project uses a sentence-aware sliding window with confidence-weighted pooling and a chunk-as-row architecture where all chunks of one article are combined inside the forward pass, so each article-ticker pair contributes exactly one loss term regardless of how many chunks it produces.

## What's in the repo

- `scripts/pipeline.py` — the full pipeline: raw data → chunk features → trained regressor and classifier → metrics. Resumable at every stage.
- `scripts/pipeline_monitor.py` — live status monitor for the pipeline.
- `scripts/hp_sweep.py` — hyperparameter sweep over hidden layers × widths × activations.
- `scripts/market_trend.py` — monthly sentiment vs. next-month SPX analysis.
- `scripts/trading_strategy.py` — backtest of three trading strategies with take-profit and stop-loss rules.
- `scripts/data_exploration.py` — generates the EDA plots used in the report.
- `scripts/sentiment_ablation.py` — ablation comparing sliding-window sentiment to naive head-only truncation.
- `configs/experiment.yaml` — project configuration (seed, horizons, rolling windows, etc.).
- `docs/report.pdf`, `docs/report.tex` — the final report and its LaTeX source.
- `results/metrics/` — summary metrics from each stage.
- `results/figures/` — the plots used in the report.

## How to run

The raw `news.csv` and `price.csv` files are not in the repo (they are large and not redistributable). Place them in `data/raw/` before running anything.

```bash
python scripts/pipeline.py        # end-to-end pipeline, stages are resumable
python scripts/hp_sweep.py        # hyperparameter sweep
python scripts/market_trend.py    # monthly sentiment vs. SPX trend
python scripts/trading_strategy.py  # trading backtest
python scripts/data_exploration.py  # EDA plots
```

Live progress while the pipeline runs:

```bash
python scripts/pipeline_monitor.py
```

All intermediate artifacts are cached, so re-running any script skips stages that are already done. Iterating on the MLP only needs to delete the relevant checkpoints — the expensive FinBERT pass is reused.

## Dependencies

Python 3.10+ with `torch`, `transformers`, `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `tqdm`, `pyarrow`, `pyyaml`. A GPU is strongly recommended for the FinBERT pass (~10 minutes on an RTX 4000 for the full corpus).
