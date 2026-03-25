# Financial News Sentiment Analysis — Experiment Design

## Overview

This project applies NLP techniques to analyze sentiment in financial news and investigate its relationship with stock price movements. We compare pretrained language models and fine-tune them for return prediction, evaluating effectiveness across multiple prediction horizons.

## Data

### Sources
- **news.csv**: Wall Street Journal articles about S&P500 firms (2017–2021). Key columns: article text, publication date, `tickers`.
- **price.csv**: Daily closing prices for S&P500 firms and SPX index (2017–2022). Key column: `ticker`.

### Preprocessing
1. Load both CSVs. For multi-ticker articles (`tickers` column may contain multiple tickers), create one (article, ticker) pair per ticker — the same article text is paired with each ticker's return separately. Drop articles with no valid ticker match.
2. Compute forward-looking returns for each (article, ticker) pair at 5 horizons:
   - `r_1d = (p_{t+1} - p_t) / p_t`
   - `r_3d = (p_{t+3} - p_t) / p_t`
   - `r_5d = (p_{t+5} - p_t) / p_t`
   - `r_10d = (p_{t+10} - p_t) / p_t`
   - `r_30d = (p_{t+30} - p_t) / p_t`
   - All day counts are **trading days**. If publication date `t` is not a trading day, shift to the next trading day first.
   - `p_t` is the **closing price of the trading day the article is attributed to** (after any non-trading-day shift).
   - **No same-day returns** — earliest target is t+1 close to prevent data leakage.
   - Articles whose longest horizon (r_30d) target date falls beyond available price data are excluded for that horizon (shorter horizons may still be valid).
3. Text cleaning: lowercase, remove special characters, truncate to model max token length.
4. Save processed dataset to `data/processed/`.

### Data Split — Expanding Rolling Window

H1 = January 1 – June 30, H2 = July 1 – December 31. Year boundaries are inclusive (e.g., "2017–2018" means Jan 1 2017 through Dec 31 2018).

- **Window 1**: Train 2017–2018, Val 2019-H1, Test 2019-H2
- **Window 2**: Train 2017–2019, Val 2020-H1, Test 2020-H2
- **Window 3**: Train 2017–2020, Val 2021-H1, Test 2021-H2

**Leakage prevention**: Test set articles have publication dates strictly after the training period. For articles near the end of a test period, longer horizons (e.g., r_30d) may have target dates that extend into the next window's time range. This is acceptable because the target return is computed from price data (not from the next window's training labels), and the FC model for each window is trained independently. However, articles whose target return date falls beyond available price data are excluded for that horizon.

**Reproducibility**: Set global random seed (42) for PyTorch, NumPy, and Python `random` module. Pin HuggingFace model revisions to specific commit hashes in `configs/experiment.yaml`.

## Models

### Pretrained Models (Zero-Shot Sentiment)

| Model | HuggingFace ID | Sentiment Score Method |
|-------|----------------|----------------------|
| FinBERT | `ProsusAI/finbert` | P(positive) - P(negative) |
| RoBERTa (sentiment) | `cardiffnlp/twitter-roberta-base-sentiment-latest` | P(positive) - P(negative) |
| Llama-3.2-1B | `meta-llama/Llama-3.2-1B-Instruct` | Prompt-based (see below) |

**BERT-base** (`bert-base-uncased`) has no sentiment head — used only as encoder for fine-tuning, not for zero-shot sentiment.

**Llama-3.2-1B prompt template:**
```
You are a financial sentiment analyst. Rate the sentiment of the following financial news article on a scale from -1.0 (very negative) to 1.0 (very positive). Respond with ONLY a single number, nothing else.

Article: {article_text_truncated_to_512_tokens}

Sentiment score:
```
- Parse output with regex `r'-?\d+\.?\d*'` to extract the first number (handles both `0.5` and `-1`).
- Clamp to [-1, 1].
- If parsing fails after 1 retry with a simplified prompt ("Rate sentiment from -1 to 1. Reply with one number only."), assign NaN and exclude from evaluation.

### Fine-Tuned Models (Return Prediction)

For each encoder (BERT-base, FinBERT, RoBERTa):
1. Extract `[CLS]` embeddings (768-dim) from **frozen, unmodified pretrained weights** — no gradient updates, no warm-up fine-tuning. This must happen before any FC training begins.
2. Cache embeddings to `results/embeddings/` (one file per encoder, keyed by article ID). Since the encoder is fully frozen and stateless, the cache is split-agnostic — the same embeddings are used across all rolling windows.
3. Train fully connected network(s) on cached embeddings.

**Two approaches (compared in report):**

**A. Separate per-horizon models** — 5 independent FC networks per encoder:
```
Embedding (768) → FC layers → Output (1 value: r_Xd)
```

**B. Single multi-output model** — 1 FC network per encoder:
```
Embedding (768) → FC layers → Output (5 values: r_1d, r_3d, r_5d, r_10d, r_30d)
```
For Approach B, use **masked loss**: when an article is missing a long-horizon target (e.g., r_30d unavailable), that output is excluded from the loss computation for that sample. This avoids dropping articles entirely and keeps the training set size comparable to Approach A. The report should note this asymmetry: Approach A uses all valid articles per horizon, while Approach B uses masked loss on the same full set.

### Hyperparameter Search

| Hyperparameter | Values |
|----------------|--------|
| Hidden layers | 1, 2, 3 |
| Neurons per layer | 128, 256, 512 |
| Activation | ReLU, GELU, LeakyReLU |
| Dropout | 0.1, 0.2, 0.3 |
| Learning rate | 1e-3, 5e-4, 1e-4 |
| Batch size | 32, 64 |
| Optimizer | Adam, AdamW |

Use **random search**: sample 25 configurations uniformly from the grid with a fixed seed (42) for reproducibility. All configs and results logged to `results/training_logs/` as JSON.

## Evaluation

### Fine-Tuned Return Prediction
- **Metrics**: R-squared, MSE per model × horizon × approach (single vs. separate) × window
- Report mean ± std across rolling windows
- **Loss curves**: training and validation loss per epoch, saved and plotted

### Binary Price Movement Prediction
- Convert to binary: return > threshold → up, ≤ threshold → down
- **Threshold strategies**:
  - Fixed: threshold = 0
  - Learned: sweep thresholds on **each rolling window's own validation set** independently to maximize macro F1. Sweep over the range [min_pred, max_pred] with step size 0.001. Learn one threshold per model × horizon × window. Apply that window's learned threshold to its own test set only.
  - Applies to **both** pretrained and fine-tuned models. For pretrained models, sweep over sentiment score range (typically [-1, 1]). For fine-tuned models, sweep over predicted return range. The different scales are handled naturally since each model's sweep uses its own output distribution.
- **Metrics**: Accuracy, macro F1-score
- Compare all models (pretrained + fine-tuned) × all horizons × both threshold strategies

### Results Tables

**Fine-tuned evaluation:**
| Model | Horizon | Approach | R² (mean±std) | MSE (mean±std) |

**Binary prediction:**
| Model | Horizon | Fixed (0) Acc/F1 | Learned Acc/F1 | Optimal Threshold |

**Single vs. separate comparison:**
| Model | Horizon | Single R²/MSE | Separate R²/MSE |

## Sentiment & Market Trend Analysis

1. For each month `t`, average all sentiment scores from articles published that month (per model).
2. Compute SPX monthly return for month `t+1`.
3. **Scatter plot**: monthly sentiment vs. SPX next-month return, per model.
4. **OLS regression**: `SPX_return_{t+1} = α + β × sentiment_t + ε`. Report β, p-value, R².
5. **Correlation**: Pearson and Spearman between monthly sentiment and SPX t+1 return.
6. **Time series overlay**: monthly sentiment and SPX return (shifted) on dual-axis plot.
7. Compare pretrained vs. fine-tuned models' alignment with market trends. For fine-tuned models, normalize predicted returns to [-1, 1] range (min-max scaling per model) before averaging as a sentiment proxy. This normalization is documented in the report methodology.

## Project Structure

```
sentiment/
├── CLAUDE.md
├── environment.yml
├── data/
│   ├── raw/                     # news.csv, price.csv (gitignored)
│   └── processed/               # cleaned, merged, split data
├── scripts/
│   ├── data_preparation.py
│   ├── pretrained_sentiment.py
│   ├── finetune.py
│   ├── evaluate.py
│   └── market_trend.py
├── utils/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── text_processing.py
│   ├── metrics.py
│   └── plotting.py
├── results/
│   ├── embeddings/
│   ├── predictions/
│   ├── training_logs/
│   ├── metrics/
│   └── figures/
├── notebooks/
│   └── results_and_analysis.ipynb
├── configs/
│   └── experiment.yaml
└── docs/
    └── superpowers/specs/
```

## Report ↔ Code Mapping

| Report Section | Data Source |
|---|---|
| Introduction | Written narrative |
| Data Description | `scripts/data_preparation.py` → `results/metrics/data_stats.json` |
| Methodology: Pretrained Models | `scripts/pretrained_sentiment.py` |
| Methodology: Fine-tuned Models | `scripts/finetune.py` + `configs/experiment.yaml` |
| Results: Fine-tuned Evaluation | `results/metrics/` → R², MSE tables, loss curves |
| Results: Binary Prediction | `results/metrics/` → accuracy, F1, learned thresholds |
| Results: Single vs. Separate | `results/metrics/` → comparison table |
| Results: Sentiment & Market Trend | `scripts/market_trend.py` → scatter plots, regression |
| Conclusion | Synthesize findings |

## Hardware

- **GPU**: NVIDIA RTX 4000 SFF Ada, 20GB VRAM
- **Environment**: Conda
- All encoder models fit comfortably in VRAM. Llama-3.2-1B runs quantized (~2-4GB).

## Key Design Decisions

1. **Strict forward-looking returns** — no same-day prediction to avoid data leakage.
2. **Expanding rolling window** — 3 windows for robust evaluation.
3. **Cached embeddings** — extract once, train many FC configurations quickly.
4. **Separate vs. single model comparison** — multi-task vs. single-task learning analysis.
5. **Learnable binary threshold** — validated on val set, compared against fixed zero.
6. **4 unique models** — BERT-base (fine-tuning encoder only), FinBERT (zero-shot + fine-tuning), RoBERTa (zero-shot + fine-tuning), Llama-3.2-1B (zero-shot only). 3 zero-shot pretrained models, 3 fine-tuning encoders.
7. **5 prediction horizons** — 1d, 3d, 5d, 10d, 30d for comprehensive analysis.
