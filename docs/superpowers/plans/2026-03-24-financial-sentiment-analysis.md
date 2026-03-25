# Financial Sentiment Analysis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a complete pipeline for extracting sentiment from financial news using pretrained and fine-tuned NLP models, evaluating prediction accuracy across multiple horizons, and analyzing sentiment-market trend relationships.

**Architecture:** Python scripts handle all computation (data prep, embedding extraction, training, evaluation). A single Jupyter notebook loads persisted results for visualization. A `progress.json` file tracks pipeline state for resumability across sessions.

**Tech Stack:** Python 3.10+, PyTorch, HuggingFace Transformers, pandas, numpy, scikit-learn, statsmodels, matplotlib, seaborn, pyyaml

**Spec:** `docs/superpowers/specs/2026-03-24-financial-sentiment-analysis-design.md`

---

## File Structure

```
sentiment/
├── CLAUDE.md                          # Session continuity — project overview, how to check progress
├── .gitignore                         # data/raw/, results/embeddings/, results/checkpoints/, *.pyc, __pycache__
├── environment.yml                    # Conda env definition
├── configs/
│   └── experiment.yaml                # Model IDs, hyperparameter grid, window definitions, seeds
├── utils/
│   ├── __init__.py
│   ├── progress.py                    # Read/write results/progress.json — all scripts use this
│   ├── data_loader.py                 # Load raw CSVs, compute returns, build splits
│   ├── text_processing.py            # Clean text, truncate to max tokens
│   └── metrics.py                     # R², MSE, accuracy, F1, threshold sweep
├── scripts/
│   ├── data_preparation.py            # Orchestrates data loading + split + saves to data/processed/
│   ├── extract_embeddings.py          # Extract [CLS] embeddings from frozen encoders → results/embeddings/
│   ├── pretrained_sentiment.py        # Run zero-shot sentiment (FinBERT, RoBERTa, Llama) → results/predictions/
│   ├── finetune.py                    # Train FC heads on cached embeddings → results/training_logs/, results/checkpoints/
│   ├── evaluate.py                    # Compute all metrics → results/metrics/
│   └── market_trend.py               # Monthly aggregation + SPX analysis → results/metrics/
├── results/
│   ├── progress.json                  # Machine-readable pipeline state tracker
│   ├── embeddings/                    # {encoder_name}_embeddings.pt
│   ├── predictions/                   # {model_name}_sentiment.parquet, {model_name}_returns.parquet
│   ├── training_logs/                 # {encoder}_{approach}_{horizon}_{window}_log.json
│   ├── checkpoints/                   # {encoder}_{approach}_{horizon}_{window}_best.pt
│   ├── metrics/                       # evaluation_results.json, data_stats.json, market_trend.json
│   └── figures/                       # All plots as PNG
├── notebooks/
│   └── results_and_analysis.ipynb     # Load results, generate all report figures and tables
└── data/
    ├── raw/                           # news.csv, price.csv (gitignored, user places here)
    └── processed/                     # processed_data.parquet, splits saved per window
```

---

## Task 1: Project Scaffolding & Configuration

**Files:**
- Create: `.gitignore`
- Create: `environment.yml`
- Create: `configs/experiment.yaml`
- Create: `utils/__init__.py`
- Create: `utils/progress.py`
- Create: `CLAUDE.md`
- Create: `results/progress.json`

- [ ] **Step 1: Create `.gitignore`**

```gitignore
# Data
data/raw/
data/processed/

# Results (large binary files)
results/embeddings/
results/checkpoints/

# Python
__pycache__/
*.pyc
*.pyo
.ipynb_checkpoints/

# Environment
.env
```

- [ ] **Step 2: Create `environment.yml`**

```yaml
name: sentiment
channels:
  - pytorch
  - nvidia
  - conda-forge
  - defaults
dependencies:
  - python=3.10
  - pytorch>=2.0
  - pytorch-cuda=12.1
  - pip
  - pip:
    - transformers>=4.36
    - accelerate>=0.25
    - bitsandbytes>=0.41
    - pandas>=2.0
    - numpy>=1.24
    - scikit-learn>=1.3
    - statsmodels>=0.14
    - matplotlib>=3.7
    - seaborn>=0.12
    - pyarrow>=14.0
    - pyyaml>=6.0
    - jupyter>=1.0
    - tqdm>=4.65
```

- [ ] **Step 3: Create `configs/experiment.yaml`**

```yaml
seed: 42

models:
  # NOTE: Pin revision hashes at project setup by running:
  #   python -c "from huggingface_hub import model_info; print(model_info('bert-base-uncased').sha)"
  # Replace "main" with the returned commit hash for reproducibility.
  bert:
    hf_id: "bert-base-uncased"
    revision: "main"  # TODO: pin to commit hash at setup
    roles: ["finetune_encoder"]
    embedding_dim: 768
  finbert:
    hf_id: "ProsusAI/finbert"
    revision: "main"  # TODO: pin to commit hash at setup
    roles: ["zero_shot", "finetune_encoder"]
    embedding_dim: 768
  roberta:
    hf_id: "cardiffnlp/twitter-roberta-base-sentiment-latest"
    revision: "main"  # TODO: pin to commit hash at setup
    roles: ["zero_shot", "finetune_encoder"]
    embedding_dim: 768
  llama:
    hf_id: "meta-llama/Llama-3.2-1B-Instruct"
    revision: "main"  # TODO: pin to commit hash at setup
    roles: ["zero_shot"]
    max_input_tokens: 512

horizons: [1, 3, 5, 10, 30]

windows:
  - name: "window_1"
    train_start: "2017-01-01"
    train_end: "2018-12-31"
    val_start: "2019-01-01"
    val_end: "2019-06-30"
    test_start: "2019-07-01"
    test_end: "2019-12-31"
  - name: "window_2"
    train_start: "2017-01-01"
    train_end: "2019-12-31"
    val_start: "2020-01-01"
    val_end: "2020-06-30"
    test_start: "2020-07-01"
    test_end: "2020-12-31"
  - name: "window_3"
    train_start: "2017-01-01"
    train_end: "2020-12-31"
    val_start: "2021-01-01"
    val_end: "2021-06-30"
    test_start: "2021-07-01"
    test_end: "2021-12-31"

hyperparameters:
  n_configs: 25
  grid:
    hidden_layers: [1, 2, 3]
    neurons: [128, 256, 512]
    activation: ["relu", "gelu", "leaky_relu"]
    dropout: [0.1, 0.2, 0.3]
    learning_rate: [0.001, 0.0005, 0.0001]
    batch_size: [32, 64]
    optimizer: ["adam", "adamw"]
  epochs: 100
  early_stopping_patience: 10

threshold_sweep:
  step: 0.001

approaches: ["separate", "single"]
```

- [ ] **Step 4: Create `utils/__init__.py`**

```python
"""Utility modules for financial sentiment analysis."""
```

- [ ] **Step 5: Create `utils/progress.py`**

```python
"""Progress tracker for pipeline resumability.

Reads/writes results/progress.json. All scripts call update_progress()
after completing a unit of work, and check_progress() before starting
to skip completed work.
"""

import json
from pathlib import Path
from datetime import datetime

PROGRESS_FILE = Path("results/progress.json")


def _load() -> dict:
    if PROGRESS_FILE.exists():
        return json.loads(PROGRESS_FILE.read_text())
    return {
        "data_preparation": "pending",
        "embeddings": {},
        "pretrained_sentiment": {},
        "finetune": {},
        "evaluate": "pending",
        "market_trend": "pending",
        "last_updated": None,
        "last_error": None,
    }


def _save(state: dict) -> None:
    PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    state["last_updated"] = datetime.now().isoformat()
    PROGRESS_FILE.write_text(json.dumps(state, indent=2))


def check_progress(stage: str, key: str | None = None) -> str:
    """Return status: 'pending', 'in_progress', 'done', or 'error'."""
    state = _load()
    if key is None:
        return state.get(stage, "pending")
    return state.get(stage, {}).get(key, "pending")


def update_progress(stage: str, key: str | None = None, status: str = "done", error: str | None = None) -> None:
    """Update progress for a stage/key."""
    state = _load()
    if key is None:
        state[stage] = status
    else:
        if stage not in state or not isinstance(state[stage], dict):
            state[stage] = {}
        state[stage][key] = status
    state["last_error"] = error
    _save(state)


def get_full_progress() -> dict:
    """Return the full progress state."""
    return _load()


def print_progress_summary() -> None:
    """Print human-readable progress summary."""
    state = _load()
    print("=" * 60)
    print("PIPELINE PROGRESS")
    print("=" * 60)
    print(f"Last updated: {state.get('last_updated', 'never')}")
    print(f"Last error: {state.get('last_error', 'none')}")
    print()
    print(f"Data preparation: {state.get('data_preparation', 'pending')}")
    print()
    print("Embeddings:")
    for k, v in state.get("embeddings", {}).items():
        print(f"  {k}: {v}")
    print()
    print("Pretrained sentiment:")
    for k, v in state.get("pretrained_sentiment", {}).items():
        print(f"  {k}: {v}")
    print()
    print("Fine-tuning:")
    for k, v in state.get("finetune", {}).items():
        print(f"  {k}: {v}")
    print()
    print(f"Evaluation: {state.get('evaluate', 'pending')}")
    print(f"Market trend: {state.get('market_trend', 'pending')}")
    print("=" * 60)
```

- [ ] **Step 6: Create initial `results/progress.json`**

```json
{
  "data_preparation": "pending",
  "embeddings": {},
  "pretrained_sentiment": {},
  "finetune": {},
  "evaluate": "pending",
  "market_trend": "pending",
  "last_updated": null,
  "last_error": null
}
```

- [ ] **Step 7: Create `CLAUDE.md`**

```markdown
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
```

- [ ] **Step 8: Create directory structure**

```bash
mkdir -p data/raw data/processed results/{embeddings,predictions,training_logs,checkpoints,metrics,figures} notebooks configs utils scripts docs/superpowers/{specs,plans}
```

- [ ] **Step 9: Commit scaffolding**

```bash
git add .gitignore environment.yml configs/experiment.yaml utils/__init__.py utils/progress.py results/progress.json CLAUDE.md
git commit -m "feat: project scaffolding with config, progress tracker, and CLAUDE.md"
```

---

## Task 2: Data Loading & Preprocessing Utilities

**Files:**
- Create: `utils/data_loader.py`
- Create: `utils/text_processing.py`

- [ ] **Step 1: Create `utils/text_processing.py`**

```python
"""Text preprocessing for financial news articles."""

import re


def clean_text(text: str) -> str:
    """Lowercase, remove special chars, normalize whitespace."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s.,;:!?'-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
```

- [ ] **Step 2: Create `utils/data_loader.py`**

```python
"""Data loading, return computation, and split creation.

Key responsibilities:
- Load news.csv and price.csv
- Expand multi-ticker articles into (article, ticker) pairs
- Compute forward-looking returns at 5 horizons using trading days only
- Shift non-trading-day publication dates to next trading day
- Create expanding rolling window splits per configs/experiment.yaml
"""

import re

import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from utils.text_processing import clean_text


def load_config() -> dict:
    return yaml.safe_load(Path("configs/experiment.yaml").read_text())


def load_raw_data(data_dir: str = "data/raw") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load news.csv and price.csv."""
    news = pd.read_csv(f"{data_dir}/news.csv")
    price = pd.read_csv(f"{data_dir}/price.csv")
    return news, price


def get_trading_days(price_df: pd.DataFrame, ticker: str = "SPX") -> pd.DatetimeIndex:
    """Get sorted trading days from price data for a reference ticker."""
    ref = price_df[price_df["ticker"] == ticker].copy()
    ref["date"] = pd.to_datetime(ref["date"])
    return ref["date"].sort_values().reset_index(drop=True)


def shift_to_next_trading_day(date: pd.Timestamp, trading_days: pd.DatetimeIndex) -> pd.Timestamp | None:
    """If date is not a trading day, shift to next trading day."""
    future = trading_days[trading_days >= date]
    if len(future) == 0:
        return None
    return future.iloc[0]


def compute_forward_returns(
    price_df: pd.DataFrame,
    ticker: str,
    base_date: pd.Timestamp,
    horizons: list[int],
    trading_days_for_ticker: pd.DatetimeIndex,
) -> dict[str, float | None]:
    """Compute forward returns from base_date for given horizons.

    Returns dict like {'r_1d': 0.012, 'r_3d': 0.034, ...}.
    Values are None if horizon extends beyond available data.
    """
    # Find index of base_date in trading days
    idx_arr = trading_days_for_ticker.get_indexer([base_date])
    if idx_arr[0] == -1:
        return {f"r_{h}d": None for h in horizons}
    base_idx = idx_arr[0]

    ticker_prices = price_df[price_df["ticker"] == ticker].set_index("date")["close"]
    p_t = ticker_prices.get(base_date)
    if p_t is None or p_t == 0:
        return {f"r_{h}d": None for h in horizons}

    results = {}
    for h in horizons:
        target_idx = base_idx + h
        if target_idx >= len(trading_days_for_ticker):
            results[f"r_{h}d"] = None
        else:
            target_date = trading_days_for_ticker.iloc[target_idx]
            p_target = ticker_prices.get(target_date)
            if p_target is None:
                results[f"r_{h}d"] = None
            else:
                results[f"r_{h}d"] = (p_target - p_t) / p_t
    return results


def build_dataset(news_df: pd.DataFrame, price_df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Build full dataset: expand multi-ticker, compute returns, clean text.

    Returns a DataFrame with columns:
    - article_id, ticker, date, text, cleaned_text
    - r_1d, r_3d, r_5d, r_10d, r_30d
    """
    horizons = config["horizons"]
    price_df = price_df.copy()
    price_df["date"] = pd.to_datetime(price_df["date"])

    # Get all trading days per ticker
    all_trading_days = {}
    for t in price_df["ticker"].unique():
        days = price_df[price_df["ticker"] == t]["date"].sort_values().reset_index(drop=True)
        all_trading_days[t] = days

    # Reference trading days (SPX) for date shifting
    spx_trading_days = all_trading_days.get("SPX", pd.DatetimeIndex([]))

    rows = []
    for idx, row in news_df.iterrows():
        # Handle multi-ticker: split on comma, semicolon, or space
        tickers_raw = str(row.get("tickers", ""))
        tickers = [t.strip() for t in re.split(r"[,;\s]+", tickers_raw) if t.strip()]

        pub_date = pd.to_datetime(row.get("date", row.get("pub_date", row.get("publication_date"))))
        if pd.isna(pub_date):
            continue

        # Shift to next trading day if needed
        shifted_date = shift_to_next_trading_day(pub_date, spx_trading_days)
        if shifted_date is None:
            continue

        text = str(row.get("text", row.get("article", row.get("headline", ""))))
        cleaned = clean_text(text)

        for ticker in tickers:
            if ticker not in all_trading_days:
                continue

            ticker_days = all_trading_days[ticker]
            # Shift for this specific ticker's trading days
            ticker_shifted = shift_to_next_trading_day(pub_date, ticker_days)
            if ticker_shifted is None:
                continue

            returns = compute_forward_returns(price_df, ticker, ticker_shifted, horizons, ticker_days)

            # Include if at least the shortest horizon is available
            if returns.get("r_1d") is None:
                continue

            rows.append({
                "article_id": idx,
                "ticker": ticker,
                "date": ticker_shifted,
                "original_pub_date": pub_date,
                "text": text,
                "cleaned_text": cleaned,
                **returns,
            })

    df = pd.DataFrame(rows)
    return df


def create_splits(df: pd.DataFrame, config: dict) -> dict:
    """Create expanding rolling window splits.

    Returns dict of {window_name: {'train': df, 'val': df, 'test': df}}.
    """
    splits = {}
    for window in config["windows"]:
        name = window["name"]
        train = df[(df["date"] >= window["train_start"]) & (df["date"] <= window["train_end"])]
        val = df[(df["date"] >= window["val_start"]) & (df["date"] <= window["val_end"])]
        test = df[(df["date"] >= window["test_start"]) & (df["date"] <= window["test_end"])]
        splits[name] = {"train": train, "val": val, "test": test}
    return splits
```

- [ ] **Step 3: Commit utilities**

```bash
git add utils/data_loader.py utils/text_processing.py
git commit -m "feat: data loading utilities with return computation and rolling window splits"
```

---

## Task 3: Data Preparation Script

**Files:**
- Create: `scripts/data_preparation.py`

- [ ] **Step 1: Create `scripts/data_preparation.py`**

```python
"""Data preparation pipeline.

Loads raw CSVs, computes forward returns, creates rolling window splits,
saves processed data and statistics. Resumable via progress.json.

Usage: python scripts/data_preparation.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.data_loader import load_config, load_raw_data, build_dataset, create_splits
from utils.progress import check_progress, update_progress


def save_data_stats(df: pd.DataFrame, splits: dict, config: dict) -> None:
    """Save dataset statistics for the report."""
    stats = {
        "total_article_ticker_pairs": len(df),
        "unique_articles": df["article_id"].nunique(),
        "unique_tickers": df["ticker"].nunique(),
        "date_range": [str(df["date"].min().date()), str(df["date"].max().date())],
        "horizons": config["horizons"],
        "horizon_availability": {},
        "windows": {},
    }

    for h in config["horizons"]:
        col = f"r_{h}d"
        stats["horizon_availability"][col] = int(df[col].notna().sum())

    for name, split in splits.items():
        stats["windows"][name] = {
            "train": len(split["train"]),
            "val": len(split["val"]),
            "test": len(split["test"]),
        }

    Path("results/metrics").mkdir(parents=True, exist_ok=True)
    Path("results/metrics/data_stats.json").write_text(json.dumps(stats, indent=2))
    print(f"Data stats saved. Total pairs: {stats['total_article_ticker_pairs']}, "
          f"Unique articles: {stats['unique_articles']}, Tickers: {stats['unique_tickers']}")


def main():
    if check_progress("data_preparation") == "done":
        print("Data preparation already completed. Skipping.")
        return

    update_progress("data_preparation", status="in_progress")

    try:
        config = load_config()

        # Set seed
        np.random.seed(config["seed"])

        print("Loading raw data...")
        news, price = load_raw_data()
        print(f"News: {len(news)} articles, Price: {len(price)} rows")

        print("Building dataset (computing returns)...")
        df = build_dataset(news, price, config)
        print(f"Built {len(df)} article-ticker pairs")

        print("Creating rolling window splits...")
        splits = create_splits(df, config)

        # Save
        Path("data/processed").mkdir(parents=True, exist_ok=True)
        df.to_parquet("data/processed/full_dataset.parquet", index=False)

        for name, split in splits.items():
            for part in ["train", "val", "test"]:
                split[part].to_parquet(f"data/processed/{name}_{part}.parquet", index=False)

        save_data_stats(df, splits, config)
        update_progress("data_preparation", status="done")
        print("Data preparation complete.")

    except Exception as e:
        update_progress("data_preparation", status="error", error=str(e))
        raise


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add scripts/data_preparation.py
git commit -m "feat: data preparation script with return computation and split creation"
```

---

## Task 4: Embedding Extraction Script

**Files:**
- Create: `scripts/extract_embeddings.py`

- [ ] **Step 1: Create `scripts/extract_embeddings.py`**

```python
"""Extract [CLS] embeddings from frozen pretrained encoders.

Extracts embeddings from BERT-base, FinBERT, and RoBERTa. Saves one .pt file
per encoder in results/embeddings/. Resumable — skips encoders already cached.

Usage: python scripts/extract_embeddings.py
"""

import sys
from pathlib import Path

import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.data_loader import load_config
from utils.progress import check_progress, update_progress


def extract_embeddings_for_model(
    model_name: str,
    hf_id: str,
    texts: list[str],
    article_ids: list,
    batch_size: int = 32,
    max_length: int = 512,
    device: str = "cuda",
) -> dict:
    """Extract [CLS] embeddings from a frozen encoder.

    Returns dict mapping article_id to embedding tensor.
    """
    print(f"Loading {model_name} ({hf_id})...")
    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    model = AutoModel.from_pretrained(hf_id).to(device)
    model.eval()

    embeddings = {}
    for i in tqdm(range(0, len(texts), batch_size), desc=f"Extracting {model_name}"):
        batch_texts = texts[i : i + batch_size]
        batch_ids = article_ids[i : i + batch_size]

        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            # [CLS] token is at position 0
            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu()

        for idx, aid in enumerate(batch_ids):
            embeddings[aid] = cls_embeddings[idx]

    # Free GPU memory
    del model
    torch.cuda.empty_cache()

    return embeddings


def main():
    config = load_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load processed data
    df = pd.read_parquet("data/processed/full_dataset.parquet")

    # Deduplicate: same article_id may appear with different tickers
    # but the text (and thus embedding) is the same
    unique_articles = df.drop_duplicates(subset="article_id")[["article_id", "cleaned_text"]]
    texts = unique_articles["cleaned_text"].tolist()
    article_ids = unique_articles["article_id"].tolist()

    print(f"Extracting embeddings for {len(texts)} unique articles")

    # Only extract for finetune_encoder models
    encoder_models = {
        name: cfg for name, cfg in config["models"].items()
        if "finetune_encoder" in cfg["roles"]
    }

    Path("results/embeddings").mkdir(parents=True, exist_ok=True)

    for model_name, model_cfg in encoder_models.items():
        if check_progress("embeddings", model_name) == "done":
            print(f"Embeddings for {model_name} already cached. Skipping.")
            continue

        update_progress("embeddings", model_name, "in_progress")

        try:
            embeddings = extract_embeddings_for_model(
                model_name=model_name,
                hf_id=model_cfg["hf_id"],
                texts=texts,
                article_ids=article_ids,
                device=device,
            )

            save_path = f"results/embeddings/{model_name}_embeddings.pt"
            torch.save(embeddings, save_path)
            print(f"Saved {model_name} embeddings to {save_path}")

            update_progress("embeddings", model_name, "done")

        except Exception as e:
            update_progress("embeddings", model_name, "error", error=str(e))
            raise

    print("All embeddings extracted.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add scripts/extract_embeddings.py
git commit -m "feat: embedding extraction script for frozen encoders with GPU batching"
```

---

## Task 5: Pretrained Sentiment Script

**Files:**
- Create: `scripts/pretrained_sentiment.py`

- [ ] **Step 1: Create `scripts/pretrained_sentiment.py`**

```python
"""Run zero-shot sentiment extraction using pretrained models.

Models: FinBERT, RoBERTa (classification heads), Llama-3.2-1B (prompt-based).
Saves sentiment scores per article to results/predictions/.
Resumable per model.

Usage: python scripts/pretrained_sentiment.py
"""

import re
import sys
from pathlib import Path

import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.data_loader import load_config
from utils.progress import check_progress, update_progress


def run_classifier_sentiment(
    model_name: str,
    hf_id: str,
    texts: list[str],
    batch_size: int = 32,
    max_length: int = 512,
    device: str = "cuda",
) -> np.ndarray:
    """Run a classification model that outputs P(pos), P(neg), P(neu).

    Returns sentiment scores: P(positive) - P(negative).
    """
    print(f"Loading {model_name} ({hf_id})...")
    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    model = AutoModelForSequenceClassification.from_pretrained(hf_id).to(device)
    model.eval()

    scores = []
    for i in tqdm(range(0, len(texts), batch_size), desc=f"Scoring {model_name}"):
        batch = texts[i : i + batch_size]
        inputs = tokenizer(
            batch, padding=True, truncation=True,
            max_length=max_length, return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()

        # Labels order varies by model — detect it
        labels = model.config.id2label
        pos_idx = next((i for i, l in labels.items() if "pos" in l.lower()), None)
        neg_idx = next((i for i, l in labels.items() if "neg" in l.lower()), None)

        if pos_idx is not None and neg_idx is not None:
            batch_scores = probs[:, pos_idx] - probs[:, neg_idx]
        else:
            # Fallback: assume last class is positive, first is negative
            batch_scores = probs[:, -1] - probs[:, 0]

        scores.extend(batch_scores.tolist())

    del model
    torch.cuda.empty_cache()
    return np.array(scores)


def run_llama_sentiment(
    hf_id: str,
    texts: list[str],
    max_input_tokens: int = 512,
    device: str = "cuda",
) -> np.ndarray:
    """Run Llama-3.2-1B for prompt-based sentiment scoring.

    Returns sentiment scores in [-1, 1]. NaN for parse failures.
    """
    print(f"Loading Llama ({hf_id})...")
    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    model = AutoModelForCausalLM.from_pretrained(
        hf_id,
        torch_dtype=torch.float16,
        device_map="auto",
        load_in_4bit=True,
    )
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompt_template = (
        "You are a financial sentiment analyst. Rate the sentiment of the following "
        "financial news article on a scale from -1.0 (very negative) to 1.0 (very positive). "
        "Respond with ONLY a single number, nothing else.\n\n"
        "Article: {text}\n\n"
        "Sentiment score:"
    )

    retry_template = "Rate sentiment from -1 to 1. Reply with one number only.\n\nArticle: {text}\n\nScore:"

    parse_regex = re.compile(r"-?\d+\.?\d*")

    scores = []
    for i, text in enumerate(tqdm(texts, desc="Scoring Llama")):
        # Truncate text to max_input_tokens using the tokenizer
        tokens = tokenizer.encode(text, add_special_tokens=False)[:max_input_tokens]
        truncated = tokenizer.decode(tokens, skip_special_tokens=True)

        score = _query_llama(model, tokenizer, prompt_template.format(text=truncated), parse_regex, device)

        if score is None:
            # Retry with simplified prompt
            score = _query_llama(model, tokenizer, retry_template.format(text=truncated), parse_regex, device)

        if score is not None:
            score = max(-1.0, min(1.0, score))  # Clamp
        else:
            score = float("nan")

        scores.append(score)

    del model
    torch.cuda.empty_cache()
    return np.array(scores)


def _query_llama(model, tokenizer, prompt, parse_regex, device) -> float | None:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=10, do_sample=False)
    response = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    match = parse_regex.search(response)
    if match:
        return float(match.group())
    return None


def main():
    config = load_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    df = pd.read_parquet("data/processed/full_dataset.parquet")
    unique_articles = df.drop_duplicates(subset="article_id")[["article_id", "cleaned_text"]]
    texts = unique_articles["cleaned_text"].tolist()
    article_ids = unique_articles["article_id"].tolist()

    Path("results/predictions").mkdir(parents=True, exist_ok=True)

    zero_shot_models = {
        name: cfg for name, cfg in config["models"].items()
        if "zero_shot" in cfg["roles"]
    }

    for model_name, model_cfg in zero_shot_models.items():
        if check_progress("pretrained_sentiment", model_name) == "done":
            print(f"Sentiment for {model_name} already computed. Skipping.")
            continue

        update_progress("pretrained_sentiment", model_name, "in_progress")

        try:
            if model_name == "llama":
                scores = run_llama_sentiment(
                    hf_id=model_cfg["hf_id"],
                    texts=texts,
                    max_input_tokens=model_cfg.get("max_input_tokens", 512),
                    device=device,
                )
            else:
                scores = run_classifier_sentiment(
                    model_name=model_name,
                    hf_id=model_cfg["hf_id"],
                    texts=texts,
                    device=device,
                )

            # Save as parquet: article_id -> sentiment score
            result_df = pd.DataFrame({
                "article_id": article_ids,
                "sentiment_score": scores,
            })
            save_path = f"results/predictions/{model_name}_sentiment.parquet"
            result_df.to_parquet(save_path, index=False)
            print(f"Saved {model_name} sentiment to {save_path}")

            nan_count = np.isnan(scores).sum()
            if nan_count > 0:
                print(f"Warning: {nan_count} NaN scores for {model_name}")

            update_progress("pretrained_sentiment", model_name, "done")

        except Exception as e:
            update_progress("pretrained_sentiment", model_name, "error", error=str(e))
            raise

    print("All pretrained sentiment extraction complete.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add scripts/pretrained_sentiment.py
git commit -m "feat: pretrained sentiment extraction (FinBERT, RoBERTa, Llama)"
```

---

## Task 6: Metrics Utilities

**Files:**
- Create: `utils/metrics.py`

- [ ] **Step 1: Create `utils/metrics.py`**

```python
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
```

- [ ] **Step 2: Commit**

```bash
git add utils/metrics.py
git commit -m "feat: evaluation metrics with threshold sweep for binary prediction"
```

---

## Task 7: Fine-Tuning Script

**Files:**
- Create: `scripts/finetune.py`

- [ ] **Step 1: Create `scripts/finetune.py`**

```python
"""Fine-tune FC networks on cached embeddings.

Trains both separate per-horizon models and single multi-output models.
Uses random search over hyperparameter grid. Saves checkpoints, training
logs, and predictions. Fully resumable via progress.json.

Usage: python scripts/finetune.py
"""

import json
import sys
import random
from pathlib import Path
from itertools import product

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.data_loader import load_config
from utils.progress import check_progress, update_progress


# ─── Model Definitions ───


class SeparateFC(nn.Module):
    """FC network for a single horizon."""

    def __init__(self, input_dim: int, hidden_layers: int, neurons: int, activation: str, dropout: float):
        super().__init__()
        layers = []
        in_dim = input_dim
        act_fn = {"relu": nn.ReLU, "gelu": nn.GELU, "leaky_relu": nn.LeakyReLU}[activation]

        for _ in range(hidden_layers):
            layers.extend([nn.Linear(in_dim, neurons), act_fn(), nn.Dropout(dropout)])
            in_dim = neurons
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class MultiOutputFC(nn.Module):
    """FC network predicting all horizons at once."""

    def __init__(self, input_dim: int, n_outputs: int, hidden_layers: int, neurons: int, activation: str, dropout: float):
        super().__init__()
        layers = []
        in_dim = input_dim
        act_fn = {"relu": nn.ReLU, "gelu": nn.GELU, "leaky_relu": nn.LeakyReLU}[activation]

        for _ in range(hidden_layers):
            layers.extend([nn.Linear(in_dim, neurons), act_fn(), nn.Dropout(dropout)])
            in_dim = neurons
        layers.append(nn.Linear(in_dim, n_outputs))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ─── Training Utilities ───


def masked_mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """MSE loss ignoring NaN targets (for multi-output model)."""
    mask = ~torch.isnan(target)
    if mask.sum() == 0:
        return torch.tensor(0.0, requires_grad=True)
    return nn.functional.mse_loss(pred[mask], target[mask])


def sample_hyperparams(config: dict) -> list[dict]:
    """Sample n_configs hyperparameter combinations using random search."""
    grid = config["hyperparameters"]["grid"]
    n = config["hyperparameters"]["n_configs"]
    rng = random.Random(config["seed"])

    configs = []
    for _ in range(n):
        cfg = {k: rng.choice(v) for k, v in grid.items()}
        configs.append(cfg)
    return configs


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    loss_fn,
    hp: dict,
    config: dict,
    device: str,
) -> tuple[nn.Module, list[dict]]:
    """Train model with early stopping. Returns model and epoch logs."""
    optimizer_cls = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW}[hp["optimizer"]]
    optimizer = optimizer_cls(model.parameters(), lr=hp["learning_rate"])

    epochs = config["hyperparameters"]["epochs"]
    patience = config["hyperparameters"]["early_stopping_patience"]

    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None
    logs = []

    for epoch in range(epochs):
        # Train
        model.train()
        train_losses = []
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(X)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # Validate
        model.eval()
        val_losses = []
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                pred = model(X)
                loss = loss_fn(pred, y)
                val_losses.append(loss.item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        logs.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_state:
        model.load_state_dict(best_state)
    return model, logs, best_val_loss


def prepare_data_loaders(
    embeddings: dict,
    split_df: pd.DataFrame,
    horizon_col: str | None,
    horizon_cols: list[str] | None,
    batch_size: int,
) -> DataLoader:
    """Build DataLoader from cached embeddings and split dataframe.

    Iterates over each row (article_id, ticker pair) in split_df.
    Embeddings are keyed by article_id (text is the same regardless of ticker),
    but labels come from each specific row to preserve the correct ticker's return.
    """
    X_list, y_list = [], []

    for _, row in split_df.iterrows():
        aid = row["article_id"]
        if aid not in embeddings:
            continue

        X_list.append(embeddings[aid])

        if horizon_col:
            y_list.append(float(row[horizon_col]))
        elif horizon_cols:
            vals = [float(row[c]) for c in horizon_cols]
            y_list.append(vals)

    if not X_list:
        return None

    X = torch.stack(X_list)
    if horizon_col:
        y = torch.tensor(y_list, dtype=torch.float32)
        # Filter NaN
        valid = ~torch.isnan(y)
        X, y = X[valid], y[valid]
    else:
        y = torch.tensor(y_list, dtype=torch.float32)
        # Keep all — masked loss handles NaN

    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


# ─── Main Pipeline ───


def main():
    config = load_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    horizons = config["horizons"]
    horizon_cols = [f"r_{h}d" for h in horizons]

    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    random.seed(config["seed"])

    hp_configs = sample_hyperparams(config)

    # Create output dirs
    for d in ["results/training_logs", "results/checkpoints", "results/predictions"]:
        Path(d).mkdir(parents=True, exist_ok=True)

    encoder_models = {
        name: cfg for name, cfg in config["models"].items()
        if "finetune_encoder" in cfg["roles"]
    }

    for encoder_name, encoder_cfg in encoder_models.items():
        print(f"\n{'='*60}")
        print(f"Encoder: {encoder_name}")
        print(f"{'='*60}")

        emb_path = f"results/embeddings/{encoder_name}_embeddings.pt"
        embeddings = torch.load(emb_path, weights_only=False)
        embedding_dim = encoder_cfg["embedding_dim"]

        for window in config["windows"]:
            wname = window["name"]

            train_df = pd.read_parquet(f"data/processed/{wname}_train.parquet")
            val_df = pd.read_parquet(f"data/processed/{wname}_val.parquet")

            # ─── Approach A: Separate per-horizon ───
            for horizon in horizons:
                h_col = f"r_{horizon}d"
                run_key = f"{encoder_name}_separate_{horizon}d_{wname}"

                if check_progress("finetune", run_key) == "done":
                    print(f"  {run_key} already done. Skipping.")
                    continue

                update_progress("finetune", run_key, "in_progress")
                print(f"\n  Training: {run_key}")

                best_val_loss = float("inf")
                best_hp = None
                best_logs = None

                for hp_idx, hp in enumerate(hp_configs):
                    train_loader = prepare_data_loaders(embeddings, train_df, h_col, None, hp["batch_size"])
                    val_loader = prepare_data_loaders(embeddings, val_df, h_col, None, hp["batch_size"])

                    if train_loader is None or val_loader is None:
                        continue

                    model = SeparateFC(
                        input_dim=embedding_dim,
                        hidden_layers=hp["hidden_layers"],
                        neurons=hp["neurons"],
                        activation=hp["activation"],
                        dropout=hp["dropout"],
                    ).to(device)

                    model, logs, epoch_best_val = train_model(
                        model, train_loader, val_loader,
                        nn.MSELoss(), hp, config, device,
                    )

                    if epoch_best_val < best_val_loss:
                        best_val_loss = epoch_best_val
                        best_hp = hp
                        best_logs = logs
                        torch.save(model.state_dict(), f"results/checkpoints/{run_key}_best.pt")

                # Save best training log
                log_data = {"hyperparameters": best_hp, "logs": best_logs, "best_val_loss": best_val_loss}
                Path(f"results/training_logs/{run_key}_log.json").write_text(json.dumps(log_data, indent=2))

                update_progress("finetune", run_key, "done")

            # ─── Approach B: Single multi-output ───
            run_key = f"{encoder_name}_single_{wname}"

            if check_progress("finetune", run_key) == "done":
                print(f"  {run_key} already done. Skipping.")
                continue

            update_progress("finetune", run_key, "in_progress")
            print(f"\n  Training: {run_key}")

            best_val_loss = float("inf")
            best_hp = None
            best_logs = None

            for hp_idx, hp in enumerate(hp_configs):
                train_loader = prepare_data_loaders(embeddings, train_df, None, horizon_cols, hp["batch_size"])
                val_loader = prepare_data_loaders(embeddings, val_df, None, horizon_cols, hp["batch_size"])

                if train_loader is None or val_loader is None:
                    continue

                model = MultiOutputFC(
                    input_dim=embedding_dim,
                    n_outputs=len(horizons),
                    hidden_layers=hp["hidden_layers"],
                    neurons=hp["neurons"],
                    activation=hp["activation"],
                    dropout=hp["dropout"],
                ).to(device)

                model, logs, epoch_best_val = train_model(
                    model, train_loader, val_loader,
                    masked_mse_loss, hp, config, device,
                )

                if epoch_best_val < best_val_loss:
                    best_val_loss = epoch_best_val
                    best_hp = hp
                    best_logs = logs
                    torch.save(model.state_dict(), f"results/checkpoints/{run_key}_best.pt")

            log_data = {"hyperparameters": best_hp, "logs": best_logs, "best_val_loss": best_val_loss}
            Path(f"results/training_logs/{run_key}_log.json").write_text(json.dumps(log_data, indent=2))

            update_progress("finetune", run_key, "done")

    # ─── Generate predictions on val and test sets ───
    if check_progress("finetune", "predictions") == "done":
        print("Predictions already generated. Skipping.")
    else:
        update_progress("finetune", "predictions", "in_progress")
        print("\n\nGenerating val and test predictions...")
        generate_predictions(config, encoder_models, device, split_type="val")
        generate_predictions(config, encoder_models, device, split_type="test")
        update_progress("finetune", "predictions", "done")
    print("Fine-tuning complete.")


def generate_predictions(config: dict, encoder_models: dict, device: str, split_type: str = "test"):
    """Load best models and generate predictions on val or test sets."""
    horizons = config["horizons"]
    horizon_cols = [f"r_{h}d" for h in horizons]

    all_predictions = []

    for encoder_name, encoder_cfg in encoder_models.items():
        embeddings = torch.load(f"results/embeddings/{encoder_name}_embeddings.pt", weights_only=False)
        embedding_dim = encoder_cfg["embedding_dim"]

        for window in config["windows"]:
            wname = window["name"]
            test_df = pd.read_parquet(f"data/processed/{wname}_{split_type}.parquet")

            # Load best hyperparams for this encoder/window
            # Separate models
            for horizon in horizons:
                run_key = f"{encoder_name}_separate_{horizon}d_{wname}"
                log_path = Path(f"results/training_logs/{run_key}_log.json")
                ckpt_path = Path(f"results/checkpoints/{run_key}_best.pt")

                if not log_path.exists() or not ckpt_path.exists():
                    continue

                log_data = json.loads(log_path.read_text())
                hp = log_data["hyperparameters"]

                model = SeparateFC(
                    input_dim=embedding_dim,
                    hidden_layers=hp["hidden_layers"],
                    neurons=hp["neurons"],
                    activation=hp["activation"],
                    dropout=hp["dropout"],
                ).to(device)
                model.load_state_dict(torch.load(ckpt_path, weights_only=True))
                model.eval()

                for _, row in test_df.iterrows():
                    aid = row["article_id"]
                    if aid not in embeddings:
                        continue
                    with torch.no_grad():
                        pred = model(embeddings[aid].unsqueeze(0).to(device)).item()
                    all_predictions.append({
                        "encoder": encoder_name,
                        "approach": "separate",
                        "horizon": f"r_{horizon}d",
                        "window": wname,
                        "article_id": aid,
                        "ticker": row["ticker"],
                        "prediction": pred,
                        "actual": row[f"r_{horizon}d"],
                    })

            # Single multi-output model
            run_key = f"{encoder_name}_single_{wname}"
            log_path = Path(f"results/training_logs/{run_key}_log.json")
            ckpt_path = Path(f"results/checkpoints/{run_key}_best.pt")

            if not log_path.exists() or not ckpt_path.exists():
                continue

            log_data = json.loads(log_path.read_text())
            hp = log_data["hyperparameters"]

            model = MultiOutputFC(
                input_dim=embedding_dim,
                n_outputs=len(horizons),
                hidden_layers=hp["hidden_layers"],
                neurons=hp["neurons"],
                activation=hp["activation"],
                dropout=hp["dropout"],
            ).to(device)
            model.load_state_dict(torch.load(ckpt_path, weights_only=True))
            model.eval()

            for _, row in test_df.iterrows():
                aid = row["article_id"]
                if aid not in embeddings:
                    continue
                with torch.no_grad():
                    preds = model(embeddings[aid].unsqueeze(0).to(device)).squeeze(0).cpu().numpy()
                for h_idx, horizon in enumerate(horizons):
                    all_predictions.append({
                        "encoder": encoder_name,
                        "approach": "single",
                        "horizon": f"r_{horizon}d",
                        "window": wname,
                        "article_id": aid,
                        "ticker": row["ticker"],
                        "prediction": float(preds[h_idx]),
                        "actual": row[f"r_{horizon}d"],
                    })

    pred_df = pd.DataFrame(all_predictions)
    pred_df.to_parquet(f"results/predictions/finetune_{split_type}_predictions.parquet", index=False)
    print(f"Saved {len(pred_df)} {split_type} predictions")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add scripts/finetune.py
git commit -m "feat: fine-tuning script with separate/single models, random search, resumability"
```

---

## Task 8: Evaluation Script

**Files:**
- Create: `scripts/evaluate.py`

- [ ] **Step 1: Create `scripts/evaluate.py`**

```python
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
```

- [ ] **Step 2: Commit**

```bash
git add scripts/evaluate.py
git commit -m "feat: evaluation script with regression, binary, and threshold metrics"
```

---

## Task 9: Market Trend Analysis Script

**Files:**
- Create: `scripts/market_trend.py`

- [ ] **Step 1: Create `scripts/market_trend.py`**

```python
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
    # predictions already have article_id, we need dates
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

        # ─── Fine-tuned models (use r_30d / separate as default for monthly analysis) ───
        pred_path = Path("results/predictions/finetune_test_predictions.parquet")
        if pred_path.exists():
            pred_df = pd.read_parquet(pred_path)

            for encoder in pred_df["encoder"].unique():
                for approach in pred_df["approach"].unique():
                    # Use r_30d for monthly trend comparison (closest to monthly horizon)
                    monthly = aggregate_monthly_finetuned(pred_df, encoder, approach, "r_30d")
                    if monthly.empty:
                        continue
                    col_name = f"sentiment_{encoder}_{approach}_r_30d"
                    label = f"Finetuned {encoder} ({approach}, r_30d)"
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
```

- [ ] **Step 2: Commit**

```bash
git add scripts/market_trend.py
git commit -m "feat: market trend analysis with OLS regression and sentiment-SPX comparison"
```

---

## Task 10: Results Notebook

**Files:**
- Create: `notebooks/results_and_analysis.ipynb`

- [ ] **Step 1: Create the notebook**

Create `notebooks/results_and_analysis.ipynb` with the following cells:

**Cell 1 (markdown):**
```markdown
# Financial News Sentiment Analysis — Results & Analysis

This notebook loads all persisted results and generates figures/tables for the report.
```

**Cell 2 (code): Setup & load results**
```python
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from IPython.display import display

# Load evaluation results
eval_results = json.loads(Path("../results/metrics/evaluation_results.json").read_text())
data_stats = json.loads(Path("../results/metrics/data_stats.json").read_text())
market_trend = json.loads(Path("../results/metrics/market_trend.json").read_text())

per_window = pd.DataFrame(eval_results["per_window"])
aggregated = pd.DataFrame(eval_results["aggregated"])

print("Data Stats:")
print(json.dumps(data_stats, indent=2))
```

**Cell 3 (markdown):**
```markdown
## 1. Data Description
```

**Cell 4 (code): Data summary tables**
```python
# Dataset overview
print(f"Total article-ticker pairs: {data_stats['total_article_ticker_pairs']}")
print(f"Unique articles: {data_stats['unique_articles']}")
print(f"Unique tickers: {data_stats['unique_tickers']}")
print(f"Date range: {data_stats['date_range']}")

# Window sizes
window_df = pd.DataFrame(data_stats["windows"]).T
display(window_df)

# Horizon availability
horizon_df = pd.DataFrame([data_stats["horizon_availability"]])
display(horizon_df)
```

**Cell 5 (markdown):**
```markdown
## 2. Fine-Tuned Model Evaluation: R² and MSE
```

**Cell 6 (code): Regression metrics heatmaps**
```python
ft = aggregated[aggregated["model_type"] == "finetuned"].copy()

for metric in ["r2_mean", "mse_mean"]:
    for approach in ["separate", "single"]:
        subset = ft[ft["approach"] == approach]
        if subset.empty:
            continue
        pivot = subset.pivot(index="model", columns="horizon", values=metric)
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.heatmap(pivot, annot=True, fmt=".4f", cmap="RdYlGn" if "r2" in metric else "RdYlGn_r", ax=ax)
        ax.set_title(f"{metric} — {approach} approach")
        fig.tight_layout()
        fig.savefig(f"../results/figures/heatmap_{metric}_{approach}.png", dpi=150)
        plt.show()
```

**Cell 7 (markdown):**
```markdown
## 3. Training Loss Curves
```

**Cell 8 (code): Loss curves**
```python
import glob

log_files = sorted(glob.glob("../results/training_logs/*_log.json"))

# Plot one example per encoder (best config, r_1d, window_1)
for encoder in ["bert", "finbert", "roberta"]:
    log_path = f"../results/training_logs/{encoder}_separate_1d_window_1_log.json"
    if not Path(log_path).exists():
        continue
    log = json.loads(Path(log_path).read_text())
    logs = pd.DataFrame(log["logs"])

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(logs["epoch"], logs["train_loss"], label="Train")
    ax.plot(logs["epoch"], logs["val_loss"], label="Validation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title(f"{encoder} — r_1d — Window 1")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"../results/figures/loss_curve_{encoder}_1d_w1.png", dpi=150)
    plt.show()
```

**Cell 9 (markdown):**
```markdown
## 4. Binary Prediction: Accuracy & F1
```

**Cell 10 (code): Binary metrics comparison**
```python
# Compare fixed vs learned threshold
binary_cols = ["model", "model_type", "approach", "horizon",
               "fixed_accuracy_mean", "fixed_f1_mean",
               "learned_accuracy_mean", "learned_f1_mean"]
binary_df = aggregated[[c for c in binary_cols if c in aggregated.columns]].copy()
display(binary_df.round(4))

# Bar chart: F1 by model and horizon
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
for ax, (threshold_type, col) in zip(axes, [("Fixed (0)", "fixed_f1_mean"), ("Learned", "learned_f1_mean")]):
    if col not in aggregated.columns:
        continue
    pivot = aggregated.pivot_table(index="model", columns="horizon", values=col, aggfunc="mean")
    pivot.plot(kind="bar", ax=ax)
    ax.set_title(f"Macro F1 — {threshold_type} Threshold")
    ax.set_ylabel("F1 Score")
    ax.legend(title="Horizon", bbox_to_anchor=(1.05, 1))
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
fig.tight_layout()
fig.savefig("../results/figures/binary_f1_comparison.png", dpi=150)
plt.show()
```

**Cell 11 (markdown):**
```markdown
## 5. Separate vs Single Model Comparison
```

**Cell 12 (code): Approach comparison**
```python
ft_only = aggregated[aggregated["model_type"] == "finetuned"]
comparison = ft_only.pivot_table(
    index=["model", "horizon"],
    columns="approach",
    values=["r2_mean", "mse_mean"],
    aggfunc="mean"
)
display(comparison.round(4))

# Difference plot
for metric in ["r2_mean", "mse_mean"]:
    if metric not in ft_only.columns:
        continue
    sep = ft_only[ft_only["approach"] == "separate"].set_index(["model", "horizon"])[metric]
    sin = ft_only[ft_only["approach"] == "single"].set_index(["model", "horizon"])[metric]
    diff = (sep - sin).reset_index()
    diff.columns = ["model", "horizon", "difference"]
    pivot = diff.pivot(index="model", columns="horizon", values="difference")
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.heatmap(pivot, annot=True, fmt=".4f", center=0, cmap="RdBu", ax=ax)
    ax.set_title(f"Separate - Single ({metric})")
    fig.tight_layout()
    fig.savefig(f"../results/figures/approach_diff_{metric}.png", dpi=150)
    plt.show()
```

**Cell 13 (markdown):**
```markdown
## 6. Sentiment & Market Trend Analysis
```

**Cell 14 (code): Market trend results**
```python
trend_df = pd.DataFrame(market_trend)
display(trend_df.round(4))

# Display saved scatter plots
from IPython.display import Image
import glob

scatter_files = sorted(glob.glob("../results/figures/scatter_*.png"))
for f in scatter_files:
    print(f"\n{Path(f).stem}")
    display(Image(filename=f, width=600))

# Time series overlay
overlay_path = "../results/figures/time_series_overlay.png"
if Path(overlay_path).exists():
    display(Image(filename=overlay_path, width=900))
```

- [ ] **Step 2: Commit**

```bash
git add notebooks/results_and_analysis.ipynb
git commit -m "feat: results notebook for report visualization"
```

---

## Task 11: Integration Test & Polish

- [ ] **Step 1: Verify all imports work**

```bash
cd /home/fao/projects/sentiment
conda activate sentiment
python -c "from utils.progress import print_progress_summary; print_progress_summary()"
python -c "from utils.data_loader import load_config; print(load_config()['models'].keys())"
python -c "from utils.metrics import sweep_threshold; print('metrics OK')"
```

- [ ] **Step 2: Verify script help text**

```bash
python scripts/data_preparation.py --help 2>/dev/null || python -c "import scripts.data_preparation"
```

- [ ] **Step 3: Add a `run_all.sh` convenience script**

```bash
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
```

```bash
chmod +x run_all.sh
```

- [ ] **Step 4: Final commit**

```bash
git add run_all.sh
git commit -m "feat: add run_all.sh convenience script for full pipeline execution"
```

---

## Execution Order & Dependencies

```
Task 1 (scaffolding) ← no deps
Task 2 (data utils) ← Task 1
Task 3 (data script) ← Task 2
Task 4 (embeddings) ← Task 2
Task 5 (pretrained) ← Task 2
Task 6 (metrics) ← no deps
Task 7 (finetune) ← Task 4, Task 6
Task 8 (evaluate) ← Task 5, Task 6, Task 7
Task 9 (market trend) ← Task 5, Task 7
Task 10 (notebook) ← Task 8, Task 9
Task 11 (integration) ← all
```

Tasks 4, 5, 6 can be implemented in parallel after Task 2.
