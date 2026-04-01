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

## Autonomous Improvement Agent

An autonomous "Scientist Agent" runs on a schedule to iteratively improve pipeline performance.

### Architecture
- **Cloud agent** (Anthropic scheduled trigger, every 2h): analyzes results, implements experiments, pushes branches
- **Server worker** (`server_worker.sh`, crontab every 10min): polls for experiment branches, trains, pushes results

### Branch Convention
- `experiment/<NNN>-<name>` — code changes (pushed by cloud agent)
- `results/<NNN>-<name>` — training results (pushed by server worker)
- Only merged to `main` if composite score improves

### Key Files
- `utils/composite_score.py` — composite score computation
- `docs/experiment_journal.md` — agent's persistent memory
- `docs/agent_prompt.md` — the scheduled trigger's instructions
- `server_worker.sh` — server-side training worker

### Composite Score
Run: `python3 -c "from utils.composite_score import print_score; print_score()"`

### Server Worker
The crontab entry (add manually):
```
*/10 * * * * cd /home/fao/projects/sentiment && bash server_worker.sh >> results/server_worker.log 2>&1
```
