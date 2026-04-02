# Experiment Journal

## Status
- **Phase:** 1 (Improving Prediction)
- **Baseline Score:** 0.4412
- **Current Best Score:** 0.4412
- **Plateau Counter:** 4
- **Last Experiment:** EXP-005

## Baseline Metrics (2026-04-01)
- Stock R² (best): 0.0039 (FinBERT, single, r_30d)
- Stock Accuracy (best): 56.17% (FinBERT, single, r_30d)
- Stock F1 (best): 0.538 (FinBERT, single, r_30d)
- Market R² (best): 0.376 (FinBERT, separate, r_30d)
- Market p-value (best): 0.045

## Experiment Log

### EXP-001: Expanding Rolling Windows (2026-04-02T01:15:00)
- **Hypothesis:** The baseline uses only 1 rolling window (2017-2018 train, 2019 val, 2020 test). The spec calls for 3 expanding windows. Adding windows covering more of 2017-2021 data will improve generalization, reduce variance in aggregated metrics, and likely improve stock R² and accuracy by training on more diverse periods.
- **What changed:** `configs/experiment.yaml` — replaced single `split` window with three expanding windows: window1 (train 2017-18, val 2019, test 2020), window2 (train 2017-19, val 2020, test 2021), window3 (train 2017, val 2018, test 2019). Also reset progress.json stages.
- **Branch:** experiment/001-expanding-windows
- **Result:** Composite score 0.2905 (market R2 collapsed from 0.376→0.040 due to regime mixing across 3 test windows)
- **Outcome:** NO IMPROVEMENT (0.2905 < 0.4412 baseline). Plateau counter: 1.

### EXP-002: Cosine Annealing LR Scheduler (2026-04-02T04:30:00)
- **Hypothesis:** Fixed LR + patience=10 leaves FC heads stuck in local minima. CosineAnnealingWarmRestarts (T_0=20 epochs, eta_min=1% of initial LR) periodically resets the LR, giving the optimizer multiple convergence attempts per run. Increased patience 10→15 ensures models get enough epochs to benefit from LR cycling. Expected improvement: stock R2 from 0.0039 toward 0.015+, boosting composite score above 0.4412.
- **What changed:** `scripts/finetune.py` — added `CosineAnnealingWarmRestarts(T_0=20, T_mult=1, eta_min=lr*0.01)` scheduler with `scheduler.step()` per epoch; also logs LR in training logs. `configs/experiment.yaml` — `early_stopping_patience` 10→15. `results/progress.json` — reset finetune/evaluate/market_trend.
- **Branch:** experiment/002-cosine-lr-scheduler
- **Result:** Composite score 0.4057 (stock R² collapsed to 0.0000 — cosine LR cycling disrupted FC head convergence; market R² also dropped 0.376→0.337 with best model shifting to RoBERTa separate r_30d p=0.061)
- **Outcome:** NO IMPROVEMENT (0.4057 < 0.4412 baseline). Plateau counter: 2.

### EXP-003: Mean-Pool Embeddings (2026-04-02T06:33:00)
- **Hypothesis:** [CLS] token embeddings from BERT-base/FinBERT/RoBERTa (not fine-tuned with sentence objectives) may not fully capture the financial sentiment signal distributed across WSJ article tokens. Mean-pooling over all non-padding hidden states provides a richer, information-dense feature vector for the FC heads. This is well-supported by the SBERT literature and is a fundamentally different representation that hasn't been tried. Expected improvement: stock R² should increase as FC heads receive better input features; market R² may improve as well.
- **What changed:** `scripts/extract_embeddings.py` — replaced `last_hidden_state[:, 0, :]` ([CLS]) with masked mean-pool `(last_hidden_state * mask).sum(1) / mask.sum(1)`. `results/progress.json` — reset embeddings, finetune, evaluate, market_trend stages.
- **Branch:** experiment/003-mean-pool-embeddings
- **Result:** Composite score 0.358 (stock R² collapsed to 0.0002 — mean-pool hurt FC heads; market R² dropped from 0.376→0.224, market sig improved but insufficient)
- **Outcome:** NO IMPROVEMENT (0.358 < 0.4412 baseline). Plateau counter: 3.

### EXP-004: Weight Decay Regularization in HP Search (2026-04-02T09:00:00)
- **Hypothesis:** The current HP grid has NO weight_decay parameter — AdamW is called with default weight_decay=0, making it identical to Adam. With a small training set (2017-2018), the FC heads likely overfit, keeping stock R² near zero. Adding weight_decay: [0.0, 0.001, 0.01, 0.1] to the search grid and passing it to the optimizer should reduce overfitting and improve stock R². Also bumping n_configs 25→40 to explore the larger space. Expected improvement: stock R² from 0.0039 toward 0.010+, boosting composite score above 0.4412.
- **What changed:** `configs/experiment.yaml` — added `weight_decay` to HP grid, bumped `n_configs` 25→40. `scripts/finetune.py` — pass `weight_decay=hp.get("weight_decay", 0.0)` to optimizer constructor. `results/progress.json` — reset finetune/evaluate/market_trend.
- **Branch:** experiment/004-weight-decay
- **Result:** Composite score 0.4061 (stock R² dropped to 0.0019; market R² dropped 0.376→0.315, best model shifted to finetuned BERT single r_30d p=0.0725. Weight decay + larger n_configs disrupted the HP search, landing on worse configurations than baseline.)
- **Outcome:** NO IMPROVEMENT (0.4061 < 0.4412 baseline). Plateau counter: 4. Remote branches could not be deleted (HTTP 403 server permission restriction).

### EXP-005: Gradient Boosting Machine heads (2026-04-02T10:35:00)
- **Hypothesis:** Neural FC heads consistently achieve stock R² ≈ 0 across all experiments — they overfit on the small 2017-2018 training set despite dropout. GradientBoostingRegressor (sklearn) uses tree ensembles that are more robust to overfitting on tabular/embedding data. Added as a NEW approach ('gbm_separate') alongside existing FC heads — the FC head training is UNCHANGED, so baseline market R² (0.376) should be preserved. GBM can only add upside: if its predictions achieve higher R² or market correlation, the composite score max() operations capture the gain. Hypothesis: stock R² from ~0 to >0.01 (GBM), market R² possibly improving above 0.376.
- **What changed:** `scripts/finetune.py` — added `to_numpy_arrays()`, `train_gbm()` helpers + GBM training loop (4 HP configs per encoder/horizon: 100/200 estimators × 3/5 depth) + GBM prediction generation. `results/progress.json` — reset finetune/evaluate/market_trend.
- **Branch:** experiment/005-gbm-heads
- **Result:** PENDING — branch pushed at 2026-04-02T10:33 UTC, results branch not yet available (checked at 2026-04-02T12:35 UTC, ~2h elapsed, within 6h timeout)
- **Outcome:** PENDING
