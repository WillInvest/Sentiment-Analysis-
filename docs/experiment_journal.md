# Experiment Journal

## Status
- **Phase:** 1 (Improving Prediction)
- **Baseline Score:** 0.4412
- **Current Best Score:** 0.4412
- **Plateau Counter:** 1
- **Last Experiment:** EXP-002

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
- **Result:** PENDING
- **Outcome:** PENDING
