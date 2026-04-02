# Experiment Journal

## Status
- **Phase:** 1 (Improving Prediction)
- **Baseline Score:** 0.4412
- **Current Best Score:** 0.4412
- **Plateau Counter:** 0
- **Last Experiment:** EXP-001

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
- **Result:** PENDING
- **Outcome:** PENDING
- **Check at 2026-04-02T02:22:00:** No results branch yet (~1h since push). Training still running. Will check next run.
