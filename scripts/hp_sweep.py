"""Hyperparameter sweep: hidden layers × neurons × activation functions.

Reuses cached chunk features + rolling windows from pipeline.py.
Trains a regressor per (config × window), saves all results to a CSV.

Grid:
    layers     ∈ {1, 2, 3}
    width      ∈ {32, 64, 128}
    activation ∈ {relu, gelu, leaky_relu}
    → 27 configs × 3 windows = 81 training runs

Usage:
    python scripts/hp_sweep.py
    python scripts/hp_sweep.py --resume   # skip already-completed (config, window) pairs
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.pipeline import (  # noqa: E402
    BATCH_ARTICLES, CHUNKS_EMB, CHUNKS_PARQUET, DATASET_PARQUET, INPUT_DIM,
    LR, MAX_EPOCHS, PATIENCE, SEED, WINDOWS,
    ChunkGroupDataset, build_groups_for_split, collate_pad, slice_df,
    update_stage, now_iso,
)

OUT_CSV = ROOT / "results/metrics/hp_sweep.csv"
LAYERS = [1, 2, 3]
WIDTHS = [32, 64, 128]
ACTIVATIONS = ["relu", "gelu", "leaky_relu"]
ACT_CLS = {"relu": nn.ReLU, "gelu": nn.GELU, "leaky_relu": nn.LeakyReLU}


class ConfigurableMLP(nn.Module):
    def __init__(self, in_dim: int, n_layers: int, width: int, activation: str, dropout: float = 0.2):
        super().__init__()
        act = ACT_CLS[activation]
        blocks = []
        prev = in_dim
        for _ in range(n_layers):
            blocks += [nn.Linear(prev, width), act(), nn.Dropout(dropout)]
            prev = width
        blocks.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*blocks)

    def forward(self, X, W, mask):
        chunk_pred = self.net(X).squeeze(-1)         # (B, K)
        chunk_pred = chunk_pred * mask
        return (chunk_pred * W).sum(dim=1)           # (B,)


def make_loaders(splits, cls_emb):
    def mk(part, shuffle):
        ds = ChunkGroupDataset(splits[part], cls_emb)
        return DataLoader(ds, batch_size=BATCH_ARTICLES, shuffle=shuffle, collate_fn=collate_pad)
    return mk("train", True), mk("val", False), mk("test", False)


def train_and_eval(window: dict, splits: dict, cls_emb: np.ndarray,
                   n_layers: int, width: int, activation: str, device: str) -> dict:
    torch.manual_seed(SEED); np.random.seed(SEED)
    model = ConfigurableMLP(INPUT_DIM, n_layers, width, activation).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()
    train_ld, val_ld, test_ld = make_loaders(splits, cls_emb)

    best_val, best_state, bad = float("inf"), None, 0
    epochs_run = 0
    for ep in range(MAX_EPOCHS):
        epochs_run = ep + 1
        model.train()
        for X, W, M, y in train_ld:
            X, W, M, y = X.to(device), W.to(device), M.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(X, W, M), y)
            loss.backward(); opt.step()

        model.eval()
        vl, n = 0.0, 0
        with torch.no_grad():
            for X, W, M, y in val_ld:
                X, W, M, y = X.to(device), W.to(device), M.to(device), y.to(device)
                vl += loss_fn(model(X, W, M), y).item() * len(y); n += len(y)
        val_loss = vl / max(n, 1)
        if val_loss < best_val - 1e-9:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break

    model.load_state_dict(best_state)
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for X, W, M, y in test_ld:
            X, W, M = X.to(device), W.to(device), M.to(device)
            preds.append(model(X, W, M).cpu().numpy())
            trues.append(y.numpy())
    pred = np.concatenate(preds) if preds else np.zeros(0)
    true = np.concatenate(trues) if trues else np.zeros(0)

    return {
        "n_test": int(len(true)),
        "epochs": int(epochs_run),
        "best_val_mse": float(best_val),
        "test_mse": float(mean_squared_error(true, pred)),
        "test_r2": float(r2_score(true, pred)) if len(true) > 1 else 0.0,
        "test_acc": float(accuracy_score(true > 0, pred > 0)),
        "test_f1": float(f1_score(true > 0, pred > 0, zero_division=0)),
    }


def load_done(csv_path: Path) -> set:
    if not csv_path.exists(): return set()
    df = pd.read_csv(csv_path)
    return set(zip(df["n_layers"], df["width"], df["activation"], df["window"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    if not args.resume and OUT_CSV.exists():
        OUT_CSV.unlink()
    done = load_done(OUT_CSV) if args.resume else set()

    print("Loading cached chunk features…")
    ds = pd.read_parquet(DATASET_PARQUET)
    chunk_meta = pd.read_parquet(CHUNKS_PARQUET)
    cls_emb = np.load(CHUNKS_EMB)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}  rows={len(ds):,}  chunks={len(chunk_meta):,}")

    # Pre-build splits per window so we don't redo the merge for every config
    print("Building splits per window…")
    window_splits = {}
    for w in WINDOWS:
        window_splits[w["name"]] = {
            part: build_groups_for_split(slice_df(ds, *w[part]), chunk_meta)
            for part in ("train", "val", "test")
        }
        s = window_splits[w["name"]]
        print(f"  {w['name']}: train={len(s['train'])} val={len(s['val'])} test={len(s['test'])}")

    grid = list(itertools.product(LAYERS, WIDTHS, ACTIVATIONS))
    total_runs = len(grid) * len(WINDOWS)
    update_stage("hp_sweep", status="in_progress", started=now_iso(),
                 total_configs=len(grid), total_runs=total_runs, completed=len(done))

    write_header = not OUT_CSV.exists()
    with OUT_CSV.open("a", newline="") as fh:
        writer = csv.writer(fh)
        if write_header:
            writer.writerow(["n_layers", "width", "activation", "window", "n_test", "epochs",
                             "best_val_mse", "test_mse", "test_r2", "test_acc", "test_f1", "elapsed_s"])
        run_idx = len(done)
        for n_layers, width, activation in grid:
            for w in WINDOWS:
                key = (n_layers, width, activation, w["name"])
                if key in done:
                    continue
                run_idx += 1
                t0 = time.time()
                update_stage("hp_sweep", status="in_progress",
                             current=f"{n_layers}L×{width}×{activation}@{w['name']}",
                             progress=f"{run_idx}/{total_runs}")
                try:
                    res = train_and_eval(w, window_splits[w["name"]], cls_emb,
                                          n_layers, width, activation, device)
                except Exception as e:
                    update_stage("hp_sweep", status="error", error=str(e), failed_at=str(key))
                    raise
                elapsed = time.time() - t0
                writer.writerow([n_layers, width, activation, w["name"], res["n_test"],
                                 res["epochs"], round(res["best_val_mse"], 8),
                                 round(res["test_mse"], 8), round(res["test_r2"], 6),
                                 round(res["test_acc"], 6), round(res["test_f1"], 6),
                                 round(elapsed, 1)])
                fh.flush()
                update_stage("hp_sweep",
                             progress=f"{run_idx}/{total_runs}",
                             last_acc=round(res["test_acc"], 4),
                             last_r2=round(res["test_r2"], 4))

    update_stage("hp_sweep", status="done", finished=now_iso(),
                 csv=str(OUT_CSV), runs=run_idx)
    print(f"\nDone. {run_idx} runs written to {OUT_CSV}")


if __name__ == "__main__":
    main()
