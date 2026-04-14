"""End-to-end training pipeline for chunk-as-row sentiment → return prediction.

Stages (each is resumable — re-running skips finished stages):
  1. data_prep         — build full_dataset.parquet from raw CSVs
  2. chunk_features    — sentence-chunk every article, run FinBERT once,
                         save per-chunk (pos, neg, neu, confidence) + [CLS] embeddings
  3. baseline          — confidence-weighted FinBERT zero-shot baseline per window
  4. train_regressor   — per-window MLP with chunk-aggregated MSE loss
  5. train_classifier  — per-window MLP with chunk-aggregated BCE loss
  6. metrics           — final R²/MSE/Acc/F1 table + JSON summary

Intermediate artifacts (all re-used on resume):
  data/processed/full_dataset.parquet
  results/predictions/finbert_chunks.parquet      (per-chunk metadata + sentiment)
  results/predictions/finbert_chunks_emb.npy      (per-chunk [CLS] embeddings, aligned by row)
  results/checkpoints/{regressor,classifier}_{w1,w2,w3}.pt
  results/predictions/{regressor,classifier}_{w1,w2,w3}_test.parquet
  results/predictions/baseline_{w1,w2,w3}_test.parquet
  results/metrics/final_metrics.json
  results/pipeline_status.json                    (live progress for the monitor)

Usage:
    python scripts/pipeline.py                  # run all stages
    python scripts/pipeline.py --stage chunk_features  # run one stage
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ───────────────────────────── paths ─────────────────────────────
DATA_RAW = ROOT / "data/raw"
DATA_PROC = ROOT / "data/processed"
RESULTS = ROOT / "results"
PRED_DIR = RESULTS / "predictions"
CKPT_DIR = RESULTS / "checkpoints"
METRICS_DIR = RESULTS / "metrics"

DATASET_PARQUET = DATA_PROC / "full_dataset.parquet"
CHUNKS_PARQUET = PRED_DIR / "finbert_chunks.parquet"
CHUNKS_EMB = PRED_DIR / "finbert_chunks_emb.npy"
STATUS_PATH = RESULTS / "pipeline_status.json"
METRICS_JSON = METRICS_DIR / "final_metrics.json"

# ───────────────────────────── config ─────────────────────────────
MODEL_ID = "ProsusAI/finbert"
SEED = 42
PURGE = pd.Timedelta(days=3)

WINDOWS = [
    {"name": "w1", "train": ("2017-01-01", "2018-12-27"),
                   "val":   ("2019-01-02", "2019-06-27"),
                   "test":  ("2019-07-02", "2019-12-27")},
    {"name": "w2", "train": ("2017-07-01", "2019-06-27"),
                   "val":   ("2019-07-02", "2019-12-27"),
                   "test":  ("2020-01-02", "2020-06-27")},
    {"name": "w3", "train": ("2018-01-01", "2019-12-27"),
                   "val":   ("2020-01-02", "2020-06-27"),
                   "test":  ("2020-07-02", "2020-12-30")},
]

# Chunk extraction
MAX_TOKENS = 510
OVERLAP_K = 2
SCORE_BATCH = 32

# Training
BATCH_ARTICLES = 64
MAX_EPOCHS = 100
PATIENCE = 10
LR = 1e-3
DROPOUT = 0.2
HIDDEN = (64, 32)
INPUT_DIM = 768 + 3       # [CLS] + [pos, neg, neu]


# ───────────────────────── status / monitor ─────────────────────────
def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_status() -> dict:
    if STATUS_PATH.exists():
        try:
            return json.loads(STATUS_PATH.read_text())
        except json.JSONDecodeError:
            pass
    return {"started": now_iso(), "stages": {}}


def save_status(s: dict) -> None:
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATUS_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(s, indent=2, default=str))
    tmp.replace(STATUS_PATH)


def update_stage(name: str, **fields) -> None:
    s = load_status()
    s["stages"].setdefault(name, {})
    s["stages"][name].update(fields)
    s["stages"][name]["updated"] = now_iso()
    save_status(s)


def stage_done(name: str) -> bool:
    return load_status().get("stages", {}).get(name, {}).get("status") == "done"


# ────────────────────── stage 1: data preparation ──────────────────────
def stage_data_prep() -> None:
    if DATASET_PARQUET.exists():
        n = len(pd.read_parquet(DATASET_PARQUET, columns=["article_id"]))
        update_stage("data_prep", status="done", rows=n, note="already on disk")
        return

    update_stage("data_prep", status="in_progress", started=now_iso())
    from utils.data_loader import build_dataset, load_config, load_raw_data

    config = load_config()
    news, price = load_raw_data()
    df = build_dataset(news, price, config)
    DATA_PROC.mkdir(parents=True, exist_ok=True)
    df.to_parquet(DATASET_PARQUET, index=False)
    update_stage("data_prep", status="done", rows=len(df), finished=now_iso())


# ──────────────────── stage 2: chunk-level FinBERT ────────────────────
_ABBREV = {"mr","mrs","ms","dr","inc","co","corp","ltd","jr","sr",
           "vs","etc","u.s","u.k","e.g","i.e","no"}
_SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[\"'A-Z0-9])")


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    out: list[str] = []
    for s in _SENT_RE.split(text):
        if out and out[-1].rstrip(".").split()[-1].lower() in _ABBREV:
            out[-1] += " " + s
        else:
            out.append(s)
    return [s for s in out if s]


def chunk_by_sentences(text: str, tokenizer,
                       max_tokens: int = MAX_TOKENS,
                       overlap_k: int = OVERLAP_K) -> list[str]:
    sents = split_sentences(text)
    if not sents:
        return []
    lens = [len(tokenizer.encode(s, add_special_tokens=False)) for s in sents]
    chunks: list[str] = []
    i, n = 0, len(sents)
    while i < n:
        cur, cl, j = [], 0, i
        while j < n and cl + lens[j] <= max_tokens:
            cur.append(sents[j]); cl += lens[j]; j += 1
        if not cur:
            ids = tokenizer.encode(sents[i], add_special_tokens=False)[:max_tokens]
            chunks.append(tokenizer.decode(ids)); i += 1
            continue
        chunks.append(" ".join(cur))
        if j >= n:
            break
        i = max(i + 1, j - overlap_k)
    return chunks


def stage_chunk_features() -> None:
    if CHUNKS_PARQUET.exists() and CHUNKS_EMB.exists():
        n = len(pd.read_parquet(CHUNKS_PARQUET, columns=["article_id"]))
        update_stage("chunk_features", status="done", chunks=n, note="already on disk")
        return

    update_stage("chunk_features", status="in_progress", started=now_iso(), phase="loading")
    df = pd.read_parquet(DATASET_PARQUET)
    text_col = "body" if "body" in df.columns else "text"
    articles = (df.drop_duplicates("article_id")[["article_id", text_col]]
                  .rename(columns={text_col: "body"})
                  .reset_index(drop=True))
    update_stage("chunk_features", phase="model_init", n_articles=len(articles))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID).to(device).eval()
    id2l = model.config.id2label
    POS = next(i for i, l in id2l.items() if "pos" in l.lower())
    NEG = next(i for i, l in id2l.items() if "neg" in l.lower())
    NEU = next(i for i, l in id2l.items() if "neu" in l.lower())

    # Chunk every article (CPU; fast)
    update_stage("chunk_features", phase="chunking")
    rows = []
    flat_chunks: list[str] = []
    bodies = articles["body"].fillna("").tolist()
    aids = articles["article_id"].astype(int).tolist()
    for ridx, body in enumerate(tqdm(bodies, desc="chunking", file=sys.stdout)):
        chunks = chunk_by_sentences(body, tok)
        n = len(chunks)
        for ci, ctext in enumerate(chunks):
            rows.append((aids[ridx], ci, n))
            flat_chunks.append(ctext)
        if (ridx + 1) % 1000 == 0:
            update_stage("chunk_features", phase="chunking",
                         progress=f"{ridx+1}/{len(articles)}",
                         total_chunks=len(flat_chunks))

    n_chunks = len(flat_chunks)
    update_stage("chunk_features", phase="scoring", total_chunks=n_chunks)

    # Score in batches, capture probs + [CLS]
    probs = np.zeros((n_chunks, 3), dtype=np.float32)
    cls_emb = np.zeros((n_chunks, 768), dtype=np.float32)
    with torch.no_grad():
        for i in tqdm(range(0, n_chunks, SCORE_BATCH), desc="finbert", file=sys.stdout):
            batch = flat_chunks[i:i+SCORE_BATCH]
            enc = tok(batch, padding=True, truncation=True,
                      max_length=512, return_tensors="pt").to(device)
            out = model(**enc, output_hidden_states=True)
            probs[i:i+len(batch)] = torch.softmax(out.logits, dim=-1).cpu().numpy()
            cls_emb[i:i+len(batch)] = out.hidden_states[-1][:, 0, :].cpu().numpy()
            if (i // SCORE_BATCH) % 10 == 0:
                update_stage("chunk_features", phase="scoring",
                             progress=f"{i+len(batch)}/{n_chunks}")

    chunk_df = pd.DataFrame(rows, columns=["article_id", "chunk_idx", "n_chunks"])
    chunk_df["pos"] = probs[:, POS]
    chunk_df["neg"] = probs[:, NEG]
    chunk_df["neu"] = probs[:, NEU]
    chunk_df["confidence"] = probs.max(axis=1)
    # row_pos = chunk's row in cls_emb / chunk_df (used to look up embeddings later)
    chunk_df["row_pos"] = np.arange(len(chunk_df), dtype=np.int64)

    PRED_DIR.mkdir(parents=True, exist_ok=True)
    chunk_df.to_parquet(CHUNKS_PARQUET, index=False)
    np.save(CHUNKS_EMB, cls_emb)
    update_stage("chunk_features", status="done",
                 chunks=int(n_chunks), articles=int(len(articles)),
                 finished=now_iso(), phase="done")


# ────────────────────── stage 3-5: per-window training ──────────────────────
def slice_df(d: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    return d[(d["date"] >= s) & (d["date"] <= e)].copy()


def build_groups_for_split(part_df: pd.DataFrame, chunk_meta: pd.DataFrame) -> list[dict]:
    """Slice-then-explode: join part_df with chunk metadata, group by (article, ticker, date)."""
    merged = part_df.merge(chunk_meta, on="article_id", how="inner")
    groups = []
    for (aid, tk, dt), g in merged.groupby(["article_id", "ticker", "date"], sort=False):
        groups.append({
            "article_id": int(aid),
            "ticker": tk,
            "date": dt,
            "row_positions": g["row_pos"].to_numpy(dtype=np.int64),
            "confidences":   g["confidence"].to_numpy(dtype=np.float32),
            "pos_arr":       g["pos"].to_numpy(dtype=np.float32),
            "neg_arr":       g["neg"].to_numpy(dtype=np.float32),
            "neu_arr":       g["neu"].to_numpy(dtype=np.float32),
            "label":         float(g["r_1d"].iloc[0]),
        })
    return groups


class ChunkGroupDataset(Dataset):
    """One sample = one (article, ticker, date) with a variable number of chunks."""

    def __init__(self, groups: list[dict], cls_emb: np.ndarray):
        self.groups = groups
        self.cls = cls_emb

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, idx: int):
        g = self.groups[idx]
        cls = self.cls[g["row_positions"]]                              # (n, 768)
        sent = np.stack([g["pos_arr"], g["neg_arr"], g["neu_arr"]], axis=1)  # (n, 3)
        feats = np.concatenate([cls, sent], axis=1).astype(np.float32)  # (n, 771)
        return feats, g["confidences"], g["label"]


def collate_pad(batch):
    feats_list, conf_list, labels = zip(*batch)
    B = len(batch)
    K = max(f.shape[0] for f in feats_list)
    D = feats_list[0].shape[1]
    X = np.zeros((B, K, D), dtype=np.float32)
    W = np.zeros((B, K), dtype=np.float32)
    M = np.zeros((B, K), dtype=np.float32)
    for i, (f, c) in enumerate(zip(feats_list, conf_list)):
        n = f.shape[0]
        X[i, :n] = f
        s = float(c.sum())
        W[i, :n] = c / s if s > 0 else 1.0 / n
        M[i, :n] = 1.0
    return (
        torch.from_numpy(X),
        torch.from_numpy(W),
        torch.from_numpy(M),
        torch.tensor(labels, dtype=torch.float32),
    )


class ChunkAggregateMLP(nn.Module):
    def __init__(self, in_dim=INPUT_DIM, h1=HIDDEN[0], h2=HIDDEN[1], dropout=DROPOUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, h1), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(h1, h2),     nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(h2, 1),
        )

    def forward(self, X, W, mask):
        # X: (B,K,D)  W: (B,K)  mask: (B,K)
        chunk_pred = self.net(X).squeeze(-1)              # (B,K)
        chunk_pred = chunk_pred * mask                    # zero out padding garbage
        return (chunk_pred * W).sum(dim=1)                # (B,)  weighted sum


def make_loaders(splits: dict, cls_emb: np.ndarray):
    def mk(part: str, shuffle: bool):
        ds = ChunkGroupDataset(splits[part], cls_emb)
        return DataLoader(ds, batch_size=BATCH_ARTICLES, shuffle=shuffle,
                          collate_fn=collate_pad, num_workers=0)
    return mk("train", True), mk("val", False), mk("test", False)


def train_one_window(head_type: str, window: dict, splits: dict, cls_emb: np.ndarray,
                     stage_name: str) -> dict:
    torch.manual_seed(SEED); np.random.seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ChunkAggregateMLP().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss() if head_type == "regressor" else nn.BCEWithLogitsLoss()
    train_ld, val_ld, test_ld = make_loaders(splits, cls_emb)

    history = {"train": [], "val": []}
    best_val, best_state, bad = float("inf"), None, 0
    for ep in range(MAX_EPOCHS):
        model.train()
        tl, n = 0.0, 0
        for X, W, M, y in train_ld:
            X, W, M, y = X.to(device), W.to(device), M.to(device), y.to(device)
            opt.zero_grad()
            pred = model(X, W, M)
            target = y if head_type == "regressor" else (y > 0).float()
            loss = loss_fn(pred, target)
            loss.backward(); opt.step()
            tl += loss.item() * len(y); n += len(y)
        train_loss = tl / max(n, 1)

        model.eval()
        vl, n = 0.0, 0
        with torch.no_grad():
            for X, W, M, y in val_ld:
                X, W, M, y = X.to(device), W.to(device), M.to(device), y.to(device)
                pred = model(X, W, M)
                target = y if head_type == "regressor" else (y > 0).float()
                vl += loss_fn(pred, target).item() * len(y); n += len(y)
        val_loss = vl / max(n, 1)

        history["train"].append(train_loss); history["val"].append(val_loss)
        update_stage(stage_name, current_window=window["name"],
                     epoch=ep + 1, train_loss=round(train_loss, 6),
                     val_loss=round(val_loss, 6),
                     best_val=round(min(best_val, val_loss), 6))

        if val_loss < best_val - 1e-9:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break

    model.load_state_dict(best_state)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state, "history": history,
                "head_type": head_type, "window": window["name"]},
               CKPT_DIR / f"{head_type}_{window['name']}.pt")

    # Test predictions
    model.eval()
    preds = []
    with torch.no_grad():
        for X, W, M, _ in test_ld:
            X, W, M = X.to(device), W.to(device), M.to(device)
            pr = model(X, W, M).cpu().numpy()
            if head_type == "classifier":
                pr = 1.0 / (1.0 + np.exp(-pr))  # sigmoid → probability
            preds.append(pr)
    preds = np.concatenate(preds) if preds else np.zeros(0)

    test_groups = splits["test"]
    out = pd.DataFrame({
        "article_id": [g["article_id"] for g in test_groups],
        "ticker":     [g["ticker"] for g in test_groups],
        "date":       [g["date"] for g in test_groups],
        "r_1d":       [g["label"] for g in test_groups],
        "pred":       preds,
    })
    out.to_parquet(PRED_DIR / f"{head_type}_{window['name']}_test.parquet", index=False)
    return history


def stage_train(head_type: str) -> None:
    stage_name = f"train_{head_type}"
    if all((CKPT_DIR / f"{head_type}_{w['name']}.pt").exists() for w in WINDOWS):
        update_stage(stage_name, status="done", note="all checkpoints already on disk")
        return

    update_stage(stage_name, status="in_progress", started=now_iso())
    ds = pd.read_parquet(DATASET_PARQUET)
    chunk_meta = pd.read_parquet(CHUNKS_PARQUET)
    cls_emb = np.load(CHUNKS_EMB)

    for w in WINDOWS:
        ckpt = CKPT_DIR / f"{head_type}_{w['name']}.pt"
        if ckpt.exists():
            update_stage(stage_name, current_window=w["name"], note="window already done")
            continue
        update_stage(stage_name, current_window=w["name"], phase="splitting")
        splits = {part: build_groups_for_split(slice_df(ds, *w[part]), chunk_meta)
                  for part in ("train", "val", "test")}
        update_stage(stage_name, current_window=w["name"], phase="training",
                     n_train=len(splits["train"]), n_val=len(splits["val"]),
                     n_test=len(splits["test"]))
        train_one_window(head_type, w, splits, cls_emb, stage_name)

    update_stage(stage_name, status="done", finished=now_iso())


# ────────────────────── stage 3 (early): zero-shot baseline ──────────────────────
def stage_baseline() -> None:
    if all((PRED_DIR / f"baseline_{w['name']}_test.parquet").exists() for w in WINDOWS):
        update_stage("baseline", status="done", note="already on disk")
        return

    update_stage("baseline", status="in_progress", started=now_iso())
    ds = pd.read_parquet(DATASET_PARQUET)
    chunk_meta = pd.read_parquet(CHUNKS_PARQUET)

    for w in WINDOWS:
        out_path = PRED_DIR / f"baseline_{w['name']}_test.parquet"
        if out_path.exists():
            continue
        test = slice_df(ds, *w["test"])
        merged = test.merge(chunk_meta, on="article_id", how="inner")
        merged["weight_num"] = merged["confidence"]
        # Group by article-ticker to compute confidence-weighted (pos − neg)
        rows = []
        for (aid, tk, dt), g in merged.groupby(["article_id", "ticker", "date"], sort=False):
            w_arr = g["confidence"].to_numpy()
            w_norm = w_arr / w_arr.sum() if w_arr.sum() > 0 else np.full(len(w_arr), 1/len(w_arr))
            score = float(((g["pos"].to_numpy() - g["neg"].to_numpy()) * w_norm).sum())
            rows.append({"article_id": int(aid), "ticker": tk, "date": dt,
                         "r_1d": float(g["r_1d"].iloc[0]), "pred": score})
        pd.DataFrame(rows).to_parquet(out_path, index=False)
        update_stage("baseline", current_window=w["name"], n=len(rows))

    update_stage("baseline", status="done", finished=now_iso())


# ────────────────────── stage 6: metrics ──────────────────────
def compute_metrics(pred_df: pd.DataFrame, head_type: str) -> dict:
    y_true = pred_df["r_1d"].to_numpy()
    pred = pred_df["pred"].to_numpy()
    out = {}
    if head_type == "regressor":
        out["mse"] = float(mean_squared_error(y_true, pred))
        out["r2"]  = float(r2_score(y_true, pred))
        out["acc"] = float(accuracy_score(y_true > 0, pred > 0))
        out["f1"]  = float(f1_score(y_true > 0, pred > 0))
    elif head_type == "classifier":
        out["acc"] = float(accuracy_score(y_true > 0, pred > 0.5))
        out["f1"]  = float(f1_score(y_true > 0, pred > 0.5))
    elif head_type == "baseline":
        out["acc"] = float(accuracy_score(y_true > 0, pred > 0))
        out["f1"]  = float(f1_score(y_true > 0, pred > 0))
    return out


def stage_metrics() -> None:
    update_stage("metrics", status="in_progress", started=now_iso())
    summary = {"windows": {}}
    for w in WINDOWS:
        wname = w["name"]
        summary["windows"][wname] = {}
        for head_type in ("regressor", "classifier", "baseline"):
            path = PRED_DIR / f"{head_type}_{wname}_test.parquet"
            if not path.exists():
                continue
            df = pd.read_parquet(path)
            summary["windows"][wname][head_type] = {"n": len(df), **compute_metrics(df, head_type)}
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_JSON.write_text(json.dumps(summary, indent=2))
    update_stage("metrics", status="done", finished=now_iso(), summary_path=str(METRICS_JSON))
    print("\nFinal metrics:")
    print(json.dumps(summary, indent=2))


# ───────────────────────────── runner ─────────────────────────────
STAGE_FUNCS = {
    "data_prep":        stage_data_prep,
    "chunk_features":   stage_chunk_features,
    "baseline":         stage_baseline,
    "train_regressor":  lambda: stage_train("regressor"),
    "train_classifier": lambda: stage_train("classifier"),
    "metrics":          stage_metrics,
}
STAGE_ORDER = ["data_prep", "chunk_features", "baseline",
               "train_regressor", "train_classifier", "metrics"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=STAGE_ORDER, help="run a single stage")
    parser.add_argument("--force", action="store_true", help="ignore stage_done check")
    args = parser.parse_args()

    stages = [args.stage] if args.stage else STAGE_ORDER
    for name in stages:
        if not args.force and stage_done(name):
            print(f"[skip] {name} (already done)")
            continue
        print(f"[run]  {name}")
        t0 = time.time()
        try:
            STAGE_FUNCS[name]()
        except Exception as e:
            update_stage(name, status="error", error=str(e))
            raise
        print(f"[done] {name}  ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
