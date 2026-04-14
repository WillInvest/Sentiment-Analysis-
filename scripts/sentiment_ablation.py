"""Run FinBERT on the full processed dataset two ways and produce an ablation:

1. Naive head-512 truncation (what the demo does).
2. Sentence-aware sliding window with confidence-weighted pooling.

Outputs:
- results/predictions/finbert_full.parquet  (per-article scores from both methods)
- results/figures/sentiment_ablation.png    (scatter + delta histogram on n_chunks>=2)

Usage:  python scripts/sentiment_ablation.py
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DATASET = ROOT / "data/processed/full_dataset.parquet"
OUT_PARQUET = ROOT / "results/predictions/finbert_full.parquet"
OUT_FIG = ROOT / "results/figures/sentiment_ablation.png"
MODEL_ID = "ProsusAI/finbert"
MAX_TOKENS = 510
OVERLAP_K = 2
BATCH_SIZE = 64

_ABBREV = {"mr","mrs","ms","dr","inc","co","corp","ltd","jr","sr","vs","etc","u.s","u.k","e.g","i.e","no"}
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


def chunk_by_sentences(text: str, tokenizer, max_tokens: int = MAX_TOKENS, overlap_k: int = OVERLAP_K) -> list[str]:
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


@torch.no_grad()
def score_texts(texts: list[str], tok, model, device: str, batch_size: int = BATCH_SIZE) -> np.ndarray:
    out = []
    for i in tqdm(range(0, len(texts), batch_size), desc="scoring", leave=False):
        enc = tok(texts[i:i+batch_size], padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
        out.append(torch.softmax(model(**enc).logits, dim=-1).cpu().numpy())
    return np.vstack(out) if out else np.zeros((0, 3))


def main():
    print(f"Loading dataset from {DATASET}")
    df = pd.read_parquet(DATASET).reset_index(drop=True)
    print(f"  {len(df):,} article-ticker rows")

    # Dedupe text per unique article — multi-ticker articles have identical body
    text_col = "body" if "body" in df.columns else "text"
    unique_articles = df.drop_duplicates(subset="article_id")[["article_id", text_col]].rename(columns={text_col: "body"}).reset_index(drop=True)
    print(f"  {len(unique_articles):,} unique articles to score")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading FinBERT on {device}")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID).to(device).eval()
    id2l = model.config.id2label
    POS = next(i for i, l in id2l.items() if "pos" in l.lower())
    NEG = next(i for i, l in id2l.items() if "neg" in l.lower())
    NEU = next(i for i, l in id2l.items() if "neu" in l.lower())
    print(f"  label map: {id2l}")

    # ---- chunk all bodies ----
    print("Chunking bodies (sentence-aware sliding window)...")
    owners: list[int] = []
    flat: list[str] = []
    n_chunks_per_article = np.zeros(len(unique_articles), dtype=np.int32)
    for ridx, body in enumerate(tqdm(unique_articles["body"].fillna("").tolist(), desc="chunking")):
        chunks = chunk_by_sentences(body, tok)
        n_chunks_per_article[ridx] = len(chunks)
        for c in chunks:
            owners.append(ridx); flat.append(c)
    owners = np.array(owners, dtype=np.int32)
    print(f"  total chunks: {len(flat):,}  (avg {len(flat)/len(unique_articles):.2f}/article, max {n_chunks_per_article.max()})")

    # ---- sliding-window pass ----
    print("Scoring chunks with FinBERT...")
    chunk_probs = score_texts(flat, tok, model, device)  # (n_chunks, 3)

    # Confidence-weighted pool per article
    print("Pooling per article...")
    pooled = np.full((len(unique_articles), 3), np.nan, dtype=np.float32)
    for ridx in range(len(unique_articles)):
        m = owners == ridx
        p = chunk_probs[m]
        if len(p) == 0:
            continue
        w = p.max(axis=1)
        w = w / w.sum()
        pooled[ridx] = (p * w[:, None]).sum(axis=0)

    # ---- naive truncation pass ----
    print("Scoring full bodies with naive head-512 truncation...")
    trunc_probs = score_texts(unique_articles["body"].fillna("").tolist(), tok, model, device)

    # ---- assemble output ----
    out = unique_articles[["article_id"]].copy()
    out["n_chunks"]    = n_chunks_per_article
    out["pos_slide"]   = pooled[:, POS]
    out["neg_slide"]   = pooled[:, NEG]
    out["neu_slide"]   = pooled[:, NEU]
    out["score_slide"] = pooled[:, POS] - pooled[:, NEG]
    out["pos_trunc"]   = trunc_probs[:, POS]
    out["neg_trunc"]   = trunc_probs[:, NEG]
    out["neu_trunc"]   = trunc_probs[:, NEU]
    out["score_trunc"] = trunc_probs[:, POS] - trunc_probs[:, NEG]

    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_PARQUET, index=False)
    print(f"  wrote {OUT_PARQUET}  ({len(out):,} rows)")

    # ---- ablation plot on n_chunks >= 2 ----
    sub = out[out["n_chunks"] >= 2].copy()
    delta = sub["score_slide"].values - sub["score_trunc"].values
    n_meaningful = int(((np.sign(sub["score_slide"]) != np.sign(sub["score_trunc"])) &
                        (sub["score_slide"].abs() > 0.05) & (sub["score_trunc"].abs() > 0.05)).sum())
    print(f"\nAblation on n_chunks>=2:")
    print(f"  articles:                {len(sub):,} / {len(out):,}")
    print(f"  mean |delta|:            {np.abs(delta).mean():.4f}")
    print(f"  meaningful sign flips:   {n_meaningful}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    sc = ax.scatter(sub["score_trunc"], sub["score_slide"], c=sub["n_chunks"],
                    cmap="viridis", s=14, alpha=0.55, edgecolor="none")
    lim = [-1.05, 1.05]
    ax.plot(lim, lim, "r--", lw=1, label="y = x (agreement)")
    ax.axhline(0, color="gray", lw=0.5); ax.axvline(0, color="gray", lw=0.5)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("Truncated (head-512)  pos − neg")
    ax.set_ylabel("Sliding window (conf-pooled)  pos − neg")
    ax.set_title(f"Article-level FinBERT sentiment\nn={len(sub):,} articles with ≥2 chunks")
    ax.legend(loc="upper left")
    plt.colorbar(sc, ax=ax, label="# chunks")

    ax = axes[1]
    ax.hist(delta, bins=60, color="#4C72B0", edgecolor="white")
    ax.axvline(0, color="red", ls="--", lw=1)
    ax.set_xlabel("Δ = sliding − truncated")
    ax.set_ylabel("# articles")
    ax.set_title(f"Score shift from full-body pooling\nmean |Δ| = {np.abs(delta).mean():.3f}, sign flips = {n_meaningful}")

    plt.tight_layout()
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, dpi=140)
    print(f"  wrote {OUT_FIG}")


if __name__ == "__main__":
    main()
