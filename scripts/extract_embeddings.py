"""Extract mean-pool embeddings from frozen pretrained encoders.

Extracts embeddings from BERT-base, FinBERT, and RoBERTa by mean-pooling over
all non-padding token hidden states (instead of [CLS] only). Saves one .pt file
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
            # Mean-pool over all non-padding tokens (masked mean)
            mask = inputs["attention_mask"].unsqueeze(-1).float()
            sum_emb = (outputs.last_hidden_state * mask).sum(dim=1)
            count = mask.sum(dim=1).clamp(min=1)
            mean_embeddings = (sum_emb / count).cpu()

        for idx, aid in enumerate(batch_ids):
            embeddings[aid] = mean_embeddings[idx]

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
