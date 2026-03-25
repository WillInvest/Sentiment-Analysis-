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
