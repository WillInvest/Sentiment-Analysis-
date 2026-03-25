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
