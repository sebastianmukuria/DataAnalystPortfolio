"""
FOMC NLP Sentiment Pipeline — Sentiment Analysis
Applies VADER and FinBERT to FOMC statement texts.
Produces sentence-level and document-level sentiment scores.
"""

import re
from typing import Optional

import numpy as np
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# ─── VADER ────────────────────────────────────────────────────────────────────

_vader = None


def get_vader() -> SentimentIntensityAnalyzer:
    global _vader
    if _vader is None:
        _vader = SentimentIntensityAnalyzer()
    return _vader


def vader_document_score(text: str) -> dict:
    """Compute VADER compound score for a full document."""
    analyzer = get_vader()
    # Split into sentences for more stable scoring
    sentences = re.split(r"(?<=[.!?])\s+", text)
    scores = [analyzer.polarity_scores(s)["compound"] for s in sentences if len(s) > 20]

    if not scores:
        scores = [analyzer.polarity_scores(text)["compound"]]

    return {
        "vader_mean": round(np.mean(scores), 4),
        "vader_std": round(np.std(scores), 4),
        "vader_min": round(np.min(scores), 4),
        "vader_max": round(np.max(scores), 4),
        "vader_n_sentences": len(scores),
        "vader_pct_positive": round(sum(s > 0.05 for s in scores) / len(scores), 4),
        "vader_pct_negative": round(sum(s < -0.05 for s in scores) / len(scores), 4),
    }


# ─── FinBERT ─────────────────────────────────────────────────────────────────

_finbert_pipeline = None


def get_finbert():
    """Lazy-load FinBERT pipeline. Requires transformers + torch."""
    global _finbert_pipeline
    if _finbert_pipeline is None:
        try:
            from transformers import pipeline
            _finbert_pipeline = pipeline(
                "text-classification",
                model="ProsusAI/finbert",
                tokenizer="ProsusAI/finbert",
                device=-1,  # CPU; set to 0 for GPU
                truncation=True,
                max_length=512,
            )
            print("[FinBERT] Model loaded")
        except ImportError:
            print("[FinBERT] transformers not installed — skipping")
    return _finbert_pipeline


def finbert_score(text: str, chunk_size: int = 400) -> dict:
    """
    Score a document with FinBERT by chunking into manageable segments.
    Returns averaged label probabilities.
    """
    pipe = get_finbert()
    if pipe is None:
        return {"finbert_positive": None, "finbert_negative": None, "finbert_neutral": None}

    words = text.split()
    chunks = [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    chunks = [c for c in chunks if len(c) > 50]

    label_scores = {"positive": [], "negative": [], "neutral": []}
    for chunk in chunks:
        try:
            result = pipe(chunk, top_k=None)
            for item in result:
                label = item["label"].lower()
                if label in label_scores:
                    label_scores[label].append(item["score"])
        except Exception:
            pass

    return {
        "finbert_positive": round(np.mean(label_scores["positive"]), 4) if label_scores["positive"] else None,
        "finbert_negative": round(np.mean(label_scores["negative"]), 4) if label_scores["negative"] else None,
        "finbert_neutral": round(np.mean(label_scores["neutral"]), 4) if label_scores["neutral"] else None,
    }


def finbert_stance(finbert_scores: dict) -> Optional[str]:
    """Map FinBERT scores to hawkish/dovish/neutral for Fed context.

    Note: FinBERT was trained on financial news where 'positive' sentiment
    often aligns with tightening expectations. We remap as follows:
      positive → Hawkish (markets view rate hikes as strength signal)
      negative → Dovish  (distress language → easing expectations)
      neutral  → Neutral
    """
    pos = finbert_scores.get("finbert_positive")
    neg = finbert_scores.get("finbert_negative")
    neu = finbert_scores.get("finbert_neutral")

    if pos is None:
        return None

    vals = {"Hawkish": pos, "Dovish": neg, "Neutral": neu}
    return max(vals, key=lambda k: vals[k] or 0)


# ─── Combined Scoring ─────────────────────────────────────────────────────────

def score_document(text: str, use_finbert: bool = False) -> dict:
    """Run all sentiment models on a single document."""
    result = {}
    result.update(vader_document_score(text))

    if use_finbert:
        fb = finbert_score(text)
        result.update(fb)
        result["finbert_stance"] = finbert_stance(fb)

    return result


def score_all_statements(
    df: pd.DataFrame,
    text_col: str = "cleaned_text",
    use_finbert: bool = False,
) -> pd.DataFrame:
    """
    Score all FOMC statements. Appends sentiment columns in-place.
    Set use_finbert=True to add FinBERT scores (slow on CPU).
    """
    scores = []
    total = len(df)
    for i, (_, row) in enumerate(df.iterrows()):
        if i % 5 == 0:
            print(f"  Scoring {i+1}/{total}...")
        text = row.get(text_col, row.get("text", ""))
        scores.append(score_document(str(text), use_finbert=use_finbert))

    scores_df = pd.DataFrame(scores)
    return pd.concat([df.reset_index(drop=True), scores_df], axis=1)
