"""
FOMC NLP Sentiment Pipeline — Text Preprocessor
Cleans and tokenizes Fed statement text for downstream analysis.
"""

import re
import string
from typing import Optional

import pandas as pd

# ─── Boilerplate patterns to strip ───────────────────────────────────────────

NOISE_PATTERNS = [
    r"For release at \d+:\d+ [ap]\.m\. E[SD]T",
    r"For immediate release",
    r"Implementation Note.*?(?=\n\n|\Z)",
    r"Voting for this action:.*?(?=\n\n|\Z)",
    r"Voting against.*?(?=\n\n|\Z)",
    r"https?://\S+",
    r"\*{3,}",
    r"\d{4}/monetary\d+[a-z]*\.htm",
]

# Policy keywords by stance
HAWKISH_TERMS = [
    "inflation", "overshoot", "tighten", "restrictive", "rate hike",
    "raise rates", "increase rates", "elevated", "persistent", "strong",
    "robust", "above target", "overheating", "wage growth", "price pressure",
    "supply chain", "energy prices", "upside risk", "balance sheet reduction",
    "quantitative tightening",
]

DOVISH_TERMS = [
    "accommodative", "stimulus", "ease", "cut rates", "lower rates",
    "support", "transitory", "below target", "slack", "unemployment",
    "labor market weakness", "downside risk", "uncertainty", "fragile",
    "quantitative easing", "asset purchases", "forward guidance",
    "zero lower bound", "near zero", "ample",
]

UNCERTAINTY_TERMS = [
    "uncertain", "uncertainty", "monitor", "assess", "data-dependent",
    "as appropriate", "gradual", "patient", "flexible", "depend",
    "evolve", "outlook", "risks",
]


def clean_text(text: str) -> str:
    """Remove boilerplate, normalize whitespace."""
    for pattern in NOISE_PATTERNS:
        text = re.sub(pattern, " ", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def count_terms(text: str, terms: list[str]) -> int:
    """Count occurrences of policy-relevant terms (case-insensitive)."""
    text_lower = text.lower()
    return sum(text_lower.count(t.lower()) for t in terms)


def extract_policy_signals(text: str) -> dict:
    """
    Extract keyword counts and derive hawkish/dovish scores.
    Returns a dictionary of signal features.
    """
    cleaned = clean_text(text)
    words = cleaned.split()
    word_count = len(words)

    hawk_count = count_terms(cleaned, HAWKISH_TERMS)
    dove_count = count_terms(cleaned, DOVISH_TERMS)
    uncertainty_count = count_terms(cleaned, UNCERTAINTY_TERMS)

    # Normalise by document length (per 100 words)
    hawk_norm = hawk_count / max(word_count, 1) * 100
    dove_norm = dove_count / max(word_count, 1) * 100

    net_hawk_score = hawk_norm - dove_norm

    # Mention of specific rate language
    mentions_rate_increase = bool(re.search(
        r"(raise|increase|hike).{0,30}(rate|federal funds)", cleaned, re.I
    ))
    mentions_rate_cut = bool(re.search(
        r"(cut|lower|reduce|decrease).{0,30}(rate|federal funds)", cleaned, re.I
    ))
    mentions_hold = bool(re.search(
        r"(maintain|hold|keep).{0,30}(rate|federal funds|target)", cleaned, re.I
    ))
    mentions_qe = bool(re.search(
        r"(asset purchase|quantitative eas|purchase program)", cleaned, re.I
    ))
    mentions_qt = bool(re.search(
        r"(balance sheet.{0,20}reduc|quantitative tight|runoff)", cleaned, re.I
    ))

    return {
        "word_count": word_count,
        "hawk_count": hawk_count,
        "dove_count": dove_count,
        "uncertainty_count": uncertainty_count,
        "hawk_norm": round(hawk_norm, 4),
        "dove_norm": round(dove_norm, 4),
        "net_hawk_score": round(net_hawk_score, 4),
        "mentions_rate_increase": int(mentions_rate_increase),
        "mentions_rate_cut": int(mentions_rate_cut),
        "mentions_hold": int(mentions_hold),
        "mentions_qe": int(mentions_qe),
        "mentions_qt": int(mentions_qt),
        "cleaned_text": cleaned,
    }


def classify_stance(net_hawk_score: float, threshold: float = 0.5) -> str:
    """
    Map net hawk score to a categorical policy stance.
    Thresholds are empirically calibrated against known FOMC periods.
    """
    if net_hawk_score > threshold:
        return "Hawkish"
    elif net_hawk_score < -threshold:
        return "Dovish"
    else:
        return "Neutral"


def process_fomc_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply preprocessing to a FOMC statements DataFrame.
    Expects columns: date, text
    Returns enriched DataFrame with all signal features.
    """
    signals = df["text"].apply(extract_policy_signals)
    signal_df = pd.DataFrame(signals.tolist())

    out = pd.concat([df[["date", "year", "url"]].reset_index(drop=True), signal_df], axis=1)
    out["stance_keyword"] = out["net_hawk_score"].apply(classify_stance)
    out["date"] = pd.to_datetime(out["date"])
    out = out.sort_values("date").reset_index(drop=True)
    return out
