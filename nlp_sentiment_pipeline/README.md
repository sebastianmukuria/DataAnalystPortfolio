# FOMC Statement NLP Sentiment Pipeline

**Type:** End-to-End NLP Pipeline + Market Correlation Analysis
**Stack:** Python, BeautifulSoup, VADER, FinBERT (HuggingFace), yfinance, Plotly, scipy
**Data Sources:** Federal Reserve website (FOMC statements 2015–2024), yfinance (SPY, BTC, TLT, GLD)

---

## Business Problem

Investment managers and macro traders spend significant time manually reading Federal Reserve meeting statements to gauge policy stance — **hawkish** (rates rising) vs **dovish** (rates falling). This is slow and subjective. A quantitative signal from NLP enables faster, more consistent positioning.

## Pipeline Architecture

```
FOMC Statements (federalreserve.gov — public)
        ↓
   Web Scraper (requests + BeautifulSoup)
        ↓
  Text Preprocessor
   - Boilerplate removal (voting records, legal notices)
   - Whitespace normalisation
        ↓
  ┌──────────────────────────────────────────────┐
  │  Sentiment Models                            │
  │  1. Keyword scoring (hawk/dove term counts)  │
  │  2. VADER (sentence-level rule-based NLP)    │
  │  3. FinBERT (ProsusAI/finbert transformer)   │
  └──────────────────────────────────────────────┘
        ↓
  Policy Stance Classification
   - Hawkish / Neutral / Dovish per meeting
        ↓
  Market Event Windows (day-0, +3 day returns)
   - SPY (equities), BTC, TLT (bonds), GLD
        ↓
  Correlation Analysis + Statistical Testing
        ↓
  Interactive Dashboard
```

## Key Findings

1. **Sentiment surprises matter more than levels** — The change in hawkishness vs last meeting is more predictive than the absolute score
2. **Hawkish surprises** → SPY mean day-0 return ≈ -0.8%; **Dovish surprises** → +0.6%
3. **BTC is 3x more volatile than SPY on FOMC days** — highly sensitive to rate policy
4. **TLT has the strongest (and most consistent) correlation** with hawk sentiment score
5. **Uncertainty language predicts volatility** — elevated uncertainty terms → higher realised vol in following days
6. The 2022 tightening cycle showed the most hawkish language in the 10-year dataset

## Project Structure

```
nlp_sentiment_pipeline/
├── notebooks/
│   └── fomc_sentiment_analysis.ipynb    # Full analysis
├── src/
│   ├── data_collection.py    # Fed statement scraper + yfinance market data
│   ├── preprocessor.py       # Text cleaning + keyword feature extraction
│   ├── sentiment_analyzer.py # VADER + FinBERT scoring
│   └── market_correlation.py # Event study + statistical tests + Plotly charts
├── data/                     # Auto-populated on first run (cached parquet)
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt

# For FinBERT (optional — requires ~500MB download):
# The notebook runs with VADER only by default (use_finbert=False)
# Set use_finbert=True for transformer-based scores

jupyter lab notebooks/fomc_sentiment_analysis.ipynb
```

Data is scraped from `federalreserve.gov` on first run and cached. Market data fetched via yfinance.

## Sentiment Features Generated

| Feature | Description |
|---------|-------------|
| `net_hawk_score` | Normalised hawkish − dovish keyword balance |
| `hawk_norm` | Hawkish terms per 100 words |
| `dove_norm` | Dovish terms per 100 words |
| `uncertainty_count` | Count of uncertainty/data-dependence language |
| `vader_mean` | Mean VADER compound score across sentences |
| `vader_pct_positive` | % sentences with positive VADER score |
| `stance_keyword` | Hawkish / Neutral / Dovish classification |
| `finbert_positive` | FinBERT positive probability (if enabled) |

## Use Cases

| Role | Application |
|------|-------------|
| Fixed Income PM | Systematic bond positioning around FOMC |
| Crypto Trader | Pre-meeting risk adjustment for BTC exposure |
| Risk Manager | Volatility scaling using uncertainty scores |
| Macro Analyst | Auto-summarised policy stance at time of release |
