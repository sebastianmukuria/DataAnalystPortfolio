# 📊 Sebastian Mukuria — Data Analyst Portfolio

## 👋 About Me
Hi, I’m Sebastian! I’ve worked as a **research analyst in the crypto industry**, where I built automation tools, wrote SQL queries, and led projects to improve data workflows in addition to blockchain research focusing on layer-1 and layer-2 networks. I also have experience as a **business data analyst in the defense sector** and co-founded a recycling startup focused on sustainability.

I earned my **B.S. in Mechanical Engineering (UC Riverside)** and I’m now pursuing an **A.A. in Data Science**.

## 🎯 Current Focus
- Building end-to-end data pipelines and ML models
- NLP and financial text analysis
- Interactive dashboards and data visualisation
- Refreshing mathematics for Machine Learning and Data Science

---

## 🚀 Projects

### 🔬 Advanced Projects

#### 📈 [Crypto Market Cycle Analysis](./crypto_market_analysis/)
> **EDA + Technical Analysis + Interactive Dashboard**

A systematic framework to identify which phase of the crypto market cycle (Accumulation → Markup → Distribution → Markdown) each major asset is in, using technical indicators and a Wyckoff-inspired classification algorithm.

- **Stack:** Python, yfinance, Plotly, pandas
- **Covers:** BTC, ETH, SOL, BNB vs SPY, GLD (2018–2024)
- **Key Result:** Markup phase delivers 2.3x better risk-adjusted returns than holding through all phases
- **Signals:** RSI, MACD, Bollinger Bands, 50/200 SMA, OBV, drawdown analysis

---

#### 💳 [Credit Risk Scoring Model](./credit_risk_model/)
> **End-to-End ML Pipeline — Classification**

A full machine learning pipeline predicting credit card default probability for a retail bank, enabling proactive customer intervention 30 days before default.

- **Stack:** Python, XGBoost, scikit-learn, SHAP, pandas
- **Dataset:** UCI Default of Credit Card Clients (30,000 customers)
- **Key Result:** XGBoost achieves AUC-ROC ~0.79, Gini ~0.58 | Catches ~70% of defaults while flagging only ~25% of customers
- **Features:** Behavioural engineering — late payment count, utilisation ratio, payment-to-bill ratio, delay trend

---

#### 🏦 [FOMC NLP Sentiment Pipeline](./nlp_sentiment_pipeline/)
> **End-to-End NLP Pipeline + Market Correlation Study**

An automated pipeline that ingests Federal Reserve meeting statements, quantifies hawkish/dovish sentiment via VADER and FinBERT, and correlates the signals with asset price reactions across equities, bonds, and Bitcoin.

- **Stack:** Python, BeautifulSoup, VADER, FinBERT (HuggingFace), yfinance, Plotly
- **Dataset:** FOMC statements 2015–2024 (scraped from federalreserve.gov)
- **Key Result:** Sentiment surprises predict SPY direction on FOMC day | BTC reacts 3x more violently than equities
- **Models:** Keyword scoring, VADER rule-based NLP, FinBERT transformer

---

### 🐍 Foundation Projects

#### ☕ [Cafe Sales Analysis](./PandasProjects/Cafe_Sales_Analysis/)
Data cleaning and EDA exercise on a messy real-world cafe sales dataset. Demonstrates handling of missing values, outliers, and category normalisation.

#### 💾 [SQL Tech Layoffs Analysis](./SQL%20Projects/Tech%20Layoffs%20Project/)
SQL queries exploring the 2022–2023 tech layoff wave — trends by company, sector, geography, and time period.

#### 🤖 [Python Automation Bot](https://github.com/sebastianmukuria/ReleaseSummarizerBot)
Slack + LLM integration to auto-summarise crypto protocol release notes, reducing manual research workload.

---

## 🛠 Technical Skills

| Category | Tools |
|----------|-------|
| **Languages** | Python, SQL |
| **Data** | pandas, NumPy, yfinance, BeautifulSoup |
| **ML** | scikit-learn, XGBoost, SHAP, imbalanced-learn |
| **NLP** | VADER, FinBERT (HuggingFace transformers) |
| **Visualisation** | Plotly, Matplotlib, seaborn |
| **Storage** | Parquet, SQLite |
| **Dev** | Git, Jupyter, VS Code |

## 🌱 Goal
This repo tracks my progression from analyst to data scientist. Long-term target: **machine learning engineering** — building production ML systems, not just notebooks.
