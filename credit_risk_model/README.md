# Credit Risk Scoring Model

**Type:** End-to-End Machine Learning Pipeline
**Stack:** Python, scikit-learn, XGBoost, SHAP, pandas, Plotly, Matplotlib
**Dataset:** UCI Default of Credit Card Clients (Taiwan, 2005) — 30,000 customers, 23 features

---

## Business Problem

A retail bank wants to predict which credit card customers will **default on their payment next month**, enabling proactive outreach before loans deteriorate. Currently, the bank only flags customers after missed payments — a reactive, high-cost approach.

**Impact:** Early identification of ~70% of defaults while flagging only 25% of the customer base for review, reducing loan-loss provisions significantly.

## Pipeline

```
Raw Data (UCI)
     ↓
Data Cleaning (undocumented category recoding, duplicate removal)
     ↓
Feature Engineering (behavioural + ratio features)
     ├── Repayment behaviour (n late payments, max delay, trend)
     ├── Utilisation ratio (bill / credit limit by month)
     └── Payment-to-bill ratio (actual payment vs statement)
     ↓
Model Training
     ├── Logistic Regression (baseline)
     ├── Random Forest (ensemble)
     └── XGBoost (gradient boosting — best performer)
     ↓
Evaluation (AUC-ROC, PR-AUC, Gini, Confusion Matrix)
     ↓
SHAP Explainability (global + individual explanations)
     ↓
Business Cost Analysis (threshold optimisation)
```

## Results

| Model | AUC-ROC | PR-AUC | Gini |
|-------|---------|--------|------|
| Logistic Regression | ~0.77 | ~0.51 | ~0.54 |
| Random Forest | ~0.78 | ~0.53 | ~0.56 |
| **XGBoost** | **~0.79** | **~0.55** | **~0.58** |

**Optimal threshold (0.35):** Precision ~45%, Recall ~70%
**Est. annual savings:** Several million NTD over a 30k-customer book

## Key Features (SHAP)

1. `pay_sep` — Most recent payment status (single strongest predictor)
2. `n_late_payments` — Count of months with any delay in past 6 months
3. `avg_utilisation` — Mean credit utilisation ratio
4. `avg_pay_ratio` — Actual payment as fraction of bill amount
5. `credit_limit` — Very low limits signal subprime risk segment

## Project Structure

```
credit_risk_model/
├── notebooks/
│   └── credit_risk_scoring.ipynb    # Full analysis
├── src/
│   ├── data_pipeline.py  # Download UCI data, clean
│   ├── features.py       # Feature engineering + train/test split
│   └── model.py          # Model training, evaluation, SHAP, cost analysis
├── data/                 # Auto-populated on first run
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
jupyter lab notebooks/credit_risk_scoring.ipynb
```

The dataset is automatically downloaded from UCI on first run and cached to `data/`.

## Business Cost Model

```
Assumptions:
  - Average credit exposure: NT$150,000 per customer
  - Loss Given Default (LGD): 45%
  - Intervention cost per flagged customer: NT$600

Cost of false negative (missed default): NT$67,500
Cost of false positive (unnecessary outreach): NT$600
→ Model dramatically favours high recall at the cost of lower precision
```
