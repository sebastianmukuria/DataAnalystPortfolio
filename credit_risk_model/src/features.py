"""
Credit Risk Scoring — Feature Engineering
Creates behavioural and financial ratio features from raw credit data.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

BILL_COLS = ["bill_sep", "bill_aug", "bill_jul", "bill_jun", "bill_may", "bill_apr"]
PAID_COLS = ["paid_sep", "paid_aug", "paid_jul", "paid_jun", "paid_may", "paid_apr"]
PAY_STAT_COLS = ["pay_sep", "pay_aug", "pay_jul", "pay_jun", "pay_may", "pay_apr"]
MONTHS = ["sep", "aug", "jul", "jun", "may", "apr"]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer features grouped into three categories:
    1. Repayment behaviour  — how often/late a customer pays
    2. Utilisation ratio    — how much of credit limit is used
    3. Payment-to-bill      — actual payment as fraction of bill
    """
    out = df.copy()

    # ── 1. Repayment behaviour ───────────────────────────────────────────────
    # Number of months with delayed payment (pay status > 0)
    out["n_late_payments"] = (df[PAY_STAT_COLS] > 0).sum(axis=1)

    # Maximum delay across all months
    out["max_delay"] = df[PAY_STAT_COLS].max(axis=1)

    # Average delay (including duly paid months)
    out["avg_delay"] = df[PAY_STAT_COLS].mean(axis=1)

    # Did customer pay duly (on time) in most recent month?
    out["paid_duly_last"] = (df["pay_sep"] <= 0).astype(int)

    # Trend in payment behaviour: is delay getting worse?
    out["delay_trend"] = df["pay_sep"] - df["pay_apr"]

    # ── 2. Utilisation ratio ─────────────────────────────────────────────────
    for month, bill_col in zip(MONTHS, BILL_COLS):
        out[f"util_{month}"] = (df[bill_col] / df["credit_limit"].replace(0, np.nan)).clip(0, 2)

    out["avg_utilisation"] = out[[f"util_{m}" for m in MONTHS]].mean(axis=1)
    out["max_utilisation"] = out[[f"util_{m}" for m in MONTHS]].max(axis=1)

    # ── 3. Payment-to-bill ratio ─────────────────────────────────────────────
    for month, paid_col, bill_col in zip(MONTHS, PAID_COLS, BILL_COLS):
        bill = df[bill_col].replace(0, np.nan)
        out[f"pay_ratio_{month}"] = (df[paid_col] / bill).clip(0, 2).fillna(0)

    out["avg_pay_ratio"] = out[[f"pay_ratio_{m}" for m in MONTHS]].mean(axis=1)

    # ── 4. Financial magnitudes ──────────────────────────────────────────────
    out["avg_bill_amt"] = df[BILL_COLS].mean(axis=1)
    out["avg_paid_amt"] = df[PAID_COLS].mean(axis=1)
    out["total_paid"] = df[PAID_COLS].sum(axis=1)
    out["total_billed"] = df[BILL_COLS].sum(axis=1)

    # Trend in bill amounts (increasing debt?)
    out["bill_trend"] = df["bill_sep"] - df["bill_apr"]

    # Log credit limit (right-skewed)
    out["log_credit_limit"] = np.log1p(df["credit_limit"])

    # Age bins
    out["age_group"] = pd.cut(
        df["age"],
        bins=[0, 25, 35, 50, 200],
        labels=["<25", "25-35", "35-50", "50+"],
    ).astype(str)

    # Categorical encoding
    out = pd.get_dummies(out, columns=["age_group"], drop_first=False)

    return out


TARGET = "default"

FEATURE_COLS = [
    # Behavioural
    "n_late_payments", "max_delay", "avg_delay",
    "paid_duly_last", "delay_trend",
    # Utilisation
    "avg_utilisation", "max_utilisation",
    "util_sep", "util_aug", "util_jul",
    # Payment ratios
    "avg_pay_ratio", "pay_ratio_sep", "pay_ratio_aug",
    # Financial magnitudes
    "avg_bill_amt", "avg_paid_amt", "total_paid", "total_billed",
    "bill_trend", "log_credit_limit",
    # Demographics
    "sex", "education", "marriage", "age",
    # Raw payment status (last 3 months)
    "pay_sep", "pay_aug", "pay_jul",
    # Age group dummies
    "age_group_<25", "age_group_25-35", "age_group_35-50", "age_group_50+",
]


def prepare_splits(df_engineered: pd.DataFrame, test_size: float = 0.2, seed: int = 42):
    """
    Returns X_train, X_test, y_train, y_test with features scaled.
    Also returns the fitted scaler for inference.
    """
    available = [c for c in FEATURE_COLS if c in df_engineered.columns]
    X = df_engineered[available]
    y = df_engineered[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=X_test.columns, index=X_test.index
    )

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler
