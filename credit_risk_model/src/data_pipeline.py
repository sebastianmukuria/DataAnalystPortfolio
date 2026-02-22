"""
Credit Risk Scoring — Data Pipeline
Downloads the UCI Default of Credit Card Clients dataset.

Business Problem:
  A bank wants to predict which customers are likely to default on their
  credit card payment next month, allowing proactive intervention to reduce
  loan-loss provisions and improve capital allocation.

Dataset:
  UCI ML Repository — Default of Credit Card Clients (Taiwan, 2005)
  30,000 customers | 23 features | Binary target: default next month
  https://archive.ics.uci.edu/dataset/350
"""

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_PATH = DATA_DIR / "credit_default_raw.parquet"

UCI_URL = (
    "https://archive.ics.uci.edu/static/public/350/"
    "default+of+credit+card+clients.zip"
)

COLUMN_MAP = {
    "ID": "id",
    "LIMIT_BAL": "credit_limit",
    "SEX": "sex",
    "EDUCATION": "education",
    "MARRIAGE": "marriage",
    "AGE": "age",
    "PAY_0": "pay_sep",
    "PAY_2": "pay_aug",
    "PAY_3": "pay_jul",
    "PAY_4": "pay_jun",
    "PAY_5": "pay_may",
    "PAY_6": "pay_apr",
    "BILL_AMT1": "bill_sep",
    "BILL_AMT2": "bill_aug",
    "BILL_AMT3": "bill_jul",
    "BILL_AMT4": "bill_jun",
    "BILL_AMT5": "bill_may",
    "BILL_AMT6": "bill_apr",
    "PAY_AMT1": "paid_sep",
    "PAY_AMT2": "paid_aug",
    "PAY_AMT3": "paid_jul",
    "PAY_AMT4": "paid_jun",
    "PAY_AMT5": "paid_may",
    "PAY_AMT6": "paid_apr",
    "default.payment.next.month": "default",
}


def download_and_parse() -> pd.DataFrame:
    """Download UCI zip, parse Excel inside, return clean DataFrame."""
    print("Downloading UCI Credit Default dataset...")
    response = requests.get(UCI_URL, timeout=60)
    response.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        xls_name = [n for n in z.namelist() if n.endswith(".xls")][0]
        with z.open(xls_name) as f:
            df = pd.read_excel(f, header=1)

    df = df.rename(columns=COLUMN_MAP)
    df = df.drop(columns=["id"], errors="ignore")
    return df


def load_raw() -> pd.DataFrame:
    """Load from cache or download fresh."""
    DATA_DIR.mkdir(exist_ok=True)
    if RAW_PATH.exists():
        print("[cache] Loading raw credit data from parquet")
        return pd.read_parquet(RAW_PATH)

    df = download_and_parse()
    df.to_parquet(RAW_PATH, index=False)
    print(f"Saved {len(df):,} rows to {RAW_PATH}")
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply domain-driven cleaning:
    - Recode education/marriage undocumented categories to 'other'
    - Clip payment status to documented range [-2, 9]
    - Remove duplicate rows
    """
    df = df.copy()

    # Education: categories 0, 5, 6 are undocumented → recode to 4 (other)
    df["education"] = df["education"].replace({0: 4, 5: 4, 6: 4})

    # Marriage: category 0 undocumented → recode to 3 (other)
    df["marriage"] = df["marriage"].replace({0: 3})

    # Payment status documented range
    pay_cols = [c for c in df.columns if c.startswith("pay_") and not c.startswith("paid_")]
    for col in pay_cols:
        df[col] = df[col].clip(-2, 9)

    df = df.drop_duplicates()
    return df


if __name__ == "__main__":
    raw = load_raw()
    cleaned = clean(raw)
    print(f"Shape: {cleaned.shape}")
    print(cleaned["default"].value_counts(normalize=True).round(3))
