"""
Crypto Market Cycle Analysis — Data Pipeline
Fetches OHLCV data for major crypto assets via yfinance.
"""

import yfinance as yf
import pandas as pd
from pathlib import Path

ASSETS = {
    "BTC": "BTC-USD",
    "ETH": "ETH-USD",
    "SOL": "SOL-USD",
    "BNB": "BNB-USD",
}

TRADITIONAL = {
    "SPY": "SPY",        # S&P 500
    "GLD": "GLD",        # Gold
    "DXY": "DX-Y.NYB",  # Dollar Index
}

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def fetch_asset(ticker: str, start: str = "2018-01-01", end: str = "2024-12-31") -> pd.DataFrame:
    """Download daily OHLCV for a single ticker."""
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    df.columns = [c.lower() for c in df.columns]
    df.index.name = "date"
    return df


def fetch_all(start: str = "2018-01-01", end: str = "2024-12-31") -> dict[str, pd.DataFrame]:
    """Fetch crypto + traditional assets, cache to parquet."""
    DATA_DIR.mkdir(exist_ok=True)
    all_assets = {**ASSETS, **TRADITIONAL}
    results = {}
    for name, ticker in all_assets.items():
        cache_path = DATA_DIR / f"{name}.parquet"
        if cache_path.exists():
            results[name] = pd.read_parquet(cache_path)
            print(f"[cache] {name}")
        else:
            print(f"[fetch] {name} ({ticker})")
            df = fetch_asset(ticker, start, end)
            df.to_parquet(cache_path)
            results[name] = df
    return results


def build_price_matrix(data: dict[str, pd.DataFrame], column: str = "close") -> pd.DataFrame:
    """Combine all close prices into a single aligned DataFrame."""
    frames = {name: df[column] for name, df in data.items() if column in df.columns}
    matrix = pd.DataFrame(frames).dropna(how="all")
    return matrix


def build_returns_matrix(price_matrix: pd.DataFrame) -> pd.DataFrame:
    """Compute daily log returns."""
    return price_matrix.pct_change().dropna()


if __name__ == "__main__":
    data = fetch_all()
    prices = build_price_matrix(data)
    print(prices.tail())
