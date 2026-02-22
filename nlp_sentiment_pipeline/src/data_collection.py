"""
FOMC NLP Sentiment Pipeline — Data Collection
Fetches Federal Reserve FOMC meeting statement texts and S&P 500 / BTC prices.

Business Problem:
  Investment managers and macro traders need to rapidly decode the Fed's
  policy stance (hawkish vs dovish) from meeting statements to anticipate
  rate decisions and position risk assets accordingly.

Data sources:
  - FOMC statements: federalreserve.gov (public, freely available)
  - Market data: yfinance (SPY, BTC-USD, TLT)
"""

import re
import time
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf
from bs4 import BeautifulSoup

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

FOMC_CALENDAR_URL = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
FOMC_HISTORICAL_URL = "https://www.federalreserve.gov/monetarypolicy/fomc_historical_year.htm"
BASE_URL = "https://www.federalreserve.gov"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (research bot; data science portfolio project)"
}


# ─── FOMC Statement Scraping ──────────────────────────────────────────────────

def get_statement_links_from_page(url: str) -> list[dict]:
    """Parse a Fed calendar/historical page and extract statement links + dates."""
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        text = a.get_text(strip=True).lower()
        if "press release" in text or "statement" in text:
            if "/newsevents/pressreleases/monetary" in href:
                full_url = BASE_URL + href if href.startswith("/") else href
                links.append({"url": full_url, "link_text": text})
    return links


def extract_statement_text(url: str) -> str:
    """Download a single FOMC press release page and extract the statement body."""
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    # Fed press releases wrap content in <div class="col-xs-12 col-sm-8 col-md-8">
    content_div = (
        soup.find("div", {"class": re.compile(r"col-xs-12.*col-sm-8")})
        or soup.find("div", {"id": "article"})
        or soup.find("div", {"class": "release-info"})
    )
    if content_div:
        return content_div.get_text(separator=" ", strip=True)
    return soup.get_text(separator=" ", strip=True)[:5000]


def build_fomc_historical_urls(start_year: int = 2015, end_year: int = 2024) -> list[str]:
    """Generate per-year FOMC historical page URLs."""
    return [
        f"https://www.federalreserve.gov/monetarypolicy/fomchistorical{year}.htm"
        for year in range(start_year, end_year + 1)
    ]


def scrape_fomc_statements(
    start_year: int = 2015,
    end_year: int = 2024,
    delay: float = 1.0,
) -> pd.DataFrame:
    """
    Scrape FOMC statement links from all historical year pages
    and download the text. Returns a DataFrame with columns:
      date, year, url, text
    """
    cache_path = DATA_DIR / "fomc_statements.parquet"
    if cache_path.exists():
        print("[cache] Loading FOMC statements from parquet")
        return pd.read_parquet(cache_path)

    DATA_DIR.mkdir(exist_ok=True)
    all_links = []

    for year in range(start_year, end_year + 1):
        url = f"https://www.federalreserve.gov/monetarypolicy/fomchistorical{year}.htm"
        print(f"Fetching index for {year}...")
        try:
            links = get_statement_links_from_page(url)
            for lnk in links:
                lnk["year"] = year
            all_links.extend(links)
            time.sleep(delay)
        except Exception as e:
            print(f"  Warning: {year} failed — {e}")

    records = []
    for i, lnk in enumerate(all_links):
        print(f"  [{i+1}/{len(all_links)}] {lnk['url']}")
        try:
            text = extract_statement_text(lnk["url"])
            # Extract date from URL pattern e.g. monetary20230201a.htm
            date_match = re.search(r"monetary(\d{8})", lnk["url"])
            date = pd.to_datetime(date_match.group(1), format="%Y%m%d") if date_match else None
            records.append({"date": date, "year": lnk["year"], "url": lnk["url"], "text": text})
            time.sleep(delay)
        except Exception as e:
            print(f"    Warning: failed — {e}")

    df = pd.DataFrame(records).dropna(subset=["text"])
    df = df.sort_values("date").reset_index(drop=True)
    df.to_parquet(cache_path, index=False)
    print(f"Saved {len(df)} FOMC statements to {cache_path}")
    return df


# ─── Market Data ──────────────────────────────────────────────────────────────

MARKET_TICKERS = {
    "SPY": "SPY",         # S&P 500 ETF
    "BTC": "BTC-USD",     # Bitcoin
    "TLT": "TLT",         # 20+ Year Treasury Bond ETF
    "GLD": "GLD",         # Gold
}


def fetch_market_data(start: str = "2015-01-01", end: str = "2025-01-01") -> pd.DataFrame:
    """Fetch daily close prices for market assets around FOMC dates."""
    cache_path = DATA_DIR / "market_prices.parquet"
    if cache_path.exists():
        print("[cache] Loading market data from parquet")
        return pd.read_parquet(cache_path)

    frames = {}
    for name, ticker in MARKET_TICKERS.items():
        df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
        frames[name] = df["Close"].squeeze()

    prices = pd.DataFrame(frames)
    prices.index.name = "date"
    prices.to_parquet(cache_path)
    print(f"Saved market data: {prices.shape}")
    return prices


def build_fomc_market_events(
    fomc_df: pd.DataFrame,
    market_df: pd.DataFrame,
    window_days: int = 3,
) -> pd.DataFrame:
    """
    For each FOMC statement date, compute:
    - Market return on day 0 (announcement day)
    - Cumulative return over [0, +window_days]
    - Pre-meeting return [-window_days, -1] (expectations)
    """
    rows = []
    market_df = market_df.copy()
    market_df.index = pd.to_datetime(market_df.index)

    for _, row in fomc_df.iterrows():
        date = pd.to_datetime(row["date"])
        if pd.isna(date):
            continue

        loc = market_df.index.searchsorted(date)
        if loc >= len(market_df) or loc < window_days:
            continue

        event = {"date": date}
        for asset in market_df.columns:
            prices = market_df[asset]
            try:
                day0 = prices.iloc[loc]
                day_prev = prices.iloc[loc - 1]
                day_fwd = prices.iloc[min(loc + window_days, len(prices) - 1)]
                day_pre = prices.iloc[max(loc - window_days, 0)]

                event[f"{asset}_day0_ret"] = (day0 - day_prev) / day_prev * 100
                event[f"{asset}_fwd_ret"] = (day_fwd - day0) / day0 * 100
                event[f"{asset}_pre_ret"] = (day0 - day_pre) / day_pre * 100
            except Exception:
                pass
        rows.append(event)

    return pd.DataFrame(rows)


if __name__ == "__main__":
    fomc = scrape_fomc_statements(2019, 2024)
    print(fomc.head())
    market = fetch_market_data()
    print(market.tail())
