"""
Crypto Market Cycle Analysis — Technical Indicators
Computes RSI, MACD, Bollinger Bands, Moving Averages, and cycle phase labels.
"""

import numpy as np
import pandas as pd


# ─── Moving Averages ─────────────────────────────────────────────────────────

def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).mean()


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


# ─── RSI ──────────────────────────────────────────────────────────────────────

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


# ─── MACD ─────────────────────────────────────────────────────────────────────

def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return pd.DataFrame({
        "macd": macd_line,
        "signal": signal_line,
        "histogram": histogram,
    })


# ─── Bollinger Bands ──────────────────────────────────────────────────────────

def bollinger_bands(series: pd.Series, window: int = 20, num_std: float = 2.0):
    mid = sma(series, window)
    std = series.rolling(window).std()
    return pd.DataFrame({
        "bb_upper": mid + num_std * std,
        "bb_mid": mid,
        "bb_lower": mid - num_std * std,
        "bb_width": (2 * num_std * std) / mid,         # band width as % of price
        "bb_pct": (series - (mid - num_std * std)) / (2 * num_std * std),  # %B
    })


# ─── Volume Indicators ────────────────────────────────────────────────────────

def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-Balance Volume."""
    direction = np.sign(close.diff()).fillna(0)
    return (direction * volume).cumsum()


def volume_sma(volume: pd.Series, window: int = 20) -> pd.Series:
    return volume.rolling(window).mean()


# ─── Market Cycle Phase ───────────────────────────────────────────────────────

def cycle_phase(close: pd.Series) -> pd.Series:
    """
    Classify each day into one of four Wyckoff-inspired market cycle phases
    based on price position relative to 50/200 SMAs and RSI.

    Phases:
      - Accumulation : price below 200 SMA, RSI recovering (30–50)
      - Markup       : price above 200 SMA, golden cross, RSI 50–70
      - Distribution : price above 200 SMA, RSI overbought (>70), momentum slowing
      - Markdown     : price below 200 SMA, RSI weak (<40)
    """
    ma50 = sma(close, 50)
    ma200 = sma(close, 200)
    rsi_vals = rsi(close, 14)

    above_200 = close > ma200
    golden_cross = ma50 > ma200

    conditions = [
        (~above_200) & (rsi_vals >= 30) & (rsi_vals <= 50),   # Accumulation
        above_200 & golden_cross & (rsi_vals >= 50) & (rsi_vals <= 70),  # Markup
        above_200 & (rsi_vals > 70),                            # Distribution
        (~above_200) & (rsi_vals < 40),                         # Markdown
    ]
    labels = ["Accumulation", "Markup", "Distribution", "Markdown"]

    phase = pd.Series("Transition", index=close.index)
    for condition, label in zip(conditions, labels):
        phase = phase.where(~condition, label)

    return phase


# ─── Full Indicator Suite ─────────────────────────────────────────────────────

def compute_all(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a OHLCV DataFrame, return it enriched with all indicators.
    Expects columns: open, high, low, close, volume
    """
    out = df.copy()

    out["sma_50"] = sma(df["close"], 50)
    out["sma_200"] = sma(df["close"], 200)
    out["ema_21"] = ema(df["close"], 21)

    out["rsi"] = rsi(df["close"])

    macd_df = macd(df["close"])
    out = pd.concat([out, macd_df], axis=1)

    bb_df = bollinger_bands(df["close"])
    out = pd.concat([out, bb_df], axis=1)

    if "volume" in df.columns:
        out["obv"] = obv(df["close"], df["volume"])
        out["vol_sma20"] = volume_sma(df["volume"])
        out["vol_ratio"] = df["volume"] / out["vol_sma20"]

    out["drawdown"] = (df["close"] / df["close"].cummax() - 1) * 100
    out["daily_return"] = df["close"].pct_change() * 100
    out["cycle_phase"] = cycle_phase(df["close"])

    return out
