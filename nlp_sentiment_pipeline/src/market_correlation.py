"""
FOMC NLP Sentiment Pipeline — Market Correlation Analysis
Tests whether FOMC statement sentiment predicts asset price moves.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats


# ─── Merge & Align ───────────────────────────────────────────────────────────

def merge_sentiment_market(
    sentiment_df: pd.DataFrame,
    market_events_df: pd.DataFrame,
    on: str = "date",
) -> pd.DataFrame:
    """Merge sentiment scores with market reaction data on FOMC dates."""
    s = sentiment_df.copy()
    s["date"] = pd.to_datetime(s["date"]).dt.normalize()

    m = market_events_df.copy()
    m["date"] = pd.to_datetime(m["date"]).dt.normalize()

    merged = pd.merge(s, m, on="date", how="inner")
    return merged


# ─── Statistical Tests ────────────────────────────────────────────────────────

def correlation_report(df: pd.DataFrame, sentiment_col: str, market_col: str) -> dict:
    """Pearson + Spearman correlations with p-values."""
    valid = df[[sentiment_col, market_col]].dropna()
    if len(valid) < 5:
        return {}

    pearson_r, pearson_p = stats.pearsonr(valid[sentiment_col], valid[market_col])
    spearman_r, spearman_p = stats.spearmanr(valid[sentiment_col], valid[market_col])

    return {
        "sentiment_col": sentiment_col,
        "market_col": market_col,
        "n": len(valid),
        "pearson_r": round(pearson_r, 4),
        "pearson_p": round(pearson_p, 4),
        "spearman_r": round(spearman_r, 4),
        "spearman_p": round(spearman_p, 4),
        "significant_5pct": pearson_p < 0.05 or spearman_p < 0.05,
    }


def stance_group_returns(df: pd.DataFrame, stance_col: str, return_col: str) -> pd.DataFrame:
    """Compare mean returns by policy stance (Hawkish / Neutral / Dovish)."""
    grouped = (
        df.groupby(stance_col)[return_col]
        .agg(["mean", "std", "count"])
        .round(3)
        .reset_index()
    )
    grouped.columns = ["stance", "mean_return", "std_return", "n"]
    return grouped


def run_correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute correlations between all sentiment signals and all market returns.
    Returns a summary DataFrame sorted by absolute Pearson r.
    """
    sentiment_cols = [
        c for c in df.columns
        if c in ["net_hawk_score", "vader_mean", "hawk_norm", "dove_norm",
                 "uncertainty_count", "finbert_positive", "finbert_negative"]
    ]
    market_cols = [
        c for c in df.columns
        if "_day0_ret" in c or "_fwd_ret" in c or "_pre_ret" in c
    ]

    rows = []
    for sc in sentiment_cols:
        for mc in market_cols:
            report = correlation_report(df, sc, mc)
            if report:
                rows.append(report)

    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values("pearson_r", key=abs, ascending=False)
    return result


# ─── Visualisations ───────────────────────────────────────────────────────────

def plot_sentiment_over_time(df: pd.DataFrame) -> go.Figure:
    """Plot FOMC hawkish score and VADER sentiment over time with regime shading."""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=("Net Hawkish Score (Keyword Model)", "VADER Sentiment"))

    # Hawkish/Dovish score
    colors = [
        "#EF4444" if v > 0 else "#3B82F6"
        for v in df["net_hawk_score"].fillna(0)
    ]
    fig.add_trace(go.Bar(
        x=df["date"], y=df["net_hawk_score"],
        marker_color=colors, name="Net Hawk Score",
    ), row=1, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.5, row=1, col=1)

    # VADER
    if "vader_mean" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["date"], y=df["vader_mean"],
            mode="lines+markers", line=dict(color="#10B981", width=2),
            marker=dict(size=6), name="VADER",
        ), row=2, col=1)
        fig.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.5, row=2, col=1)

    fig.update_layout(
        template="plotly_dark", height=600,
        title="FOMC Statement Sentiment — 2015–2024",
        showlegend=True,
    )
    return fig


def plot_stance_market_response(
    df: pd.DataFrame,
    stance_col: str = "stance_keyword",
    return_col: str = "SPY_day0_ret",
    asset: str = "SPY",
) -> go.Figure:
    """Box plot: market return on FOMC day by policy stance."""
    color_map = {"Hawkish": "#EF4444", "Neutral": "#6B7280", "Dovish": "#3B82F6"}

    fig = px.box(
        df.dropna(subset=[stance_col, return_col]),
        x=stance_col, y=return_col,
        color=stance_col,
        color_discrete_map=color_map,
        points="all",
        title=f"{asset} Day-0 Return by FOMC Stance",
        template="plotly_dark",
        category_orders={stance_col: ["Hawkish", "Neutral", "Dovish"]},
    )
    fig.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.5)
    fig.update_layout(height=450, showlegend=False)
    return fig


def plot_scatter_sentiment_return(
    df: pd.DataFrame,
    x_col: str = "net_hawk_score",
    y_col: str = "SPY_day0_ret",
    color_col: str = "stance_keyword",
) -> go.Figure:
    """Scatter: sentiment score vs asset return with regression line."""
    valid = df[[x_col, y_col, color_col, "date"]].dropna()
    color_map = {"Hawkish": "#EF4444", "Neutral": "#6B7280", "Dovish": "#3B82F6"}

    fig = px.scatter(
        valid, x=x_col, y=y_col,
        color=color_col, color_discrete_map=color_map,
        hover_data=["date"],
        trendline="ols",
        title=f"FOMC Sentiment vs {y_col.replace('_', ' ').title()}",
        template="plotly_dark",
    )
    fig.update_layout(height=450)
    return fig


def plot_btc_vs_spy_response(df: pd.DataFrame) -> go.Figure:
    """Compare BTC and SPY day-0 reactions across FOMC meetings."""
    fig = go.Figure()

    for asset, color in [("SPY", "#3B82F6"), ("BTC", "#F59E0B")]:
        col = f"{asset}_day0_ret"
        if col in df.columns:
            fig.add_trace(go.Bar(
                x=df["date"], y=df[col],
                name=asset, marker_color=color, opacity=0.8,
            ))

    fig.update_layout(
        template="plotly_dark",
        barmode="group",
        title="FOMC Day-0 Market Reactions: BTC vs SPY",
        yaxis_title="Return (%)",
        height=450,
    )
    return fig
