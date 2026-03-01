"""
Crypto Market Cycle Analysis — Visualizations
Plotly-based interactive charts for the dashboard.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

PHASE_COLORS = {
    "Accumulation": "#3B82F6",   # blue
    "Markup":       "#10B981",   # green
    "Distribution": "#F59E0B",   # amber
    "Markdown":     "#EF4444",   # red
    "Transition":   "#6B7280",   # gray
}


def candlestick_with_indicators(df: pd.DataFrame, asset: str = "BTC") -> go.Figure:
    """
    Multi-panel chart:
    Panel 1 — Candlestick + SMA50/200 + Bollinger Bands, colored by cycle phase
    Panel 2 — Volume w/ 20-day avg
    Panel 3 — RSI with overbought/oversold lines
    Panel 4 — MACD histogram + signal
    """
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        row_heights=[0.5, 0.15, 0.175, 0.175],
        vertical_spacing=0.03,
        subplot_titles=(
            f"{asset} Price + Market Cycle Phases",
            "Volume",
            "RSI (14)",
            "MACD (12, 26, 9)",
        ),
    )

    # ── Panel 1: Price ──────────────────────────────────────────────────────
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["open"], high=df["high"],
        low=df["low"], close=df["close"],
        name="Price", increasing_line_color="#10B981",
        decreasing_line_color="#EF4444", showlegend=False,
    ), row=1, col=1)

    for ma, color, dash in [("sma_50", "#F59E0B", "dot"), ("sma_200", "#818CF8", "solid")]:
        if ma in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df[ma], name=ma.upper().replace("_", " "),
                line=dict(color=color, width=1.5, dash=dash),
            ), row=1, col=1)

    for band, color, fill in [
        ("bb_upper", "rgba(99,102,241,0.3)", "tonexty"),
        ("bb_lower", "rgba(99,102,241,0.3)", None),
    ]:
        if band in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df[band], name=band.replace("_", " ").title(),
                line=dict(color="rgba(99,102,241,0.5)", width=1, dash="dash"),
                fill=fill, fillcolor="rgba(99,102,241,0.05)",
                showlegend=False,
            ), row=1, col=1)

    # Phase background shading
    if "cycle_phase" in df.columns:
        phase_changes = df["cycle_phase"].ne(df["cycle_phase"].shift())
        phase_starts = df.index[phase_changes].tolist()
        phase_starts.append(df.index[-1])
        for i in range(len(phase_starts) - 1):
            phase = df.loc[phase_starts[i], "cycle_phase"]
            fig.add_vrect(
                x0=phase_starts[i], x1=phase_starts[i + 1],
                fillcolor=PHASE_COLORS.get(phase, "#6B7280"),
                opacity=0.08, layer="below", line_width=0,
                row=1, col=1,
            )

    # ── Panel 2: Volume ─────────────────────────────────────────────────────
    if "volume" in df.columns:
        colors = ["#10B981" if r >= 0 else "#EF4444" for r in df["daily_return"].fillna(0)]
        fig.add_trace(go.Bar(
            x=df.index, y=df["volume"], name="Volume",
            marker_color=colors, showlegend=False,
        ), row=2, col=1)
        if "vol_sma20" in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df["vol_sma20"], name="Vol SMA 20",
                line=dict(color="#F59E0B", width=1.5),
            ), row=2, col=1)

    # ── Panel 3: RSI ─────────────────────────────────────────────────────────
    if "rsi" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["rsi"], name="RSI",
            line=dict(color="#818CF8", width=1.5),
        ), row=3, col=1)
        for level, color in [(70, "rgba(239,68,68,0.5)"), (30, "rgba(16,185,129,0.5)")]:
            fig.add_hline(y=level, line_dash="dash", line_color=color, row=3, col=1)
        fig.add_hrect(y0=30, y1=70, fillcolor="rgba(107,114,128,0.05)",
                      line_width=0, row=3, col=1)

    # ── Panel 4: MACD ─────────────────────────────────────────────────────────
    if "macd" in df.columns:
        histo_colors = ["#10B981" if v >= 0 else "#EF4444"
                        for v in df["histogram"].fillna(0)]
        fig.add_trace(go.Bar(
            x=df.index, y=df["histogram"], name="MACD Histogram",
            marker_color=histo_colors, showlegend=False,
        ), row=4, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=df["macd"], name="MACD",
            line=dict(color="#3B82F6", width=1.5),
        ), row=4, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=df["signal"], name="Signal",
            line=dict(color="#F59E0B", width=1.5),
        ), row=4, col=1)

    fig.update_layout(
        template="plotly_dark",
        height=900,
        title=f"{asset} — Market Cycle Dashboard",
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=60, r=20, t=80, b=20),
    )
    return fig


def correlation_heatmap(returns: pd.DataFrame) -> go.Figure:
    """Correlation matrix heatmap for crypto + traditional assets."""
    corr = returns.corr()
    fig = px.imshow(
        corr,
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        text_auto=".2f",
        title="Asset Return Correlations",
        template="plotly_dark",
    )
    fig.update_layout(height=500)
    return fig


def cycle_phase_distribution(df: pd.DataFrame, asset: str = "BTC") -> go.Figure:
    """Pie chart showing time spent in each market cycle phase."""
    counts = df["cycle_phase"].value_counts()
    fig = px.pie(
        values=counts.values,
        names=counts.index,
        color=counts.index,
        color_discrete_map=PHASE_COLORS,
        title=f"{asset} — Time Spent in Each Cycle Phase",
        template="plotly_dark",
        hole=0.4,
    )
    return fig


def drawdown_chart(df: pd.DataFrame, asset: str = "BTC") -> go.Figure:
    """Historical drawdown from all-time-high."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index, y=df["drawdown"],
        fill="tozeroy", fillcolor="rgba(239,68,68,0.2)",
        line=dict(color="#EF4444", width=1.5),
        name="Drawdown %",
    ))
    fig.update_layout(
        template="plotly_dark",
        title=f"{asset} — Drawdown from ATH (%)",
        yaxis_title="Drawdown (%)",
        height=350,
    )
    return fig


def returns_heatmap_by_month(df: pd.DataFrame, asset: str = "BTC") -> go.Figure:
    """Monthly returns calendar heatmap."""
    df2 = df.copy()
    df2["year"] = df2.index.year
    df2["month"] = df2.index.month
    monthly = df2.groupby(["year", "month"])["daily_return"].sum().reset_index()
    pivot = monthly.pivot(index="year", columns="month", values="daily_return")
    month_names = ["Jan","Feb","Mar","Apr","May","Jun",
                   "Jul","Aug","Sep","Oct","Nov","Dec"]
    pivot.columns = [month_names[m - 1] for m in pivot.columns]

    fig = px.imshow(
        pivot,
        color_continuous_scale="RdYlGn",
        text_auto=".1f",
        title=f"{asset} — Monthly Returns Heatmap (%)",
        template="plotly_dark",
        aspect="auto",
    )
    fig.update_layout(height=400)
    return fig


def phase_returns_boxplot(df: pd.DataFrame, asset: str = "BTC") -> go.Figure:
    """Box plot of daily returns by cycle phase."""
    fig = px.box(
        df.reset_index(),
        x="cycle_phase",
        y="daily_return",
        color="cycle_phase",
        color_discrete_map=PHASE_COLORS,
        title=f"{asset} — Daily Return Distribution by Cycle Phase",
        template="plotly_dark",
        points="outliers",
        category_orders={"cycle_phase": ["Accumulation","Markup","Distribution","Markdown","Transition"]},
    )
    fig.update_layout(height=400, showlegend=False)
    return fig
