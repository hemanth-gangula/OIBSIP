"""
visualizations.py
-----------------
ALL dashboard charts are Plotly interactive (returned as JSON strings).
matplotlib / seaborn imports are kept for Jupyter Notebook compatibility
but are NOT used for dashboard rendering.
"""

import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.utils import PlotlyJSONEncoder

# matplotlib / seaborn retained for Jupyter Notebook — NOT used by Flask routes
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns


# ---------------------------------------------------------------------------
# Shared layout helpers (NO margin here — set per-chart to avoid conflicts)
# ---------------------------------------------------------------------------

def _dark_layout(**extra) -> dict:
    """Return a dark-theme layout dict merged with any extra kwargs."""
    base = dict(
        paper_bgcolor="#0f172a",
        plot_bgcolor="#1e293b",
        font=dict(color="#f1f5f9", family="Inter, system-ui, sans-serif"),
        legend=dict(
            font=dict(color="#f1f5f9", size=11),
            bgcolor="#1e293b",
            bordercolor="#334155",
            borderwidth=1,
        ),
        hoverlabel=dict(
            bgcolor="#1e293b",
            bordercolor="#6366f1",
            font=dict(color="#f1f5f9", size=12),
        ),
    )
    base.update(extra)
    return base


_AXIS_X = dict(
    tickfont=dict(size=11, color="#94a3b8"),
    gridcolor="#334155",
    zerolinecolor="#334155",
    linecolor="#334155",
)
_AXIS_Y = dict(
    tickfont=dict(size=11, color="#94a3b8"),
    gridcolor="#334155",
    zerolinecolor="#334155",
    linecolor="#334155",
)

COLORS = [
    "#6366f1", "#8b5cf6", "#ec4899", "#f43f5e", "#f97316",
    "#eab308", "#22c55e", "#14b8a6", "#0ea5e9", "#3b82f6",
    "#a855f7", "#06b6d4", "#84cc16", "#fb923c", "#f472b6",
]


def _title(text: str) -> dict:
    return dict(text=text, font=dict(size=15, color="#f1f5f9"),
                x=0.5, xanchor="center")


def _axis_title(text: str) -> dict:
    return dict(text=text, font=dict(size=12, color="#cbd5e1"))


def _json(fig: go.Figure) -> str:
    return json.dumps(fig, cls=PlotlyJSONEncoder)


# ===========================================================================
# APPS DASHBOARD — 7 charts
# ===========================================================================

def chart_category_bar(data: dict) -> str:
    """Interactive horizontal bar — apps per category (top 20)."""
    cats = data["categories"][:20]
    counts = data["counts"][:20]
    bar_colors = (COLORS * 2)[:len(cats)]

    fig = go.Figure(go.Bar(
        x=counts[::-1],
        y=cats[::-1],
        orientation="h",
        marker=dict(color=bar_colors[::-1], line=dict(width=0)),
        hovertemplate="<b>%{y}</b><br>Apps: %{x:,}<extra></extra>",
        text=[f"{c:,}" for c in counts[::-1]],
        textposition="outside",
        textfont=dict(size=10, color="#cbd5e1"),
    ))
    fig.update_layout(
        _dark_layout(
            title=_title("App Distribution by Category (Top 20)"),
            xaxis=dict(**_AXIS_X, title=_axis_title("Number of Apps")),
            yaxis=dict(**_AXIS_Y, autorange=True),
            height=max(420, len(cats) * 26),
            margin=dict(t=56, b=40, l=180, r=80),
        )
    )
    return _json(fig)


def chart_rating_distribution(data: dict) -> str:
    """Interactive bar — rating distribution buckets."""
    mean_r   = data["overall_mean"]
    median_r = data["overall_median"]
    bins     = data["rating_bins"]
    rcts     = data["rating_counts"]

    fig = go.Figure(go.Bar(
        x=bins,
        y=rcts,
        marker=dict(
            color=rcts,
            colorscale="Viridis",
            showscale=False,
            line=dict(width=0),
        ),
        hovertemplate="<b>%{x}</b><br>Apps: %{y:,}<extra></extra>",
        text=[f"{v:,}" for v in rcts],
        textposition="outside",
        textfont=dict(size=10, color="#cbd5e1"),
    ))
    fig.update_layout(
        _dark_layout(
            title=_title(f"Rating Distribution  (Mean: {mean_r} | Median: {median_r})"),
            xaxis=dict(**_AXIS_X, title=_axis_title("Rating Range"), tickangle=-30),
            yaxis=dict(**_AXIS_Y, title=_axis_title("Number of Apps")),
            height=400,
            margin=dict(t=56, b=60, l=60, r=24),
        )
    )
    return _json(fig)


def chart_avg_rating_by_category(data: dict) -> str:
    """Interactive horizontal bar — average rating per category (top 15)."""
    cats   = data["avg_by_category"][:15]
    vals   = data["avg_by_category_vals"][:15]
    mean_r = data["overall_mean"]

    bar_colors = [
        "#22c55e" if v >= 4.3 else "#eab308" if v >= 4.0 else "#f97316" if v >= 3.5 else "#f43f5e"
        for v in vals
    ]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=vals[::-1],
        y=cats[::-1],
        orientation="h",
        marker=dict(color=bar_colors[::-1], line=dict(width=0)),
        hovertemplate="<b>%{y}</b><br>Avg Rating: %{x:.2f}<extra></extra>",
        text=[f"{v:.2f}" for v in vals[::-1]],
        textposition="outside",
        textfont=dict(size=10, color="#cbd5e1"),
    ))
    fig.add_vline(
        x=mean_r, line_dash="dash", line_color="#f97316", line_width=2,
        annotation=dict(
            text=f"Mean: {mean_r}",
            font=dict(color="#f97316", size=11),
            bgcolor="rgba(15,23,42,0.8)",
        ),
    )
    fig.update_layout(
        _dark_layout(
            title=_title("Average App Rating by Category (Top 15)"),
            xaxis=dict(**_AXIS_X, title=_axis_title("Average Rating"), range=[0, 5.5]),
            yaxis=dict(**_AXIS_Y, autorange=True),
            height=max(400, len(cats) * 28),
            margin=dict(t=56, b=40, l=180, r=80),
        )
    )
    return _json(fig)


def chart_size_vs_installs(data: dict) -> str:
    """Interactive scatter — app size vs installs with Pearson r annotation."""
    sizes    = data["size_data"]
    installs = data["installs_data"]
    r        = data["correlation"]
    strength = data["strength"]
    direction = data["direction"]

    log_inst = [np.log10(max(i, 1)) for i in installs]

    fig = go.Figure(go.Scatter(
        x=sizes,
        y=log_inst,
        mode="markers",
        marker=dict(color="#6366f1", size=5, opacity=0.45, line=dict(width=0)),
        hovertemplate=(
            "<b>Size:</b> %{x:.1f} MB<br>"
            "<b>Installs:</b> %{customdata:,}<extra></extra>"
        ),
        customdata=installs,
    ))
    fig.add_annotation(
        x=0.98, y=0.97,
        xref="paper", yref="paper",
        text=f"<b>r = {r}</b><br>{strength.title()} {direction} relationship",
        showarrow=False,
        font=dict(size=12, color="#f97316"),
        align="right",
        bgcolor="rgba(15,23,42,0.8)",
        bordercolor="#f97316",
        borderwidth=1,
        borderpad=6,
    )
    tick_vals = [3, 4, 5, 6, 7, 8]
    tick_text = ["1K", "10K", "100K", "1M", "10M", "100M"]
    fig.update_layout(
        _dark_layout(
            title=_title(f"App Size vs. Installs  (Pearson r = {r}  |  {strength.title()} {direction})"),
            xaxis=dict(**_AXIS_X, title=_axis_title("App Size (MB)")),
            yaxis=dict(
                **_AXIS_Y,
                title=_axis_title("Installs (log scale)"),
                tickvals=tick_vals,
                ticktext=tick_text,
            ),
            height=480,
            margin=dict(t=56, b=60, l=80, r=24),
        )
    )
    return _json(fig)


def chart_free_vs_paid(data: dict) -> str:
    """Interactive donut — free vs paid distribution."""
    labels = ["Free", "Paid"]
    counts = [data["free_count"], data["paid_count"]]
    total  = sum(counts)

    fig = go.Figure(go.Pie(
        labels=labels,
        values=counts,
        hole=0.55,
        marker=dict(
            colors=["#6366f1", "#ec4899"],
            line=dict(color="#0f172a", width=3),
        ),
        textinfo="label+percent",
        textfont=dict(size=13, color="#f1f5f9"),
        hovertemplate=(
            "<b>%{label}</b><br>"
            "Count: %{value:,}<br>"
            "Share: %{percent}<extra></extra>"
        ),
        pull=[0.03, 0.03],
    ))
    fig.add_annotation(
        x=0.5, y=0.5,
        text=f"<b>{total:,}</b><br>Total Apps",
        showarrow=False,
        font=dict(size=14, color="#f1f5f9"),
        xref="paper", yref="paper",
    )
    fig.update_layout(
        _dark_layout(
            title=_title("Free vs. Paid App Distribution"),
            height=400,
            margin=dict(t=56, b=20, l=20, r=20),
        )
    )
    return _json(fig)


def chart_price_distribution(data: dict) -> str:
    """Interactive bar — paid app price buckets."""
    magma = ["#f0f921", "#fca636", "#e16462", "#b12a90", "#6a00a8", "#0d0887"]

    fig = go.Figure(go.Bar(
        x=data["price_buckets"],
        y=data["price_bucket_counts"],
        marker=dict(color=magma, line=dict(width=0)),
        hovertemplate="<b>%{x}</b><br>Apps: %{y:,}<extra></extra>",
        text=[f"{v:,}" for v in data["price_bucket_counts"]],
        textposition="outside",
        textfont=dict(size=11, color="#cbd5e1"),
    ))
    fig.update_layout(
        _dark_layout(
            title=_title("Price Distribution - Paid Apps"),
            xaxis=dict(**_AXIS_X, title=_axis_title("Price Range (USD)")),
            yaxis=dict(**_AXIS_Y, title=_axis_title("Number of Paid Apps")),
            height=400,
            margin=dict(t=56, b=60, l=60, r=24),
        )
    )
    return _json(fig)


def chart_revenue_by_category(data: dict) -> str:
    """Interactive horizontal bar — estimated revenue by category."""
    cats = data["revenue_categories"]
    rev  = data["revenue_estimates"]

    def _fmt(v):
        if v >= 1_000_000:
            return f"${v/1_000_000:.1f}M"
        if v >= 1_000:
            return f"${v/1_000:.0f}K"
        return f"${v:.0f}"

    plasma     = px.colors.sequential.Plasma
    n          = max(len(cats) - 1, 1)
    bar_colors = [plasma[int(i / n * (len(plasma) - 1))] for i in range(len(cats))]

    fig = go.Figure(go.Bar(
        x=rev[::-1],
        y=cats[::-1],
        orientation="h",
        marker=dict(color=bar_colors[::-1], line=dict(width=0)),
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Est. Revenue: $%{x:,.0f}<br>"
            "<i>Indicative estimate only</i><extra></extra>"
        ),
        text=[_fmt(v) for v in rev[::-1]],
        textposition="outside",
        textfont=dict(size=10, color="#cbd5e1"),
    ))
    fig.update_layout(
        _dark_layout(
            title=_title("Estimated Revenue by Category - Paid Apps (Indicative Only)"),
            xaxis=dict(**_AXIS_X, title=_axis_title("Estimated Revenue (USD)"), tickformat="$,.0f"),
            yaxis=dict(**_AXIS_Y, autorange=True),
            height=max(380, len(cats) * 34),
            margin=dict(t=56, b=40, l=180, r=100),
        )
    )
    return _json(fig)


# ===========================================================================
# REVIEWS DASHBOARD — 4 charts
# ===========================================================================

def chart_sentiment_donut(dist: dict) -> str:
    """Interactive donut — positive / negative / neutral sentiment."""
    labels = ["Positive", "Negative", "Neutral"]
    counts = [dist[l]["count"] for l in labels]
    total  = dist["total"]

    fig = go.Figure(go.Pie(
        labels=labels,
        values=counts,
        hole=0.55,
        marker=dict(
            colors=["#22c55e", "#f43f5e", "#94a3b8"],
            line=dict(color="#0f172a", width=3),
        ),
        textinfo="label+percent",
        textfont=dict(size=13, color="#f1f5f9"),
        hovertemplate=(
            "<b>%{label}</b><br>"
            "Count: %{value:,}<br>"
            "Share: %{percent}<extra></extra>"
        ),
        pull=[0.04, 0.04, 0.02],
    ))
    fig.add_annotation(
        x=0.5, y=0.5,
        text=f"<b>{total:,}</b><br>Reviews",
        showarrow=False,
        font=dict(size=14, color="#f1f5f9"),
        xref="paper", yref="paper",
    )
    fig.update_layout(
        _dark_layout(
            title=_title("VADER Sentiment Distribution"),
            height=400,
            margin=dict(t=56, b=20, l=20, r=20),
        )
    )
    return _json(fig)


def chart_sentiment_bar(dist: dict) -> str:
    """Interactive bar — sentiment counts with percentage labels."""
    labels = ["Positive", "Negative", "Neutral"]
    counts = [dist[l]["count"] for l in labels]
    pcts   = [dist[l]["pct"]   for l in labels]
    colors = ["#22c55e", "#f43f5e", "#94a3b8"]

    fig = go.Figure(go.Bar(
        x=labels,
        y=counts,
        marker=dict(color=colors, line=dict(width=0)),
        hovertemplate=(
            "<b>%{x}</b><br>"
            "Count: %{y:,}<br>"
            "Percentage: %{customdata}%<extra></extra>"
        ),
        customdata=pcts,
        text=[f"{c:,}  ({p}%)" for c, p in zip(counts, pcts)],
        textposition="outside",
        textfont=dict(size=11, color="#cbd5e1"),
    ))
    fig.update_layout(
        _dark_layout(
            title=_title("Sentiment Classification (VADER)"),
            xaxis=dict(**_AXIS_X),
            yaxis=dict(**_AXIS_Y, title=_axis_title("Number of Reviews")),
            height=420,
            margin=dict(t=56, b=40, l=60, r=24),
        )
    )
    return _json(fig)


def chart_compound_histogram(df: pd.DataFrame) -> str:
    """Interactive histogram of VADER compound scores with threshold lines."""
    mean_val = float(df["vader_compound"].mean())
    scores   = df["vader_compound"].tolist()

    fig = go.Figure(go.Histogram(
        x=scores,
        nbinsx=50,
        marker=dict(color="#6366f1", opacity=0.85, line=dict(color="#0f172a", width=0.4)),
        hovertemplate="Score: %{x:.2f}<br>Count: %{y:,}<extra></extra>",
        name="Reviews",
    ))
    for x_val, color, label, pos in [
        (0.05,     "#22c55e", "Positive (+0.05)",     "top right"),
        (-0.05,    "#f43f5e", "Negative (-0.05)",     "top left"),
        (mean_val, "#f97316", f"Mean ({mean_val:.3f})", "top right"),
    ]:
        fig.add_vline(
            x=x_val,
            line_dash="dash" if x_val != mean_val else "solid",
            line_color=color,
            line_width=2,
            annotation=dict(
                text=label,
                font=dict(color=color, size=10),
                bgcolor="rgba(15,23,42,0.8)",
            ),
        )
    fig.update_layout(
        _dark_layout(
            title=_title("Distribution of VADER Compound Scores"),
            xaxis=dict(**_AXIS_X, title=_axis_title("VADER Compound Score (-1 to +1)")),
            yaxis=dict(**_AXIS_Y, title=_axis_title("Number of Reviews")),
            height=420,
            showlegend=False,
            margin=dict(t=56, b=60, l=60, r=24),
        )
    )
    return _json(fig)


def chart_sentiment_by_category(cat_data: dict) -> str:
    """Interactive stacked bar — sentiment counts by category (top 15)."""
    cats  = cat_data["categories"][:15]
    pos_c = cat_data["positive_counts"][:15]
    neg_c = cat_data["negative_counts"][:15]
    neu_c = cat_data["neutral_counts"][:15]
    pos_p = cat_data["positive_pct"][:15]
    neg_p = cat_data["negative_pct"][:15]
    neu_p = [round(100 - a - b, 1) for a, b in zip(pos_p, neg_p)]

    fig = go.Figure()
    for label, vals, color, pcts in [
        ("Positive", pos_c, "#22c55e", pos_p),
        ("Neutral",  neu_c, "#94a3b8", neu_p),
        ("Negative", neg_c, "#f43f5e", neg_p),
    ]:
        fig.add_trace(go.Bar(
            name=label,
            x=cats,
            y=vals,
            marker=dict(color=color, line=dict(width=0)),
            hovertemplate=(
                f"<b>%{{x}}</b><br>"
                f"{label}: %{{y:,}}<br>"
                f"Category share: %{{customdata}}%<extra></extra>"
            ),
            customdata=pcts,
        ))
    fig.update_layout(
        _dark_layout(
            barmode="stack",
            title=_title("Sentiment by Category (Top 15 by Positive %)"),
            xaxis=dict(**_AXIS_X, tickangle=-35),
            yaxis=dict(**_AXIS_Y, title=_axis_title("Number of Reviews")),
            height=480,
            margin=dict(t=56, b=100, l=60, r=24),
        )
    )
    return _json(fig)


# ===========================================================================
# Backward-compat aliases (used nowhere in the new app.py, kept for safety)
# ===========================================================================

def plotly_interactive_category(data: dict, pricing_data: dict) -> str:
    return chart_category_bar(data)


def plotly_interactive_sentiment(dist: dict, cat_data) -> str:
    if cat_data:
        return chart_sentiment_by_category(cat_data)
    return chart_sentiment_donut(dist)
