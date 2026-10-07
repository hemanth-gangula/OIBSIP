"""
visualizations.py
-----------------
All chart generation for the Google Play Store Analytics application.
Returns base64-encoded PNG images (Matplotlib/Seaborn) or JSON (Plotly).
"""

import io
import base64
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json

# ─────────────────────────────────────────────
# THEME
# ─────────────────────────────────────────────
DARK_BG   = '#0F0F1A'
CARD_BG   = '#1A1A2E'
PRIMARY   = '#6C63FF'
SECONDARY = '#3ECFCF'
ACCENT    = '#FF6584'
TEXT_CLR  = '#E0E0E0'
GRID_CLR  = '#2A2A3E'

PALETTE_CAT = [
    '#6C63FF','#3ECFCF','#FF6584','#F39C12','#2ECC71',
    '#E74C3C','#9B59B6','#1ABC9C','#E67E22','#3498DB',
]

def _fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=130, bbox_inches='tight',
                facecolor=DARK_BG, edgecolor='none')
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return b64

def _base_style():
    plt.rcParams.update({
        'figure.facecolor': DARK_BG,
        'axes.facecolor':   CARD_BG,
        'axes.edgecolor':   GRID_CLR,
        'axes.labelcolor':  TEXT_CLR,
        'xtick.color':      TEXT_CLR,
        'ytick.color':      TEXT_CLR,
        'text.color':       TEXT_CLR,
        'grid.color':       GRID_CLR,
        'grid.linewidth':   0.5,
        'font.family':      'sans-serif',
        'font.size':        10,
    })


# ─────────────────────────────────────────────
# 1. CATEGORY DISTRIBUTION (Bar)
# ─────────────────────────────────────────────

def chart_category_distribution(category_data):
    _base_style()
    counts = pd.DataFrame(category_data['category_counts'])
    if 'App_Count' not in counts.columns:
        counts.columns = ['Category', 'App_Count']
    top = counts.head(20).sort_values('App_Count')

    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.barh(top['Category'], top['App_Count'],
                   color=PALETTE_CAT * 4, edgecolor='none', height=0.7)
    for bar in bars:
        ax.text(bar.get_width() + top['App_Count'].max() * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f'{int(bar.get_width()):,}', va='center', fontsize=8, color=TEXT_CLR)
    ax.set_xlabel('Number of Apps', labelpad=10)
    ax.set_title('Top 20 Categories by App Count', pad=15, fontsize=14, fontweight='bold', color=TEXT_CLR)
    ax.grid(axis='x', linestyle='--', alpha=0.4)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right', 'bottom']].set_visible(False)
    plt.tight_layout()
    return _fig_to_b64(fig)


# ─────────────────────────────────────────────
# 2. RATING DISTRIBUTION (Histogram)
# ─────────────────────────────────────────────

def chart_rating_distribution(apps_df):
    _base_style()
    rated = apps_df.dropna(subset=['Rating'])
    fig, ax = plt.subplots(figsize=(9, 5))
    n, bins, patches = ax.hist(rated['Rating'], bins=30, color=PRIMARY,
                                edgecolor=CARD_BG, linewidth=0.5, alpha=0.9)
    # Colour gradient
    for patch, val in zip(patches, bins):
        patch.set_facecolor(plt.cm.cool(val / 5.0))
    avg = rated['Rating'].mean()
    ax.axvline(avg, color=ACCENT, linestyle='--', linewidth=1.8,
               label=f'Mean: {avg:.2f}')
    ax.set_xlabel('Rating', labelpad=10)
    ax.set_ylabel('Number of Apps', labelpad=10)
    ax.set_title('App Rating Distribution', pad=15, fontsize=14, fontweight='bold', color=TEXT_CLR)
    ax.legend(facecolor=CARD_BG, edgecolor=GRID_CLR, labelcolor=TEXT_CLR)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    return _fig_to_b64(fig)


# ─────────────────────────────────────────────
# 3. AVERAGE RATING BY CATEGORY (Horizontal bar)
# ─────────────────────────────────────────────

def chart_avg_rating_by_category(rating_data):
    _base_style()
    df = pd.DataFrame(rating_data['avg_by_category']).head(20)
    df = df.sort_values('Avg_Rating')
    colors = [SECONDARY if r >= 4.0 else (PRIMARY if r >= 3.5 else ACCENT)
              for r in df['Avg_Rating']]

    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.barh(df['Category'], df['Avg_Rating'], color=colors,
                   edgecolor='none', height=0.7)
    for bar in bars:
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                f'{bar.get_width():.2f}', va='center', fontsize=8, color=TEXT_CLR)
    ax.axvline(rating_data['avg_overall'], color=ACCENT, linestyle='--',
               linewidth=1.5, label=f"Overall avg: {rating_data['avg_overall']:.2f}★")
    ax.set_xlim(0, 5.2)
    ax.set_xlabel('Average Rating', labelpad=10)
    ax.set_title('Average Rating by Category (Top 20)', pad=15, fontsize=14,
                 fontweight='bold', color=TEXT_CLR)
    ax.legend(facecolor=CARD_BG, edgecolor=GRID_CLR, labelcolor=TEXT_CLR)
    ax.grid(axis='x', linestyle='--', alpha=0.4)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right', 'bottom']].set_visible(False)
    plt.tight_layout()
    return _fig_to_b64(fig)


# ─────────────────────────────────────────────
# 4. SIZE vs INSTALLS (Scatter)
# ─────────────────────────────────────────────

def chart_size_vs_installs(size_installs_data):
    _base_style()
    if not size_installs_data.get('scatter_data'):
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.text(0.5, 0.5, 'Insufficient data', transform=ax.transAxes,
                ha='center', va='center', color=TEXT_CLR, fontsize=14)
        ax.set_facecolor(CARD_BG)
        return _fig_to_b64(fig)

    df = pd.DataFrame(size_installs_data['scatter_data'])
    fig, ax = plt.subplots(figsize=(9, 6))
    sc = ax.scatter(df['Size_MB'], df['Installs_Numeric'],
                    alpha=0.55, s=18, c=df['Size_MB'],
                    cmap='cool', edgecolors='none')
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label('Size (MB)', color=TEXT_CLR)
    cbar.ax.yaxis.set_tick_params(color=TEXT_CLR)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=TEXT_CLR)
    ax.set_xlabel('App Size (MB)', labelpad=10)
    ax.set_ylabel('Installs', labelpad=10)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f'{x/1e6:.0f}M' if x >= 1e6 else (f'{x/1e3:.0f}K' if x >= 1e3 else str(int(x)))
    ))
    corr = size_installs_data.get('correlation', 'N/A')
    ax.set_title(f'App Size vs Installs  (r = {corr})', pad=15,
                 fontsize=14, fontweight='bold', color=TEXT_CLR)
    ax.grid(linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False)
    # Trend line
    try:
        m, b = np.polyfit(df['Size_MB'], np.log1p(df['Installs_Numeric']), 1)
        x_line = np.linspace(df['Size_MB'].min(), df['Size_MB'].max(), 200)
        ax.plot(x_line, np.expm1(m * x_line + b), color=ACCENT,
                linewidth=1.8, linestyle='--', label='Trend')
        ax.legend(facecolor=CARD_BG, edgecolor=GRID_CLR, labelcolor=TEXT_CLR)
    except Exception:
        pass
    plt.tight_layout()
    return _fig_to_b64(fig)


# ─────────────────────────────────────────────
# 5. FREE vs PAID (Pie / Donut)
# ─────────────────────────────────────────────

def chart_free_vs_paid(pricing_data):
    _base_style()
    labels = ['Free', 'Paid']
    sizes  = [pricing_data['free_count'], pricing_data['paid_count']]
    colors = [SECONDARY, ACCENT]
    explode = (0.04, 0.04)

    fig, ax = plt.subplots(figsize=(7, 6))
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, autopct='%1.1f%%',
        colors=colors, explode=explode,
        wedgeprops=dict(width=0.55, edgecolor=DARK_BG, linewidth=2),
        textprops=dict(color=TEXT_CLR),
        startangle=90,
    )
    for at in autotexts:
        at.set_fontsize(12)
        at.set_fontweight('bold')
    ax.set_title('Free vs Paid App Distribution', pad=15, fontsize=14,
                 fontweight='bold', color=TEXT_CLR)
    ax.text(0, 0, f"Total\n{pricing_data['free_count'] + pricing_data['paid_count']:,}",
            ha='center', va='center', fontsize=11, fontweight='bold', color=TEXT_CLR)
    plt.tight_layout()
    return _fig_to_b64(fig)


# ─────────────────────────────────────────────
# 6. PAID APP PRICE DISTRIBUTION (Bar)
# ─────────────────────────────────────────────

def chart_price_distribution(pricing_data):
    _base_style()
    df = pd.DataFrame(pricing_data['price_distribution'])
    if df.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, 'No paid apps in dataset', transform=ax.transAxes,
                ha='center', va='center', color=TEXT_CLR, fontsize=13)
        return _fig_to_b64(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(df['Range'].astype(str), df['Count'],
                  color=PALETTE_CAT[:len(df)], edgecolor='none', width=0.6)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(df['Count']) * 0.01,
                f'{int(bar.get_height()):,}', ha='center', va='bottom', fontsize=9, color=TEXT_CLR)
    ax.set_xlabel('Price Range', labelpad=10)
    ax.set_ylabel('Number of Paid Apps', labelpad=10)
    ax.set_title('Paid App Price Distribution', pad=15, fontsize=14,
                 fontweight='bold', color=TEXT_CLR)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    return _fig_to_b64(fig)


# ─────────────────────────────────────────────
# 7. ESTIMATED REVENUE BY CATEGORY (Bar)
# ─────────────────────────────────────────────

def chart_revenue_by_category(pricing_data):
    _base_style()
    df = pd.DataFrame(pricing_data['revenue_by_category'])
    if df.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, 'No revenue data available', transform=ax.transAxes,
                ha='center', va='center', color=TEXT_CLR, fontsize=13)
        return _fig_to_b64(fig)

    df = df.sort_values('Est_Revenue')
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(df['Category'], df['Est_Revenue'],
                   color=PRIMARY, edgecolor='none', height=0.6)
    for bar in bars:
        val = bar.get_width()
        label = f'${val/1e6:.1f}M' if val >= 1e6 else f'${val/1e3:.0f}K'
        ax.text(val + df['Est_Revenue'].max() * 0.01,
                bar.get_y() + bar.get_height() / 2,
                label, va='center', fontsize=8, color=TEXT_CLR)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f'${x/1e6:.0f}M' if x >= 1e6 else f'${x/1e3:.0f}K'
    ))
    ax.set_xlabel('Estimated Revenue (USD)', labelpad=10)
    ax.set_title('Estimated Revenue by Category (Top 10 Paid)', pad=15,
                 fontsize=14, fontweight='bold', color=TEXT_CLR)
    ax.grid(axis='x', linestyle='--', alpha=0.4)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right', 'bottom']].set_visible(False)
    plt.tight_layout()
    return _fig_to_b64(fig)


# ─────────────────────────────────────────────
# 8. SENTIMENT DISTRIBUTION (Bar)
# ─────────────────────────────────────────────

def chart_sentiment_distribution(sentiment_data):
    _base_style()
    labels = ['Positive', 'Negative', 'Neutral']
    counts = [
        sentiment_data['positive_count'],
        sentiment_data['negative_count'],
        sentiment_data['neutral_count'],
    ]
    colors = ['#2ECC71', '#E74C3C', '#F39C12']

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Bar chart
    bars = axes[0].bar(labels, counts, color=colors, edgecolor='none', width=0.55)
    for bar in bars:
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(counts) * 0.01,
                     f'{int(bar.get_height()):,}', ha='center', va='bottom',
                     fontsize=10, fontweight='bold', color=TEXT_CLR)
    axes[0].set_ylabel('Number of Reviews', labelpad=10)
    axes[0].set_title('Sentiment Count', pad=12, fontsize=13, fontweight='bold', color=TEXT_CLR)
    axes[0].grid(axis='y', linestyle='--', alpha=0.4)
    axes[0].set_axisbelow(True)
    axes[0].spines[['top', 'right']].set_visible(False)

    # Donut
    pcts = [sentiment_data['positive_pct'], sentiment_data['negative_pct'], sentiment_data['neutral_pct']]
    wedges, texts, autotexts = axes[1].pie(
        pcts, labels=labels, autopct='%1.1f%%', colors=colors,
        wedgeprops=dict(width=0.55, edgecolor=DARK_BG, linewidth=2),
        textprops=dict(color=TEXT_CLR), startangle=90,
    )
    for at in autotexts:
        at.set_fontsize(11)
        at.set_fontweight('bold')
    axes[1].set_title('Sentiment Share', pad=12, fontsize=13, fontweight='bold', color=TEXT_CLR)
    axes[1].text(0, 0, f"{sentiment_data['total_reviews']:,}\nReviews",
                 ha='center', va='center', fontsize=10, fontweight='bold', color=TEXT_CLR)

    fig.suptitle('User Review Sentiment Analysis', fontsize=15, fontweight='bold',
                 color=TEXT_CLR, y=1.02)
    plt.tight_layout()
    return _fig_to_b64(fig)


# ─────────────────────────────────────────────
# 9. SENTIMENT BY CATEGORY (Stacked Bar)
# ─────────────────────────────────────────────

def chart_sentiment_by_category(sentiment_cat_data):
    _base_style()
    df = pd.DataFrame(sentiment_cat_data['by_category']).head(15)
    if df.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, 'No merged data available', transform=ax.transAxes,
                ha='center', va='center', color=TEXT_CLR, fontsize=13)
        return _fig_to_b64(fig)

    df = df.sort_values('Total', ascending=True)
    pos_col = 'Positive' if 'Positive' in df.columns else 'positive'
    neg_col = 'Negative' if 'Negative' in df.columns else 'negative'
    neu_col = 'Neutral'  if 'Neutral'  in df.columns else 'neutral'

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(df['Category'], df.get(pos_col, 0), color='#2ECC71', label='Positive', height=0.65)
    ax.barh(df['Category'], df.get(neu_col, 0), color='#F39C12', label='Neutral',  height=0.65,
            left=df.get(pos_col, 0))
    ax.barh(df['Category'], df.get(neg_col, 0), color='#E74C3C', label='Negative', height=0.65,
            left=df.get(pos_col, 0) + df.get(neu_col, 0))

    ax.set_xlabel('Review Count', labelpad=10)
    ax.set_title('Sentiment Distribution by Category', pad=15, fontsize=14,
                 fontweight='bold', color=TEXT_CLR)
    ax.legend(facecolor=CARD_BG, edgecolor=GRID_CLR, labelcolor=TEXT_CLR, loc='lower right')
    ax.grid(axis='x', linestyle='--', alpha=0.4)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right', 'bottom']].set_visible(False)
    plt.tight_layout()
    return _fig_to_b64(fig)


# ─────────────────────────────────────────────
# 10. INTERACTIVE PLOTLY – Multi-tab analysis
# ─────────────────────────────────────────────

def chart_interactive_plotly(apps_df, category_data, rating_data, pricing_data, sentiment_data):
    """
    Returns a Plotly figure JSON with 4 subplots:
      Tab 1 – Category bubble chart (count × avg rating × installs)
      Tab 2 – Rating heatmap by category
      Tab 3 – Price vs Installs scatter (paid apps)
      Tab 4 – Sentiment polarity strip
    """
    plotly_dark = dict(
        paper_bgcolor='#0F0F1A',
        plot_bgcolor='#1A1A2E',
        font=dict(color='#E0E0E0', family='sans-serif'),
        colorway=['#6C63FF','#3ECFCF','#FF6584','#F39C12','#2ECC71',
                  '#E74C3C','#9B59B6','#1ABC9C','#E67E22','#3498DB'],
    )

    # ── Sub-figure 1: Bubble chart – category installs vs avg rating ──
    try:
        cat_df   = pd.DataFrame(category_data['category_counts'])
        if 'App_Count' not in cat_df.columns:
            cat_df.columns = ['Category', 'App_Count']
        rat_df   = pd.DataFrame(rating_data['avg_by_category'])
        bubble   = cat_df.merge(rat_df, on='Category').head(30)
        # Median installs per category
        inst_cat = apps_df.dropna(subset=['Installs_Numeric']).groupby('Category')['Installs_Numeric'].median().reset_index()
        inst_cat.columns = ['Category', 'Median_Installs']
        bubble   = bubble.merge(inst_cat, on='Category', how='left')
        bubble['Median_Installs'] = bubble['Median_Installs'].fillna(1)

        fig_bubble = px.scatter(
            bubble, x='Avg_Rating', y='App_Count',
            size='Median_Installs', color='Category',
            hover_name='Category',
            hover_data={'Avg_Rating': ':.2f', 'App_Count': ':,', 'Median_Installs': ':,.0f'},
            size_max=55,
            title='Category Overview: Rating vs App Count (bubble = median installs)',
            labels={'Avg_Rating': 'Average Rating', 'App_Count': 'Number of Apps'},
        )
        fig_bubble.update_layout(**plotly_dark, height=520)
        fig_bubble.update_xaxes(gridcolor='#2A2A3E')
        fig_bubble.update_yaxes(gridcolor='#2A2A3E')
        bubble_json = fig_bubble.to_json()
    except Exception as e:
        bubble_json = None

    # ── Sub-figure 2: Rating distribution violin / box by type ──
    try:
        sub = apps_df.dropna(subset=['Rating', 'Type'])
        sub = sub[sub['Type'].isin(['Free', 'Paid'])]
        fig_violin = px.violin(
            sub, y='Rating', x='Type', color='Type',
            box=True, points='outliers',
            color_discrete_map={'Free': '#3ECFCF', 'Paid': '#FF6584'},
            title='Rating Distribution: Free vs Paid Apps',
            labels={'Rating': 'App Rating', 'Type': 'App Type'},
        )
        fig_violin.update_layout(**plotly_dark, height=450)
        violin_json = fig_violin.to_json()
    except Exception as e:
        violin_json = None

    # ── Sub-figure 3: Top 10 categories by install volume ──
    try:
        inst_cat2 = (
            apps_df.dropna(subset=['Installs_Numeric'])
            .groupby('Category')['Installs_Numeric']
            .sum()
            .reset_index()
            .sort_values('Installs_Numeric', ascending=False)
            .head(10)
        )
        inst_cat2['Installs_B'] = (inst_cat2['Installs_Numeric'] / 1e9).round(2)
        fig_inst = px.bar(
            inst_cat2, x='Category', y='Installs_B',
            color='Installs_B', color_continuous_scale='Viridis',
            title='Top 10 Categories by Total Installs (Billions)',
            labels={'Installs_B': 'Total Installs (B)', 'Category': 'Category'},
            text='Installs_B',
        )
        fig_inst.update_traces(texttemplate='%{text:.1f}B', textposition='outside')
        fig_inst.update_layout(**plotly_dark, height=480,
                               coloraxis_showscale=False)
        fig_inst.update_xaxes(tickangle=-35, gridcolor='#2A2A3E')
        fig_inst.update_yaxes(gridcolor='#2A2A3E')
        inst_json = fig_inst.to_json()
    except Exception as e:
        inst_json = None

    # ── Sub-figure 4: Sunburst – Type → Category → Rating bucket ──
    try:
        sub4 = apps_df.dropna(subset=['Type', 'Category', 'Rating']).copy()
        sub4 = sub4[sub4['Type'].isin(['Free', 'Paid'])]
        sub4['Rating_Bucket'] = pd.cut(sub4['Rating'],
                                       bins=[0, 2, 3, 4, 5],
                                       labels=['<2', '2-3', '3-4', '4-5'])
        sub4['Rating_Bucket'] = sub4['Rating_Bucket'].astype(str)
        fig_sun = px.sunburst(
            sub4.sample(min(3000, len(sub4)), random_state=42),
            path=['Type', 'Category', 'Rating_Bucket'],
            title='App Landscape: Type → Category → Rating Bucket',
            color_discrete_sequence=px.colors.qualitative.Vivid,
        )
        fig_sun.update_layout(**plotly_dark, height=540)
        sun_json = fig_sun.to_json()
    except Exception as e:
        sun_json = None

    return {
        'bubble':  bubble_json,
        'violin':  violin_json,
        'installs': inst_json,
        'sunburst': sun_json,
    }
