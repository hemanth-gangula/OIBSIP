import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # non-interactive backend — IMPORTANT for script execution
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from textblob import TextBlob

pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', 100)
pd.set_option('display.float_format', '{:.2f}'.format)

plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
sns.set_style('whitegrid')
sns.set_palette('husl')

# Use absolute paths (script runs from workdir)
BASE_DIR = r'c:\Users\Admib\OneDrive\Documents\OIBSIP\DataAnalytics-L2-GooglePlayStoreAnalysis'
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')
FIGURES_DIR = os.path.join(OUTPUT_DIR, 'figures')
CLEANED_DIR = os.path.join(OUTPUT_DIR, 'cleaned_data')

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(CLEANED_DIR, exist_ok=True)

APPS_PATH = os.path.join(DATA_DIR, 'googleplaystore.csv')
REVIEWS_PATH = os.path.join(DATA_DIR, 'googleplaystore_user_reviews.csv')

df_apps = pd.read_csv(APPS_PATH)
print(f'Apps dataset loaded: {df_apps.shape[0]:,} rows x {df_apps.shape[1]} columns')
df_reviews = pd.read_csv(REVIEWS_PATH)
print(f'Reviews dataset loaded: {df_reviews.shape[0]:,} rows x {df_reviews.shape[1]} columns')

# --- CLEANING ---
apps = df_apps.copy()
reviews = df_reviews.copy()

duplicates_before = apps.duplicated().sum()
apps = apps.drop_duplicates()
print(f'Removed {duplicates_before} duplicate app rows.')

bad_rows = apps[apps['Category'].str.match(r'^\d', na=False)]
print(f'Bad rows (Category starts with digit): {len(bad_rows)}')
apps = apps[~apps['Category'].str.match(r'^\d', na=False)]

apps['Rating'] = pd.to_numeric(apps['Rating'], errors='coerce')
apps.loc[~apps['Rating'].between(1.0, 5.0, inclusive='both'), 'Rating'] = np.nan

apps['Reviews'] = pd.to_numeric(apps['Reviews'], errors='coerce')

def parse_size(size_str):
    if pd.isna(size_str) or str(size_str).strip().lower() in ['varies with device', 'nan', '']:
        return np.nan
    s = str(size_str).strip()
    if s.endswith('M'):
        return float(s[:-1])
    elif s.endswith('k'):
        return float(s[:-1]) / 1024
    else:
        try:
            return float(s)
        except ValueError:
            return np.nan

apps['Size_MB'] = apps['Size'].apply(parse_size)
apps['Installs_Clean'] = apps['Installs'].str.replace(r'[+,]', '', regex=True)
apps['Installs_Clean'] = pd.to_numeric(apps['Installs_Clean'], errors='coerce')
apps['Price_Clean'] = apps['Price'].str.replace(r'[\$]', '', regex=True)
apps['Price_Clean'] = pd.to_numeric(apps['Price_Clean'], errors='coerce')
apps['Type'] = apps['Type'].str.strip()
apps.loc[~apps['Type'].isin(['Free', 'Paid']), 'Type'] = np.nan
apps['Last_Updated'] = pd.to_datetime(apps['Last Updated'], errors='coerce', format='%B %d, %Y')
apps_clean = apps.dropna(subset=['App', 'Category'])
print(f'Apps after cleaning: {apps_clean.shape}')

reviews['Translated_Review'] = reviews['Translated_Review'].replace('nan', np.nan)
reviews_clean = reviews.dropna(subset=['Translated_Review'])
reviews_clean = reviews_clean.drop_duplicates()
print(f'Reviews after cleaning: {reviews_clean.shape}')

apps_clean.to_csv(os.path.join(CLEANED_DIR, 'cleaned_apps.csv'), index=False)
reviews_clean.to_csv(os.path.join(CLEANED_DIR, 'cleaned_reviews.csv'), index=False)
print('Cleaned datasets saved.')

# --- CATEGORY DISTRIBUTION ---
category_counts = apps_clean['Category'].value_counts().reset_index()
category_counts.columns = ['Category', 'App_Count']
print(f'Total categories: {category_counts.shape[0]}')

fig, ax = plt.subplots(figsize=(14, 8))
ax.barh(category_counts['Category'][::-1], category_counts['App_Count'][::-1],
        color=sns.color_palette('husl', len(category_counts)))
ax.set_xlabel('Number of Apps')
ax.set_title('App Distribution by Category on Google Play Store', fontsize=15, fontweight='bold')
ax.set_ylabel('Category')
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'category_distribution.png'), dpi=150, bbox_inches='tight')
plt.close()
print('Saved: category_distribution.png')

# --- RATINGS ANALYSIS ---
category_ratings = apps_clean.groupby('Category')['Rating'].agg(['mean', 'count']).reset_index()
category_ratings.columns = ['Category', 'Avg_Rating', 'App_Count']
category_ratings = category_ratings[category_ratings['App_Count'] >= 10].sort_values('Avg_Rating', ascending=False)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(apps_clean['Rating'].dropna(), bins=20, color='steelblue', edgecolor='white', alpha=0.8)
axes[0].set_xlabel('Rating')
axes[0].set_ylabel('Number of Apps')
axes[0].set_title('Distribution of App Ratings')
axes[0].axvline(apps_clean['Rating'].mean(), color='red', linestyle='--',
                label=f'Mean: {apps_clean["Rating"].mean():.2f}')
axes[0].legend()
axes[1].barh(category_ratings['Category'][::-1], category_ratings['Avg_Rating'][::-1], color='coral')
axes[1].set_xlabel('Average Rating')
axes[1].set_title('Average Rating by Category\n(min. 10 apps)')
axes[1].set_xlim(3.5, 5.0)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'ratings_analysis.png'), dpi=150, bbox_inches='tight')
plt.close()
print('Saved: ratings_analysis.png')

# --- SIZE VS INSTALLS ---
size_install = apps_clean.dropna(subset=['Size_MB', 'Installs_Clean']).copy()
size_install = size_install[size_install['Installs_Clean'] > 0]
size_install['Log_Installs'] = np.log10(size_install['Installs_Clean'])
print(f'Apps with both size and installs data: {len(size_install)}')

fig, ax = plt.subplots(figsize=(12, 6))
scatter = ax.scatter(size_install['Size_MB'], size_install['Log_Installs'],
                    alpha=0.3, c=size_install['Rating'].fillna(3.0), cmap='RdYlGn',
                    s=20, vmin=1, vmax=5)
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Rating')
ax.set_xlabel('App Size (MB)')
ax.set_ylabel('Installs (log10 scale)')
ax.set_title('App Size vs Installs (color = Rating)', fontsize=14)
from numpy.polynomial.polynomial import polyfit
c_fit, m_fit = polyfit(size_install['Size_MB'], size_install['Log_Installs'], 1)
ax.plot(sorted(size_install['Size_MB']), [m_fit*x + c_fit for x in sorted(size_install['Size_MB'])],
       'b-', alpha=0.5, linewidth=1.5, label='Trend')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'size_vs_installs.png'), dpi=150, bbox_inches='tight')
plt.close()
corr = size_install[['Size_MB', 'Installs_Clean']].corr().iloc[0, 1]
log_corr = size_install[['Size_MB', 'Log_Installs']].corr().iloc[0, 1]
print(f'Saved: size_vs_installs.png | Correlation: {corr:.4f}, Log correlation: {log_corr:.4f}')

# --- PRICING ANALYSIS ---
type_counts = apps_clean['Type'].value_counts()
paid_apps = apps_clean[apps_clean['Price_Clean'] > 0]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].pie(type_counts.values, labels=type_counts.index, autopct='%1.1f%%',
           colors=['#4CAF50', '#FF5722'], startangle=90, textprops={'fontsize': 12})
axes[0].set_title('Free vs Paid Apps Distribution')
if len(paid_apps) > 0:
    axes[1].hist(paid_apps['Price_Clean'].clip(upper=30), bins=30, color='orange', edgecolor='white', alpha=0.8)
    axes[1].set_xlabel('Price (USD)')
    axes[1].set_ylabel('Number of Apps')
    axes[1].set_title(f'Paid App Price Distribution (clipped at $30, n={len(paid_apps)})')
    axes[1].axvline(paid_apps['Price_Clean'].median(), color='red', linestyle='--',
                   label=f'Median: ${paid_apps["Price_Clean"].median():.2f}')
    axes[1].legend()
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'pricing_analysis.png'), dpi=150, bbox_inches='tight')
plt.close()
print('Saved: pricing_analysis.png')

# Revenue estimate
paid_with_installs = paid_apps.dropna(subset=['Installs_Clean'])
paid_with_installs = paid_with_installs[paid_with_installs['Installs_Clean'] > 0].copy()
paid_with_installs['Theoretical_Gross'] = paid_with_installs['Price_Clean'] * paid_with_installs['Installs_Clean']
rev_by_cat = paid_with_installs.groupby('Category')['Theoretical_Gross'].sum().sort_values(ascending=False).head(10)

fig, ax = plt.subplots(figsize=(12, 6))
rev_by_cat.plot(kind='barh', ax=ax, color='coral')
ax.set_xlabel('Theoretical Gross Revenue Estimate (USD)')
ax.set_title('Top 10 Categories: Theoretical Gross Revenue Estimate\n(Price x Installs - NOT actual revenue)', fontsize=13)
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x/1e6:.0f}M'))
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'revenue_estimate.png'), dpi=150, bbox_inches='tight')
plt.close()
print('Saved: revenue_estimate.png')

# --- SENTIMENT ANALYSIS ---
print('Running TextBlob sentiment analysis...')

def get_textblob_sentiment(text):
    if pd.isna(text) or str(text).strip() == '':
        return None, None
    analysis = TextBlob(str(text))
    return analysis.sentiment.polarity, analysis.sentiment.subjectivity

def classify_sentiment(polarity):
    if pd.isna(polarity):
        return 'Unknown'
    if polarity > 0.05:
        return 'Positive'
    elif polarity < -0.05:
        return 'Negative'
    else:
        return 'Neutral'

reviews_clean = reviews_clean.copy()
reviews_clean[['TB_Polarity', 'TB_Subjectivity']] = reviews_clean['Translated_Review'].apply(
    lambda x: pd.Series(get_textblob_sentiment(x))
)
reviews_clean['TB_Sentiment'] = reviews_clean['TB_Polarity'].apply(classify_sentiment)
print('TextBlob sentiment distribution:')
print(reviews_clean['TB_Sentiment'].value_counts())

sentiment_counts = reviews_clean['TB_Sentiment'].value_counts()
colors_map = {'Positive': '#4CAF50', 'Neutral': '#FFC107', 'Negative': '#F44336', 'Unknown': '#9E9E9E'}
bar_colors = [colors_map.get(s, '#888888') for s in sentiment_counts.index]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].bar(sentiment_counts.index, sentiment_counts.values, color=bar_colors, edgecolor='white')
axes[0].set_xlabel('Sentiment')
axes[0].set_ylabel('Number of Reviews')
axes[0].set_title('Review Sentiment Distribution (TextBlob)')
for i, v in enumerate(sentiment_counts.values):
    axes[0].text(i, v + 100, str(v), ha='center', fontweight='bold')
axes[1].hist(reviews_clean['TB_Polarity'].dropna(), bins=40, color='steelblue', edgecolor='white', alpha=0.8)
axes[1].axvline(0.05, color='green', linestyle='--', alpha=0.7, label='Positive threshold')
axes[1].axvline(-0.05, color='red', linestyle='--', alpha=0.7, label='Negative threshold')
axes[1].set_xlabel('Polarity Score')
axes[1].set_ylabel('Number of Reviews')
axes[1].set_title('Distribution of TextBlob Polarity Scores')
axes[1].legend()
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'sentiment_distribution.png'), dpi=150, bbox_inches='tight')
plt.close()
print('Saved: sentiment_distribution.png')

# --- SENTIMENT BY CATEGORY ---
apps_for_join = apps_clean[['App', 'Category']].drop_duplicates(subset='App')
reviews_with_cat = reviews_clean.merge(apps_for_join, on='App', how='left')
reviews_cat = reviews_with_cat.dropna(subset=['Category', 'TB_Sentiment'])
reviews_cat = reviews_cat[reviews_cat['TB_Sentiment'] != 'Unknown']
sentiment_cat = reviews_cat.groupby(['Category', 'TB_Sentiment']).size().unstack(fill_value=0)
sentiment_cat = sentiment_cat[sentiment_cat.sum(axis=1) >= 50]
sentiment_cat_pct = sentiment_cat.div(sentiment_cat.sum(axis=1), axis=0) * 100
print(f'Categories with >= 50 reviews: {len(sentiment_cat)}')

fig, ax = plt.subplots(figsize=(14, 8))
cols_to_plot = [c for c in ['Positive', 'Neutral', 'Negative'] if c in sentiment_cat_pct.columns]
plot_colors = [{'Positive': '#4CAF50', 'Neutral': '#FFC107', 'Negative': '#F44336'}[c] for c in cols_to_plot]
sentiment_cat_pct[cols_to_plot].plot(kind='barh', stacked=True, ax=ax, color=plot_colors)
ax.set_xlabel('Percentage of Reviews')
ax.set_title('Sentiment Distribution by Category (categories with >= 50 reviews)', fontsize=14)
ax.set_ylabel('Category')
ax.legend(title='Sentiment', bbox_to_anchor=(1.01, 1), loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'sentiment_by_category.png'), dpi=150, bbox_inches='tight')
plt.close()
print('Saved: sentiment_by_category.png')

# --- INTERACTIVE PLOTLY ---
category_summary = apps_clean.groupby('Category').agg(
    App_Count=('App', 'count'),
    Avg_Rating=('Rating', 'mean'),
    Avg_Installs=('Installs_Clean', 'mean')
).reset_index().dropna()
category_summary = category_summary.sort_values('App_Count', ascending=False)

fig_plotly = px.scatter(
    category_summary,
    x='App_Count',
    y='Avg_Rating',
    size='Avg_Installs',
    color='Category',
    hover_name='Category',
    hover_data={'App_Count': True, 'Avg_Rating': ':.2f', 'Avg_Installs': ':.0f', 'Category': False},
    title='Google Play Store: Category Landscape<br><sup>X=App Count | Y=Avg Rating | Bubble Size=Avg Installs</sup>',
    labels={'App_Count': 'Number of Apps', 'Avg_Rating': 'Average Rating'},
    height=600
)
fig_plotly.update_layout(showlegend=False, plot_bgcolor='white')
fig_plotly.write_html(os.path.join(FIGURES_DIR, 'interactive_category_landscape.html'))
print('Saved: interactive_category_landscape.html')

top_cats = category_summary.head(20)
fig_bar = px.bar(
    top_cats,
    x='App_Count',
    y='Category',
    orientation='h',
    color='Avg_Rating',
    color_continuous_scale='RdYlGn',
    hover_data={'App_Count': True, 'Avg_Rating': ':.2f'},
    title='Top 20 Most Competitive Categories (colored by Avg Rating)',
    labels={'App_Count': 'Number of Apps', 'Category': 'Category'},
    height=600
)
fig_bar.update_layout(yaxis={'categoryorder': 'total ascending'})
fig_bar.write_html(os.path.join(FIGURES_DIR, 'interactive_top_categories.html'))
print('Saved: interactive_top_categories.html')

# --- MERGED DATASET ---
merged = apps_clean.merge(reviews_clean[['App', 'TB_Sentiment', 'TB_Polarity', 'TB_Subjectivity']],
                          on='App', how='left')
merged.to_csv(os.path.join(CLEANED_DIR, 'merged_data.csv'), index=False)
print(f'Saved merged_data.csv: {merged.shape}')

# --- SUMMARY REPORT ---
print('\n' + '='*60)
print('ANALYSIS COMPLETE - SUMMARY')
print('='*60)
print(f'Apps cleaned: {len(apps_clean):,}')
print(f'Categories: {apps_clean["Category"].nunique()}')
print(f'Reviews cleaned: {len(reviews_clean):,}')
print(f'Overall avg rating: {apps_clean["Rating"].mean():.2f}')
type_pct = apps_clean['Type'].value_counts(normalize=True)*100
print(f'Free apps: {type_pct.get("Free", 0):.1f}%  Paid apps: {type_pct.get("Paid", 0):.1f}%')
sentiment_pct = reviews_clean['TB_Sentiment'].value_counts(normalize=True)*100
for s, p in sentiment_pct.items():
    print(f'  {s}: {p:.1f}%')
print('\nFiles generated:')
for fname in sorted(os.listdir(FIGURES_DIR)):
    fpath = os.path.join(FIGURES_DIR, fname)
    fsize = os.path.getsize(fpath)
    print(f'  outputs/figures/{fname}  ({fsize:,} bytes)')
for fname in sorted(os.listdir(CLEANED_DIR)):
    fpath = os.path.join(CLEANED_DIR, fname)
    fsize = os.path.getsize(fpath)
    print(f'  outputs/cleaned_data/{fname}  ({fsize:,} bytes)')
print('='*60)
