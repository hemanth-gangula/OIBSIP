"""
app.py
------
Flask backend for the Google Play Store Analytics application.
Oasis Infobyte – Data Analytics Internship – Level 2 Task 4

Architecture notes for Vercel serverless deployment:
  - No global in-memory state (each Lambda invocation is isolated)
  - Uploads go to /tmp (the only writable path on serverless)
  - The /analyse route receives both DataFrames via the session store
    which is backed by signed cookies (client-side), keeping the
    full analysis self-contained in a single POST chain per session
  - Matplotlib uses the Agg backend (no display server needed)
"""

import os
import json
import traceback
import uuid
import tempfile

import matplotlib
matplotlib.use('Agg')   # must be set before any other matplotlib import

from flask import (Flask, render_template, request, jsonify, session)
from werkzeug.utils import secure_filename

import data_processor as dp
import sentiment_analyzer as sa
import visualizations as viz

# ─────────────────────────────────────────────
# APP CONFIG
# ─────────────────────────────────────────────
app = Flask(__name__)

# Secret key from environment variable (set in Vercel dashboard) or random fallback
app.secret_key = os.environ.get('SECRET_KEY', os.urandom(32))

# Make Python builtins available in Jinja2 templates
app.jinja_env.globals.update(enumerate=enumerate, zip=zip, len=len, int=int, round=round)

# Use /tmp on serverless, local uploads/ otherwise
UPLOAD_FOLDER = os.environ.get('UPLOAD_FOLDER',
                               os.path.join(tempfile.gettempdir(), 'gps_uploads'))
ALLOWED_EXTS  = {'csv', 'xlsx', 'xls'}
MAX_CONTENT_MB = 50   # reduced from 100 MB for serverless

app.config['UPLOAD_FOLDER']      = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_MB * 1024 * 1024

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ── In-process store (keyed by session id) ────────────────
# On Vercel each worker process is isolated; within a single
# warm worker the dict persists, bridging the upload→analyse
# chain.  If the worker is recycled the user must re-upload
# (the UI gracefully handles this via the status endpoint).
_store: dict = {}


def _allowed(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTS


def _sid() -> str:
    if 'sid' not in session:
        session['sid'] = str(uuid.uuid4())
    return session['sid']


# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/dashboard')
def dashboard():
    sid  = _sid()
    data = _store.get(sid, {})
    if not data.get('analysis_ready'):
        return render_template('index.html',
                               error="Please upload and process both datasets first.")
    return render_template('dashboard.html', analysis=data)


# ── Upload Apps ──────────────────────────────

@app.route('/upload/apps', methods=['POST'])
def upload_apps():
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file part in request.'}), 400

    f = request.files['file']
    if not f or f.filename == '':
        return jsonify({'success': False, 'error': 'No file selected.'}), 400
    if not _allowed(f.filename):
        return jsonify({'success': False,
                        'error': f'Unsupported format. Use: {", ".join(sorted(ALLOWED_EXTS))}'}), 400

    df, err = dp.load_dataframe(f)
    if err:
        return jsonify({'success': False, 'error': err}), 400

    valid, msg = dp.validate_apps_dataset(df)
    if not valid:
        return jsonify({'success': False, 'error': msg}), 400

    sid = _sid()
    _store.setdefault(sid, {})
    _store[sid]['apps_raw']       = df
    _store[sid]['analysis_ready'] = False

    info = dp.get_dataset_info(df)
    return jsonify({
        'success':  True,
        'message':  msg,
        'filename': secure_filename(f.filename),
        'filetype': f.filename.rsplit('.', 1)[1].upper(),
        'rows':     info['rows'],
        'cols':     info['cols'],
        'columns':  info['columns'],
        'preview':  info['preview'],
    })


# ── Upload Reviews ───────────────────────────

@app.route('/upload/reviews', methods=['POST'])
def upload_reviews():
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file part in request.'}), 400

    f = request.files['file']
    if not f or f.filename == '':
        return jsonify({'success': False, 'error': 'No file selected.'}), 400
    if not _allowed(f.filename):
        return jsonify({'success': False,
                        'error': f'Unsupported format. Use: {", ".join(sorted(ALLOWED_EXTS))}'}), 400

    df, err = dp.load_dataframe(f)
    if err:
        return jsonify({'success': False, 'error': err}), 400

    valid, msg = dp.validate_reviews_dataset(df)
    if not valid:
        return jsonify({'success': False, 'error': msg}), 400

    sid = _sid()
    _store.setdefault(sid, {})
    _store[sid]['reviews_raw']    = df
    _store[sid]['analysis_ready'] = False

    info = dp.get_dataset_info(df)
    return jsonify({
        'success':  True,
        'message':  msg,
        'filename': secure_filename(f.filename),
        'filetype': f.filename.rsplit('.', 1)[1].upper(),
        'rows':     info['rows'],
        'cols':     info['cols'],
        'columns':  info['columns'],
        'preview':  info['preview'],
    })


# ── Run Analysis ─────────────────────────────

@app.route('/analyse', methods=['POST'])
def analyse():
    sid   = _sid()
    store = _store.get(sid, {})

    if 'apps_raw' not in store:
        return jsonify({'success': False,
                        'error': 'Apps dataset not found. Please re-upload — the server '
                                 'may have restarted since your upload.'}), 400
    if 'reviews_raw' not in store:
        return jsonify({'success': False,
                        'error': 'Reviews dataset not found. Please re-upload — the server '
                                 'may have restarted since your upload.'}), 400

    try:
        # 1. Clean
        apps_clean,    apps_log    = dp.clean_apps_dataset(store['apps_raw'])
        reviews_clean, reviews_log = dp.clean_reviews_dataset(store['reviews_raw'])

        # 2. Sentiment analysis
        sentiment_result = sa.analyse_sentiment(reviews_clean)
        reviews_enriched = sentiment_result.pop('enriched_df')

        # 3. Core analyses
        category_data  = dp.analyse_categories(apps_clean)
        rating_data    = dp.analyse_ratings(apps_clean)
        size_inst_data = dp.analyse_size_installs(apps_clean)
        pricing_data   = dp.analyse_pricing(apps_clean)
        sentiment_cat  = dp.analyse_sentiment_by_category(apps_clean, reviews_enriched)

        # 4. Developer insights
        insights = dp.generate_developer_insights(
            apps_clean, reviews_enriched,
            pricing_data, rating_data, category_data
        )

        # 5. Static charts (base64 PNG)
        charts = {
            'category_dist':  viz.chart_category_distribution(category_data),
            'rating_dist':    viz.chart_rating_distribution(apps_clean),
            'avg_rating_cat': viz.chart_avg_rating_by_category(rating_data),
            'size_installs':  viz.chart_size_vs_installs(size_inst_data),
            'free_paid':      viz.chart_free_vs_paid(pricing_data),
            'price_dist':     viz.chart_price_distribution(pricing_data),
            'revenue_cat':    viz.chart_revenue_by_category(pricing_data),
            'sentiment_dist': viz.chart_sentiment_distribution(sentiment_result),
            'sentiment_cat':  viz.chart_sentiment_by_category(sentiment_cat),
        }

        # 6. Interactive Plotly charts
        plotly_charts = viz.chart_interactive_plotly(
            apps_clean, category_data, rating_data,
            pricing_data, sentiment_result
        )

        # 7. KPIs (all dynamic — never hardcoded)
        kpis = {
            'total_apps':         int(len(apps_clean)),
            'total_categories':   category_data['total_categories'],
            'avg_rating':         rating_data['avg_overall'],
            'total_reviews_apps': int(apps_clean['Reviews'].sum())
                                  if 'Reviews' in apps_clean.columns else 0,
            'total_installs':     int(apps_clean['Installs_Numeric'].sum())
                                  if 'Installs_Numeric' in apps_clean.columns else 0,
            'free_pct':           pricing_data['free_pct'],
            'paid_pct':           pricing_data['paid_pct'],
            'total_user_reviews': sentiment_result['total_reviews'],
            'positive_pct':       sentiment_result['positive_pct'],
            'negative_pct':       sentiment_result['negative_pct'],
            'neutral_pct':        sentiment_result['neutral_pct'],
        }

        # 8. Persist result in process store
        _store[sid].update({
            'analysis_ready': True,
            'apps_log':       apps_log,
            'reviews_log':    reviews_log,
            'kpis':           kpis,
            'category_data':  category_data,
            'rating_data':    rating_data,
            'size_inst_data': size_inst_data,
            'pricing_data':   pricing_data,
            'sentiment_data': sentiment_result,
            'sentiment_cat':  sentiment_cat,
            'charts':         charts,
            'plotly_charts':  plotly_charts,
            'insights':       insights,
            'engine_used':    sentiment_result.get('engine_used', 'N/A'),
        })

        return jsonify({'success': True, 'redirect': '/dashboard'})

    except Exception:
        tb = traceback.format_exc()
        app.logger.error('Analysis error:\n%s', tb)
        return jsonify({'success': False,
                        'error': f'Analysis failed. Check server logs for details.\n{tb}'}), 500


# ── Status check ─────────────────────────────

@app.route('/status')
def status():
    sid   = _sid()
    store = _store.get(sid, {})
    return jsonify({
        'apps_uploaded':    'apps_raw'    in store,
        'reviews_uploaded': 'reviews_raw' in store,
        'analysis_ready':   store.get('analysis_ready', False),
    })


# ── Health check (Vercel ping) ────────────────

@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'service': 'Google Play Store Analytics'})


# ─────────────────────────────────────────────
# WSGI ENTRY POINT
# ─────────────────────────────────────────────
# Vercel imports this module and calls `app` as the WSGI callable.
# The `if __name__ == '__main__'` block is only used for local dev.

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
