"""
app.py — Flask web application for the Autocomplete & Autocorrect NLP project.

Provides page routes and JSON API endpoints that delegate to the existing
src/ NLP modules.  Model data is loaded once from data/processed/model_data.json
and cached as a module-level singleton.
"""

import sys
import os

# ---------------------------------------------------------------------------
# Ensure src/ is on the path before any src imports
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import json
import re
import math
import logging
from collections import Counter
from typing import Any, Dict, Optional

from flask import (
    Flask,
    jsonify,
    render_template,
    request,
    send_file,
    send_from_directory,
    abort,
)
from flask_cors import CORS

# src/ imports (available after sys.path manipulation above)
from autocomplete import BigramModel, TrigramModel, get_autocomplete_suggestions
from autocorrect import EditDistanceCorrector, ProbabilisticCorrector

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = Flask(
    __name__,
    template_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates"),
    static_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)), "public"),
    static_url_path="/static",
)
CORS(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Project root (absolute, so paths work regardless of cwd)
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
_models: Optional[Dict[str, Any]] = None


def load_models() -> Dict[str, Any]:
    """
    Load and reconstruct all NLP models from model_data.json.

    Returns a dict with keys:
        vocab            – {word: freq}
        bigram_model     – fitted BigramModel
        trigram_model    – fitted TrigramModel
        edit_corrector   – EditDistanceCorrector
        prob_corrector   – ProbabilisticCorrector

    The result is cached in the module-level _models singleton; subsequent
    calls return the cached dict immediately.
    """
    global _models
    if _models is not None:
        return _models

    try:
        model_path = os.path.join(PROJECT_ROOT, "data", "processed", "model_data.json")
        with open(model_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        vocab: Dict[str, int] = data["vocab"]          # {word: freq}
        bigram_json: Dict[str, list] = data["bigram"]  # {word: [[next, prob], ...]}
        trigram_json: Dict[str, list] = data["trigram"] # {w1_w2: [[w3, prob], ...]}

        # ------------------------------------------------------------------
        # Reconstruct BigramModel
        # ------------------------------------------------------------------
        bigram_model = BigramModel()
        bigram_model.unigram_counts = Counter(vocab)

        bigram_counts: Dict[tuple, int] = {}
        for word, items in bigram_json.items():
            word_freq = vocab.get(word, 1)
            for item in items:
                next_word, prob = item[0], item[1]
                # Approximate count = prob * unigram_freq, minimum 1
                approx = max(1, int(prob * word_freq))
                bigram_counts[(word, next_word)] = approx
        bigram_model.bigram_counts = Counter(bigram_counts)
        bigram_model.fitted = True

        # ------------------------------------------------------------------
        # Reconstruct TrigramModel
        # ------------------------------------------------------------------
        trigram_model = TrigramModel()

        # Trigram bigram_counts (the (w1,w2) context counts)
        # We approximate using bigram probabilities: count(w1,w2) ≈ prob(w2|w1) * freq(w1)
        # The simplest approach: reuse bigram_counts from above as-is and
        # additionally build from trigram keys for contexts not seen in bigrams.
        trigram_bigram_counts: Dict[tuple, int] = dict(bigram_counts)

        trigram_counts: Dict[tuple, int] = {}
        for key, items in trigram_json.items():
            # key is 'w1_w2' — split on first underscore only
            parts = key.split("_", 1)
            if len(parts) != 2:
                continue
            w1, w2 = parts[0], parts[1]
            context_key = (w1, w2)
            # Ensure context exists; approximate w1_w2 count as 100 (relative scale)
            if context_key not in trigram_bigram_counts:
                trigram_bigram_counts[context_key] = 100

            ctx_count = trigram_bigram_counts[context_key]
            for item in items:
                w3, prob = item[0], item[1]
                approx = max(1, int(prob * ctx_count))
                trigram_counts[(w1, w2, w3)] = approx

        trigram_model.bigram_counts = Counter(trigram_bigram_counts)
        trigram_model.trigram_counts = Counter(trigram_counts)
        trigram_model.bigram_model = bigram_model
        trigram_model.fitted = True

        # ------------------------------------------------------------------
        # Correctors
        # ------------------------------------------------------------------
        edit_corrector = EditDistanceCorrector(vocab)
        prob_corrector = ProbabilisticCorrector(vocab)

        _models = {
            "vocab": vocab,
            "bigram_model": bigram_model,
            "trigram_model": trigram_model,
            "edit_corrector": edit_corrector,
            "prob_corrector": prob_corrector,
        }
        logger.info(
            "Models loaded: vocab=%d, bigrams=%d, trigrams=%d",
            len(vocab),
            len(bigram_model.bigram_counts),
            len(trigram_model.trigram_counts),
        )
        return _models

    except Exception as exc:
        logger.error("Failed to load models: %s", exc)
        raise


# ---------------------------------------------------------------------------
# Preload models at startup (non-fatal; individual routes handle 503)
# ---------------------------------------------------------------------------
try:
    load_models()
except Exception:
    logger.warning("Model preload failed; will retry on first request.")


# ---------------------------------------------------------------------------
# Page routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/autocomplete")
def autocomplete_page():
    return render_template("autocomplete.html")


@app.route("/autocorrect")
def autocorrect_page():
    return render_template("autocorrect.html")


@app.route("/analytics")
def analytics_page():
    return render_template("analytics.html")


@app.route("/models")
def models_page():
    return render_template("models.html")


@app.route("/about")
def about_page():
    return render_template("about.html")


@app.route("/health")
def health():
    models_loaded = _models is not None
    return jsonify({"status": "ok", "models_loaded": models_loaded})


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def _models_or_503():
    """Return loaded models dict, or raise a 503-carrying tuple."""
    try:
        return load_models(), None
    except Exception as exc:
        return None, (jsonify({"error": "Models not available", "detail": str(exc)}), 503)


def _validate_text(body: dict, key: str, max_len: int = 500):
    """Extract and validate a text field from request JSON.

    Returns (value, error_response) where error_response is None on success.
    """
    if not body or key not in body:
        return None, (jsonify({"error": f"Missing required field: '{key}'"}), 400)
    value = str(body[key]).strip()
    if not value:
        return None, (jsonify({"error": f"'{key}' must not be empty"}), 400)
    if len(value) > max_len:
        return None, (
            jsonify({"error": f"'{key}' exceeds maximum length of {max_len} characters"}),
            400,
        )
    return value, None


# ---------------------------------------------------------------------------
# POST /api/autocomplete
# ---------------------------------------------------------------------------

@app.route("/api/autocomplete", methods=["POST"])
def api_autocomplete():
    """
    Generate next-word autocomplete suggestions.

    Request JSON:
        text    (str, required)         : input text, max 500 chars
        model   (str, optional)         : 'bigram' | 'trigram' | 'both' (default 'both')
        top_k   (int, optional)         : 1-10 (default 3)

    Response JSON:
        bigram          : [{word, score, rank}, ...]
        trigram         : [{word, score, rank}, ...]
        context_words   : [word, ...]
        model_used      : str
    """
    try:
        models, err = _models_or_503()
        if err:
            return err

        body = request.get_json(silent=True) or {}

        # Validate text
        text, err = _validate_text(body, "text", 500)
        if err:
            return err

        # Validate model param
        model_choice = str(body.get("model", "both")).strip().lower()
        if model_choice not in ("bigram", "trigram", "both"):
            return jsonify({"error": "model must be 'bigram', 'trigram', or 'both'"}), 400

        # Validate top_k
        try:
            top_k = int(body.get("top_k", 3))
        except (TypeError, ValueError):
            return jsonify({"error": "top_k must be an integer"}), 400
        if not (1 <= top_k <= 10):
            return jsonify({"error": "top_k must be between 1 and 10"}), 400

        context_words = re.findall(r"[a-z]+", text.lower())

        suggestions = get_autocomplete_suggestions(
            text,
            models["bigram_model"],
            models["trigram_model"],
            top_k=top_k,
        )

        def format_preds(preds):
            return [
                {"word": word, "score": round(score, 6), "rank": rank + 1}
                for rank, (word, score) in enumerate(preds)
            ]

        bigram_results = format_preds(suggestions["bigram"]) if model_choice in ("bigram", "both") else []
        trigram_results = format_preds(suggestions["trigram"]) if model_choice in ("trigram", "both") else []

        return jsonify(
            {
                "bigram": bigram_results,
                "trigram": trigram_results,
                "context_words": context_words,
                "model_used": model_choice,
            }
        )

    except Exception as exc:
        logger.error("Error in /api/autocomplete: %s", exc, exc_info=True)
        return jsonify({"error": "Internal server error"}), 500


# ---------------------------------------------------------------------------
# POST /api/autocorrect
# ---------------------------------------------------------------------------

@app.route("/api/autocorrect", methods=["POST"])
def api_autocorrect():
    """
    Suggest spelling corrections for a single word.

    Request JSON:
        word        (str, required)         : single word, max 100 chars
        approach    (str, optional)         : 'edit_distance' | 'probabilistic' | 'both' (default 'both')

    Response JSON:
        edit_distance   : [{word, edit_distance, frequency, rank}, ...]
        probabilistic   : [{word, probability, rank}, ...]
        input_word      : str
        is_correct      : bool
        approach_used   : str
    """
    try:
        models, err = _models_or_503()
        if err:
            return err

        body = request.get_json(silent=True) or {}

        # Validate word
        word, err = _validate_text(body, "word", 100)
        if err:
            return err

        # Must be a single lowercase alphabetic word
        if not re.fullmatch(r"[a-z]+", word.lower()):
            return jsonify({"error": "word must contain only alphabetic characters (a-z)"}), 400

        word = word.lower()

        # Validate approach
        approach = str(body.get("approach", "both")).strip().lower()
        if approach not in ("edit_distance", "probabilistic", "both"):
            return jsonify(
                {"error": "approach must be 'edit_distance', 'probabilistic', or 'both'"}
            ), 400

        ed_results = []
        prob_results = []
        is_correct = False

        if approach in ("edit_distance", "both"):
            raw = models["edit_corrector"].correct(word)
            for rank, (candidate, edit_dist, freq) in enumerate(raw):
                ed_results.append(
                    {
                        "word": candidate,
                        "edit_distance": edit_dist,
                        "frequency": freq,
                        "rank": rank + 1,
                    }
                )
            if ed_results and ed_results[0]["edit_distance"] == 0:
                is_correct = True

        if approach in ("probabilistic", "both"):
            raw = models["prob_corrector"].correct(word)
            for rank, (candidate, prob) in enumerate(raw):
                prob_results.append(
                    {
                        "word": candidate,
                        "probability": round(prob, 8),
                        "rank": rank + 1,
                    }
                )
            if not is_correct and prob_results and prob_results[0]["probability"] == 1.0:
                is_correct = True

        return jsonify(
            {
                "edit_distance": ed_results,
                "probabilistic": prob_results,
                "input_word": word,
                "is_correct": is_correct,
                "approach_used": approach,
            }
        )

    except Exception as exc:
        logger.error("Error in /api/autocorrect: %s", exc, exc_info=True)
        return jsonify({"error": "Internal server error"}), 500


# ---------------------------------------------------------------------------
# GET /api/metrics
# ---------------------------------------------------------------------------

@app.route("/api/metrics", methods=["GET"])
def api_metrics():
    """Return baseline evaluation metrics (previously reported results)."""
    return jsonify(
        {
            "autocomplete": {
                "bigram": {"top1": 0.305, "top3": 0.570, "mrr": 0.422},
                "trigram": {"top1": 0.650, "top3": 0.800, "mrr": 0.720},
            },
            "autocorrect": {
                "edit_distance": {"top1": 0.480, "top3": 0.580, "n_tested": 50},
                "probabilistic": {"top1": 0.480, "top3": 0.580, "n_tested": 50},
            },
        }
    )


# ---------------------------------------------------------------------------
# GET /api/analytics
# ---------------------------------------------------------------------------

@app.route("/api/analytics", methods=["GET"])
def api_analytics():
    """
    Return corpus analytics derived from the loaded vocabulary.

    Response JSON:
        top_20_words    : [{word, freq}, ...] (top 20 by frequency)
        vocab_stats     : {total_words, unique_words, avg_freq}
        freq_buckets    : {'1': count, '2-5': count, '6-20': count,
                           '21-100': count, '100+': count}
    """
    try:
        models, err = _models_or_503()
        if err:
            return err

        vocab = models["vocab"]

        # Top-20 words by frequency
        sorted_vocab = sorted(vocab.items(), key=lambda x: x[1], reverse=True)
        top_20 = [{"word": w, "freq": f} for w, f in sorted_vocab[:20]]

        # Vocab stats
        total_words = sum(vocab.values())
        unique_words = len(vocab)
        avg_freq = round(total_words / unique_words, 2) if unique_words else 0.0

        # Frequency distribution buckets
        buckets = {"1": 0, "2-5": 0, "6-20": 0, "21-100": 0, "100+": 0}
        for freq in vocab.values():
            if freq == 1:
                buckets["1"] += 1
            elif freq <= 5:
                buckets["2-5"] += 1
            elif freq <= 20:
                buckets["6-20"] += 1
            elif freq <= 100:
                buckets["21-100"] += 1
            else:
                buckets["100+"] += 1

        return jsonify(
            {
                "top_20_words": top_20,
                "vocab_stats": {
                    "total_words": total_words,
                    "unique_words": unique_words,
                    "avg_freq": avg_freq,
                },
                "freq_buckets": buckets,
            }
        )

    except Exception as exc:
        logger.error("Error in /api/analytics: %s", exc, exc_info=True)
        return jsonify({"error": "Internal server error"}), 500


# ---------------------------------------------------------------------------
# GET /api/download/<filename>
# ---------------------------------------------------------------------------

_ALLOWED_DOWNLOADS = {"autocomplete_predictions.csv", "autocorrect_results.csv"}


@app.route("/api/download/<filename>")
def api_download(filename: str):
    """Serve CSV prediction outputs. Only whitelisted filenames are allowed."""
    if filename not in _ALLOWED_DOWNLOADS:
        return jsonify({"error": "File not found"}), 404
    predictions_dir = os.path.join(PROJECT_ROOT, "outputs", "predictions")
    filepath = os.path.join(predictions_dir, filename)
    if not os.path.isfile(filepath):
        return jsonify({"error": "File not found"}), 404
    return send_file(filepath, as_attachment=True)


# ---------------------------------------------------------------------------
# Static asset fallback routes (local dev — Vercel CDN serves public/ directly)
# ---------------------------------------------------------------------------

@app.route("/css/<path:filename>")
def serve_css(filename: str):
    css_dir = os.path.join(PROJECT_ROOT, "public", "css")
    return send_from_directory(css_dir, filename)


@app.route("/js/<path:filename>")
def serve_js(filename: str):
    js_dir = os.path.join(PROJECT_ROOT, "public", "js")
    return send_from_directory(js_dir, filename)


# ---------------------------------------------------------------------------
# Error handlers
# ---------------------------------------------------------------------------

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Not found"}), 404


@app.errorhandler(500)
def internal_error(e):
    return jsonify({"error": "Internal server error"}), 500


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug)
