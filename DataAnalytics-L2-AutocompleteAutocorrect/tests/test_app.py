"""
tests/test_app.py — Flask test-client tests for app.py.

Covers page routes, API endpoints, validation, and error handling.
"""

import json
import pytest

from app import app as flask_app


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def client():
    flask_app.config["TESTING"] = True
    return flask_app.test_client()


# ---------------------------------------------------------------------------
# Page routes
# ---------------------------------------------------------------------------

class TestPageRoutes:
    def test_index_ok(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert b"Previously Reported" in resp.data

    def test_autocomplete_page_ok(self, client):
        resp = client.get("/autocomplete")
        assert resp.status_code == 200
        assert b"textarea" in resp.data

    def test_autocorrect_page_ok(self, client):
        resp = client.get("/autocorrect")
        assert resp.status_code == 200

    def test_analytics_page_ok(self, client):
        resp = client.get("/analytics")
        assert resp.status_code == 200

    def test_models_page_ok(self, client):
        resp = client.get("/models")
        assert resp.status_code == 200
        assert b"statistical" in resp.data

    def test_about_page_ok(self, client):
        resp = client.get("/about")
        assert resp.status_code == 200

    def test_health_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "ok"

    def test_nonexistent_404(self, client):
        resp = client.get("/nonexistent")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# POST /api/autocomplete
# ---------------------------------------------------------------------------

class TestApiAutocomplete:
    def test_valid_request_returns_bigram_key(self, client):
        resp = client.post(
            "/api/autocomplete",
            json={"text": "the quick"},
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert "bigram" in data

    def test_missing_body_returns_400(self, client):
        resp = client.post("/api/autocomplete", json={})
        assert resp.status_code == 400

    def test_empty_text_returns_400(self, client):
        resp = client.post("/api/autocomplete", json={"text": ""})
        assert resp.status_code == 400

    def test_text_too_long_returns_400(self, client):
        resp = client.post("/api/autocomplete", json={"text": "x" * 501})
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# POST /api/autocorrect
# ---------------------------------------------------------------------------

class TestApiAutocorrect:
    def test_valid_word_returns_200(self, client):
        resp = client.post(
            "/api/autocorrect",
            json={"word": "speling"},
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert "edit_distance" in data or "probabilistic" in data

    def test_missing_body_returns_400(self, client):
        resp = client.post("/api/autocorrect", json={})
        assert resp.status_code == 400

    def test_multi_word_returns_400(self, client):
        resp = client.post(
            "/api/autocorrect",
            json={"word": "hello world"},
        )
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# GET /api/metrics
# ---------------------------------------------------------------------------

class TestApiMetrics:
    def test_returns_200_with_expected_keys(self, client):
        resp = client.get("/api/metrics")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "autocomplete" in data
        assert "autocorrect" in data


# ---------------------------------------------------------------------------
# GET /api/analytics
# ---------------------------------------------------------------------------

class TestApiAnalytics:
    def test_returns_200_with_top_20_words(self, client):
        resp = client.get("/api/analytics")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "top_20_words" in data
        assert len(data["top_20_words"]) == 20
