"""
api/index.py
------------
Vercel serverless entry point.
Imports the Flask app from the Task 4 subfolder and exposes it as `app`
so Vercel's Python runtime can detect and serve it.
"""
import sys
import os

# Add the Task 4 project directory to sys.path so all imports resolve
_task4_dir = os.path.join(os.path.dirname(__file__), "..", "DataAnalytics-L2-GooglePlayStoreAnalysis")
sys.path.insert(0, os.path.abspath(_task4_dir))

# Set MPLCONFIGDIR before any matplotlib import
os.environ.setdefault("MPLCONFIGDIR", "/tmp")

from app import app  # noqa: F401 — Vercel detects `app` as the WSGI handler
