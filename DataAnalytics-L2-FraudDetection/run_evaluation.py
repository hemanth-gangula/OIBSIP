"""
Runner script for evaluation pipeline.
Execute from the project root directory.
"""
import sys
import os

# Add src/ to path so evaluate_models can import preprocess
src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
sys.path.insert(0, src_dir)

from evaluate_models import main
main()
