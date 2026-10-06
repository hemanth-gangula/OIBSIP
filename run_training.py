"""
Runner script — executed from the project root so relative imports resolve.
"""
import sys
import os

# Add src/ to path
src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
sys.path.insert(0, src_dir)

from train_models import main
main()
