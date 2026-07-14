# ABOUTME: Pytest bootstrap so tests can import top-level repo modules (caft_pca, etc.).
# ABOUTME: Adds the repository root to sys.path.
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
