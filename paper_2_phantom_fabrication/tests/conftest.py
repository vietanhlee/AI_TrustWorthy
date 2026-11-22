# -*- coding: utf-8 -*-
"""
Pytest configuration for Paper 2 test suite.
Ensures repository root is always in sys.path and Intel OpenMP duplicate library check is handled.
"""

import os
import sys
from pathlib import Path

# Fix Intel OpenMP duplicate runtime on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Ensure repo root is on sys.path
TEST_DIR = Path(__file__).resolve().parent
PAPER2_DIR = TEST_DIR.parent
REPO_ROOT = PAPER2_DIR.parent

for p in [str(REPO_ROOT), str(PAPER2_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)
