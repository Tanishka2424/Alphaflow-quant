"""
conftest.py
-----------
Pytest configuration file.
Adds the project root to sys.path so all test files can
import from src/ and config.py without path gymnastics.

Location: F:/commodity_trading_project/conftest.py  (project root)
"""

import sys
import os

# Insert project root at the front of the path
sys.path.insert(0, os.path.dirname(__file__))
