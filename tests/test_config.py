"""
tests/test_config.py
--------------------
Sanity checks on config.py to catch accidental misconfigurations.
These tests run instantly — no model, no network needed.

Run with:
    pytest tests/test_config.py -v

Location: F:/commodity_trading_project/tests/test_config.py
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


def test_supported_assets_not_empty():
    assert len(config.SUPPORTED_ASSETS) > 0


def test_all_assets_have_required_keys():
    for symbol, meta in config.SUPPORTED_ASSETS.items():
        assert "name"     in meta, f"{symbol} missing 'name'"
        assert "type"     in meta, f"{symbol} missing 'type'"
        assert "interval" in meta, f"{symbol} missing 'interval'"


def test_window_size_is_positive_int():
    assert isinstance(config.WINDOW_SIZE, int)
    assert config.WINDOW_SIZE > 0


def test_n_features_matches_feature_cols():
    """N_FEATURES must equal the length of FEATURE_COLS."""
    assert config.N_FEATURES == len(config.FEATURE_COLS)


def test_threshold_in_valid_range():
    assert 0.0 < config.DEFAULT_THRESHOLD < 1.0


def test_default_capital_positive():
    assert config.DEFAULT_CAPITAL > 0


def test_feature_cols_has_no_duplicates():
    assert len(config.FEATURE_COLS) == len(set(config.FEATURE_COLS))


def test_model_path_is_string():
    assert isinstance(config.MODEL_PATH, str)


def test_annualisation_factor_positive():
    assert config.ANNUALISATION_FACTOR > 0
