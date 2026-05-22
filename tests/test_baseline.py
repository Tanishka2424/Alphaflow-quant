"""
tests/test_baseline.py
----------------------
Tests for XGBoostBaseline — training, walk-forward, feature importance.
No internet or TensorFlow needed.

Run with:
    pytest tests/test_baseline.py -v

Location: F:/commodity_trading_project/tests/test_baseline.py
"""

import sys
import os
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.model.baseline import XGBoostBaseline


# ---------------------------------------------------------------------------
# Fixture — synthetic features + labels
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_data():
    """200 rows of synthetic feature data with binary labels."""
    np.random.seed(42)
    n = 200
    from config import FEATURE_COLS
    df = pd.DataFrame(
        np.random.randn(n, len(FEATURE_COLS)),
        columns=FEATURE_COLS,
    )
    labels = pd.Series(np.random.randint(0, 2, n))
    return df, labels


@pytest.fixture
def trained_baseline(synthetic_data):
    features_df, labels = synthetic_data
    model = XGBoostBaseline()
    model.train(features_df, labels)
    return model, features_df, labels


# ---------------------------------------------------------------------------
# make_labels
# ---------------------------------------------------------------------------

def test_make_labels_binary(synthetic_data):
    features_df, _ = synthetic_data
    close = pd.Series(np.linspace(75, 80, len(features_df)))
    df    = pd.DataFrame({"Close": close})
    labels = XGBoostBaseline.make_labels(df)
    assert set(labels.dropna().unique()).issubset({0, 1})


def test_make_labels_last_row_nan(synthetic_data):
    features_df, _ = synthetic_data
    close  = pd.Series(np.linspace(75, 80, len(features_df)))
    df     = pd.DataFrame({"Close": close})
    labels = XGBoostBaseline.make_labels(df)
    assert pd.isna(labels.iloc[-1])


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------

def test_train_sets_model(synthetic_data):
    features_df, labels = synthetic_data
    b = XGBoostBaseline()
    b.train(features_df, labels)
    assert b.model is not None


# ---------------------------------------------------------------------------
# predict_proba
# ---------------------------------------------------------------------------

def test_predict_proba_in_range(trained_baseline):
    model, features_df, _ = trained_baseline
    row  = features_df.iloc[[0]]
    prob = model.predict_proba(row)
    assert 0.0 <= prob <= 1.0


def test_predict_proba_returns_float(trained_baseline):
    model, features_df, _ = trained_baseline
    prob = model.predict_proba(features_df.iloc[[0]])
    assert isinstance(prob, float)


# ---------------------------------------------------------------------------
# get_feature_importance
# ---------------------------------------------------------------------------

def test_feature_importance_has_all_features(trained_baseline):
    from config import FEATURE_COLS
    model, _, _ = trained_baseline
    importance  = model.get_feature_importance()
    for feat in FEATURE_COLS:
        assert feat in importance


def test_feature_importance_values_sum_to_one(trained_baseline):
    model, _, _ = trained_baseline
    importance  = model.get_feature_importance()
    total       = sum(importance.values())
    assert total == pytest.approx(1.0, abs=0.05)


# ---------------------------------------------------------------------------
# walk_forward_validate
# ---------------------------------------------------------------------------

def test_walk_forward_returns_dict(synthetic_data):
    features_df, labels = synthetic_data
    b      = XGBoostBaseline()
    result = b.walk_forward_validate(features_df, labels, n_splits=3)
    assert isinstance(result, dict)


def test_walk_forward_has_required_keys(synthetic_data):
    features_df, labels = synthetic_data
    b      = XGBoostBaseline()
    result = b.walk_forward_validate(features_df, labels, n_splits=3)
    for key in ["mean_accuracy", "std_accuracy", "mean_auc", "folds"]:
        assert key in result


def test_walk_forward_accuracy_in_range(synthetic_data):
    features_df, labels = synthetic_data
    b      = XGBoostBaseline()
    result = b.walk_forward_validate(features_df, labels, n_splits=3)
    assert 0 <= result["mean_accuracy"] <= 100


def test_walk_forward_correct_fold_count(synthetic_data):
    features_df, labels = synthetic_data
    b      = XGBoostBaseline()
    result = b.walk_forward_validate(features_df, labels, n_splits=3)
    assert len(result["folds"]) == 3
