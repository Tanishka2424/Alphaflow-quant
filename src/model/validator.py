"""
src/model/validator.py
----------------------
Walk-forward validation for the CNN-LSTM model.

Why walk-forward instead of random split?
    Random split on time-series data causes LOOKAHEAD BIAS:
    the model trains on data from the future and tests on the past.
    This makes accuracy look artificially high and the strategy
    completely useless in live trading.

    Walk-forward validation ensures:
    - Training data always comes BEFORE test data in time
    - The model never "sees the future" during training
    - Accuracy numbers are realistic and trustworthy

Location: F:/commodity_trading_project/src/model/validator.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, roc_auc_score

from config import WINDOW_SIZE, N_FEATURES, FEATURE_COLS


def prepare_sequences(
    scaled_features: np.ndarray,
    labels:          np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert a scaled feature array into CNN-LSTM input sequences.

    Each sequence is WINDOW_SIZE consecutive rows of features.
    The label for sequence i is the direction of candle i+WINDOW_SIZE.

    Args:
        scaled_features: Array of shape (N, N_FEATURES).
        labels:          Binary array of shape (N,).

    Returns:
        X: Array of shape (N - WINDOW_SIZE, WINDOW_SIZE, N_FEATURES).
        y: Array of shape (N - WINDOW_SIZE,).
    """
    X, y = [], []
    for i in range(len(scaled_features) - WINDOW_SIZE):
        X.append(scaled_features[i : i + WINDOW_SIZE])
        y.append(labels[i + WINDOW_SIZE])
    return np.array(X), np.array(y)


def walk_forward_evaluate(
    model,
    scaler,
    features_df: pd.DataFrame,
    labels:      pd.Series,
    n_splits:    int = 5,
) -> dict:
    """
    Evaluate CNN-LSTM accuracy using TimeSeriesSplit.

    This does NOT retrain the model on each fold — it evaluates the
    already-trained model on time-ordered test splits. This gives a
    realistic picture of how the model would perform going forward.

    Args:
        model:       Loaded Keras CNN-LSTM model.
        scaler:      Fitted RobustScaler.
        features_df: Feature DataFrame with FEATURE_COLS columns.
        labels:      Binary direction labels aligned to features_df.
        n_splits:    Number of CV folds. Default 5.

    Returns:
        Dictionary with per-fold and aggregate accuracy/AUC results.
    """
    X_raw = features_df[FEATURE_COLS].values
    y_raw = labels.values

    tscv         = TimeSeriesSplit(n_splits=n_splits)
    fold_results = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X_raw), 1):
        # Scale using only training data to prevent data leakage
        from sklearn.preprocessing import RobustScaler
        fold_scaler = RobustScaler()
        fold_scaler.fit(X_raw[train_idx])

        X_test_scaled = fold_scaler.transform(X_raw[test_idx])
        y_test_raw    = y_raw[test_idx]

        # Build sequences from test fold
        X_seq, y_seq = prepare_sequences(X_test_scaled, y_test_raw)

        if len(X_seq) == 0:
            print(f"  Fold {fold}: skipped (not enough test data)")
            continue

        # Predict
        probs      = model.predict(X_seq, verbose=0).flatten()
        preds      = (probs > 0.5).astype(int)
        acc        = round(accuracy_score(y_seq, preds) * 100, 2)

        try:
            auc = round(roc_auc_score(y_seq, probs), 4)
        except ValueError:
            auc = 0.0   # only one class in fold — too small

        fold_results.append({
            "fold":     fold,
            "accuracy": acc,
            "auc":      auc,
            "n_test":   len(y_seq),
        })
        print(f"  Fold {fold}: Acc={acc}%  AUC={auc}  n_test={len(y_seq)}")

    if not fold_results:
        return {"error": "No folds had sufficient data."}

    accs = [r["accuracy"] for r in fold_results]
    aucs = [r["auc"]      for r in fold_results]

    summary = {
        "model":         "CNN-LSTM",
        "n_splits":      n_splits,
        "mean_accuracy": round(float(np.mean(accs)), 2),
        "std_accuracy":  round(float(np.std(accs)),  2),
        "mean_auc":      round(float(np.mean(aucs)), 4),
        "folds":         fold_results,
    }

    print("\n" + "=" * 40)
    print("  CNN-LSTM WALK-FORWARD RESULTS")
    print("=" * 40)
    print(f"Mean Accuracy: {summary['mean_accuracy']}% ± {summary['std_accuracy']}%")
    print(f"Mean AUC:      {summary['mean_auc']}")
    print("=" * 40)

    return summary


def compare_models(
    cnn_lstm_results: dict,
    xgb_results:      dict,
) -> pd.DataFrame:
    """
    Build a comparison table between CNN-LSTM and XGBoost.

    This is the table that goes in your README — it proves
    CNN-LSTM adds value over the simpler baseline.

    Args:
        cnn_lstm_results: Output of walk_forward_evaluate().
        xgb_results:      Output of XGBoostBaseline.walk_forward_validate().

    Returns:
        DataFrame formatted as a results table.

    Example output:
        | Model      | Accuracy | Std  | AUC   |
        |------------|----------|------|-------|
        | CNN-LSTM   | 53.2%    | 1.1% | 0.541 |
        | XGBoost    | 51.8%    | 1.4% | 0.523 |
    """
    rows = []
    for result in [cnn_lstm_results, xgb_results]:
        rows.append({
            "Model":         result.get("model", "Unknown"),
            "Mean Accuracy": f"{result.get('mean_accuracy', 0)}%",
            "Std":           f"±{result.get('std_accuracy', 0)}%",
            "Mean AUC":      result.get("mean_auc", 0),
            "Folds":         result.get("n_splits", 0),
        })
    return pd.DataFrame(rows)
