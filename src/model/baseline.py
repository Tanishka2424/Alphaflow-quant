"""
src/model/baseline.py
---------------------
XGBoost baseline model trained on the same features as CNN-LSTM.

Why a baseline?
    Without comparing against something simpler, we can't prove the
    CNN-LSTM is actually learning anything useful. A baseline that achieves
    51% accuracy makes a CNN-LSTM at 53% meaningful. Without it, the number
    is just a number.

    XGBoost is ideal as a baseline because:
    - It uses the same tabular features (no sequence/temporal structure)
    - It's fast to train and interpret
    - It's a strong model on its own (not a toy)

Location: F:/commodity_trading_project/src/model/baseline.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
)
import xgboost as xgb

from config import FEATURE_COLS


class XGBoostBaseline:
    """
    XGBoost classifier trained on flattened technical indicator features.

    Unlike CNN-LSTM which sees a sequence of 60 windows, XGBoost sees
    only the most recent row of features. This tests whether temporal
    context (sequence modelling) actually adds value.

    Attributes:
        model:     Trained XGBClassifier.
        cv_results: Walk-forward cross-validation results.
    """

    def __init__(self) -> None:
        self.model      = None
        self.cv_results = None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, features_df: pd.DataFrame, labels: pd.Series) -> None:
        """
        Train XGBoost on the full dataset.

        Args:
            features_df: DataFrame with FEATURE_COLS columns.
            labels:      Binary series — 1 if next candle UP, 0 if DOWN.
        """
        self.model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
            verbosity=0,
        )
        self.model.fit(features_df[FEATURE_COLS], labels)
        print("XGBoost baseline trained.")

    # ------------------------------------------------------------------
    # Walk-forward validation
    # ------------------------------------------------------------------

    def walk_forward_validate(
        self,
        features_df: pd.DataFrame,
        labels:      pd.Series,
        n_splits:    int = 5,
    ) -> dict:
        """
        Evaluate using TimeSeriesSplit — no lookahead bias.

        Standard train/test split on time-series data leaks future
        information into training (lookahead bias). TimeSeriesSplit
        ensures training always uses only past data to predict future.

        Fold structure (n_splits=5):
            Fold 1: Train [0:20%]   Test [20:40%]
            Fold 2: Train [0:40%]   Test [40:60%]
            Fold 3: Train [0:60%]   Test [60:80%]  ← grows forward
            ...

        Args:
            features_df: Feature DataFrame.
            labels:      Binary target series.
            n_splits:    Number of CV folds. Default 5.

        Returns:
            Dictionary with mean accuracy, AUC, precision, recall
            and per-fold breakdown.
        """
        tscv       = TimeSeriesSplit(n_splits=n_splits)
        X          = features_df[FEATURE_COLS].values
        y          = labels.values

        fold_results = []

        for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            model = xgb.XGBClassifier(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                use_label_encoder=False,
                eval_metric="logloss",
                random_state=42,
                verbosity=0,
            )
            model.fit(X_train, y_train)

            y_pred      = model.predict(X_test)
            y_pred_prob = model.predict_proba(X_test)[:, 1]

            fold_results.append({
                "fold":      fold,
                "accuracy":  round(accuracy_score(y_test, y_pred) * 100, 2),
                "auc":       round(roc_auc_score(y_test, y_pred_prob), 4),
                "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
                "recall":    round(recall_score(y_test, y_pred, zero_division=0), 4),
                "n_test":    len(y_test),
            })

            print(
                f"  Fold {fold}: "
                f"Acc={fold_results[-1]['accuracy']}%  "
                f"AUC={fold_results[-1]['auc']}  "
                f"n_test={len(y_test)}"
            )

        # Aggregate across folds
        accs  = [r["accuracy"]  for r in fold_results]
        aucs  = [r["auc"]       for r in fold_results]
        precs = [r["precision"] for r in fold_results]
        recs  = [r["recall"]    for r in fold_results]

        summary = {
            "model":            "XGBoost",
            "n_splits":         n_splits,
            "mean_accuracy":    round(float(np.mean(accs)),  2),
            "std_accuracy":     round(float(np.std(accs)),   2),
            "mean_auc":         round(float(np.mean(aucs)),  4),
            "mean_precision":   round(float(np.mean(precs)), 4),
            "mean_recall":      round(float(np.mean(recs)),  4),
            "folds":            fold_results,
        }

        self.cv_results = summary
        self._print_summary(summary)
        return summary

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_proba(self, features_row: pd.DataFrame) -> float:
        """
        Predict UP probability for a single row of features.

        Args:
            features_row: Single-row DataFrame with FEATURE_COLS.

        Returns:
            Float probability in [0, 1].
        """
        assert self.model is not None, "Call train() first."
        prob = self.model.predict_proba(
            features_row[FEATURE_COLS]
        )[0][1]
        return float(prob)

    def get_feature_importance(self) -> dict[str, float]:
        """
        Return XGBoost feature importances sorted descending.

        These are based on how often each feature is used in tree splits,
        weighted by improvement in the objective. Useful to compare
        against SHAP values from the CNN-LSTM.

        Returns:
            Dictionary mapping feature name → importance score.
        """
        assert self.model is not None, "Call train() first."
        scores = dict(zip(FEATURE_COLS, self.model.feature_importances_))
        total  = sum(scores.values()) or 1
        return {feat: round(val / total, 4) for feat, val in scores.items()}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def make_labels(full_df: pd.DataFrame) -> pd.Series:
        """
        Create binary direction labels from close prices.

        Label = 1 if next candle closes HIGHER than current candle.
        Label = 0 otherwise.

        Args:
            full_df: DataFrame with 'Close' column.

        Returns:
            Binary series aligned to full_df index (last row is NaN).
        """
        shifted = full_df["Close"].shift(-1)
        return (shifted > full_df["Close"]).where(shifted.notna()).astype("Int64")

    @staticmethod
    def _print_summary(summary: dict) -> None:
        print("\n" + "=" * 40)
        print("   XGBOOST WALK-FORWARD RESULTS")
        print("=" * 40)
        print(f"Folds:          {summary['n_splits']}")
        print(f"Mean Accuracy:  {summary['mean_accuracy']}% ± {summary['std_accuracy']}%")
        print(f"Mean AUC:       {summary['mean_auc']}")
        print(f"Mean Precision: {summary['mean_precision']}")
        print(f"Mean Recall:    {summary['mean_recall']}")
        print("=" * 40)
