"""
src/model/predictor.py
----------------------
Clean wrapper around the trained CNN-LSTM Keras model.
Handles loading, inference, and SHAP value computation.

Location: F:/commodity_trading_project/src/model/predictor.py
"""

from __future__ import annotations

import json
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from joblib import load

from config import (
    MODEL_PATH,
    SCALER_PATH,
    SHAP_PATH,
    FEATURE_COLS,
    WINDOW_SIZE,
    N_FEATURES,
)


class CNNLSTMPredictor:
    """
    Wrapper around the trained CNN-LSTM model.

    Responsibilities:
    - Load model and scaler from disk
    - Scale input windows
    - Run inference and return probability
    - Compute SHAP feature importance values
    - Cache SHAP results to disk so the dashboard loads fast

    Attributes:
        model:   Loaded Keras model.
        scaler:  Fitted RobustScaler.
        _shap_explainer: Cached SHAP DeepExplainer (lazy-loaded).
    """

    def __init__(
        self,
        model_path:  str = MODEL_PATH,
        scaler_path: str = SCALER_PATH,
    ) -> None:
        self.model_path  = model_path
        self.scaler_path = scaler_path
        self.model       = None
        self.scaler      = None
        self._shap_explainer = None

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load model weights and scaler from disk."""
        print(f"Loading model  → {self.model_path}")
        self.model  = tf.keras.models.load_model(self.model_path)
        print(f"Loading scaler → {self.scaler_path}")
        self.scaler = load(self.scaler_path)
        print("Model and scaler loaded successfully.")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_proba(self, window: np.ndarray) -> float:
        """
        Run inference on a single scaled window.

        Args:
            window: Numpy array of shape (WINDOW_SIZE, N_FEATURES).
                    Must already be scaled with self.scaler.

        Returns:
            Float probability in [0, 1].
            > 0.5 means model predicts price will go UP.
        """
        assert window.shape == (WINDOW_SIZE, N_FEATURES), (
            f"Expected window shape ({WINDOW_SIZE}, {N_FEATURES}), "
            f"got {window.shape}"
        )
        x    = window[np.newaxis, ...]          # shape: (1, 60, 8)
        prob = float(self.model.predict(x, verbose=0)[0][0])
        return prob

    def predict_batch(self, windows: np.ndarray) -> np.ndarray:
        """
        Run inference on multiple windows at once (faster than looping).

        Args:
            windows: Numpy array of shape (N, WINDOW_SIZE, N_FEATURES).

        Returns:
            1D numpy array of probabilities, shape (N,).
        """
        return self.model.predict(windows, verbose=0).flatten()

    def scale_window(self, raw_features: pd.DataFrame) -> np.ndarray:
        """
        Scale a features DataFrame using the fitted RobustScaler.

        Args:
            raw_features: DataFrame with FEATURE_COLS columns.
                          Must have at least WINDOW_SIZE rows.

        Returns:
            Scaled numpy array of shape (WINDOW_SIZE, N_FEATURES).
        """
        tail   = raw_features.tail(WINDOW_SIZE)
        scaled = self.scaler.transform(tail)
        return scaled

    # ------------------------------------------------------------------
    # SHAP explainability
    # ------------------------------------------------------------------

    def compute_shap_values(
        self,
        background_windows: np.ndarray,
        explain_windows:    np.ndarray,
    ) -> dict[str, float]:
        """
        Compute mean absolute SHAP values per feature.

        Uses DeepExplainer which is designed for TensorFlow/Keras models.
        SHAP values tell us: "how much did each feature push the prediction
        away from the baseline (average prediction)?"

        Args:
            background_windows: Array shape (N_bg, WINDOW_SIZE, N_FEATURES).
                                 Used as the reference distribution.
                                 Use 50-100 random training samples.
            explain_windows:    Array shape (N_ex, WINDOW_SIZE, N_FEATURES).
                                 Windows to explain. Use recent 20-50 samples.

        Returns:
            Dictionary mapping feature name → mean |SHAP| value.
            Higher value = more important feature for this prediction.

        Example:
            {
                "rsi":         0.182,
                "macd":        0.143,
                "volatility":  0.091,
                ...
            }
        """
        try:
            import shap
            if self._shap_explainer is None:
                print("Building SHAP DeepExplainer (first time, takes ~30 sec)...")
                self._shap_explainer = shap.DeepExplainer(
                    self.model,
                    background_windows,
                )

            print(f"Computing SHAP values for {len(explain_windows)} windows...")
            shap_vals = self._shap_explainer.shap_values(explain_windows)

            # shap_vals shape: (N_ex, WINDOW_SIZE, N_FEATURES)
            # Average over samples and time steps → shape: (N_FEATURES,)
            mean_abs = np.abs(shap_vals).mean(axis=(0, 1))

            result = {
                feat: round(float(val), 4)
                for feat, val in zip(FEATURE_COLS, mean_abs)
            }

            # Sort descending by importance
            result = dict(
                sorted(result.items(), key=lambda x: x[1], reverse=True)
            )
            return result

        except ImportError:
            print("SHAP not installed. Run: pip install shap")
            return {feat: 0.0 for feat in FEATURE_COLS}
        except Exception as e:
            print(f"SHAP computation failed: {e}")
            return {feat: 0.0 for feat in FEATURE_COLS}

    def save_shap_values(self, shap_dict: dict[str, float]) -> None:
        """
        Save SHAP values to JSON so the dashboard loads them instantly
        without recomputing every time.

        Args:
            shap_dict: Output of compute_shap_values().
        """
        os.makedirs(os.path.dirname(SHAP_PATH), exist_ok=True)
        with open(SHAP_PATH, "w") as f:
            json.dump(shap_dict, f, indent=2)
        print(f"SHAP values saved → {SHAP_PATH}")

    @staticmethod
    def load_shap_values() -> dict[str, float] | None:
        """
        Load previously saved SHAP values from disk.

        Returns:
            Dictionary of feature → importance, or None if file not found.
        """
        if not os.path.exists(SHAP_PATH):
            return None
        with open(SHAP_PATH) as f:
            return json.load(f)
