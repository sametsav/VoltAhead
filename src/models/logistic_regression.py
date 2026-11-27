
"""Reusable inference utilities for VoltAhead Logistic Regression models."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def load_logistic_regression_model(artifact_path):
    """Load a trusted, previously saved Logistic Regression artifact."""
    artifact_path = Path(artifact_path)

    if not artifact_path.is_file():
        raise FileNotFoundError(
            f"Model artifact not found: {artifact_path}"
        )

    artifact = joblib.load(artifact_path)

    required_keys = {
        "model_name",
        "model_version",
        "pipeline",
        "feature_columns",
        "threshold",
    }

    missing_keys = required_keys - set(artifact)

    if missing_keys:
        raise ValueError(
            f"Model artifact is missing required keys: {missing_keys}"
        )

    if artifact["model_name"] != "logistic_regression":
        raise ValueError("Artifact is not a Logistic Regression model.")

    if artifact["model_version"] not in {"initial", "optimized"}:
        raise ValueError("Unsupported Logistic Regression model version.")

    if artifact["model_version"] == "optimized":
        if "calibrator" not in artifact:
            raise ValueError(
                "Optimized artifact is missing its calibrator."
            )

    return artifact


def predict_logistic_regression(artifact, features):
    """
    Generate availability probabilities and binary predictions.

    Returns a DataFrame with:
    - y_prob: P(Available = 1)
    - y_pred: predicted class using the saved threshold
    """
    if not isinstance(features, pd.DataFrame):
        raise TypeError("Features must be a pandas DataFrame.")

    required_features = artifact["feature_columns"]

    missing_features = [
        feature for feature in required_features
        if feature not in features.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing required features: {missing_features}"
        )

    X = features[required_features]

    # Generate probabilities with the saved preprocessing pipeline
    probabilities = artifact["pipeline"].predict_proba(X)[:, 1]

    # Apply calibration only when available
    if artifact.get("calibration_method") == "isotonic":
        probabilities = artifact["calibrator"].predict(
            probabilities
        )

    probabilities = np.asarray(probabilities, dtype=float)

    if not np.isfinite(probabilities).all():
        raise ValueError("Model produced non-finite probabilities.")

    if not np.all((probabilities >= 0) & (probabilities <= 1)):
        raise ValueError("Model produced probabilities outside [0, 1].")

    threshold = float(artifact["threshold"])

    predictions = (probabilities >= threshold).astype(int)

    return pd.DataFrame(
        {
            "y_prob": probabilities,
            "y_pred": predictions,
        },
        index=features.index,
    )
