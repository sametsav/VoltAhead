"""LightGBM model utilities for VoltAhead."""

import numpy as np
import pandas as pd

from lightgbm import LGBMClassifier


CONTRACT_VERSION = "1.0"


def fit_station_categories(data):
    """Learn station categories from fitting data only."""

    return sorted(
        data["station_id"].dropna().unique().tolist()
    )


def prepare_features(data, feature_columns, station_categories):
    """Apply the shared feature order and categorical vocabulary."""

    X = data.loc[:, feature_columns].copy()

    if "station_id" in feature_columns:
        X["station_id"] = pd.Categorical(
            X["station_id"],
            categories=station_categories,
            ordered=False
        )

    return X


def create_initial_model(seed=42, n_jobs=4):
    """Create the initial horizon-aware LightGBM classifier."""

    return LGBMClassifier(
        objective="binary",
        boosting_type="gbdt",
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=6,
        min_child_samples=50,
        random_state=seed,
        n_jobs=n_jobs,
        device_type="cpu",
        deterministic=True,
        force_col_wise=True,
        verbosity=-1
    )


def predict_availability(model, X):
    """Return the probability of availability (Class 1)."""

    class_index = list(model.classes_).index(1)

    probabilities = model.predict_proba(X)[:, class_index]

    if not np.isfinite(probabilities).all():
        raise ValueError("Non-finite probabilities detected.")

    if not ((probabilities >= 0) & (probabilities <= 1)).all():
        raise ValueError("Probabilities outside [0, 1].")

    return probabilities


def probability_to_logit(probabilities):
    """Convert probabilities into clipped logit values."""

    probabilities = np.clip(
        np.asarray(probabilities),
        1e-12,
        1 - 1e-12
    )

    return np.log(
        probabilities / (1 - probabilities)
    ).reshape(-1, 1)


def predict_calibrated_availability(
    model,
    features,
    calibrator
):
    """Return calibrated P(Available = 1)."""

    raw_probabilities = predict_availability(
        model,
        features
    )

    if calibrator is None:
        return raw_probabilities

    class_index = list(calibrator.classes_).index(1)

    calibrated_probabilities = (
        calibrator.predict_proba(
            probability_to_logit(raw_probabilities)
        )[:, class_index]
    )

    if not np.isfinite(calibrated_probabilities).all():
        raise ValueError(
            "Non-finite calibrated probabilities detected."
        )

    if not (
        (calibrated_probabilities >= 0)
        & (calibrated_probabilities <= 1)
    ).all():
        raise ValueError(
            "Calibrated probabilities outside [0, 1]."
        )

    return calibrated_probabilities


def predict_labels(probabilities, threshold=0.5):
    """Convert availability probabilities into binary labels."""

    return (
        np.asarray(probabilities) >= threshold
    ).astype(np.int8)


def predict_from_artifact(artifact, data):
    """Generate probabilities and labels from a saved model artifact."""

    if artifact["contract_version"] != CONTRACT_VERSION:
        raise ValueError("Unsupported inference contract version.")

    X = prepare_features(
        data,
        artifact["feature_columns"],
        artifact["station_categories"]
    )

    probabilities = predict_calibrated_availability(
        artifact["model"],
        X,
        artifact["calibrator"]
    )

    labels = predict_labels(
        probabilities,
        artifact["threshold"]
    )

    return probabilities, labels