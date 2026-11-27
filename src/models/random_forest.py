"""Random Forest model implementation for VoltAhead."""

import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline


# Shared model input features
FEATURE_COLUMNS = [
    "station_id",
    "hour",
    "day_of_week",
    "is_weekend",
    "current_idle",
    "current_busy",
    "station_capacity",
    "occupancy_rate",
    "idle_lag_5",
    "idle_lag_10",
    "idle_lag_15",
    "idle_lag_30",
    "idle_lag_40",
    "idle_lag_60",
    "idle_lag_120",
    "prediction_horizon"
]


# Initial Random Forest configuration
DEFAULT_RF_PARAMS = {
    "n_estimators": 100,
    "max_depth": 15,
    "min_samples_leaf": 5,
    "max_features": "sqrt",
    "class_weight": None,
    "random_state": 42,
    "n_jobs": 4
}


def build_preprocessor(feature_columns=None):
    """Create the feature preprocessing transformer."""

    if feature_columns is None:
        feature_columns = FEATURE_COLUMNS

    categorical_features = [
        feature for feature in feature_columns
        if feature == "station_id"
    ]

    numerical_features = [
        feature for feature in feature_columns
        if feature != "station_id"
    ]

    transformers = []

    if categorical_features:
        transformers.append((
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=True,
                dtype=np.float32
            ),
            categorical_features
        ))

    if numerical_features:
        transformers.append((
            "numerical",
            "passthrough",
            numerical_features
        ))

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=1.0
    )


def build_random_forest_pipeline(
    feature_columns=None,
    model_params=None
):
    """Build a configurable Random Forest pipeline."""

    params = DEFAULT_RF_PARAMS.copy()

    if model_params is not None:
        params.update(model_params)

    preprocessor = build_preprocessor(feature_columns)

    classifier = RandomForestClassifier(**params)

    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ])