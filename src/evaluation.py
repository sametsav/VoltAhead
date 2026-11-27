"""
VoltAhead - Shared Evaluation Module

Provides standardized prediction validation,
classification metrics, and performance analysis
for all VoltAhead models.
"""

from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix
)


# ---------------------------------------------------------
# 1. Shared Constants
# ---------------------------------------------------------

PREDICTION_COLUMNS = [
    "station_id",
    "time",
    "target_time",
    "prediction_horizon",
    "y_true",
    "y_pred",
    "y_prob",
    "model_name",
    "model_version",
    "threshold"
]

KEY_COLUMNS = [
    "station_id",
    "time",
    "prediction_horizon"
]

PREDICTION_HORIZONS = [30, 40, 60]

AVAILABLE_CLASS = 1
UNAVAILABLE_CLASS = 0

DEFAULT_THRESHOLD = 0.5

# ---------------------------------------------------------
# 2. Prediction Validation
# ---------------------------------------------------------

def validate_prediction_output(predictions, reference=None):
    """Validate prediction format and observation integrity."""

    if not isinstance(predictions, pd.DataFrame):
        raise TypeError("Predictions must be a pandas DataFrame.")

    if predictions.empty:
        raise ValueError("Prediction table is empty.")

    # Check required columns
    missing_columns = [
        col for col in PREDICTION_COLUMNS
        if col not in predictions.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing prediction columns: {missing_columns}"
        )

    # Check missing values
    if predictions[PREDICTION_COLUMNS].isna().any().any():
        raise ValueError("Prediction table contains missing values.")

    # Check model identity
    if len(predictions[["model_name", "model_version"]]
           .drop_duplicates()) != 1:
        raise ValueError(
            "Expected exactly one model and model version."
        )

    # Check unique observation keys
    if predictions.duplicated(KEY_COLUMNS).any():
        raise ValueError("Duplicate observation keys detected.")

    # Check prediction horizons
    if not predictions["prediction_horizon"].isin(
        PREDICTION_HORIZONS
    ).all():
        raise ValueError("Invalid prediction horizon.")

    # Check binary labels
    for col in ["y_true", "y_pred"]:
        if not predictions[col].isin([0, 1]).all():
            raise ValueError(f"Invalid binary labels in {col}.")

    # Check probability range
    if not predictions["y_prob"].between(0, 1).all():
        raise ValueError("Probabilities must be between 0 and 1.")

    # Check threshold range
    if not predictions["threshold"].between(0, 1).all():
        raise ValueError("Threshold must be between 0 and 1.")

    # Check timestamp alignment
    expected_target_time = (
        predictions["time"]
        + pd.to_timedelta(
            predictions["prediction_horizon"], unit="min"
        )
    )

    if not predictions["target_time"].eq(
        expected_target_time
    ).all():
        raise ValueError("Incorrect target timestamp alignment.")

    # Check threshold consistency
    expected_predictions = (
        predictions["y_prob"] >= predictions["threshold"]
    ).astype(int)

    if not predictions["y_pred"].eq(expected_predictions).all():
        raise ValueError("Predictions are inconsistent with threshold.")


    # Check consistency with the reference dataset
    if reference is not None:

        reference_columns = KEY_COLUMNS + [
            "target_time", "target"
        ]

        if not isinstance(reference, pd.DataFrame):
            raise TypeError("Reference must be a DataFrame.")

        if not set(reference_columns).issubset(reference.columns):
            raise ValueError("Reference dataset has missing columns.")

        if reference[reference_columns].isna().any().any():
            raise ValueError("Reference contains missing values.")

        if reference.duplicated(KEY_COLUMNS).any():
            raise ValueError("Reference contains duplicate keys.")

        # Match observations using their unique keys
        comparison = predictions.merge(
            reference[reference_columns],
            on=KEY_COLUMNS,
            how="outer",
            suffixes=("_pred", "_ref"),
            indicator=True,
            validate="one_to_one"
        )

        # Check missing and unexpected observations
        missing_count = (
            comparison["_merge"] == "right_only"
        ).sum()

        unexpected_count = (
            comparison["_merge"] == "left_only"
        ).sum()

        if missing_count or unexpected_count:
            raise ValueError(
                f"Missing observations: {missing_count}, "
                f"Unexpected observations: {unexpected_count}"
            )

        # Check true labels
        if not comparison["y_true"].eq(
            comparison["target"]
        ).all():
            raise ValueError("True labels do not match reference.")

        # Check target timestamps
        if not comparison["target_time_pred"].eq(
            comparison["target_time_ref"]
        ).all():
            raise ValueError("Target timestamps do not match reference.")

    return True

# ---------------------------------------------------------
# 3. Classification Metrics
# ---------------------------------------------------------

def calculate_classification_metrics(y_true, y_pred, y_prob):
    """Calculate standardized binary classification metrics."""

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_prob = np.asarray(y_prob)

    # Convert available-class probabilities
    # into unavailable-class probabilities
    y_true_0 = (y_true == UNAVAILABLE_CLASS).astype(int)
    y_prob_0 = 1 - y_prob

    # Check whether both target classes are present
    both_classes = len(np.unique(y_true)) == 2

    # Confusion matrix with fixed class order
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    )

    metrics = {
        "n_observations": len(y_true),

        "accuracy": accuracy_score(y_true, y_pred),

        "balanced_accuracy": (
            balanced_accuracy_score(y_true, y_pred)
            if both_classes else np.nan
        ),

        "precision_0": precision_score(
            y_true,
            y_pred,
            pos_label=0,
            zero_division=np.nan
        ),

        "recall_0": recall_score(
            y_true,
            y_pred,
            pos_label=0,
            zero_division=np.nan
        ),

        "f1_0": f1_score(
            y_true,
            y_pred,
            pos_label=0,
            zero_division=np.nan
        ),

        "roc_auc": (
            roc_auc_score(y_true, y_prob)
            if both_classes else np.nan
        ),

        "ap_0": (
            average_precision_score(y_true_0, y_prob_0)
            if both_classes else np.nan
        ),

        "brier_score": brier_score_loss(
            y_true,
            y_prob,
            pos_label=1
        ),

        "actual_0_predicted_0": int(cm[0, 0]),
        "actual_0_predicted_1": int(cm[0, 1]),
        "actual_1_predicted_0": int(cm[1, 0]),
        "actual_1_predicted_1": int(cm[1, 1])
    }

    return metrics

# ---------------------------------------------------------
# 4. Overall Evaluation
# ---------------------------------------------------------

def evaluate_predictions(predictions, training_station_ids):
    """Evaluate overall model performance."""

    validate_prediction_output(predictions)

    metrics = calculate_classification_metrics(
        predictions["y_true"],
        predictions["y_pred"],
        predictions["y_prob"]
    )

    result = {
        "model_name": predictions["model_name"].iloc[0],
        "model_version": predictions["model_version"].iloc[0],
        "evaluation_group": "overall",
        "n_stations": predictions["station_id"].nunique(),
        **metrics
    }

    return pd.DataFrame([result])

# ---------------------------------------------------------
# 5. Horizon-Based Evaluation
# ---------------------------------------------------------

def evaluate_by_horizon(predictions, training_station_ids):
    """Evaluate model performance for each prediction horizon."""

    validate_prediction_output(predictions)

    results = []

    for horizon in PREDICTION_HORIZONS:

        subset = predictions[
            predictions["prediction_horizon"] == horizon
        ]

        if subset.empty:
            continue

        metrics = calculate_classification_metrics(
            subset["y_true"],
            subset["y_pred"],
            subset["y_prob"]
        )

        result = {
            "model_name": subset["model_name"].iloc[0],
            "model_version": subset["model_version"].iloc[0],
            "evaluation_group": "horizon",
            "prediction_horizon": horizon,
            "n_stations": subset["station_id"].nunique(),
            **metrics
        }

        results.append(result)

    return pd.DataFrame(results)

# ---------------------------------------------------------
# 6. Seen / Unseen Station Evaluation
# ---------------------------------------------------------

def evaluate_by_station_group(predictions, training_station_ids):
    """Evaluate seen and unseen station performance separately."""

    validate_prediction_output(predictions)

    training_station_ids = set(training_station_ids)

    if not training_station_ids:
        raise ValueError("Training station IDs cannot be empty.")

    # Identify stations observed during training
    seen_mask = predictions["station_id"].isin(
        training_station_ids
    )

    groups = {
        "seen": predictions[seen_mask],
        "unseen": predictions[~seen_mask]
    }

    results = []

    for group_name, subset in groups.items():

        if subset.empty:
            continue

        metrics = calculate_classification_metrics(
            subset["y_true"],
            subset["y_pred"],
            subset["y_prob"]
        )

        result = {
            "model_name": subset["model_name"].iloc[0],
            "model_version": subset["model_version"].iloc[0],
            "evaluation_group": group_name,
            "n_stations": subset["station_id"].nunique(),
            **metrics
        }

        results.append(result)

    return pd.DataFrame(results)

# ---------------------------------------------------------
# 7. Evaluation Results Export
# ---------------------------------------------------------

def export_evaluation_results(
    predictions,
    training_station_ids,
    reference,
    output_dir="outputs/tables"
):
    """Validate, evaluate and export model performance."""

    # Verify complete reference coverage
    validate_prediction_output(
        predictions,
        reference=reference
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_name = predictions["model_name"].iloc[0]
    model_version = predictions["model_version"].iloc[0]

    prefix = f"{model_name}_{model_version}"

    # Calculate evaluation results
    overall = evaluate_predictions(
        predictions, training_station_ids
    )

    by_horizon = evaluate_by_horizon(
        predictions, training_station_ids
    )

    by_station_group = evaluate_by_station_group(
        predictions, training_station_ids
    )

    results = {
        "overall": overall,
        "by_horizon": by_horizon,
        "by_station_group": by_station_group
    }

    # Export each evaluation table
    for evaluation_name, table in results.items():

        output_path = (
            output_dir
            / f"{prefix}_{evaluation_name}.csv"
        )

        table.to_csv(
            output_path,
            index=False,
            na_rep="N/A"
        )

        print(f"Saved: {output_path}")

    return results