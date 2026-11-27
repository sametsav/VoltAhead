# VoltAhead – Shared Evaluation Module

## 1. Purpose

The shared evaluation module provides a standardized evaluation framework for all VoltAhead models, ensuring consistent metrics, prediction validation, and fair model comparison.

**Module:** `src/evaluation.py`

## 2. Prediction Output Format

Each model must produce a prediction DataFrame containing the following columns:

| Column | Description |
|---|---|
| station_id | Charging station identifier |
| time | Observation timestamp |
| target_time | Future prediction timestamp |
| prediction_horizon | Prediction horizon: 30, 40 or 60 minutes |
| y_true | Actual availability label |
| y_pred | Predicted availability label |
| y_prob | Probability of availability (Class 1) |
| model_name | Model identifier |
| model_version | Initial or optimized model version |
| threshold | Classification decision threshold |

Class definitions:

- Class 0: Unavailable
- Class 1: Available

The unique observation key is `(station_id, time, prediction_horizon)`.

Predicted labels must satisfy `y_pred = (y_prob >= threshold)`.

## 3. Prediction Validation

The `validate_prediction_output()` function checks:

- Required columns and missing values
- Duplicate observation keys
- Valid prediction horizons
- Binary target and prediction labels
- Probability and threshold ranges
- Timestamp alignment
- Threshold consistency
- Model and version consistency
- Complete observation coverage against the reference dataset
- True label consistency with the reference dataset

## 4. Evaluation Metrics

All models are evaluated using:

- Accuracy
- Balanced Accuracy
- Class-0 Precision
- Class-0 Recall
- Class-0 F1-score
- ROC-AUC
- Class-0 Average Precision (AP₀)
- Brier Score
- Confusion Matrix

The primary model selection metric is Class-0 Average Precision on seen validation stations.

Undefined metrics are represented as NaN internally and exported as N/A.

## 5. Evaluation Groups

Performance is evaluated at three levels:

- Overall performance
- Prediction horizon: 30, 40 and 60 minutes
- Seen and unseen station groups

Seen stations are identified using the shared training station IDs.

## 6. Public Functions

- `validate_prediction_output(predictions, reference=None)`
- `calculate_classification_metrics(y_true, y_pred, y_prob)`
- `evaluate_predictions(predictions, training_station_ids)`
- `evaluate_by_horizon(predictions, training_station_ids)`
- `evaluate_by_station_group(predictions, training_station_ids)`
- `export_evaluation_results(predictions, training_station_ids, reference, output_dir="outputs/tables")`

## 7. CSV Output

The export function generates three standardized CSV files:

- `{model_name}_{model_version}_overall.csv`
- `{model_name}_{model_version}_by_horizon.csv`
- `{model_name}_{model_version}_by_station_group.csv`

Use a separate output directory when evaluating different dataset splits to prevent overwriting results.

## 8. Testing

Run the unit tests from the repository root:

`python -m unittest discover -s tests -v`

The module has successfully passed 13 unit tests, a full validation dataset integration test covering 618,726 observations, and a CSV export test.

## 9. Evaluation Protocol

All models must use the shared chronological dataset splits and identical validation observations.

Model tuning, calibration and threshold selection must be performed using training-period data only.

The official test dataset must remain untouched until final model selection is completed.

Training and inference times are recorded separately as model metadata.

Baseline probability and threshold handling must be agreed upon before official baseline evaluation.