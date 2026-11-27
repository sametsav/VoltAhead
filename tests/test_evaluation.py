"""Unit tests for the VoltAhead shared evaluation module."""

import unittest

import numpy as np
import pandas as pd

from src.evaluation import (
    validate_prediction_output,
    calculate_classification_metrics,
    evaluate_predictions,
    evaluate_by_horizon,
    evaluate_by_station_group
)


class TestEvaluation(unittest.TestCase):

    def setUp(self):
        """Create a synthetic prediction dataset."""

        self.predictions = pd.DataFrame({
            "station_id": [1001, 1002, 1001, 1948],
            "time": pd.to_datetime(
                ["2023-01-06 10:00"] * 4
            ),
            "prediction_horizon": [30, 30, 60, 60],
            "y_true": [1, 0, 1, 0],
            "y_prob": [0.9, 0.2, 0.8, 0.7]
        })

        self.predictions["target_time"] = (
            self.predictions["time"]
            + pd.to_timedelta(
                self.predictions["prediction_horizon"],
                unit="min"
            )
        )

        self.predictions["threshold"] = 0.5

        self.predictions["y_pred"] = (
            self.predictions["y_prob"] >= 0.5
        ).astype(int)

        self.predictions["model_name"] = "random_forest"
        self.predictions["model_version"] = "initial"

        self.reference = self.predictions[[
            "station_id",
            "time",
            "target_time",
            "prediction_horizon",
            "y_true"
        ]].rename(columns={"y_true": "target"})

        self.training_station_ids = [1001, 1002]

    def test_valid_predictions(self):
        self.assertTrue(
            validate_prediction_output(
                self.predictions, self.reference
            )
        )

    def test_missing_column(self):
        invalid = self.predictions.drop(columns=["y_prob"])

        with self.assertRaises(ValueError):
            validate_prediction_output(invalid)

    def test_duplicate_keys(self):
        invalid = pd.concat([
            self.predictions,
            self.predictions.iloc[:1]
        ], ignore_index=True)

        with self.assertRaises(ValueError):
            validate_prediction_output(invalid)

    def test_invalid_probability(self):
        invalid = self.predictions.copy()
        invalid.loc[0, "y_prob"] = 1.5

        with self.assertRaises(ValueError):
            validate_prediction_output(invalid)

    def test_threshold_consistency(self):
        invalid = self.predictions.copy()
        invalid.loc[0, "y_pred"] = 0

        with self.assertRaises(ValueError):
            validate_prediction_output(invalid)

    def test_missing_observation(self):
        invalid = self.predictions.iloc[:-1]

        with self.assertRaises(ValueError):
            validate_prediction_output(
                invalid, self.reference
            )

    def test_incorrect_true_label(self):
        invalid = self.predictions.copy()
        invalid.loc[0, "y_true"] = 0

        with self.assertRaises(ValueError):
            validate_prediction_output(
                invalid, self.reference
            )

    def test_shuffled_predictions(self):
        shuffled = self.predictions.sample(
            frac=1, random_state=42
        )

        self.assertTrue(
            validate_prediction_output(
                shuffled, self.reference
            )
        )

    def test_classification_metrics(self):
        metrics = calculate_classification_metrics(
            [1, 1, 0, 0],
            [1, 1, 0, 1],
            [0.9, 0.8, 0.2, 0.7]
        )

        self.assertAlmostEqual(metrics["accuracy"], 0.75)
        self.assertAlmostEqual(metrics["precision_0"], 1.0)
        self.assertAlmostEqual(metrics["recall_0"], 0.5)
        self.assertAlmostEqual(metrics["f1_0"], 2 / 3)
        self.assertAlmostEqual(metrics["ap_0"], 1.0)
        self.assertAlmostEqual(metrics["roc_auc"], 1.0)
        self.assertAlmostEqual(metrics["brier_score"], 0.145)

    def test_single_class(self):
        metrics = calculate_classification_metrics(
            [1, 1, 1],
            [1, 1, 1],
            [0.8, 0.9, 0.95]
        )

        self.assertTrue(np.isnan(metrics["roc_auc"]))
        self.assertTrue(np.isnan(metrics["ap_0"]))
        self.assertTrue(np.isnan(metrics["balanced_accuracy"]))
        self.assertTrue(np.isnan(metrics["precision_0"]))
        self.assertTrue(np.isnan(metrics["recall_0"]))
        self.assertTrue(np.isnan(metrics["f1_0"]))

        self.assertEqual(metrics["accuracy"], 1.0)

    def test_overall_evaluation(self):
        result = evaluate_predictions(
            self.predictions,
            self.training_station_ids
        )

        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["n_observations"], 4)
        self.assertAlmostEqual(result.iloc[0]["accuracy"], 0.75)

    def test_horizon_evaluation(self):
        result = evaluate_by_horizon(
            self.predictions,
            self.training_station_ids
        )

        self.assertEqual(len(result), 2)
        self.assertEqual(result["n_observations"].sum(), 4)

    def test_station_group_evaluation(self):
        result = evaluate_by_station_group(
            self.predictions,
            self.training_station_ids
        ).set_index("evaluation_group")

        self.assertEqual(result.loc["seen", "n_observations"], 3)
        self.assertEqual(result.loc["unseen", "n_observations"], 1)

        self.assertEqual(result.loc["seen", "n_stations"], 2)
        self.assertEqual(result.loc["unseen", "n_stations"], 1)


if __name__ == "__main__":
    unittest.main()