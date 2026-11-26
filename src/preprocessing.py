
"""
VoltAhead — Shared Data Preprocessing and Feature Engineering

Creates consistent, model-ready datasets for 30-, 40-, and
60-minute EV charging availability prediction.

Team-approved configuration:
- Dataset: December 1, 2022 – January 21, 2023
- Missing value strategy: Complete-case
- Split strategy: Chronological
- Unseen stations: Retained
"""

import numpy as np
import pandas as pd

import data_loader as dl


# ---------------------------------------------------------
# Shared Configuration
# ---------------------------------------------------------

PREDICTION_HORIZONS = [30, 40, 60]
LAG_INTERVALS = [5, 10, 15, 30, 40, 60, 120]

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
    "prediction_horizon",
]

DATASET_START = pd.Timestamp("2022-12-01")
DATASET_END = pd.Timestamp("2023-01-22")

SPLIT_DATES = {
    "Train": ("2022-12-01", "2023-01-06"),
    "Validation": ("2023-01-06", "2023-01-14"),
    "Test": ("2023-01-14", "2023-01-22"),
}

# Reserved for future robustness evaluation
ROBUSTNESS_START = pd.Timestamp("2023-01-22")
ROBUSTNESS_END = pd.Timestamp("2023-03-01")


# ---------------------------------------------------------
# 1. Data Loading
# ---------------------------------------------------------

def load_base_data():
    """Load the shared dataset while preserving missing values."""

    df = dl.load_data(columns=["idle", "busy"])
    df = dl.attach_station_info(df)

    return df.sort_index()


# ---------------------------------------------------------
# 2. Data Validation
# ---------------------------------------------------------

def validate_and_clean_data(df):
    """Check station data consistency without filling missing values."""

    required = ["idle", "busy", "charge_count"]

    if df.index.has_duplicates:
        raise ValueError("Duplicate station-time observations found.")

    missing_columns = sorted(set(required) - set(df.columns))

    if missing_columns:
        raise ValueError(
            f"Missing required base columns: {missing_columns}"
        )

    # Missing idle/busy observations are intentionally retained until
    # the common complete-case filtering step.

    valid = df["idle"].notna() & df["busy"].notna()

    invalid = valid & (
        (df["idle"] < 0)
        | (df["busy"] < 0)
        | (df["idle"] + df["busy"] != df["charge_count"])
    )

    if invalid.any():
        raise ValueError(
            "Invalid idle/busy/capacity observations found."
        )

    if df["charge_count"].isna().any() or (
        df["charge_count"] <= 0
    ).any():
        raise ValueError("Invalid station capacity found.")

    return df


# ---------------------------------------------------------
# 3. Temporal Features
# ---------------------------------------------------------

def create_temporal_features(df):
    """Create time-based prediction features."""

    time = df.index.get_level_values("time")

    df["hour"] = time.hour
    df["day_of_week"] = time.day_of_week
    df["is_weekend"] = (time.day_of_week >= 5).astype(int)

    return df


# ---------------------------------------------------------
# 4. Current Station Features
# ---------------------------------------------------------

def create_station_features(df):
    """Create current charging station features."""

    df["current_idle"] = df["idle"]
    df["current_busy"] = df["busy"]
    df["station_capacity"] = df["charge_count"]

    df["occupancy_rate"] = (
        df["current_busy"]
        / df["station_capacity"].where(
            df["station_capacity"] > 0
        )
    )

    return df


# ---------------------------------------------------------
# 5. Historical Lag Features
# ---------------------------------------------------------

def create_lag_features(df):
    """Generate historical idle counts for each station."""

    for lag in LAG_INTERVALS:
        df[f"idle_lag_{lag}"] = (
            df.groupby(level="station_id")["idle"]
            .shift(lag // 5)
        )

    return df


# ---------------------------------------------------------
# 6. Future Target Labels
# ---------------------------------------------------------

def create_target_labels(df):
    """Create binary future availability labels."""

    for horizon in PREDICTION_HORIZONS:

        future_idle = (
            df.groupby(level="station_id")["idle"]
            .shift(-(horizon // 5))
        )

        df[f"target_{horizon}"] = (
            (future_idle > 0)
            .astype(float)
            .where(future_idle.notna())
        )

    return df


# ---------------------------------------------------------
# 7. Horizon-Based Dataset
# ---------------------------------------------------------

def create_horizon_dataset(df):
    """Create one prediction example per station-time-horizon."""

    time = df.index.get_level_values("time")

    # Select the 52-day period only after generating
    # lags and targets on the original 5-minute grid.
    period_mask = (
        (time >= DATASET_START)
        & (time < DATASET_END)
    )

    base_df = df.loc[period_mask].reset_index()

    horizon_datasets = []

    for horizon in PREDICTION_HORIZONS:

        horizon_df = base_df[
            ["station_id", "time"]
            + [
                col for col in FEATURE_COLUMNS
                if col not in ("station_id", "prediction_horizon")
            ]
            + [f"target_{horizon}"]
        ].copy()

        horizon_df["prediction_horizon"] = horizon

        horizon_df["target_time"] = (
            horizon_df["time"]
            + pd.to_timedelta(horizon, unit="min")
        )

        horizon_df = horizon_df.rename(
            columns={f"target_{horizon}": "target"}
        )

        horizon_datasets.append(horizon_df)

    model_df = pd.concat(
        horizon_datasets,
        ignore_index=True
    )

    model_df = model_df[
        ["station_id", "time", "target_time"]
        + [col for col in FEATURE_COLUMNS if col != "station_id"]
        + ["target"]
    ]

    return model_df


# ---------------------------------------------------------
# 8. Complete-Case Filtering
# ---------------------------------------------------------

def handle_missing_values(model_df):
    """Apply the team-approved complete-case strategy."""

    # Current features, lag features, and labels must be valid.
    required = FEATURE_COLUMNS + ["target"]

    valid_mask = model_df[required].notna().all(axis=1)

    valid_mask &= model_df["target"].isin([0, 1])
    valid_mask &= model_df["station_capacity"] > 0
    valid_mask &= model_df["occupancy_rate"].between(0, 1)

    valid_mask &= (
        model_df["time"] >= DATASET_START
    ) & (
        model_df["time"] < DATASET_END
    ) & (
        model_df["target_time"] < DATASET_END
    )

    return model_df.loc[valid_mask].copy()


# ---------------------------------------------------------
# 9. Chronological Splitting
# ---------------------------------------------------------

def chronological_split(model_df):
    """Split examples without allowing target-time leakage."""

    datasets = {}

    for split_name, (start, end) in SPLIT_DATES.items():

        start = pd.Timestamp(start)
        end = pd.Timestamp(end)

        mask = (
            (model_df["time"] >= start)
            & (model_df["time"] < end)
            & (model_df["target_time"] < end)
        )

        datasets[split_name] = (
            model_df.loc[mask]
            .sort_values(
                ["time", "station_id", "prediction_horizon"]
            )
            .reset_index(drop=True)
        )

    return (
        datasets["Train"],
        datasets["Validation"],
        datasets["Test"],
    )


# ---------------------------------------------------------
# 10. Final Validation
# ---------------------------------------------------------

def validate_processed_data(train, validation, test):
    """Validate the shared model-ready dataset contract and chronology."""

    datasets = {
        "Train": train,
        "Validation": validation,
        "Test": test,
    }

    required_columns = (
        ["station_id", "time", "target_time"]
        + [
            col for col in FEATURE_COLUMNS
            if col != "station_id"
        ]
        + ["target"]
    )

    forbidden = {
        "time", "target_time", "target",
        "future_idle", "target_30",
        "target_40", "target_60",
    }

    # Common feature contract
    assert len(FEATURE_COLUMNS) == 16
    assert len(set(FEATURE_COLUMNS)) == 16
    assert not forbidden.intersection(FEATURE_COLUMNS)

    reference_dtypes = train.dtypes

    for split_name, dataset in datasets.items():

        start, end = [
            pd.Timestamp(value)
            for value in SPLIT_DATES[split_name]
        ]

        # Exact shared schema and consistent Pandas dtypes
        assert dataset.columns.tolist() == required_columns
        assert dataset.dtypes.equals(reference_dtypes)
        assert not dataset.empty

        # Complete features and binary target labels
        assert dataset[FEATURE_COLUMNS].notna().all().all()
        assert dataset["target"].isin([0, 1]).all()

        # Both classes must be represented
        assert set(dataset["target"].unique()) == {0.0, 1.0}

        # Prediction horizons
        assert dataset["prediction_horizon"].isin(
            PREDICTION_HORIZONS
        ).all()

        # Station ID integrity
        assert dataset["station_id"].notna().all()

        # Station capacity and charging counts
        assert (dataset["station_capacity"] > 0).all()
        assert (dataset["current_idle"] >= 0).all()
        assert (dataset["current_busy"] >= 0).all()

        assert np.allclose(
            dataset["current_idle"]
            + dataset["current_busy"],
            dataset["station_capacity"]
        )

        # Occupancy rate consistency
        assert dataset["occupancy_rate"].between(0, 1).all()

        assert np.allclose(
            dataset["occupancy_rate"],
            dataset["current_busy"]
            / dataset["station_capacity"]
        )

        # Chronological split integrity
        assert dataset["time"].ge(start).all()
        assert dataset["time"].lt(end).all()
        assert dataset["target_time"].ge(start).all()
        assert dataset["target_time"].lt(end).all()

        # Verify target timestamp alignment
        assert (
            dataset["target_time"]
            == dataset["time"]
            + pd.to_timedelta(
                dataset["prediction_horizon"],
                unit="min"
            )
        ).all()

        # Unique observation identifiers
        assert not dataset.duplicated(
            ["station_id", "time", "prediction_horizon"]
        ).any()

    # Prevent chronological target leakage
    assert (
        train["target_time"].max()
        < validation["time"].min()
    )

    assert (
        validation["target_time"].max()
        < test["time"].min()
    )

    return True


# ---------------------------------------------------------
# 11. Main Pipeline
# ---------------------------------------------------------

def run_preprocessing():
    """Run the complete shared preprocessing pipeline."""

    df = load_base_data()

    df = validate_and_clean_data(df)

    df = create_temporal_features(df)
    df = create_station_features(df)
    df = create_lag_features(df)
    df = create_target_labels(df)

    model_df = create_horizon_dataset(df)
    model_df = handle_missing_values(model_df)

    train, validation, test = chronological_split(model_df)

    validate_processed_data(train, validation, test)

    return train, validation, test
