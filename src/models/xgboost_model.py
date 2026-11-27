"""
XGBoost model for VoltAhead (availability 30/40/60 min ahead).

* Target: 1 = Available, 0 = Unavailable.  ``y_prob`` = P(Available = 1).
* Inputs: the 16 shared ``FEATURE_COLUMNS``; ``station_id`` is one-hot encoded
  with ``handle_unknown="ignore"`` so unseen stations get an all-zero station
  block (the model then relies on the other 15 features).
* Artifacts are plain dicts holding scikit-learn / xgboost objects only
  (no custom classes), mirroring the Logistic Regression artifact layout.
"""
from __future__ import annotations

import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

SEED = 42
MODEL_NAME = "xgboost"
CATEGORICAL_FEATURES = ["station_id"]
FEATURE_COLUMNS = [
    "station_id", "hour", "day_of_week", "is_weekend", "current_idle", "current_busy",
    "station_capacity", "occupancy_rate", "idle_lag_5", "idle_lag_10", "idle_lag_15",
    "idle_lag_30", "idle_lag_40", "idle_lag_60", "idle_lag_120", "prediction_horizon",
]
PREDICTION_COLUMNS = [
    "station_id", "time", "target_time", "prediction_horizon",
    "y_true", "y_pred", "y_prob", "model_name", "model_version", "threshold",
]

# Conservative initial configuration (Step 3)
DEFAULT_XGB_PARAMS = {
    "objective": "binary:logistic",
    "tree_method": "hist",
    "learning_rate": 0.05,
    "max_depth": 4,
    "n_estimators": 200,
    "min_child_weight": 1,
    "subsample": 1.0,
    "colsample_bytree": 1.0,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
    "eval_metric": "logloss",
    "random_state": SEED,
}


def build_preprocessor(feature_columns=FEATURE_COLUMNS) -> ColumnTransformer:
    cats = [c for c in CATEGORICAL_FEATURES if c in feature_columns]
    nums = [c for c in feature_columns if c not in cats]
    return ColumnTransformer(
        [("station", OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32), cats),
         ("numeric", "passthrough", nums)],
        verbose_feature_names_out=False,
    ).set_output(transform="default")


def build_xgboost_pipeline(params=None, feature_columns=FEATURE_COLUMNS, n_jobs=None) -> Pipeline:
    p = dict(DEFAULT_XGB_PARAMS)
    if params:
        p.update(params)
    if n_jobs is not None:
        p["n_jobs"] = n_jobs
    return Pipeline([("preprocess", build_preprocessor(feature_columns)),
                     ("model", XGBClassifier(**p))])


def class0_sample_weights(y, weight_for_class0=1.0):
    """Sample weights that up-weight Unavailable (class 0) rows; label definition is unchanged."""
    y = np.asarray(y)
    return np.where(y == 0, float(weight_for_class0), 1.0)


def fit_xgboost(pipeline, X, y, sample_weight=None, eval_set=None, early_stopping_rounds=None):
    """Fit and return (pipeline, seconds).  eval_set = (X_val, y_val) in raw feature space."""
    pre = pipeline.named_steps["preprocess"]
    model = pipeline.named_steps["model"]
    t0 = time.perf_counter()
    Xt = pre.fit_transform(X)
    kw = {}
    if sample_weight is not None:
        kw["sample_weight"] = sample_weight
    if eval_set is not None:
        Xv, yv = eval_set
        kw["eval_set"] = [(pre.transform(Xv), yv)]
        kw["verbose"] = False
        if early_stopping_rounds:
            model.set_params(early_stopping_rounds=early_stopping_rounds)
    model.fit(Xt, y, **kw)
    return pipeline, time.perf_counter() - t0


def raw_probability(pipeline, X):
    """Uncalibrated P(Available = 1)."""
    return pipeline.predict_proba(X)[:, 1]


# ----------------------------------------------------------------- calibration
def fit_calibrator(raw_prob, y, method="sigmoid"):
    """Fit on calibration-period predictions of an ALREADY FITTED estimator."""
    raw_prob = np.asarray(raw_prob, dtype="float64")
    if method == "sigmoid":
        z = _logit(raw_prob).reshape(-1, 1)
        m = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000).fit(z, y)
    elif method == "isotonic":
        m = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(raw_prob, y)
    else:
        raise ValueError(method)
    return {"method": method, "model": m}


def apply_calibrator(calibrator, raw_prob):
    raw_prob = np.asarray(raw_prob, dtype="float64")
    if calibrator is None:
        return raw_prob
    if calibrator["method"] == "sigmoid":
        return calibrator["model"].predict_proba(_logit(raw_prob).reshape(-1, 1))[:, 1]
    return calibrator["model"].predict(raw_prob)


def _logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


# ---------------------------------------------------------- artifact + predict
def make_artifact(pipeline, version, threshold=0.5, calibrator=None, feature_columns=FEATURE_COLUMNS, extra=None):
    a = {"model_name": MODEL_NAME, "model_version": version, "pipeline": pipeline,
         "feature_columns": list(feature_columns), "threshold": float(threshold),
         "calibrator": calibrator}
    if extra:
        a["metadata"] = extra
    return a


def save_xgboost_artifact(artifact, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, path)
    return path


def load_xgboost_artifact(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Model artifact not found: {path}")
    a = joblib.load(path)
    missing = {"model_name", "model_version", "pipeline", "feature_columns", "threshold"} - set(a)
    if missing:
        raise ValueError(f"Model artifact is missing required keys: {missing}")
    if a["model_name"] != MODEL_NAME:
        raise ValueError("Artifact is not an XGBoost model.")
    if a["model_version"] not in {"initial", "optimized"}:
        raise ValueError("Unsupported XGBoost model version.")
    if a["model_version"] == "optimized" and a.get("calibrator", None) is None and "calibrator" not in a:
        raise ValueError("Optimized artifact is missing its calibrator key.")
    return a


def predict_xgboost(artifact, features):
    """Return DataFrame with y_prob = P(Available=1) (calibrated if the artifact has a calibrator) and y_pred."""
    if not isinstance(features, pd.DataFrame):
        raise TypeError("features must be a pandas DataFrame")
    missing = set(artifact["feature_columns"]) - set(features.columns)
    if missing:
        raise ValueError(f"Missing feature columns: {sorted(missing)}")
    raw = raw_probability(artifact["pipeline"], features[artifact["feature_columns"]])
    prob = apply_calibrator(artifact.get("calibrator"), raw)
    return pd.DataFrame({"y_prob": prob, "y_pred": (prob >= artifact["threshold"]).astype(int)},
                        index=features.index)


def build_prediction_frame(df, artifact, version=None):
    """Shared-schema prediction frame for a processed frame that has ``target``."""
    t0 = time.perf_counter()
    p = predict_xgboost(artifact, df)
    sec = time.perf_counter() - t0
    out = df[["station_id", "time", "target_time", "prediction_horizon"]].copy()
    out["y_true"] = df["target"].astype(int).to_numpy()
    out["y_pred"] = p["y_pred"].to_numpy()
    out["y_prob"] = p["y_prob"].to_numpy()
    out["model_name"] = MODEL_NAME
    out["model_version"] = version or artifact["model_version"]
    out["threshold"] = float(artifact["threshold"])
    return out[PREDICTION_COLUMNS].reset_index(drop=True), sec
