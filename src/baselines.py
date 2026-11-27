"""
Persistence and Majority baselines for VoltAhead.

All probabilities are estimated from the approved fitting data only
(Dec 1-29, both ``time`` and ``target_time`` inside the period).  Validation
and test labels are never used.  ``y_prob`` is always P(Available = 1).

Model identifiers / versions (one pair per prediction frame, as required by
``evaluation.validate_prediction_output``):

    persistence / deterministic   y_prob in {0.0, 1.0}  (current_idle > 0)
    persistence / probability     y_prob = P(target=1 | current_available, horizon)
    majority    / global          y_prob = global available share (required baseline)
    majority    / horizon         y_prob = available share per horizon (extra)

Smoothing rule (persistence/probability and majority/horizon)
-------------------------------------------------------------
    p = (k + m * prior) / (n + m)       m = SMOOTHING_STRENGTH (10 pseudo-observations)
k = available labels in the group, n = group size,
prior = global available share of the same fitting data
        (persistence: share for the same horizon; majority/horizon: global share).
An empty group (n = 0) falls back to the prior.  With ~100k+ rows per group the
smoothing is practically inactive; it only protects tiny/empty groups.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

THRESHOLD = 0.5
SMOOTHING_STRENGTH = 10
HORIZONS = [30, 40, 60]
PREDICTION_COLUMNS = [
    "station_id", "time", "target_time", "prediction_horizon",
    "y_true", "y_pred", "y_prob", "model_name", "model_version", "threshold",
]


def current_available(df: pd.DataFrame) -> np.ndarray:
    """1 if current_idle > 0 else 0."""
    return (df["current_idle"].to_numpy() > 0).astype(int)


def _smooth(k, n, prior, m=SMOOTHING_STRENGTH):
    return (k + m * prior) / (n + m)


# ---------------------------------------------------------------- persistence
def fit_persistence_transitions(fit_df: pd.DataFrame, m: int = SMOOTHING_STRENGTH) -> pd.DataFrame:
    """Six-row table: current_available x prediction_horizon."""
    d = pd.DataFrame({
        "current_available": current_available(fit_df),
        "prediction_horizon": fit_df["prediction_horizon"].to_numpy(),
        "target": fit_df["target"].to_numpy().astype(int),
    })
    grid = pd.MultiIndex.from_product([[0, 1], HORIZONS],
                                      names=["current_available", "prediction_horizon"]).to_frame(index=False)
    g = d.groupby(["current_available", "prediction_horizon"])["target"].agg(
        n_observations="size", n_available="sum").reset_index()
    t = grid.merge(g, how="left").fillna({"n_observations": 0, "n_available": 0})
    t[["n_observations", "n_available"]] = t[["n_observations", "n_available"]].astype(int)
    t["n_unavailable"] = t["n_observations"] - t["n_available"]
    prior = d.groupby("prediction_horizon")["target"].mean()
    t["prior_horizon_rate"] = t["prediction_horizon"].map(prior).fillna(d["target"].mean())
    t["raw_available_rate"] = np.where(t["n_observations"] > 0, t["n_available"] / t["n_observations"].clip(lower=1), np.nan)
    t["smoothed_available_rate"] = _smooth(t["n_available"], t["n_observations"], t["prior_horizon_rate"], m)
    t["smoothing_strength"] = m
    t["fallback_used"] = t["n_observations"] == 0
    t["current_state"] = t["current_available"].map({0: "unavailable", 1: "available"})
    return t[["current_state", "current_available", "prediction_horizon", "n_observations", "n_available",
              "n_unavailable", "raw_available_rate", "prior_horizon_rate", "smoothed_available_rate",
              "smoothing_strength", "fallback_used"]]


def _frame(df, y_prob, name, version, threshold=THRESHOLD):
    y_prob = np.asarray(y_prob, dtype="float64")
    out = df[["station_id", "time", "target_time", "prediction_horizon"]].copy()
    out["y_true"] = df["target"].astype(int).to_numpy()
    out["y_pred"] = (y_prob >= threshold).astype(int)
    out["y_prob"] = y_prob
    out["model_name"] = name
    out["model_version"] = version
    out["threshold"] = float(threshold)
    return out[PREDICTION_COLUMNS].reset_index(drop=True)


def predict_persistence_deterministic(df: pd.DataFrame) -> pd.DataFrame:
    """Original rule: current_idle > 0 -> available. y_prob is the hard 0/1 score."""
    return _frame(df, current_available(df).astype(float), "persistence", "deterministic")


def predict_persistence_probability(df: pd.DataFrame, transitions: pd.DataFrame) -> pd.DataFrame:
    key = transitions.set_index(["current_available", "prediction_horizon"])["smoothed_available_rate"]
    idx = pd.MultiIndex.from_arrays([current_available(df), df["prediction_horizon"].to_numpy()])
    return _frame(df, key.reindex(idx).to_numpy(), "persistence", "probability")


# ------------------------------------------------------------------- majority
def fit_majority(fit_df: pd.DataFrame, m: int = SMOOTHING_STRENGTH) -> pd.DataFrame:
    """Global row plus per-horizon rows (the horizon rows are the optional extra)."""
    y = fit_df["target"].astype(int)
    n, k = len(y), int(y.sum())
    rows = [{"scope": "global", "prediction_horizon": "all", "n_observations": n, "n_available": k,
             "n_unavailable": n - k, "available_rate": k / n, "probability_used": k / n,
             "majority_class": int(k / n >= 0.5), "required": True}]
    for h in HORIZONS:
        yh = y[fit_df["prediction_horizon"].to_numpy() == h]
        nh, kh = len(yh), int(yh.sum())
        rows.append({"scope": "horizon", "prediction_horizon": h, "n_observations": nh, "n_available": kh,
                     "n_unavailable": nh - kh, "available_rate": kh / nh if nh else np.nan,
                     "probability_used": _smooth(kh, nh, k / n, m), "majority_class": int(_smooth(kh, nh, k / n, m) >= 0.5),
                     "required": False})
    return pd.DataFrame(rows)


def predict_majority_global(df, table):
    p = float(table.loc[table["scope"] == "global", "probability_used"].iloc[0])
    return _frame(df, np.full(len(df), p), "majority", "global")


def predict_majority_horizon(df, table):
    t = table[table["scope"] == "horizon"].copy()
    t["prediction_horizon"] = t["prediction_horizon"].astype(int)
    p = df["prediction_horizon"].map(t.set_index("prediction_horizon")["probability_used"])
    return _frame(df, p.to_numpy(), "majority", "horizon")


def predict_all_baselines(df, transitions, majority_table):
    return {
        ("persistence", "deterministic"): predict_persistence_deterministic(df),
        ("persistence", "probability"): predict_persistence_probability(df, transitions),
        ("majority", "global"): predict_majority_global(df, majority_table),
        ("majority", "horizon"): predict_majority_horizon(df, majority_table),
    }
