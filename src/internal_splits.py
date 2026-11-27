"""
Approved internal chronological splits (Efe's task, Section 4).

Both the prediction timestamp (``time``) and the future target timestamp
(``target_time``) must lie inside a period, so no label of one period is
observed after that period ends.
"""
import pandas as pd

INTERNAL_SPLITS = {
    "model_fit":   ("2022-12-01", "2022-12-24"),   # Dec 1-23
    "tuning":      ("2022-12-24", "2022-12-30"),   # Dec 24-29
    "calibration": ("2022-12-30", "2023-01-03"),   # Dec 30-Jan 2
    "threshold":   ("2023-01-03", "2023-01-06"),   # Jan 3-5
}
FIT_WITH_TUNING = ("2022-12-01", "2022-12-30")     # Dec 1-29 (refit / baseline fit)
DEV_SAMPLE_STEP = 4                                # every 4th 5-min timestamp


def period_mask(df, start, end):
    """Rows whose time AND target_time are in [start, end)."""
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    return (df["time"] >= s) & (df["time"] < e) & (df["target_time"] >= s) & (df["target_time"] < e)


def get_period(df, name):
    start, end = FIT_WITH_TUNING if name == "fit_tuning" else INTERNAL_SPLITS[name]
    return df.loc[period_mask(df, start, end)]


def development_sample(df_model_fit):
    """Deterministic 25% sample: keep every 4th distinct 5-minute timestamp."""
    ts = pd.Index(sorted(df_model_fit["time"].unique()))
    keep = ts[::DEV_SAMPLE_STEP]
    return df_model_fit[df_model_fit["time"].isin(keep)]
