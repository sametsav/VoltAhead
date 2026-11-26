"""
Shared data access layer for the VoltAhead project.

Dataset: ``VoltAhead_Dataset`` - 100 EV charging stations selected from UrbanEV
(Shenzhen), 5-minute observations from 2022-12-01 to 2023-02-28. It consists of
two CSV files:

    selected_stations.csv         one row per station (metadata)
    selected_stations_data.csv    long format, one row per station x 5-min step

Everybody on the team should load data ONLY through this module so that we all
share the same station IDs, the same timestamp grid and the same sample.

Quick start
-----------
>>> import sys; sys.path.append("../src")        # from a notebook in notebooks/
>>> import data_loader as dl
>>> info = dl.load_station_info()                  # 100 rows, one per station
>>> df   = dl.load_data()                          # long: MultiIndex (station_id, time)
>>> busy = dl.load_stations_wide(variable="busy")  # time x station matrix
>>> ids  = dl.get_sample_station_ids()             # shared 20-station dev sample
>>> s    = dl.load_station(ids[0])                 # one station, full 5-min grid

Where is the data?
------------------
Put both CSV files in ``<project>/data/`` (or in a sub-folder of it, e.g.
``data/VoltAhead_Dataset/``). To keep them somewhere else, set the environment
variable ``VOLTAHEAD_DATA_DIR`` to the folder that contains the two files.
The first call converts the 300 MB CSV to a Parquet cache in ``data/.cache/``
(ignored by git) so later loads take about a second; this needs ``pyarrow``
and silently falls back to the CSV if it is missing.

Conventions (identical for everyone)
------------------------------------
* ``station_id`` and ``TAZID`` are ``int``.
* Time index is a tz-naive ``DatetimeIndex`` named ``time``: a 5-minute grid from
  2022-12-01 00:00 to 2023-02-28 23:55 (25,920 steps). The files carry no time
  zone; the coordinates are in Shenzhen, so treat timestamps as local time.
* The 7 static columns that the big CSV repeats on every row (longitude,
  latitude, slow_count, fast_count, charge_count, TAZID, distance_km) are NOT
  returned by ``load_data``; they live in ``load_station_info()``. Use
  ``attach_station_info(df)`` if you want them on every row.
* A missing observation is NOT a missing row: the row exists but ALL 10 value
  columns are NaN (see notebooks/01_data_quality.ipynb). Use ``valid_mask`` to
  separate real observations from gaps. ``load_data`` re-indexes to the full
  grid by default (so nothing disappears silently); ``reindex=False`` returns
  the file exactly as stored, which is what data-quality checks need.
* Returned frames are copies; the in-memory cache is never modified.
"""

from __future__ import annotations

import os
import random
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parents[1]

STATIONS_FILE = "selected_stations.csv"
DATA_FILE = "selected_stations_data.csv"

START = pd.Timestamp("2022-12-01 00:00:00")
END = pd.Timestamp("2023-02-28 23:55:00")
FREQ = "5min"
N_STEPS = 25_920  # 90 days x 288

STATION_COLUMNS = [
    "busy", "idle", "s_price", "e_price",
    "fast_busy", "fast_idle", "slow_busy", "slow_idle",
    "duration", "volume",
]
STATIC_COLUMNS = [
    "longitude", "latitude", "slow_count", "fast_count",
    "charge_count", "TAZID", "distance_km",
]

# Shared development sample.
SAMPLE_SEED = 42
SAMPLE_SIZE = 20
SAMPLE_MAX_NAN_SHARE = 0.5


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
def get_data_root() -> Path:
    """Return the folder that contains the two dataset CSV files."""
    candidates = []
    env = os.environ.get("VOLTAHEAD_DATA_DIR")
    if env:
        candidates.append(Path(env).expanduser())
    data_dir = PROJECT_ROOT / "data"
    candidates.append(data_dir)
    if data_dir.is_dir():
        candidates += sorted(p for p in data_dir.iterdir() if p.is_dir() and not p.name.startswith("."))

    for c in candidates:
        if (c / STATIONS_FILE).is_file() and (c / DATA_FILE).is_file():
            return c
    tried = "\n  ".join(str(c) for c in candidates) or "(none)"
    raise FileNotFoundError(
        f"Could not find {STATIONS_FILE} and {DATA_FILE}. Put them into the project's "
        "data/ folder (see README) or set VOLTAHEAD_DATA_DIR. Looked in:\n  " + tried
    )


def stations_path() -> Path:
    return get_data_root() / STATIONS_FILE


def data_path() -> Path:
    return get_data_root() / DATA_FILE


# --------------------------------------------------------------------------- #
# Time grid
# --------------------------------------------------------------------------- #
def expected_index() -> pd.DatetimeIndex:
    """The complete, gap-free 5-minute grid the dataset is supposed to cover."""
    return pd.date_range(START, END, freq=FREQ, name="time")


# --------------------------------------------------------------------------- #
# Station metadata
# --------------------------------------------------------------------------- #
def load_station_info() -> pd.DataFrame:
    """
    Station metadata, one row per station, sorted by ``station_id``.

    Columns: station_id, longitude, latitude, slow_count, fast_count,
    charge_count (= capacity = slow + fast), TAZID (traffic zone),
    distance_km (distance of the station to the dataset's reference point; the
    point itself is not stored in the files).

    Note: selected_stations.csv is sorted by distance_km; this function sorts by
    station_id to match the order of the time-series file.
    """
    df = pd.read_csv(stations_path())
    df["station_id"] = df["station_id"].astype(int)
    df["TAZID"] = df["TAZID"].astype(int)
    return df.sort_values("station_id").reset_index(drop=True)


def list_station_ids() -> list[int]:
    """Sorted list of all station IDs (100)."""
    return load_station_info()["station_id"].tolist()


# --------------------------------------------------------------------------- #
# Time series
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def _load_stored() -> pd.DataFrame:
    """The big CSV exactly as stored (value columns only), cached in memory/Parquet."""
    path = data_path()
    stat = path.stat()
    cache = path.parent / ".cache" / f"{path.stem}_{stat.st_size}_{stat.st_mtime_ns}.parquet"

    if cache.is_file():
        try:
            return pd.read_parquet(cache)
        except Exception:  # corrupt cache or no pyarrow -> fall back to CSV
            pass

    df = pd.read_csv(
        path,
        usecols=["station_id", "time", *STATION_COLUMNS],
        dtype={c: "float32" for c in STATION_COLUMNS} | {"station_id": "int32"},
        parse_dates=["time"],
    )
    df = df.sort_values(["station_id", "time"], kind="stable").set_index(["station_id", "time"])

    try:
        cache.parent.mkdir(exist_ok=True)
        df.to_parquet(cache)
    except Exception:
        pass
    return df


def load_data(
    station_ids: Iterable[int] | None = None,
    columns: Sequence[str] | None = None,
    reindex: bool = True,
) -> pd.DataFrame:
    """
    Time series in long format with a (station_id, time) MultiIndex.

    Parameters
    ----------
    station_ids : stations to load (default: all 100).
    columns     : subset of STATION_COLUMNS (default: all 10).
    reindex     : if True (default) every station covers the full expected grid
                  (25,920 rows) and missing timestamps would show up as NaN
                  rows. If False the data is returned exactly as stored
                  (duplicates / gaps untouched) - use this for quality checks.
    """
    df = _load_stored()
    if station_ids is not None:
        ids = [int(i) for i in station_ids]
        df = df.loc[df.index.get_level_values("station_id").isin(ids)]
    if columns is not None:
        df = df[list(columns)]
    df = df.copy()
    if reindex:
        ids = df.index.get_level_values("station_id").unique()
        df = df[~df.index.duplicated(keep="first")]
        full = pd.MultiIndex.from_product([ids, expected_index()], names=["station_id", "time"])
        df = df.reindex(full)
    return df


def load_station(
    station_id: int,
    columns: Sequence[str] | None = None,
    reindex: bool = True,
) -> pd.DataFrame:
    """One station's 5-minute series indexed by ``time``."""
    df = load_data([station_id], columns, reindex)
    return df.droplevel("station_id")


def load_stations_wide(
    station_ids: Iterable[int] | None = None,
    variable: str = "busy",
    reindex: bool = True,
) -> pd.DataFrame:
    """One variable as a ``time x station_id`` matrix (default: all stations)."""
    df = load_data(station_ids, [variable], reindex)
    wide = df[variable].unstack("station_id")
    wide.columns = wide.columns.astype(int)
    return wide


def attach_station_info(df: pd.DataFrame) -> pd.DataFrame:
    """Add the static station columns (capacity, TAZID, ...) to a frame that has a station_id index level."""
    info = load_station_info().set_index("station_id")[STATIC_COLUMNS]
    ids = df.index.get_level_values("station_id")
    out = df.copy()
    for c in STATIC_COLUMNS:
        out[c] = info[c].reindex(ids).to_numpy()
    return out


def valid_mask(df: pd.DataFrame) -> pd.Series:
    """True for rows that hold a real observation (gaps are all-NaN rows)."""
    return df["busy"].notna()


def station_nan_share() -> pd.Series:
    """Share of all-NaN rows per station over the full period (index: station_id)."""
    df = load_data(columns=["busy"])
    return df["busy"].isna().groupby(level="station_id").mean().rename("nan_share")


# --------------------------------------------------------------------------- #
# Shared development sample
# --------------------------------------------------------------------------- #
def get_sample_station_ids(
    n: int = SAMPLE_SIZE,
    seed: int = SAMPLE_SEED,
    max_nan_share: float | None = SAMPLE_MAX_NAN_SHARE,
) -> list[int]:
    """
    Deterministic shared subset of stations for fast development.

    Everyone calling this with the defaults gets exactly the same stations, so
    results are comparable between notebooks. Final runs should use all
    stations (``list_station_ids()``).

    max_nan_share : only stations whose share of all-NaN rows is at most this
        value are eligible (default 0.5, which drops 10 of the 100 stations).
        Pass None to draw from all 100.
    """
    ids = list_station_ids()
    if max_nan_share is not None:
        share = station_nan_share()
        ids = [i for i in ids if share.loc[i] <= max_nan_share]
    rng = random.Random(seed)
    return sorted(rng.sample(ids, min(n, len(ids))))
