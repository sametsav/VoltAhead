"""
Shared data access layer for the VoltAhead project (UrbanEV dataset).

Everybody on the team should load data ONLY through this module so that we all
share the same station IDs, the same timestamp structure and the same sample
of stations.

Quick start
-----------
>>> import sys; sys.path.append("../src")            # from a notebook in notebooks/
>>> import data_loader as dl
>>> ids = dl.get_sample_station_ids()                 # shared 50-station subset (cleaned pool)
>>> df  = dl.load_station(ids[0])                    # one station, 5-min grid
>>> busy = dl.load_stations_wide(ids, "busy")        # time x station matrix
>>> zones = dl.load_zone_variable("occupancy")       # time x TAZID matrix

Where is the data?
------------------
By default the loader looks inside ``<project>/data/``. It accepts both layouts:

    data/UrbanEVDataset/20220901-20230228_station-raw/...
    data/20220901-20230228_station-raw/...

To keep the data somewhere else, set the environment variable
``VOLTAHEAD_DATA_DIR`` to the folder that contains the two dataset folders.

Conventions (identical for everyone)
------------------------------------
* ``station_id`` is always a Python/NumPy ``int`` (the raw pile file stores it as float).
* ``TAZID`` (zone id) is always ``int``.
* Time index is a tz-naive ``DatetimeIndex`` named ``time``, 5-minute grid from
  2022-09-01 00:00 to 2023-02-28 23:55 (52,128 steps) for the 5-min data and
  hourly (4,344 steps) for the 1-hour zone data. The files carry no time zone;
  the station coordinates are in Shenzhen, so treat timestamps as local time.
* In the raw station files a missing observation is NOT a missing row: the row
  exists but ALL value columns are NaN (see notebooks/01_data_quality.ipynb).
  Use ``valid_mask`` / ``.dropna()`` to separate real observations from gaps.
* ``load_station`` re-indexes to the full grid by default, so missing
  timestamps appear as NaN rows instead of silently disappearing. Pass
  ``reindex=False`` to get the file exactly as stored.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR_NAME = "20220901-20230228_station-raw"
ZONE_DIR_NAME = "20220901-20230228_zone-cleaned-aggregated"

START = pd.Timestamp("2022-09-01 00:00:00")
END_5MIN = pd.Timestamp("2023-02-28 23:55:00")
END_1H = pd.Timestamp("2023-02-28 23:00:00")

# Shared sample used by every notebook while we develop code.
SAMPLE_SEED = 42
SAMPLE_SIZE = 50

STATION_COLUMNS = [
    "busy", "idle", "s_price", "e_price",
    "fast_busy", "fast_idle", "slow_busy", "slow_idle",
    "duration", "volume",
]
ZONE_VARIABLES = ["occupancy", "duration", "volume", "volume-11kw", "e_price", "s_price"]

# Smallest dtypes that are safe for each raw column (cuts memory ~3x).
_STATION_DTYPES = {
    "busy": "float32", "idle": "float32",
    "fast_busy": "float32", "fast_idle": "float32",
    "slow_busy": "float32", "slow_idle": "float32",
    "s_price": "float32", "e_price": "float32",
    "duration": "float32", "volume": "float32",
}


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
def get_data_root() -> Path:
    """Return the folder that contains the two UrbanEV dataset folders."""
    candidates = []
    env = os.environ.get("VOLTAHEAD_DATA_DIR")
    if env:
        candidates.append(Path(env).expanduser())
    data_dir = PROJECT_ROOT / "data"
    candidates += [data_dir, data_dir / "UrbanEVDataset"]
    if data_dir.is_dir():
        candidates += sorted(p for p in data_dir.iterdir() if p.is_dir())

    for c in candidates:
        if (c / RAW_DIR_NAME).is_dir() and (c / ZONE_DIR_NAME).is_dir():
            return c
    tried = "\n  ".join(str(c) for c in candidates) or "(none)"
    raise FileNotFoundError(
        "UrbanEV dataset not found. Extract it into the project's data/ folder "
        "(see README) or set VOLTAHEAD_DATA_DIR. Looked in:\n  " + tried
    )


def raw_dir() -> Path:
    return get_data_root() / RAW_DIR_NAME


def zone_dir() -> Path:
    return get_data_root() / ZONE_DIR_NAME


# --------------------------------------------------------------------------- #
# Time grids
# --------------------------------------------------------------------------- #
def expected_index(freq: str = "5min") -> pd.DatetimeIndex:
    """The complete, gap-free time grid the dataset is supposed to cover."""
    if freq in ("5min", "5T"):
        return pd.date_range(START, END_5MIN, freq="5min", name="time")
    if freq in ("1hour", "1h", "h", "H"):
        return pd.date_range(START, END_1H, freq="1h", name="time")
    raise ValueError(f"Unknown freq {freq!r}; use '5min' or '1hour'.")


# --------------------------------------------------------------------------- #
# Static tables
# --------------------------------------------------------------------------- #
def load_station_info(source: str = "raw") -> pd.DataFrame:
    """
    Station metadata, one row per station, ``station_id`` as int column.

    source="raw"  : all 1,682 stations; has slow_count, fast_count, charge_count,
                    longitude, latitude, TAZID.
    source="zone" : the 1,362 stations kept after cleaning; has charge_count,
                    longitude, latitude, TAZID only.
    """
    path = {"raw": raw_dir(), "zone": zone_dir()}[source] / "station_information.csv"
    df = pd.read_csv(path)
    df["station_id"] = df["station_id"].astype(int)
    df["TAZID"] = df["TAZID"].astype(int)
    return df.sort_values("station_id").reset_index(drop=True)


def load_pile_info() -> pd.DataFrame:
    """Rated power / type (AC or DC) of every individual charging pile."""
    df = pd.read_csv(raw_dir() / "pile_rated_power.csv")
    df["station_id"] = df["station_id"].astype(int)
    df["pile_id"] = df["pile_id"].astype("int64")
    return df


def load_zone_info() -> pd.DataFrame:
    """Traffic analysis zone (TAZ) table: id, centroid, charge_count, area, perimeter."""
    df = pd.read_csv(zone_dir() / "zone-information.csv")
    df["TAZID"] = df["TAZID"].astype(int)
    return df


def load_zone_adjacency() -> pd.DataFrame:
    """Zone x zone 0/1 adjacency matrix (index and columns are TAZID)."""
    df = pd.read_csv(zone_dir() / "adj.csv")
    ids = df.columns.astype(int)
    df = df.iloc[:, : len(ids)]
    df.columns = ids
    df.index = ids
    return df


def load_zone_distance() -> pd.DataFrame:
    """Zone x zone distance matrix in metres (index and columns are TAZID)."""
    df = pd.read_csv(zone_dir() / "distance.csv")
    ids = df.columns.astype(int)
    df.columns = ids
    df.index = ids
    return df


def load_station_distance() -> pd.DataFrame:
    """Station x station distance matrix in metres (index and columns are station_id)."""
    df = pd.read_csv(raw_dir() / "station_distance.csv", index_col=0)
    df.index = df.index.astype(int)
    df.columns = df.columns.astype(int)
    return df


def list_station_ids(source: str = "raw") -> list[int]:
    """Sorted list of station IDs (from the metadata table, not from the folder)."""
    return load_station_info(source)["station_id"].tolist()


def get_sample_station_ids(
    n: int = SAMPLE_SIZE, seed: int = SAMPLE_SEED, pool: str = "cleaned"
) -> list[int]:
    """
    Deterministic shared subset of stations for fast development.

    Everyone calling this with the defaults gets exactly the same stations, so
    results are comparable between notebooks. Switch to ``list_station_ids()``
    for the final full-dataset runs.

    pool="cleaned" (default): draw from the 1,362 stations that survive in the
        zone-cleaned dataset (every one has < 30 % NaN rows).
    pool="all": draw from all 1,682 raw stations (includes 320 stations that
        the dataset authors dropped, many with > 50 % NaN rows).

    The raw 5-minute file of each station is always what ``load_station`` reads.
    """
    if pool not in ("cleaned", "all"):
        raise ValueError("pool must be 'cleaned' or 'all'")
    ids = list_station_ids("zone" if pool == "cleaned" else "raw")
    rng = random.Random(seed)
    return sorted(rng.sample(ids, min(n, len(ids))))


def valid_mask(df: pd.DataFrame) -> pd.Series:
    """True for rows that hold a real observation (raw gaps are all-NaN rows)."""
    return df["busy"].notna()


# --------------------------------------------------------------------------- #
# Station-level time series (raw, 5-minute)
# --------------------------------------------------------------------------- #
def station_file(station_id: int) -> Path:
    return raw_dir() / "charge_5min" / f"{int(station_id)}.csv"


def load_station(
    station_id: int,
    columns: Sequence[str] | None = None,
    reindex: bool = True,
) -> pd.DataFrame:
    """
    One station's 5-minute series, indexed by ``time``.

    Parameters
    ----------
    columns : subset of STATION_COLUMNS (default: all).
    reindex : if True (default) the result covers the full expected grid and
              missing timestamps are NaN rows. If False the file is returned
              as stored (duplicates and gaps untouched) - use this for data
              quality checks.
    """
    usecols = None
    if columns is not None:
        usecols = ["time", *columns]
    df = pd.read_csv(
        station_file(station_id),
        usecols=usecols,
        parse_dates=["time"],
        dtype={k: v for k, v in _STATION_DTYPES.items() if usecols is None or k in usecols},
    )
    df = df.set_index("time")
    if reindex:
        df = df[~df.index.duplicated(keep="first")].reindex(expected_index("5min"))
    return df


def load_stations(
    station_ids: Iterable[int] | None = None,
    columns: Sequence[str] | None = None,
    reindex: bool = True,
) -> pd.DataFrame:
    """
    Several stations stacked in long format with a (station_id, time) MultiIndex.
    ``station_ids=None`` loads the shared sample, NOT the full dataset.
    """
    ids = list(station_ids) if station_ids is not None else get_sample_station_ids()
    frames = {int(i): load_station(i, columns, reindex) for i in ids}
    out = pd.concat(frames, names=["station_id"])
    return out


def load_stations_wide(
    station_ids: Iterable[int] | None = None,
    variable: str = "busy",
    reindex: bool = True,
) -> pd.DataFrame:
    """
    One variable for many stations as a ``time x station_id`` matrix.
    ``station_ids=None`` loads the shared sample.
    """
    ids = list(station_ids) if station_ids is not None else get_sample_station_ids()
    cols = {int(i): load_station(i, [variable], reindex)[variable] for i in ids}
    wide = pd.DataFrame(cols)
    wide.columns.name = "station_id"
    return wide


# --------------------------------------------------------------------------- #
# Zone-level matrices (cleaned + aggregated)
# --------------------------------------------------------------------------- #
def load_zone_variable(variable: str = "occupancy", freq: str = "5min") -> pd.DataFrame:
    """
    A zone-level variable as a ``time x TAZID`` matrix.

    variable : one of ZONE_VARIABLES.
    freq     : "5min" (52,128 rows) or "1hour" (4,344 rows).
    """
    if variable not in ZONE_VARIABLES:
        raise ValueError(f"variable must be one of {ZONE_VARIABLES}")
    sub = {"5min": "charge_5min", "1hour": "charge_1hour"}.get(freq)
    if sub is None:
        raise ValueError("freq must be '5min' or '1hour'")
    df = pd.read_csv(zone_dir() / sub / f"{variable}.csv", parse_dates=["time"])
    df = df.set_index("time")
    df.columns = df.columns.astype(int)
    df.columns.name = "TAZID"
    return df.astype("float32")
