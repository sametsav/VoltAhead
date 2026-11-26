import pandas as pd
from pathlib import Path
import data_loader as dl

ROOT = Path(__file__).resolve().parents[1]

# Read selected station metadata
stations = pd.read_csv(
    ROOT / "outputs/tables/selected_stations.csv"
)

ids = stations["station_id"].astype(int).tolist()

assert len(ids) == 100
assert len(set(ids)) == 100

# Study period
START = "2022-12-01 00:00:00"
END = "2023-02-28 23:55:00"

frames = []

# Load and filter each station
for i, station_id in enumerate(ids, start=1):

    df = dl.load_station(station_id, reindex=True)

    df = df.loc[START:END].copy()

    df["station_id"] = station_id

    frames.append(df.reset_index())

    print(f"[{i}/100] Station {station_id} loaded")

# Combine all stations
dataset = pd.concat(frames, ignore_index=True)

# Add station metadata
dataset = dataset.merge(
    stations,
    on="station_id",
    how="left",
    validate="many_to_one"
)

# Sort observations
dataset = dataset.sort_values(
    ["station_id", "time"]
).reset_index(drop=True)

# Move station_id and time to the beginning
columns = ["station_id", "time"]

columns += [
    c for c in dataset.columns
    if c not in columns
]

dataset = dataset[columns]

# Validate dataset
assert len(dataset) == 2592000
assert dataset["station_id"].nunique() == 100

# Save combined dataset
output = ROOT / "data/selected_stations_data.csv"

dataset.to_csv(output, index=False)

print("\nDataset created successfully!")
print("Shape:", dataset.shape)
print("Stations:", dataset["station_id"].nunique())
print("Start:", dataset["time"].min())
print("End:", dataset["time"].max())
print("Missing idle values:", dataset["idle"].isna().sum())
print("Saved to:", output)