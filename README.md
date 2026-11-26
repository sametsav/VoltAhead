# VoltAhead

EV Charging Station Availability Prediction and Recommendation System.

## Project Overview

VoltAhead is a machine learning-based system that predicts future EV charging station availability using historical charging data and recommends suitable charging stations.

**Course:** ISE308 – Artificial Intelligence and Expert Systems

## Dataset

We use the UrbanEV dataset.

Download: https://doi.org/10.5061/dryad.np5hqc04z

Download and extract the dataset into the `data/` directory.

Do not upload dataset files to GitHub.

## Project Structure

```text
VoltAhead/
├── data/
├── notebooks/
├── src/
│   └── data_loader.py
├── outputs/
│   ├── figures/
│   └── tables/
├── .gitignore
├── requirements.txt
└── README.md
```

## Installation

### 1. Clone the repository

```bash
git clone REPOSITORY_URL
cd VoltAhead
```

### 2. Create a virtual environment

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows (PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up VS Code

- Install the Python extension.
- Install the Jupyter extension.
- Open the VoltAhead folder.
- Select `.venv` as the notebook kernel.

## Step 1 – Exploratory Data Analysis

| Team Member | Responsibility | Notebook |
|---|---|---|
| Efe | Dataset Structure & Data Quality | 01_data_quality.ipynb |
| Nehir | Temporal Analysis & Autocorrelation | 02_temporal_autocorrelation.ipynb |
| Zümra | Charging Patterns & Temporal Usage | 03_charging_patterns.ipynb |
| Samet | Station Availability & Target Distribution | 04_availability_distribution.ipynb |

## Branches

Each team member should work on their own branch.

- Efe: `eda/data-quality`
- Nehir: `eda/temporal-analysis`
- Zümra: `eda/charging-patterns`
- Samet: `eda/availability`
- Step 2 – Shared Preprocessing: `preprocessing/feature-engineering`

Create your branch:

```bash
git checkout -b YOUR_BRANCH_NAME
```

Save and upload your work:

```bash
git add .
git commit -m "Describe your changes"
git push -u origin YOUR_BRANCH_NAME
```

Create a Pull Request when your work is ready for review.

## EDA Deliverables

Each team member should provide:

- One executable Jupyter Notebook.
- Relevant visualizations and analysis tables.
- Key findings.
- Implications for machine learning.

Save notebooks in `notebooks/`, figures in `outputs/figures/`, and tables in `outputs/tables/`.

## Collaboration Rules

- Do not push directly to the main branch during development.
- Do not upload raw datasets or virtual environments.
- Use descriptive commit messages.
- Document your findings in English.
- Keep notebooks reproducible.


## Step 2 – Data Preprocessing & Feature Engineering

### Objective

Step 2 prepares a shared, reproducible dataset for machine learning models that predict future EV charging station availability.

The preprocessing pipeline is implemented in `src/preprocessing.py`, with analysis, validation, and visualizations documented in `notebooks/05_preprocessing.ipynb`.

### Dataset Configuration

The team selected the following configuration based on data quality, missing value analysis, and chronological split comparisons:

| Property | Configuration |
|---|---|
| Dataset period | December 1, 2022 – January 21, 2023 |
| Dataset length | 52 days |
| Missing value strategy | Complete-case |
| Prediction horizons | 30, 40, 60 minutes |
| Training period | Dec 1, 2022 – Jan 5, 2023 |
| Validation period | Jan 6 – Jan 13, 2023 |
| Test period | Jan 14 – Jan 21, 2023 |
| Input features | 16 candidate features |
| Target | Binary future charger availability |

The 52-day dataset was selected because it provides more consistent observation coverage than the full 90-day period.

The full dataset has a missing observation rate of approximately 25.46%, compared with 9.79% for the selected period.

**Limitation:** The shorter period excludes January 22 – February 28, 2023. This remaining 38-day period is reserved for additional robustness evaluation after final model selection.

### Input Features

The shared dataset contains the following 16 candidate input features:

- `station_id`
- `hour`
- `day_of_week`
- `is_weekend`
- `current_idle`
- `current_busy`
- `station_capacity`
- `occupancy_rate`
- `idle_lag_5`
- `idle_lag_10`
- `idle_lag_15`
- `idle_lag_30`
- `idle_lag_40`
- `idle_lag_60`
- `idle_lag_120`
- `prediction_horizon`

The target represents future availability:

- `1`: At least one charger is available at the prediction horizon.
- `0`: No charger is available at the prediction horizon.

Observations with missing target values are excluded.

### Generate the Model-Ready Datasets

After installing dependencies and setting up the UrbanEV dataset according to the Dataset section, run the shared preprocessing pipeline from a notebook:

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd().parent / "src"))

from preprocessing import run_preprocessing, FEATURE_COLUMNS

train, validation, test = run_preprocessing()
```

The pipeline loads data through the existing shared data loader, creates features and targets, applies complete-case filtering, performs chronological splitting, and validates the resulting datasets.

### Machine Learning Interface

All team members should use the same feature list and target definition:

```python
X_train = train[FEATURE_COLUMNS]
y_train = train["target"]

X_val = validation[FEATURE_COLUMNS]
y_val = validation["target"]

X_test = test[FEATURE_COLUMNS]
y_test = test["target"]
```

`time` and `target_time` are retained for validation but are not included in the model input features.

Model-specific encoding, scaling, and class imbalance handling should be performed in the appropriate training pipelines.

The test set must not be used for model selection or hyperparameter tuning.

### Final Dataset Summary

| Dataset | Prediction Examples | Stations |
|---|---:|---:|
| Training | 2,798,569 | 91 |
| Validation | 618,726 | 90 |
| Testing | 620,433 | 93 |
| **Total** | **4,037,728** | — |

The final datasets contain no missing input features or missing target labels.

### Unseen Stations

The validation dataset contains one station not observed during training: `1948`.

The test dataset contains two stations not observed during training: `1790` and `1948`.

These stations are retained in the datasets and can be evaluated separately.

The primary modeling objective is future availability prediction for existing stations.

### Outputs

The following model-ready datasets are generated locally:

```text
data/processed/train.parquet
data/processed/validation.parquet
data/processed/test.parquet
```

These files are excluded from Git and can be regenerated using the shared preprocessing pipeline.

Analysis figures are available in `outputs/figures/`, and validation and summary tables are available in `outputs/tables/`.

### Validation

The final preprocessing pipeline verifies:

- Valid binary target labels
- Complete input features
- Correct prediction horizon values
- Valid station features and historical lag features
- No duplicate station-time-horizon combinations
- Chronological split boundaries
- Correct target timestamp alignment
- No future target information in the model input feature list

The reusable module was verified to reproduce the notebook-generated datasets exactly.
