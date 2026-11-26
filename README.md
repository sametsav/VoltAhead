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
