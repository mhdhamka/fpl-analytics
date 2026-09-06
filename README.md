<div align="center">

# Premier League Performance Analytics Pipeline (`pl-analytics`)

A production-grade, modular Python analytics and machine learning pipeline built to ingest, process, visualize, and model real-time player data from the Premier League. Designed with clean architecture principles to deliver data-driven insights into player performance, expected metrics ($xG$), and points prediction.

[Live Demo](https://fpremier-league-analytics.streamlit.app/) · [Report Bug](https://github.com/mhdhamka/pl-analytics/issues) · [Request Feature](https://github.com/mhdhamka/pl-analytics/issues)

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/Python-3.x-3776ab?logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.x-150458?logo=pandas&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-EB6864?logo=xgboost&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-3776AB?logo=python&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3776AB?logo=python&logoColor=white)

</div>

---

## Overview

The **Premier League Performance Analytics Pipeline** is an end-to-end data engineering and predictive modeling project tailored for sports analytics and Fantasy Premier League (FPL) management. 

Managing raw football data can be messy and fragmented. This project automates the entire lifecycle—from pulling live data straight off official league endpoints to structuring clean datasets, generating exploratory data visualizations, and deploying machine learning models to forecast player performance. It highlights professional software engineering standards, featuring modular design (`src/`), a master orchestration script (`main.py`), and interactive EDA notebooks.

---

## Tech Stack & Libraries

* **Language:** Python 3.x
* **Data Manipulation & Analysis:** Pandas, NumPy
* **Data Visualization:** Seaborn, Matplotlib
* **Machine Learning:** Scikit-learn (Random Forest Regressor), XGBoost
* **Data Ingestion & DevOps:** Requests (Live FPL API), Pytest,

---

## Core Features & Pipeline Steps

1. **Automated Data Ingestion (`src/ingest.py`)**: Implements fault-tolerant HTTP integration to pull live bootstrap data straight from the official Premier League API, structuring raw JSON payloads into version-controlled backups.

2. **Robust Data Wrangling & Engineering (`src/clean.py`)**: Executes programmatic relational mapping (merging player attributes with team and position metadata), handles missing data vectors, and engineers advanced features like scaled market valuations (`market_value_m`).

3. **Automated Visualization Engine (`src/visualize.py`)**: Programmatically generates and serializes publication-grade analytics plots using Seaborn and Matplotlib with automated directory provisioning:
   * Top 10 Goalscorers Bar Chart (`outputs/figures/top_goalscorers.png`)
   * Expected Goals ($xG$) vs. Actual Goals Scatter Plot (`outputs/figures/xg_vs_actual_goals.png`)

4. **Exploratory Data Analysis (`notebooks/01_eda.ipynb`)**: Features a structured exploratory notebook containing statistical distributions, positional performance benchmarking, and feature correlation heatmaps.

5. **Advanced Predictive Modeling & Benchmarking (`src/model.py`)**: Benchmarks supervised regressors (**Random Forest vs. XGBoost**) using rigorous **5-Fold Cross-Validation** to forecast player fantasy point returns based on underlying core metrics, yielding high predictive fidelity ($R^2 \approx 0.84$) and automated feature importance tracking.

6. **Centralized Configuration Architecture (`src/config.py`)**: Implements a single-source-of-truth configuration pattern managing all global paths, API endpoints, and feature vectors, completely eradicating hardcoded magic strings from the codebase.

---

## Getting Started & Installation

### 1. Clone the Repository

```cmd
git clone https://github.com/mhdhamka/pl-analytics.git
cd pl-analytics

```

### 2. Install Dependencies

```cmd
pip install -r requirements.txt

```

### 3. Run the Entire Pipeline

Execute the master execution script to fetch fresh data, clean it, generate updated charts, and train the machine learning model in one command:

```cmd
python main.py

```

### 4. Run Unit Tests

To execute the test suite locally using pytest:

```cmd
python -m pytest

```

---

## Model Performance Overview

* **Algorithm:** Random Forest Regressor
* **Evaluation Strategy**: 5-Fold Cross-Validation & Hold-out Test Split
* **Target Variable:** Total Points
* **Performance ($R^2$ Score):** ~0.84
* **Key Drivers:** Minutes played and goals scored hold the highest feature importances in driving overall player output.

### Sample Visualization: Top Goalscorers
![Top Goalscorers](outputs/figures/top_goalscorers.png)

---

## Project Architecture

```text
pl-analytics/
│
├── .github/
│   └── workflows/
│       └── pipeline.yml    # GitHub Actions CI/CD automation workflow
│
├── data/
│   ├── raw/                # Original API responses or raw CSV backups
│   └── processed/          # Cleaned datasets saved as CSV
│
├── notebooks/
│   └── 01_eda.ipynb        # Jupyter notebooks for exploratory data analysis
│
├── src/
│   ├── __init__.py         # Makes src a package
│   ├── ingest.py           # API connection and live data fetching[cite: 3]
│   ├── clean.py            # Preprocessing, filtering, and feature engineering[cite: 2]
│   ├── visualize.py        # Matplotlib/Seaborn automated plotting functions[cite: 1]
│   └── model.py            # Scikit-learn machine learning regression model
│
├── tests/
│   └── test_config.py      # Pytest unit tests for configuration and setup
│
├── outputs/
│   └── figures/            # Saved chart images and visual exports (top scorers, xG plots)
│
├── .gitignore              # Files to ignore (e.g., venv, __pycache__)
├── requirements.txt        # Project dependencies (pandas, requests, seaborn, scikit-learn)
└── main.py                 # Master execution script to run the pipeline end-to-end[cite: 1, 2, 3]

```

---

## Contributing

Issues and pull requests are welcome. If you're picking up one of the Roadmap items above, please open an issue first so effort isn't duplicated.

1. Fork the repo
2. Create a feature branch (`git checkout -b feature/real-data-integration`)
3. Commit your changes
4. Open a pull request describing what changed and why

---

## License

Distributed under the MIT License.

---

<div align="center">

If you found this project interesting, consider giving it a star ⭐

Crafted by **[@mhdhamka](https://github.com/mhdhamka)**

</div>

