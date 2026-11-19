# Architecture

### System Overview

`fpl-analytics` is organized into modular ingestion, cleaning, feature engineering, visualization, training, and inference layers:

1. **Ingestion (`src/ingest.py`):** Pulls live FPL data with retry/backoff logic, saves latest snapshots under `data/raw/`, and writes timestamped runs under `data/history/<timestamp>/`.
2. **Cleaning & Features (`src/clean.py`, `src/features.py`):** Merges player, team, and position tables, validates schemas with Pandera, and builds per-90, value, and position features into `data/processed/players_cleaned.csv`.
3. **Visualization & Modeling (`src/visualize.py`, `src/model.py`):** Generates charts under `outputs/figures/` and benchmarks/persists the trained model (`models/points_model.joblib` + `models/metadata.json`).
4. **Serving (`src/predict.py`, `app.py`, `api/main.py`):** Shares a single inference layer across the Streamlit dashboard and FastAPI service.

---

### Tech Stack & Libraries

- **Language:** Python 3.x
- **Data Manipulation & Validation:** Pandas, NumPy, Pandera
- **Web Services & UI:** FastAPI, Uvicorn, Streamlit
- **Machine Learning:** Scikit-learn (RandomizedSearchCV, Random Forest), XGBoost, Joblib
- **Data Visualization:** Seaborn, Matplotlib
- **Ingestion, DevOps & Testing:** Requests (FPL API), Pytest, Docker, DevContainers, GitHub Actions CI/CD

---

### Project Structure

```text
fpl-analytics/
├── .devcontainer/
│   └── devcontainer.json
├── .github/
│   └── workflows/
│       └── pipeline.yml             # CI: lint, test, smoke-test, docker build
├── api/
│   ├── __init__.py
│   └── main.py                      # FastAPI service (/predict, /players, /model/metadata)
├── assets/
│   └── images/
│       └── pl.png
├── data/
│   ├── history/                     # Timestamped snapshot runs (e.g., 20261119T174042Z/)
│   ├── processed/
│   │   └── players_cleaned.csv
│   └── raw/
│       ├── players_raw.csv
│       ├── positions_raw.csv
│       └── teams_raw.csv
├── docs/
│   ├── ARCHITECTURE.md
│   ├── PIPELINE.md
│   ├── SETUP.md
│   └── notebooks/
│       └── fpl_eda.ipynb            # Exploratory analysis notebook
├── logs/
│   └── pipeline.log                 # Rotating pipeline log output
├── models/
│   ├── metadata.json                # Saved model metrics and metadata
│   └── points_model.joblib          # Persisted trained regression pipeline
├── outputs/
│   └── figures/
│       ├── feature_importance.png
│       ├── top_goalscorers.png
│       ├── value_picks.png
│       └── xg_vs_actual_goals.png
├── scripts/
│   └── generate_sample_data.py      # Offline/sample dataset generator
├── src/
│   ├── __init__.py
│   ├── clean.py                     # Merge, schema validation, and cleaning
│   ├── config.py                    # Paths, constants, and env overrides
│   ├── features.py                  # Per-90 rates, value metrics, and encodings
│   ├── ingest.py                    # FPL API fetch with retry/backoff and snapshots
│   ├── logging_config.py            # Console + rotating file logger setup
│   ├── model.py                     # Model training, tuning, evaluation, and saving
│   ├── predict.py                   # Shared inference loader and prediction helpers
│   └── visualize.py                 # Chart generation utilities
├── tests/
│   ├── conftest.py
│   ├── test_api.py
│   ├── test_clean.py
│   ├── test_config.py
│   ├── test_features.py
│   ├── test_ingest.py
│   └── test_model.py
├── app.py                           # Streamlit dashboard
├── docker-compose.yml
├── Dockerfile                       # Multi-target build for Streamlit and FastAPI
├── main.py                          # End-to-end pipeline entrypoint
├── Makefile
├── pyproject.toml
├── README.md
├── requirements-dev.txt
└── requirements.txt
```

---