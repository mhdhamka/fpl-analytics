## Core Features & Pipeline Steps

1. **Fault-Tolerant Data Ingestion (`src/ingest.py`)**: 
   * Pulls live bootstrap data straight from official league endpoints.
   * Implements robust retry logic with **exponential backoff** to handle network hiccups gracefully.
   * Maintains a timestamped historical ledger of every fetch (`data/history/`) instead of overwriting files.

2. **Rigorous Data Wrangling & Schema Validation (`src/clean.py`)**: 
   * Programmatically executes relational mapping (merging player attributes with team and position metadata).
   * Validates raw and cleaned data frames against an explicit schema via **`pandera`** (with a dependency-free fallback if not installed).
   * Engineers advanced attributes: per-90 rate metrics, a points-per-£million value index, and one-hot encoded player positions so that **position is a real model feature** rather than ignored.

3. **Automated Visualization Engine (`src/visualize.py`)**: 
   * Programmatically generates and serializes publication-grade analytics charts with automated directory provisioning:
     * Top 10 Goalscorers Bar Chart (`outputs/figures/top_goalscorers.png`)
     * Expected Goals ($xG$) vs. Actual Goals Scatter Plot (`outputs/figures/xg_vs_actual_goals.png`)

4. **Exploratory Data Analysis (`notebooks/01_eda.ipynb`)**: 
   * Features a structured exploratory notebook containing statistical distributions, positional performance benchmarking, and feature correlation heatmaps.

5. **Optimized Predictive Modeling & Persistence (`src/model.py`)**: 
   * Benchmarks supervised regressors (**Random Forest vs. XGBoost**) utilizing **Randomized Hyperparameter Search** and rigorous 5-Fold Cross-Validation.
   * **Persists the winning pipeline to disk** alongside metadata, meaning the app and API load pre-trained models instantly rather than forcing runtime retraining on every session or request.

6. **Dual Consumption Layers (App & API)**: 
   * **Streamlit Dashboard (`app.py`)**: Interactive UI for exploring data, viewing metrics, and checking player point forecasts.
   * **FastAPI Service (`api/main.py`)**: High-performance backend providing endpoints (`/predict`, `/players`, `/model/metadata`) for external application integration.

7. **Structured Logging & Enterprise Standards**: 
   * Replaces traditional `print()` debugging with a centralized logging configuration (`src/logging_config.py`) writing cleanly formatted outputs to both the console and a **rotating log file**.

---