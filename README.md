
# Premier League Performance Analytics Pipeline (`pl-analytics`)

A production-grade, modular Python analytics and machine learning pipeline built to ingest, process, visualize, and model real-time player data from the Premier League. Designed with clean architecture principles to deliver data-driven insights into player performance, expected metrics ($xG$), and points prediction.

---

## Project Architecture

```text
pl-analytics/
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
├── outputs/
│   └── figures/            # Saved chart images and visual exports (top scorers, xG plots)
│
├── .gitignore              # Files to ignore (e.g., venv, __pycache__)
├── requirements.txt        # Project dependencies (pandas, requests, seaborn, scikit-learn)
└── main.py                 # Master execution script to run the pipeline end-to-end[cite: 1, 2, 3]

```

---

## Tech Stack & Libraries

* **Language:** Python 3.x
* **Data Manipulation & Analysis:** Pandas, NumPy
* **Data Visualization:** Seaborn, Matplotlib
* **Machine Learning:** Scikit-learn (Random Forest Regressor)
* **Data Ingestion:** Requests (Live Fantasy Premier League API)



---

## Core Features & Pipeline Steps

1. **Live Data Ingestion (`src/ingest.py`)**: Connects directly to the official Premier League data endpoints to pull live player statistics, team details, and position mapping.


2. **Data Cleaning & Feature Engineering (`src/clean.py`)**: Handles null values, maps categorical IDs to readable text names (teams and positions), and creates custom features like scaled market values (`market_value_m`).


3. **Automated Visualization (`src/visualize.py`)**: Automatically renders and exports publication-quality plots:
* Top 10 Goalscorers Bar Chart (`outputs/figures/top_goalscorers.png`)


* Expected Goals ($xG$) vs. Actual Goals Scatter Plot (`outputs/figures/xg_vs_actual_goals.png`)


4. **Exploratory Data Analysis (`notebooks/01_eda.ipynb`)**: An interactive notebook covering statistical summaries, position-wise performance breakdowns, and feature correlation heatmaps.
5. **Predictive Modeling (`src/model.py`)**: Trains a supervised **Random Forest Regressor** to predict player total fantasy/performance points based on underlying metrics (Minutes played, goals, assists, clean sheets, and $xG$), achieving strong predictive accuracy ($R^2 \approx 0.84$).

---

## Getting Started & Installation

### 1. Clone the Repository

```cmd
git clone [https://github.com/yourusername/pl-analytics.git](https://github.com/yourusername/pl-analytics.git)
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

---

## Model Performance Overview

* **Algorithm:** Random Forest Regressor
* **Target Variable:** Total Points
* **Performance ($R^2$ Score):** ~0.84
* **Key Drivers:** Minutes played and goals scored hold the highest feature importances in driving overall player output.

```

---

