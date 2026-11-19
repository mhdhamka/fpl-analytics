## Getting Started & Installation

### 1. Clone the Repository

```cmd
git clone https://github.com/mhdhamka/fpl-analytics.git
cd fpl-analytics

```

### 2. Install Dependencies

```cmd
pip install -r requirements.txt

pip install -r requirements-dev.txt

```

### 3. Run the Entire Pipeline

Execute the master execution script to fetch fresh data, clean it, generate updated charts, and train the machine learning model in one command:

```bash
# Option A: run against the live FPL API
python main.py

# Option B: no network / just trying it out
python scripts/generate_sample_data.py
python main.py   # will use the sample data already in data/raw/

streamlit run app.py                 # dashboard at localhost:8501
uvicorn api.main:app --reload        # API + docs at localhost:8000/docs

```
Or with Docker:

```bash
docker compose up --build
```


### 4. Run Unit Tests

To execute the test suite locally using pytest:

```bash
pip install -r requirements-dev.txt

# For Windows PowerShell:
$env:PYTHONPATH="."; pytest

# For Windows CMD:
set PYTHONPATH=. && pytest

# For Linux / macOS:
PYTHONPATH=. pytest

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