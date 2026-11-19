<div align="center">

# Fantasy Premier League Performance Analytics Pipeline (`fpl-analytics`)

A modular Python analytics and machine learning pipeline built to ingest, process, visualize, and model real-time player data from the Premier League. Designed with clean architecture principles to deliver data-driven insights into player performance, expected metrics ($xG$), and points prediction.

[Live Demo](https://fpremier-league-analytics.streamlit.app/) · [Documentation](./docs) · [Architecture & Repository Layout](./docs/ARCHITECTURE.md) · [Pipeline & Modules Guide](./docs/PIPELINE.md) · [Getting Started](./docs/SETUP.md) · [Report Bug](https://github.com/mhdhamka/fpl-analytics/issues) · [Request Feature](https://github.com/mhdhamka/fpl-analytics/issues)

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

The **Fantasy Premier League Performance Analytics Pipeline** is an end-to-end data engineering and predictive modeling project tailored for sports analytics and Fantasy Premier League (FPL) management. 

Managing raw football data can be messy and fragmented. This project automates the entire lifecycle—from pulling live data straight off official league endpoints to structuring clean datasets, generating exploratory data visualizations, and deploying machine learning models to forecast player performance. It highlights professional software engineering standards, featuring modular design (`src/`), a master orchestration script (`main.py`), and interactive EDA notebooks.

---

## Tech Stack & Libraries

* **Language:** Python 3.x
* **Data Manipulation & Analysis:** Pandas, NumPy, Pandera (runtime schema validation)
* **Web Services & UI:** FastAPI, Uvicorn, Streamlit
* **Machine Learning:** Scikit-learn (RandomizedSearchCV, Random Forest), XGBoost
* **Data Visualization:** Seaborn, Matplotlib
* **Data Ingestion & DevOps:** Requests (Live FPL API), Pytest, Docker, GitHub Actions CI/CD

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

Developed & Maintained by **[@mhdhamka](https://github.com/mhdhamka)**

</div>


