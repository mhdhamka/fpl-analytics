<div align="center">

# Fantasy Premier League Performance Analytics Pipeline (`fpl-analytics`)

> A modular Python analytics and machine learning pipeline built to ingest, process, visualize, and model real-time player data from the Premier League. Designed with clean architecture principles to deliver data-driven insights into player performance, expected metrics ($xG$), and points prediction.

[Live Demo](https://fpremier-league-analytics.streamlit.app/) · [Documentation](./docs) · [Architecture](./docs/ARCHITECTURE.md) · [Pipeline](./docs/PIPELINE.md) · [Getting Started](./docs/SETUP.md) · [Report Bug](https://github.com/mhdhamka/fpl-analytics/issues) · [Request Feature](https://github.com/mhdhamka/fpl-analytics/issues)

![License](https://img.shields.io/badge/LICENSE-MIT-0EA5E9?style=flat-square&labelColor=0F172A)
![Python](https://img.shields.io/badge/PYTHON-3.x-38BDF8?style=flat-square&labelColor=0F172A&logo=python&logoColor=38BDF8)
![Pandas](https://img.shields.io/badge/PANDAS-2.x-A855F7?style=flat-square&labelColor=0F172A&logo=pandas&logoColor=A855F7)
![Scikit-Learn](https://img.shields.io/badge/SCIKIT--LEARN-FB923C?style=flat-square&labelColor=0F172A&logo=scikit-learn&logoColor=FB923C)
![XGBoost](https://img.shields.io/badge/XGBOOST-34D399?style=flat-square&labelColor=0F172A)
![Seaborn](https://img.shields.io/badge/SEABORN-60A5FA?style=flat-square&labelColor=0F172A)
![Matplotlib](https://img.shields.io/badge/MATPLOTLIB-F472B6?style=flat-square&labelColor=0F172A)

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


