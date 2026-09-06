.PHONY: install install-dev sample-data pipeline test lint fmt run-app run-api docker-up clean

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements-dev.txt

sample-data:
	PYTHONPATH=. python scripts/generate_sample_data.py

pipeline:
	PYTHONPATH=. python main.py

test:
	PYTHONPATH=. pytest

lint:
	ruff check src api tests main.py app.py

fmt:
	black src api tests main.py app.py

run-app:
	streamlit run app.py

run-api:
	uvicorn api.main:app --reload

docker-up:
	docker compose up --build

clean:
	rm -rf data/processed/* outputs/figures/* models/* logs/*
