PY ?= .venv/bin/python
PYTHON ?= python3

.PHONY: help venv test lint check parity bench bench-full backtest data dashboard tables check-tables clean

help:
	@echo "make venv         - create .venv and install CPU + dev requirements (PYTHON=python3.12 to pick one)"
	@echo "make test         - run the test suite (GPU tests skip off-GPU)"
	@echo "make lint         - ruff check"
	@echo "make check        - lint + test + check-tables, what CI runs"
	@echo "make parity       - CPU/GPU numerical parity report"
	@echo "make bench        - quick benchmark sweep (50/500 assets)"
	@echo "make bench-full   - full sweep (50/500/3000 assets, 10y daily)"
	@echo "make backtest     - rolling backtest vs equal-weight benchmark"
	@echo "make data         - download and cache the S&P 500 universe"
	@echo "make dashboard    - launch the Streamlit dashboard"
	@echo "make tables       - regenerate README/docs tables from benchmarks/results"
	@echo "make check-tables - fail if any generated table is stale"

venv:
	$(PYTHON) -m venv .venv
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt -r requirements-dev.txt

test:
	$(PY) -m pytest tests/ -v

lint:
	$(PY) -m ruff check .

check: lint test check-tables

parity:
	$(PY) -m pipeline.parity_tests --n 200 --days 2000

bench:
	$(PY) -m benchmarks.run_benchmarks --sizes 50 500 --days 1260 --runs 5

bench-full:
	$(PY) -m benchmarks.run_benchmarks --sizes 50 500 3000 --days 2520 --runs 5

backtest:
	$(PY) -m backtest.run_backtest --source synthetic --n 200 --frequency QE

data:
	$(PY) -m data.download_universe --source yfinance --n 500 --start 2014-01-01

dashboard:
	$(PY) -m streamlit run dashboard/app.py

tables:
	$(PY) -m benchmarks.render_tables

check-tables:
	$(PY) -m benchmarks.render_tables --check

clean:
	rm -rf __pycache__ */__pycache__ .pytest_cache .ruff_cache
	rm -f benchmarks/results/*.csv benchmarks/results/*.png
