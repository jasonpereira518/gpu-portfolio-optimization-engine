PY ?= .venv/bin/python

.PHONY: help venv test parity bench bench-full backtest data dashboard clean

help:
	@echo "make venv       - create .venv and install CPU requirements"
	@echo "make test       - run the test suite (GPU tests skip off-GPU)"
	@echo "make parity     - CPU/GPU numerical parity report"
	@echo "make bench      - quick benchmark sweep (50/500 assets)"
	@echo "make bench-full - full sweep (50/500/3000 assets, 10y daily)"
	@echo "make backtest   - rolling backtest vs equal-weight benchmark"
	@echo "make data       - download and cache the S&P 500 universe"
	@echo "make dashboard  - launch the Streamlit dashboard"

venv:
	python3 -m venv .venv
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt

test:
	$(PY) -m pytest tests/ -v

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
	.venv/bin/streamlit run dashboard/app.py

clean:
	rm -rf __pycache__ */__pycache__ .pytest_cache
	rm -f benchmarks/results/*.csv benchmarks/results/*.png
