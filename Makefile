.PHONY: install test figures ruff

install:
	python3 -m pip install -e ".[dev]"

test:
	python3 -m pytest -q

figures:
	python3 plot_cascade.py

ruff:
	python3 -m ruff check .
