# Simple project workflow — run `make help` to see targets.

PY ?= python
NB_DIR ?= jupyter_notebooks
DATA_DIR ?= data

.PHONY: help install check fix lint format sync nb2py py2nb regen app clean nbstrip pre-commit

help:
	@echo "Usage:"
	@echo "  make install    - install dev tools and project deps"
	@echo "  make lint       - run Ruff lint only"
	@echo "  make format     - run Black format only"
	@echo "  make nbstrip    - strip notebook outputs (run before commit)"
	@echo "  make pre-commit - run lint, format, and nbstrip (recommended before commit)"
	@echo "  make app        - run the Dash app locally"
	@echo "  make clean      - remove caches and temp files"

install:
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt

lint:
	ruff check .

format:
	black .

nbstrip:
	nbstripout $(NB_DIR)/*.ipynb

pre-commit: lint format nbstrip
	@echo "✅ Pre-commit checks complete"

app:
	streamlit run app.py

clean:
	@find . -name "__pycache__" -type d -prune -exec rm -rf {} +
	@find . -name ".ruff_cache" -type d -prune -exec rm -rf {} +
	@find . -name ".pytest_cache" -type d -prune -exec rm -rf {} +
