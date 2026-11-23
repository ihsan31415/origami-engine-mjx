# Makefile for origami-engine-mjx
# Usage: make <target>

PYTHON ?= python
VENV  ?= .venv
PY    ?= $(VENV)/bin/python

.PHONY: help venv install test example examples lint format typecheck docs-check clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  %-12s %s\n", $$1, $$2}'

venv: ## Create a virtual environment
	python -m venv $(VENV)

install: venv ## Install the package and dev dependencies in editable mode
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -e ".[dev]"

test: ## Run the test suite
	$(PY) -m pytest -q

example: ## Run the quick-start example (CPU, MJX-JAX)
	$(PY) examples/01_evaluate_cpu_jax.py

examples: ## Run all example scripts
	@for f in examples/0*.py; do echo "== $$f"; $(PY) "$$f"; done

lint: ## Run ruff lint + format check
	$(PY) -m ruff check .
	$(PY) -m ruff format --check .

format: ## Apply ruff formatting
	$(PY) -m ruff format .
	$(PY) -m ruff check --fix .

typecheck: ## Run mypy on the MJX package
	$(PY) -m mypy origami_engine_mjx --config-file pyproject.toml

docs-check: ## Verify Markdown links and heading anchors
	$(PY) scripts/check_docs_links.py

clean: ## Remove generated artifacts and caches
	rm -f examples/*.png
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
