.PHONY: install install-dev setup clean build run help test

# Python version and installation variables
PYTHON := python3
VENV := venv

help:
	@echo "JFKReveal - Declassified JFK Documents Analysis Tool"
	@echo ""
	@echo "Available commands:"
	@echo "  make install      Install the package"
	@echo "  make install-dev  Install the package in development mode"
	@echo "  make setup        Create virtual environment and install dependencies"
	@echo "  make clean        Remove build artifacts and temporary files"
	@echo "  make build        Build package distribution files"
	@echo "  make run          Run the JFK document analysis pipeline"
	@echo "  make test         Run tests"
	@echo ""
	@echo "Optional parameters for 'make run':"
	@echo "  SKIP_SCRAPING=1   Skip document scraping"
	@echo "  SKIP_PROCESSING=1 Skip document processing"
	@echo ""
	@echo "Example: make run SKIP_SCRAPING=1"

setup:
	$(PYTHON) -m venv $(VENV)
	$(VENV)/bin/pip install --upgrade pip
	$(VENV)/bin/pip install -r requirements.txt

install:
	$(PYTHON) -m pip install .

install-dev:
	$(PYTHON) -m pip install -e .

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf **/__pycache__/
	rm -rf .pytest_cache
	rm -rf .coverage
	rm -rf htmlcov
	rm -rf .tox/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

build:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install --upgrade build
	$(PYTHON) -m build

test:
	$(PYTHON) -m pytest

run:
	$(eval ARGS := )
	$(if $(SKIP_SCRAPING),$(eval ARGS += --skip-scraping))
	$(if $(SKIP_PROCESSING),$(eval ARGS += --skip-processing))
	$(PYTHON) -m jfkreveal $(ARGS)