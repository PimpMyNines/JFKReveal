.PHONY: install install-dev setup clean build run help test setup-ollama

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
	@echo "  make setup-ollama Setup Ollama with the recommended embedding model"
	@echo ""
	@echo "Optional parameters for 'make run':"
	@echo "  SKIP_SCRAPING=1           Skip document scraping"
	@echo "  SKIP_PROCESSING=1         Skip document processing"
	@echo "  SKIP_ANALYSIS=1           Skip document analysis"
	@echo "  USE_EXISTING_PROCESSED=1  Skip OCR but use existing processed documents"
	@echo "  MAX_WORKERS=20            Number of documents to process in parallel (default: 20)"
	@echo "  NO_CLEAN_TEXT=1           Disable text cleaning for OCR documents"
	@echo ""
	@echo "Example: make run SKIP_SCRAPING=1 USE_EXISTING_PROCESSED=1"
	@echo ""
	@echo "Optional parameters for 'make setup-ollama':"
	@echo "  MODEL=nomic-embed-text    Embedding model to install (default)"
	@echo "  MODEL=mxbai-embed-large   State-of-the-art large embedding model"
	@echo "  MODEL=all-minilm          Lightweight embedding model"
	@echo ""
	@echo "Example: make setup-ollama MODEL=mxbai-embed-large"

setup:
	$(PYTHON) -m venv $(VENV)
	$(VENV)/bin/pip install --upgrade pip
	$(VENV)/bin/pip install -r requirements.txt
	$(VENV)/bin/python -m spacy download en_core_web_sm

install:
	$(PYTHON) -m pip install .

install-dev:
	$(VENV)/bin/pip install -e .

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf **/__pycache__/
	rm -rf **/*.egg-info/
	rm -rf .pytest_cache
	rm -rf .coverage
	rm -rf htmlcov
	rm -rf .tox/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete

build: clean
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install --upgrade build
	@echo "Building package with pyproject.toml configuration..."
	$(PYTHON) -m build
	@echo "\n==========================================================\n"
	@echo "IMPORTANT: After installation, you'll need to download the"
	@echo "spaCy language model with this command:"
	@echo "python -m spacy download en_core_web_sm"
	@echo "\n==========================================================\n"

test:
	$(VENV)/bin/python -m pytest

setup-ollama:
	@mkdir -p tools
	$(VENV)/bin/python tools/setup_ollama.py $(if $(MODEL),--model $(MODEL),)

run: install-dev
	$(eval ARGS := )
	$(if $(SKIP_SCRAPING),$(eval ARGS += --skip-scraping))
	$(if $(SKIP_PROCESSING),$(eval ARGS += --skip-processing))
	$(if $(SKIP_ANALYSIS),$(eval ARGS += --skip-analysis))
	$(if $(USE_EXISTING_PROCESSED),$(eval ARGS += --use-existing-processed))
	$(if $(MAX_WORKERS),$(eval ARGS += --max-workers $(MAX_WORKERS)))
	$(if $(NO_CLEAN_TEXT),$(eval ARGS += --no-clean-text))
	$(VENV)/bin/python -m jfkreveal $(ARGS)