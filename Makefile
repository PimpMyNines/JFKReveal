.PHONY: install install-dev setup clean build run help test test-unit test-integration test-e2e setup-ollama search analyze reports dashboard examples version

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
	@echo "  make search       Search for documents matching a query"
	@echo "  make analyze      Analyze documents for a specific topic"
	@echo "  make reports      Generate analysis reports"
	@echo "  make dashboard    Launch interactive visualization dashboard"
	@echo "  make test         Run all tests"
	@echo "  make test-unit    Run unit tests only"
	@echo "  make test-integration Run integration tests only"
	@echo "  make test-e2e     Run end-to-end tests only (skips slow tests)"
	@echo "  make setup-ollama Setup Ollama with the recommended embedding model"
	@echo "  make examples     Show usage examples for all commands"
	@echo "  make version      Show version information"
	@echo ""
	@echo "Optional parameters for 'make run':"
	@echo "  SKIP_SCRAPING=1           Skip document scraping"
	@echo "  SKIP_PROCESSING=1         Skip document processing"
	@echo "  SKIP_ANALYSIS=1           Skip document analysis"
	@echo "  USE_EXISTING_PROCESSED=1  Skip OCR but use existing processed documents"
	@echo "  MAX_WORKERS=20            Number of documents to process in parallel (default: 20)"
	@echo "  NO_CLEAN_TEXT=1           Disable text cleaning for OCR documents"
	@echo "  NO_OCR=1                  Disable OCR for scanned documents"
	@echo "  OCR_RESOLUTION=2.0        OCR resolution scaling factor (default: 2.0)"
	@echo "  OCR_LANGUAGE=eng          OCR language (default: eng)"
	@echo "  LOG_LEVEL=INFO            Logging level (DEBUG|INFO|WARNING|ERROR|CRITICAL)"
	@echo "  FAIL_FAST=1               Stop pipeline on first error"
	@echo "  PROGRESS_BAR=0            Disable progress bar display"
	@echo "  MODEL=gpt-4o              LLM model to use for analysis"
	@echo ""
	@echo "Examples:"
	@echo "  make run SKIP_SCRAPING=1 USE_EXISTING_PROCESSED=1"
	@echo "  make search QUERY=\"Lee Harvey Oswald CIA connection\""
	@echo "  make analyze TOPIC=\"CIA involvement\""
	@echo ""
	@echo "For more detailed examples and options:"
	@echo "  make examples"

setup:
	$(PYTHON) -m venv $(VENV)
	$(VENV)/bin/pip install --upgrade pip
	$(VENV)/bin/pip install -r requirements.txt
	$(VENV)/bin/pip install colorama tqdm
	$(VENV)/bin/python -m spacy download en_core_web_sm

install:
	$(PYTHON) -m pip install .

install-dev:
	$(PYTHON) -m pip install -e .
	$(PYTHON) -m pip install colorama tqdm

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
	# Create temporary requirements file if it doesn't exist
	test -f requirements.txt || touch requirements.txt
	$(PYTHON) -m build

test:
	$(VENV)/bin/python -m pytest

test-unit:
	$(VENV)/bin/python -m pytest tests/unit/ -v

test-integration:
	$(VENV)/bin/python -m pytest tests/integration/ -v

test-e2e:
	$(VENV)/bin/python -m pytest tests/e2e/ -v --skip-slow

setup-ollama:
	@mkdir -p tools
	$(VENV)/bin/python tools/setup_ollama.py $(if $(MODEL),--model $(MODEL),)

examples:
	$(VENV)/bin/python -m jfkreveal --examples

version:
	$(VENV)/bin/python -m jfkreveal --version

# Main pipeline run command
run:
	$(eval ARGS := )
	$(if $(SKIP_SCRAPING),$(eval ARGS += --skip-scraping))
	$(if $(SKIP_PROCESSING),$(eval ARGS += --skip-processing))
	$(if $(SKIP_ANALYSIS),$(eval ARGS += --skip-analysis))
	$(if $(USE_EXISTING_PROCESSED),$(eval ARGS += --use-existing-processed))
	$(if $(MAX_WORKERS),$(eval ARGS += --max-workers $(MAX_WORKERS)))
	$(if $(NO_CLEAN_TEXT),$(eval ARGS += --no-clean-text))
	$(if $(NO_OCR),$(eval ARGS += --no-ocr))
	$(if $(OCR_RESOLUTION),$(eval ARGS += --ocr-resolution $(OCR_RESOLUTION)))
	$(if $(OCR_LANGUAGE),$(eval ARGS += --ocr-language $(OCR_LANGUAGE)))
	$(if $(LOG_LEVEL),$(eval ARGS += --log-level $(LOG_LEVEL)))
	$(if $(FAIL_FAST),$(eval ARGS += --fail-fast))
	$(if $(PROGRESS_BAR)$(filter 0,$(PROGRESS_BAR)),$(eval ARGS += --no-progress-bar))
	$(if $(MODEL),$(eval ARGS += --model $(MODEL)))
	$(VENV)/bin/python -m jfkreveal $(ARGS)

# Document search command
search:
	$(eval ARGS := search)
	$(if $(QUERY),$(eval ARGS += "$(QUERY)"))
	$(if $(LIMIT),$(eval ARGS += --limit $(LIMIT)))
	$(if $(FORMAT),$(eval ARGS += --format $(FORMAT)))
	$(if $(OUTPUT_FILE),$(eval ARGS += --output-file "$(OUTPUT_FILE)"))
	$(if $(SEARCH_TYPE),$(eval ARGS += --search-type $(SEARCH_TYPE)))
	$(if $(RERANK),$(eval ARGS += --rerank))
	$(if $(LOG_LEVEL),$(eval ARGS += --log-level $(LOG_LEVEL)))
	$(VENV)/bin/python -m jfkreveal $(ARGS)

# Topic analysis command
analyze:
	$(eval ARGS := run-analysis)
	$(if $(TOPIC),$(eval ARGS += --topic "$(TOPIC)"))
	$(if $(MAX_DOCUMENTS),$(eval ARGS += --max-documents $(MAX_DOCUMENTS)))
	$(if $(OUTPUT_FILE),$(eval ARGS += --output-file "$(OUTPUT_FILE)"))
	$(if $(MAX_RETRIES),$(eval ARGS += --max-retries $(MAX_RETRIES)))
	$(if $(MODEL),$(eval ARGS += --model $(MODEL)))
	$(if $(LOG_LEVEL),$(eval ARGS += --log-level $(LOG_LEVEL)))
	$(VENV)/bin/python -m jfkreveal $(ARGS)

# Report generation command
reports:
	$(eval ARGS := generate-report)
	$(if $(REPORT_TYPES),$(eval ARGS += --report-types $(REPORT_TYPES)))
	$(if $(TEMPLATE_DIR),$(eval ARGS += --template-dir "$(TEMPLATE_DIR)"))
	$(if $(CUSTOM_CSS),$(eval ARGS += --custom-css "$(CUSTOM_CSS)"))
	$(if $(INCLUDE_EVIDENCE)$(filter 0,$(INCLUDE_EVIDENCE)),$(eval ARGS += --no-include-evidence))
	$(if $(INCLUDE_SOURCES)$(filter 0,$(INCLUDE_SOURCES)),$(eval ARGS += --no-include-sources))
	$(if $(MODEL),$(eval ARGS += --model $(MODEL)))
	$(if $(LOG_LEVEL),$(eval ARGS += --log-level $(LOG_LEVEL)))
	$(VENV)/bin/python -m jfkreveal $(ARGS)

# Dashboard launch command
dashboard:
	$(eval ARGS := view-dashboard)
	$(if $(HOST),$(eval ARGS += --host $(HOST)))
	$(if $(PORT),$(eval ARGS += --port $(PORT)))
	$(if $(DEBUG),$(eval ARGS += --debug))
	$(if $(THEME),$(eval ARGS += --theme $(THEME)))
	$(if $(DATA_DIR),$(eval ARGS += --data-dir "$(DATA_DIR)"))
	$(if $(LOG_LEVEL),$(eval ARGS += --log-level $(LOG_LEVEL)))
	$(VENV)/bin/python -m jfkreveal $(ARGS)