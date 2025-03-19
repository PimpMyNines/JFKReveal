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
	@echo "  SKIP_SCRAPING=1           Skip document scraping"
	@echo "  SKIP_PROCESSING=1         Skip document processing"
	@echo "  SKIP_ANALYSIS=1           Skip document analysis"
	@echo "  MODEL_PROVIDER=anthropic  Use Anthropic models instead of OpenAI"
	@echo "  OPENAI_API_KEY=key        Specify OpenAI API key"
	@echo "  ANTHROPIC_API_KEY=key     Specify Anthropic API key"
	@echo "  ENABLE_AUDIT_LOGGING=true Enable detailed thought process logging"
	@echo ""
	@echo "Examples:"
	@echo "  make run SKIP_SCRAPING=1 SKIP_PROCESSING=1"
	@echo "  make run MODEL_PROVIDER=anthropic"
	@echo "  make run MODEL_PROVIDER=anthropic ANTHROPIC_ANALYSIS_MODEL=claude-3-7-opus-20240620"
	@echo "  make run MODEL_PROVIDER=xai XAI_ANALYSIS_MODEL=grok-2"
	@echo "  make run-all-models      Run analysis with multiple models for comparison"

setup:
	$(PYTHON) -m venv $(VENV)
	$(VENV)/bin/pip install --upgrade pip
	$(VENV)/bin/pip install -r requirements.txt

install: setup
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

build: install
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install --upgrade build
	$(PYTHON) -m build

test: build
	$(VENV)/bin/python -m pytest

run: build
	$(eval ARGS := )
	$(if $(SKIP_SCRAPING),$(eval ARGS += --skip-scraping))
	$(if $(SKIP_PROCESSING),$(eval ARGS += --skip-processing))
	$(if $(SKIP_ANALYSIS),$(eval ARGS += --skip-analysis))
	$(if $(MODEL_PROVIDER),$(eval ARGS += --model-provider $(MODEL_PROVIDER)))
	$(if $(OPENAI_API_KEY),$(eval ARGS += --openai-api-key $(OPENAI_API_KEY)))
	$(if $(ANTHROPIC_API_KEY),$(eval ARGS += --anthropic-api-key $(ANTHROPIC_API_KEY)))
	$(if $(XAI_API_KEY),$(eval ARGS += --xai-api-key $(XAI_API_KEY)))
	
	# Set environment variables if provided
	$(if $(OPENAI_EMBEDDING_MODEL),$(eval export OPENAI_EMBEDDING_MODEL=$(OPENAI_EMBEDDING_MODEL)))
	$(if $(OPENAI_ANALYSIS_MODEL),$(eval export OPENAI_ANALYSIS_MODEL=$(OPENAI_ANALYSIS_MODEL)))
	$(if $(OPENAI_REPORT_MODEL),$(eval export OPENAI_REPORT_MODEL=$(OPENAI_REPORT_MODEL)))
	$(if $(ANTHROPIC_ANALYSIS_MODEL),$(eval export ANTHROPIC_ANALYSIS_MODEL=$(ANTHROPIC_ANALYSIS_MODEL)))
	$(if $(ANTHROPIC_REPORT_MODEL),$(eval export ANTHROPIC_REPORT_MODEL=$(ANTHROPIC_REPORT_MODEL)))
	$(if $(XAI_ANALYSIS_MODEL),$(eval export XAI_ANALYSIS_MODEL=$(XAI_ANALYSIS_MODEL)))
	$(if $(XAI_REPORT_MODEL),$(eval export XAI_REPORT_MODEL=$(XAI_REPORT_MODEL)))
	$(if $(ENABLE_AUDIT_LOGGING),$(eval export ENABLE_AUDIT_LOGGING=$(ENABLE_AUDIT_LOGGING)))
	
	$(VENV)/bin/python -m jfkreveal $(ARGS)

# Run analysis with all models for comparison
run-all-models: build
	@echo "Running analysis with multiple models for comparison..."
	@echo "=== Running with OpenAI GPT-4o ==="
	$(MAKE) run MODEL_PROVIDER=openai SKIP_SCRAPING=1 SKIP_PROCESSING=1
	@echo "=== Running with Anthropic Claude 3.7 Sonnet ==="
	$(MAKE) run MODEL_PROVIDER=anthropic ANTHROPIC_ANALYSIS_MODEL=claude-3-7-sonnet-20240620 SKIP_SCRAPING=1 SKIP_PROCESSING=1
	@echo "=== Running with xAI Grok-2 ==="
	$(MAKE) run MODEL_PROVIDER=xai XAI_ANALYSIS_MODEL=grok-2 SKIP_SCRAPING=1 SKIP_PROCESSING=1
	@echo "=== All model comparisons completed ==="
	$(VENV)/bin/python -m jfkreveal.tools.generate_comparison