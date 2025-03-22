#!/bin/bash
# Run tests for the multi-model functionality

echo "Running tests for multi-model functionality..."

# Run unit tests for the model registry
echo "Testing ModelRegistry..."
python -m pytest tests/unit/test_model_registry.py -v

# Run unit tests for the model configuration
echo "Testing ModelConfiguration..."
python -m pytest tests/unit/test_model_config.py -v

# Run unit tests for multi-model report generation
echo "Testing multi-model report generation..."
python -m pytest tests/unit/test_findings_report.py -v

# Run all tests with coverage
echo "Running all tests with coverage..."
python -m pytest tests/unit/ --cov=src/jfkreveal/utils --cov=src/jfkreveal/summarization --cov-report=term

echo "Tests completed."