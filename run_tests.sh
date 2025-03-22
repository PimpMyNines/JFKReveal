#!/bin/bash
set -e

# Install test dependencies if needed
pip install pytest pytest-cov pytest-mock

# Set environment variables for testing
export OPENAI_API_KEY="sk-dummy-key-for-testing"
export OPENAI_EMBEDDING_MODEL="text-embedding-3-large"
export OPENAI_ANALYSIS_MODEL="gpt-4.5-preview"

# Run unit tests with coverage measurement
echo "Running unit tests..."
python -m pytest tests/ -v -m "not integration and not e2e" --cov=src/jfkreveal --cov-report=term

# Report coverage statistics
COVERAGE_PERCENT=$(python -m pytest --cov=src/jfkreveal | grep TOTAL | awk '{print $4}' | sed 's/%//')

echo "Total coverage: $COVERAGE_PERCENT%"

# Check if coverage is at least 90%
if (( $(echo "$COVERAGE_PERCENT < 90" | bc -l) )); then
    echo "Coverage below 90%. Please add more tests."
    exit 1
else
    echo "Coverage meets or exceeds 90% target."
fi

# Optionally run integration tests (slower)
if [ "$RUN_INTEGRATION" = "1" ]; then
    echo "Running integration tests..."
    python -m pytest tests/ -v -m "integration" 
fi

# Optionally run end-to-end tests (slowest, requires more setup)
if [ "$RUN_E2E" = "1" ]; then
    echo "Running end-to-end tests..."
    python -m pytest tests/ -v -m "e2e" 
fi

echo "All tests completed successfully!" 