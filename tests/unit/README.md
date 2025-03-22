# Unit Tests

This directory contains all unit tests for the JFKReveal project. Unit tests focus on testing individual components in isolation.

## What to Test Here

- Individual classes and functions
- Components that can be tested in isolation
- Mocking of external dependencies

## Running Tests

From the project root:

```bash
# Run all unit tests
pytest tests/unit

# Run specific test file
pytest tests/unit/test_archives_gov_scraper.py

# Run with coverage
pytest tests/unit --cov=src/jfkreveal
```

## Test Structure

Each test file should:
1. Test a single component
2. Use mocks for external dependencies
3. Have clear test cases
4. Test both success and error cases 