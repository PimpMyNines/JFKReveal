# Integration Tests

This directory contains all integration tests for the JFKReveal project. Integration tests focus on testing the interaction between multiple components.

## What to Test Here

- Interactions between two or more components
- Data flow between components
- Contract testing between interfaces

## Running Tests

From the project root:

```bash
# Run all integration tests
pytest tests/integration

# Run specific test file
pytest tests/integration/test_integration.py

# Run with verbose output
pytest tests/integration -v
```

## Test Structure

Integration tests should:
1. Test components working together
2. Test realistic data flows
3. Focus on the interactions and boundaries
4. Validate that components can work together correctly 