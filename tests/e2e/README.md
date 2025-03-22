# End-to-End Tests

This directory contains all end-to-end tests for the JFKReveal project. E2E tests verify the system works correctly from start to finish.

## What to Test Here

- Complete user workflows
- System functionality with real dependencies
- API integrations with external services

## Running Tests

From the project root:

```bash
# Run all E2E tests
pytest tests/e2e

# Run specific test file
pytest tests/e2e/test_e2e.py

# Run with verbose output
pytest tests/e2e -v
```

## Test Structure

E2E tests should:
1. Test complete workflows from start to finish
2. Use real services and dependencies when possible
3. Validate system behavior from a user's perspective
4. Include appropriate setup and teardown for external resources

## Note

These tests may be slow and require API credentials. They're conditionally run in CI to avoid hitting rate limits. 