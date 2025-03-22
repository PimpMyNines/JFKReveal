# End-to-End Tests

This directory contains all end-to-end tests for the JFKReveal project. E2E tests verify the system works correctly from start to finish.

## What to Test Here

- Complete user workflows
- System functionality with real dependencies
- API integrations with external services

## Running Tests

From the project root:

```bash
# Run all E2E tests with mock API
pytest tests/e2e -v

# Run with real API credentials (if available)
RUN_E2E_FULL=1 pytest tests/e2e -v

# Run specific test file
pytest tests/e2e/test_e2e.py -v

# Run slow tests
RUN_SLOW_TESTS=1 pytest tests/e2e -v
```

## Test Options

The E2E tests have several modes controlled by environment variables:

- `RUN_E2E_FULL=1`: Execute tests that require real API credentials
- `RUN_SLOW_TESTS=1`: Execute tests that take longer to run
- `E2E_MAX_RETRIES=n`: Set maximum number of retries for rate-limited API calls (default: 5)
- `E2E_RETRY_DELAY=n`: Set base delay in seconds between retries (default: 5)

## API Credential Management

Tests automatically handle API credential scenarios:

1. **Real API Credentials**: If `OPENAI_API_KEY` is available, tests can use it
2. **Mock Mode**: If no API key is available, tests automatically use mock mode
3. **CI Environment**: In CI, tests use a simplified mode to avoid rate limits
4. **Rate Limit Handling**: All API calls have built-in retry with exponential backoff

## Test Structure

E2E tests should:
1. Test complete workflows from start to finish
2. Use real services and dependencies when possible
3. Validate system behavior from a user's perspective
4. Include appropriate setup and teardown for external resources
5. Apply proper API mocking for CI environments

## GitHub Actions Integration

The CI pipeline is configured to run E2E tests conditionally:

- On main branch: Always run with mock mode
- With API credentials: Run non-intensive tests
- Without API credentials: Skip API-dependent tests

## Adding New E2E Tests

When adding new E2E tests:

1. Use the `@pytest.mark.e2e` decorator to identify E2E tests
2. Use the `api_key_manager` fixture for credential handling
3. Use `@retry_on_rate_limit` for API functions that may hit rate limits
4. Implement proper mock fallbacks for CI environment
5. Add skip conditions for tests that should only run in specific environments