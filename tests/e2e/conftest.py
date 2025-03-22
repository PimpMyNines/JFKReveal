"""
Configuration and fixtures for E2E tests.
"""
import os
import pytest
from typing import Dict, Any

# Environment variable to control running specific test types
# These help with running different test types in different environments
ENV_VAR_CONFIGS = {
    # Set to run tests that require real API credentials
    "RUN_E2E_FULL": {
        "description": "Set to run full E2E tests with real API",
        "default": False,
    },
    # Set to run tests that may be slow but don't require real API
    "RUN_SLOW_TESTS": {
        "description": "Set to run slow tests",
        "default": False,
    },
    # Set to control rate limiting behavior
    "E2E_MAX_RETRIES": {
        "description": "Maximum number of retries for rate-limited API calls",
        "default": 5,
    },
    "E2E_RETRY_DELAY": {
        "description": "Base delay (seconds) between retries",
        "default": 5,
    },
}


def pytest_configure(config):
    """Add E2E marker to pytest configuration."""
    config.addinivalue_line(
        "markers", "e2e: mark test as an end-to-end test that interacts with external services"
    )


def pytest_collection_modifyitems(items):
    """Skip E2E tests if CI=true and not specifically enabled."""
    if os.environ.get("CI") == "true" and not os.environ.get("RUN_E2E_IN_CI"):
        skip_marker = pytest.mark.skip(reason="E2E tests skipped in CI. Set RUN_E2E_IN_CI=1 to run")
        for item in items:
            if "e2e" in item.keywords:
                item.add_marker(skip_marker)


# Register custom command line options
def pytest_addoption(parser):
    """Add E2E test options to pytest command line."""
    parser.addoption(
        "--run-e2e-full", 
        action="store_true", 
        default=False,
        help="Run full E2E tests including those requiring real API credentials"
    )
    parser.addoption(
        "--run-slow-tests", 
        action="store_true", 
        default=False,
        help="Run slow tests that might take significant time"
    )


# Set environment variables based on command line options
@pytest.hookimpl(tryfirst=True)
def pytest_configure(config):
    """Configure environment based on pytest options."""
    if config.getoption("--run-e2e-full"):
        os.environ["RUN_E2E_FULL"] = "1"
    if config.getoption("--run-slow-tests"):
        os.environ["RUN_SLOW_TESTS"] = "1"


@pytest.fixture(scope="session")
def e2e_config() -> Dict[str, Any]:
    """
    Return E2E test configuration derived from environment variables.
    
    This provides a single place to access all E2E test configuration.
    """
    config = {}
    
    # Process all configured environment variables
    for var_name, var_config in ENV_VAR_CONFIGS.items():
        env_val = os.environ.get(var_name)
        if var_config.get("type") == "int":
            # Parse integers
            config[var_name] = int(env_val) if env_val else var_config["default"]
        elif var_config.get("type") == "float":
            # Parse floats
            config[var_name] = float(env_val) if env_val else var_config["default"]
        elif var_config.get("type") == "bool" or not var_config.get("type"):
            # Parse booleans (default for no type)
            config[var_name] = bool(env_val) if env_val is not None else var_config["default"]
        else:
            # Strings and others
            config[var_name] = env_val if env_val is not None else var_config["default"]
    
    # Add CI detection
    config["IN_CI"] = os.environ.get("CI") == "true"
    
    return config