# JFKReveal - Declassified JFK Documents Analysis
__version__ = '0.1.0'

# Import logging functionality from the dedicated module
from .utils.logger import (
    setup_logging,
    get_logger,
    log_execution_time,
    log_function_calls,
    LoggingManager
)

# Set default logging configuration
setup_logging()
