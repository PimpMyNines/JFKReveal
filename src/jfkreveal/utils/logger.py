"""
Centralized logging configuration for the JFKReveal project.

This module provides consistent logging configuration and helper functions
for all components of the JFKReveal system. It ensures that all modules
use the same logging format, levels, and behavior.
"""
import logging
import sys
import os
import time
from typing import Optional, Dict, Any, Union
from functools import wraps
import traceback

# Default logger configuration
DEFAULT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DEFAULT_CONSOLE_FORMAT = "%(levelname)s - %(message)s"
DEFAULT_LOG_LEVEL = logging.INFO

# Format type configurations
LOG_FORMATS = {
    "standard": {
        "file": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        "console": "%(levelname)s - %(message)s"
    },
    "detailed": {
        "file": "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
        "console": "%(asctime)s - %(levelname)s - %(message)s"
    },
    "minimal": {
        "file": "%(asctime)s: %(message)s",
        "console": "%(message)s"
    }
}

# Global configuration
config = {
    "initialized": False,
    "log_level": DEFAULT_LOG_LEVEL,
    "log_file": "jfkreveal.log",
    "console": True,
    "module_levels": {}  # Custom levels for specific modules
}


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = "jfkreveal.log",
    console: bool = True,
    file_format: str = DEFAULT_LOG_FORMAT,
    console_format: str = DEFAULT_CONSOLE_FORMAT,
    module_levels: Optional[Dict[str, int]] = None,
    format_type: str = "standard"
) -> logging.Logger:
    """
    Configure logging for the JFKReveal package.
    
    Args:
        level: Default logging level (default: logging.INFO)
        log_file: Path to log file (default: jfkreveal.log, None to disable file logging)
        console: Whether to log to console (default: True)
        file_format: Format for file logging
        console_format: Format for console logging
        module_levels: Dict mapping module names to specific log levels
        format_type: Predefined format type (standard, detailed, minimal)
        
    Returns:
        Configured root logger
    """
    # Update global config
    config["initialized"] = True
    config["log_level"] = level
    config["log_file"] = log_file
    config["console"] = console
    config["module_levels"] = module_levels or {}
    
    # Create root logger
    logger = logging.getLogger("jfkreveal")
    logger.setLevel(level)
    logger.handlers = []  # Remove any existing handlers
    
    # Get format definitions based on format_type if specified
    if format_type in LOG_FORMATS:
        file_format = LOG_FORMATS[format_type]["file"]
        console_format = LOG_FORMATS[format_type]["console"]

    # Create formatters
    file_formatter = logging.Formatter(file_format)
    console_formatter = logging.Formatter(console_format)
    
    # Add file handler if requested
    if log_file:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
        except (IOError, PermissionError) as e:
            sys.stderr.write(f"Warning: Could not create log file at {log_file}: {e}\n")
    
    # Add console handler if requested
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    # Configure module-specific log levels
    if module_levels:
        for module_name, module_level in module_levels.items():
            module_logger = logging.getLogger(f"jfkreveal.{module_name}")
            module_logger.setLevel(module_level)
    
    # Set propagate=False to avoid duplicate logging
    logger.propagate = False
    
    return logger


def get_logger(module_name: str) -> logging.Logger:
    """
    Get a logger for a specific module with proper configuration.
    
    Args:
        module_name: Name of the module (e.g., 'database.document_processor')
        
    Returns:
        Configured logger for the module
    """
    logger_name = f"jfkreveal.{module_name}"
    logger = logging.getLogger(logger_name)
    
    # Set module-specific level if configured
    if module_name in config.get("module_levels", {}):
        logger.setLevel(config["module_levels"][module_name])
    
    # Ensure at least base configuration exists
    if not config.get("initialized", False):
        setup_logging()
    
    return logger


def log_execution_time(logger: Optional[logging.Logger] = None, level: int = logging.DEBUG):
    """
    Decorator to log function execution time.
    
    Args:
        logger: Logger to use (if None, uses the module's logger)
        level: Logging level to use
        
    Returns:
        Decorator function
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get logger from the function's module if not provided
            nonlocal logger
            if logger is None:
                logger = logging.getLogger(func.__module__)
            
            # Log start
            start_time = time.time()
            func_name = func.__qualname__
            
            try:
                # Execute function
                result = func(*args, **kwargs)
                
                # Log end with execution time
                execution_time = time.time() - start_time
                logger.log(level, f"Function {func_name} executed in {execution_time:.4f} seconds")
                
                return result
            except Exception as e:
                # Log exception with execution time
                execution_time = time.time() - start_time
                logger.error(f"Exception in {func_name} after {execution_time:.4f} seconds: {e}")
                logger.debug(f"Traceback: {traceback.format_exc()}")
                raise
        
        return wrapper
    
    return decorator


def log_function_calls(logger: Optional[logging.Logger] = None, level: int = logging.DEBUG):
    """
    Decorator to log function calls with arguments.
    
    Args:
        logger: Logger to use (if None, uses the module's logger)
        level: Logging level to use
        
    Returns:
        Decorator function
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get logger from the function's module if not provided
            nonlocal logger
            if logger is None:
                logger = logging.getLogger(func.__module__)
            
            # Convert args to string, limiting length for large arguments
            args_str = [
                repr(arg)[:100] + "..." if len(repr(arg)) > 100 else repr(arg)
                for arg in args
            ]
            
            # Convert kwargs to string, limiting length for large values
            kwargs_str = [
                f"{k}={repr(v)[:100] + '...' if len(repr(v)) > 100 else repr(v)}"
                for k, v in kwargs.items()
            ]
            
            # Combined arguments string
            args_kwargs_str = ", ".join(args_str + kwargs_str)
            
            func_name = func.__qualname__
            logger.log(level, f"Calling {func_name}({args_kwargs_str})")
            
            try:
                result = func(*args, **kwargs)
                
                # Log return value, limiting length for large results
                result_str = repr(result)
                if len(result_str) > 100:
                    result_str = result_str[:100] + "..."
                    
                logger.log(level, f"{func_name} returned: {result_str}")
                return result
            except Exception as e:
                logger.error(f"Exception in {func_name}: {e}")
                logger.debug(f"Traceback: {traceback.format_exc()}")
                raise
        
        return wrapper
    
    return decorator


class LoggingManager:
    """
    Helper class for managing logging configuration at runtime.
    """
    
    @staticmethod
    def set_level(level: Union[int, str], module: Optional[str] = None):
        """
        Set log level for the entire application or a specific module.
        
        Args:
            level: Logging level (can be int or string like 'INFO')
            module: Optional module name to set level for
        """
        # Convert string level to int if needed
        if isinstance(level, str):
            level = getattr(logging, level.upper())
        
        if module:
            # Set for specific module
            logger = logging.getLogger(f"jfkreveal.{module}")
            logger.setLevel(level)
            config["module_levels"][module] = level
        else:
            # Set for all loggers
            root_logger = logging.getLogger("jfkreveal")
            root_logger.setLevel(level)
            config["log_level"] = level
    
    @staticmethod
    def get_level(module: Optional[str] = None) -> int:
        """
        Get the current log level for the application or a specific module.
        
        Args:
            module: Optional module name to get level for
            
        Returns:
            Current log level
        """
        if module:
            logger = logging.getLogger(f"jfkreveal.{module}")
            return logger.getEffectiveLevel()
        else:
            return logging.getLogger("jfkreveal").level
    
    @staticmethod
    def disable_console_logging():
        """Disable console logging for all loggers."""
        root_logger = logging.getLogger("jfkreveal")
        for handler in root_logger.handlers[:]:
            if isinstance(handler, logging.StreamHandler) and handler.stream == sys.stdout:
                root_logger.removeHandler(handler)
        config["console"] = False
    
    @staticmethod
    def enable_console_logging(format_str: str = DEFAULT_CONSOLE_FORMAT):
        """
        Enable console logging for all loggers.
        
        Args:
            format_str: Format string for console output
        """
        if not config["console"]:
            root_logger = logging.getLogger("jfkreveal")
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(logging.Formatter(format_str))
            root_logger.addHandler(console_handler)
            config["console"] = True