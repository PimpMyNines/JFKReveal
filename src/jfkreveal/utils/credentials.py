"""
Credential management utilities for JFKReveal.

This module provides a robust credential management system with support for
multiple credential sources, fallbacks, and rotation mechanisms. It also 
includes error handling and validation for different API providers.
"""
import os
import json
import logging
import time
import random
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Optional, List, Any, Type, TypeVar, Callable, Union
from datetime import datetime, timedelta
import dotenv

from ..interfaces import ICredentialProvider
from ..utils.logger import get_logger

# Create logger for this module
logger = get_logger("credentials")

# Define credential sources
T = TypeVar('T', bound='BaseCredentialSource')


class CredentialError(Exception):
    """Base exception for credential-related errors."""
    pass


class CredentialNotFoundError(CredentialError):
    """Exception raised when a credential is not found in any source."""
    pass


class CredentialValidationError(CredentialError):
    """Exception raised when a credential fails validation."""
    pass


class APIRateLimitError(CredentialError):
    """Exception raised when API rate limits are hit."""
    pass


class BaseCredentialSource(ABC):
    """Base class for credential sources."""
    
    def __init__(self, name: str, priority: int = 100):
        """
        Initialize the credential source.
        
        Args:
            name: Name of the credential source
            priority: Priority of the source (lower numbers are tried first)
        """
        self.name = name
        self.priority = priority
    
    @abstractmethod
    def get_credential(self, name: str) -> Optional[str]:
        """
        Get a credential by name.
        
        Args:
            name: The name of the credential
            
        Returns:
            The credential value, or None if not found
        """
        pass
    
    @abstractmethod
    def set_credential(self, name: str, value: str) -> bool:
        """
        Set a credential by name.
        
        Args:
            name: The name of the credential
            value: The credential value
            
        Returns:
            True if successful, False otherwise
        """
        pass
    
    def has_credential(self, name: str) -> bool:
        """
        Check if a credential exists.
        
        Args:
            name: The name of the credential
            
        Returns:
            True if the credential exists, False otherwise
        """
        return self.get_credential(name) is not None


class MemoryCredentialSource(BaseCredentialSource):
    """In-memory credential source."""
    
    def __init__(self, name: str = "memory", priority: int = 10):
        """Initialize in-memory credential source."""
        super().__init__(name, priority)
        self._credentials: Dict[str, str] = {}
    
    def get_credential(self, name: str) -> Optional[str]:
        """Get a credential from memory."""
        return self._credentials.get(name)
    
    def set_credential(self, name: str, value: str) -> bool:
        """Set a credential in memory."""
        self._credentials[name] = value
        return True
    
    def clear_credential(self, name: str) -> bool:
        """Remove a credential from memory."""
        if name in self._credentials:
            del self._credentials[name]
            return True
        return False


class EnvironmentCredentialSource(BaseCredentialSource):
    """Environment variable credential source."""
    
    def __init__(self, name: str = "environment", priority: int = 20, load_dotenv: bool = True):
        """
        Initialize environment credential source.
        
        Args:
            name: Name of the credential source
            priority: Priority of the source
            load_dotenv: Whether to load credentials from .env file
        """
        super().__init__(name, priority)
        if load_dotenv:
            # Load .env file from current directory and parent directories
            dotenv.load_dotenv()
    
    def get_credential(self, name: str) -> Optional[str]:
        """Get a credential from environment variables."""
        return os.environ.get(name)
    
    def set_credential(self, name: str, value: str) -> bool:
        """
        Set a credential in environment variables.
        
        Note: This only sets for the current process, not persistently.
        """
        os.environ[name] = value
        return True


class FileCredentialSource(BaseCredentialSource):
    """File-based credential source."""
    
    def __init__(
        self, 
        file_path: str, 
        name: str = "file", 
        priority: int = 30,
        create_if_missing: bool = True
    ):
        """
        Initialize file-based credential source.
        
        Args:
            file_path: Path to the credentials file
            name: Name of the credential source
            priority: Priority of the source
            create_if_missing: Whether to create the file if it doesn't exist
        """
        super().__init__(name, priority)
        self.file_path = Path(file_path)
        self._credentials: Dict[str, str] = {}
        
        # Create directory if it doesn't exist
        if create_if_missing and not self.file_path.parent.exists():
            self.file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load credentials from file if it exists
        self._load_credentials()
    
    def _load_credentials(self) -> None:
        """Load credentials from file."""
        if self.file_path.exists():
            try:
                with open(self.file_path, "r") as f:
                    self._credentials = json.load(f)
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse credentials file: {self.file_path}")
            except Exception as e:
                logger.error(f"Error loading credentials from file: {e}")
    
    def _save_credentials(self) -> bool:
        """Save credentials to file."""
        try:
            with open(self.file_path, "w") as f:
                json.dump(self._credentials, f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Error saving credentials to file: {e}")
            return False
    
    def get_credential(self, name: str) -> Optional[str]:
        """Get a credential from file."""
        # Reload from file in case it changed
        self._load_credentials()
        return self._credentials.get(name)
    
    def set_credential(self, name: str, value: str) -> bool:
        """Set a credential in file."""
        self._credentials[name] = value
        return self._save_credentials()


class RotatingCredentialSource(BaseCredentialSource):
    """Credential source that rotates between multiple credentials of the same type."""
    
    def __init__(
        self, 
        name: str = "rotating", 
        priority: int = 50,
        credential_prefix: str = "",
        cooldown_seconds: int = 60
    ):
        """
        Initialize rotating credential source.
        
        Args:
            name: Name of the credential source
            priority: Priority of the source
            credential_prefix: Prefix for credential names (e.g., "OPENAI_API_KEY_")
            cooldown_seconds: Cooldown period after using a credential
        """
        super().__init__(name, priority)
        self.credential_prefix = credential_prefix
        self.cooldown_seconds = cooldown_seconds
        self._last_used: Dict[str, float] = {}  # Credential name -> timestamp
        self._source = MemoryCredentialSource(f"{name}_internal", 1)
    
    def get_available_credentials(self, base_name: str) -> List[str]:
        """
        Get list of available credentials with the given base name.
        
        Args:
            base_name: Base name of the credential (e.g., "OPENAI_API_KEY")
            
        Returns:
            List of available credential names
        """
        # If using a prefix, we need to strip it for matching
        match_name = base_name
        if self.credential_prefix and base_name.startswith(self.credential_prefix):
            match_name = base_name[len(self.credential_prefix):]
        
        # Find all credentials with this name or prefix
        credentials = []
        
        # Case 1: Check for exact match
        if self._source.has_credential(base_name):
            credentials.append(base_name)
        
        # Case 2: Check for numbered credentials (e.g., OPENAI_API_KEY_1, OPENAI_API_KEY_2)
        i = 1
        while self._source.has_credential(f"{base_name}_{i}"):
            credentials.append(f"{base_name}_{i}")
            i += 1
        
        return credentials
    
    def get_credential(self, name: str) -> Optional[str]:
        """
        Get a credential by name, rotating if multiple are available.
        
        Args:
            name: The name of the credential
            
        Returns:
            The credential value, or None if not found
        """
        # Get available credentials
        credentials = self.get_available_credentials(name)
        if not credentials:
            return None
        
        # Filter out credentials that are on cooldown
        current_time = time.time()
        available_credentials = [
            cred for cred in credentials 
            if current_time - self._last_used.get(cred, 0) > self.cooldown_seconds
        ]
        
        # If all credentials are on cooldown, use the least recently used one
        if not available_credentials and credentials:
            available_credentials = [min(
                credentials, 
                key=lambda c: self._last_used.get(c, 0)
            )]
        
        # If we have multiple available credentials, select one randomly
        if available_credentials:
            credential_name = random.choice(available_credentials)
            # Mark as used
            self._last_used[credential_name] = current_time
            return self._source.get_credential(credential_name)
        
        return None
    
    def set_credential(self, name: str, value: str) -> bool:
        """Set a credential."""
        return self._source.set_credential(name, value)
    
    def add_credential(self, base_name: str, value: str) -> str:
        """
        Add a new credential to the rotation.
        
        Args:
            base_name: Base name of the credential (e.g., "OPENAI_API_KEY")
            value: Credential value
            
        Returns:
            The full name of the credential that was added
        """
        # Get existing credentials
        credentials = self.get_available_credentials(base_name)
        
        # Determine next index
        next_index = 1
        if credentials:
            # Extract indices of existing credentials
            indices = []
            for cred in credentials:
                if cred == base_name:
                    indices.append(0)
                elif cred.startswith(f"{base_name}_"):
                    try:
                        idx = int(cred[len(base_name) + 1:])
                        indices.append(idx)
                    except ValueError:
                        pass
            
            if indices:
                next_index = max(indices) + 1
        
        # Create new credential name
        credential_name = f"{base_name}_{next_index}"
        
        # Set the credential
        self._source.set_credential(credential_name, value)
        
        return credential_name
    
    def mark_credential_as_used(self, name: str) -> None:
        """
        Mark a credential as just used, enforcing cooldown.
        
        Args:
            name: Name of the credential
        """
        self._last_used[name] = time.time()
    
    def mark_credential_as_rate_limited(self, name: str, duration: int = 60) -> None:
        """
        Mark a credential as rate limited for a longer cooldown.
        
        Args:
            name: Name of the credential
            duration: Duration of the rate limit in seconds
        """
        # Set last used time to current time + duration - cooldown
        # This effectively adds the duration to the normal cooldown
        self._last_used[name] = time.time() + duration - self.cooldown_seconds


class APIValidationResult:
    """Result of API credential validation."""
    
    def __init__(
        self, 
        is_valid: bool, 
        message: str = "", 
        rate_limited: bool = False,
        rate_limit_reset: Optional[int] = None
    ):
        """
        Initialize validation result.
        
        Args:
            is_valid: Whether the credential is valid
            message: Message providing additional details
            rate_limited: Whether the credential is currently rate limited
            rate_limit_reset: When the rate limit will reset (seconds from now)
        """
        self.is_valid = is_valid
        self.message = message
        self.rate_limited = rate_limited
        self.rate_limit_reset = rate_limit_reset


class BaseAPIValidator(ABC):
    """Base class for API credential validators."""
    
    @abstractmethod
    def validate(self, api_key: str) -> APIValidationResult:
        """
        Validate an API key.
        
        Args:
            api_key: The API key to validate
            
        Returns:
            Validation result
        """
        pass


class OpenAIValidator(BaseAPIValidator):
    """Validator for OpenAI API credentials."""
    
    def validate(self, api_key: str) -> APIValidationResult:
        """Validate an OpenAI API key."""
        try:
            import openai
            from openai import OpenAI
            
            # Create a client with the API key
            client = OpenAI(api_key=api_key)
            
            # Make a simple request to check if the key is valid
            # Using models.list since it's a lightweight call
            client.models.list()
            
            return APIValidationResult(
                is_valid=True,
                message="OpenAI API key is valid"
            )
        except openai.RateLimitError as e:
            # Key is valid but rate limited
            reset_time = None
            # Try to extract reset time from error message if available
            # This is a simplification; actual implementation would need to parse headers
            
            return APIValidationResult(
                is_valid=True,
                message="OpenAI API key is valid but rate limited",
                rate_limited=True,
                rate_limit_reset=reset_time or 60  # Default to 60 seconds
            )
        except openai.AuthenticationError:
            return APIValidationResult(
                is_valid=False,
                message="Invalid OpenAI API key"
            )
        except Exception as e:
            logger.error(f"Error validating OpenAI API key: {e}")
            return APIValidationResult(
                is_valid=False,
                message=f"Error validating OpenAI API key: {str(e)}"
            )


class AzureOpenAIValidator(BaseAPIValidator):
    """Validator for Azure OpenAI API credentials."""
    
    def validate(self, api_key: str) -> APIValidationResult:
        """Validate an Azure OpenAI API key."""
        try:
            # We also need an endpoint for Azure OpenAI
            endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
            if not endpoint:
                return APIValidationResult(
                    is_valid=False,
                    message="Azure OpenAI endpoint is missing"
                )
            
            import openai
            from openai import AzureOpenAI
            
            # Create a client with the API key and endpoint
            client = AzureOpenAI(
                api_key=api_key,
                azure_endpoint=endpoint,
                api_version="2023-05-15"  # Use appropriate version
            )
            
            # Make a simple request to check if the key is valid
            # This requires a deployment name which should be in env vars
            deployment_name = os.environ.get("AZURE_OPENAI_DEPLOYMENT")
            if not deployment_name:
                return APIValidationResult(
                    is_valid=False,
                    message="Azure OpenAI deployment name is missing"
                )
                
            # Check models to validate credentials
            client.models.list()
            
            return APIValidationResult(
                is_valid=True,
                message="Azure OpenAI API key is valid"
            )
        except openai.RateLimitError:
            # Key is valid but rate limited
            return APIValidationResult(
                is_valid=True,
                message="Azure OpenAI API key is valid but rate limited",
                rate_limited=True,
                rate_limit_reset=60  # Default to 60 seconds
            )
        except openai.AuthenticationError:
            return APIValidationResult(
                is_valid=False,
                message="Invalid Azure OpenAI API key"
            )
        except Exception as e:
            logger.error(f"Error validating Azure OpenAI API key: {e}")
            return APIValidationResult(
                is_valid=False,
                message=f"Error validating Azure OpenAI API key: {str(e)}"
            )


class AnthropicValidator(BaseAPIValidator):
    """Validator for Anthropic API credentials."""
    
    def validate(self, api_key: str) -> APIValidationResult:
        """Validate an Anthropic API key."""
        try:
            # Anthropic requires their Python package
            try:
                import anthropic
            except ImportError:
                logger.warning("Anthropic Python package not installed")
                return APIValidationResult(
                    is_valid=False,
                    message="Anthropic Python package not installed"
                )
            
            # Create a client with the API key
            client = anthropic.Anthropic(api_key=api_key)
            
            # Make a minimal request to validate the key
            # Usually a short model list call is best, but we'll use a minimal completion
            client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=1,
                messages=[
                    {"role": "user", "content": "Hello"}
                ]
            )
            
            return APIValidationResult(
                is_valid=True,
                message="Anthropic API key is valid"
            )
        except anthropic.RateLimitError:
            # Key is valid but rate limited
            return APIValidationResult(
                is_valid=True,
                message="Anthropic API key is valid but rate limited",
                rate_limited=True,
                rate_limit_reset=60  # Default to 60 seconds
            )
        except anthropic.AuthenticationError:
            return APIValidationResult(
                is_valid=False,
                message="Invalid Anthropic API key"
            )
        except Exception as e:
            logger.error(f"Error validating Anthropic API key: {e}")
            return APIValidationResult(
                is_valid=False,
                message=f"Error validating Anthropic API key: {str(e)}"
            )


class CredentialManager(ICredentialProvider):
    """
    Advanced credential manager with multiple sources and fallbacks.
    
    This manager can handle different credential sources with priorities,
    validate credentials, handle rotation, and provide detailed errors.
    """
    
    def __init__(self):
        """Initialize the credential manager."""
        self.sources: List[BaseCredentialSource] = []
        self.validators: Dict[str, BaseAPIValidator] = {}
        self.in_memory_source = MemoryCredentialSource()
        self.add_source(self.in_memory_source)
        
        # Add environment credential source by default
        self.add_source(EnvironmentCredentialSource())
        
        # Keep track of credential errors
        self._credential_errors: Dict[str, List[str]] = {}
        
        # Register validators
        self.register_validator("OPENAI_API_KEY", OpenAIValidator())
        self.register_validator("AZURE_OPENAI_API_KEY", AzureOpenAIValidator())
        self.register_validator("ANTHROPIC_API_KEY", AnthropicValidator())
    
    def add_source(self, source: BaseCredentialSource) -> None:
        """
        Add a credential source.
        
        Args:
            source: The credential source to add
        """
        self.sources.append(source)
        # Sort sources by priority
        self.sources.sort(key=lambda s: s.priority)
    
    def register_validator(self, credential_name: str, validator: BaseAPIValidator) -> None:
        """
        Register a validator for a credential type.
        
        Args:
            credential_name: The name of the credential to validate
            validator: The validator to use
        """
        self.validators[credential_name] = validator
    
    def get_credential(self, name: str) -> Optional[str]:
        """
        Get a credential by name, trying all sources in priority order.
        
        Args:
            name: The name of the credential
            
        Returns:
            The credential value, or None if not found
        """
        # Reset errors for this credential
        self._credential_errors[name] = []
        
        # Try each source in priority order
        for source in self.sources:
            credential = source.get_credential(name)
            if credential:
                logger.debug(f"Found credential {name} in source {source.name}")
                return credential
        
        # Not found in any source
        self._credential_errors[name].append(f"Credential {name} not found in any source")
        logger.warning(f"Credential {name} not found in any source")
        return None
    
    def set_credential(self, name: str, value: str) -> None:
        """
        Set a credential, storing it in the in-memory source.
        
        Args:
            name: The name of the credential
            value: The credential value
        """
        # Store in in-memory source
        self.in_memory_source.set_credential(name, value)
    
    def validate_credential(self, name: str, value: Optional[str] = None) -> APIValidationResult:
        """
        Validate a credential using the appropriate validator.
        
        Args:
            name: The name of the credential to validate
            value: The credential value to validate, or None to get from sources
            
        Returns:
            Validation result
            
        Raises:
            CredentialNotFoundError: If the credential is not found and no value is provided
            CredentialValidationError: If validation fails
        """
        # Get the credential value if not provided
        if value is None:
            value = self.get_credential(name)
            if value is None:
                raise CredentialNotFoundError(f"Credential {name} not found")
        
        # Check if we have a validator for this credential
        if name in self.validators:
            result = self.validators[name].validate(value)
            if not result.is_valid:
                self._credential_errors[name].append(result.message)
            return result
        
        # If no validator, assume it's valid
        return APIValidationResult(is_valid=True, message=f"No validator available for {name}")
    
    def get_with_fallback(
        self, 
        names: List[str], 
        validate: bool = True,
        raise_on_missing: bool = False
    ) -> Optional[str]:
        """
        Get a credential with fallbacks.
        
        Args:
            names: List of credential names to try in order
            validate: Whether to validate credentials
            raise_on_missing: Whether to raise an exception if no credential is found
            
        Returns:
            The first valid credential found, or None if none are found/valid
            
        Raises:
            CredentialNotFoundError: If no credential is found and raise_on_missing is True
        """
        for name in names:
            credential = self.get_credential(name)
            if credential:
                if validate and name in self.validators:
                    result = self.validate_credential(name, credential)
                    if not result.is_valid:
                        continue
                    if result.rate_limited:
                        logger.warning(f"Credential {name} is currently rate limited")
                        continue
                return credential
        
        # No valid credential found
        if raise_on_missing:
            raise CredentialNotFoundError(f"No valid credential found among: {', '.join(names)}")
        
        return None
    
    def get_all_providers(self, name: str) -> Dict[str, str]:
        """
        Get all credentials with the given name from all sources.
        
        Args:
            name: Base name of the credential
            
        Returns:
            Dictionary mapping source names to credential values
        """
        results = {}
        for source in self.sources:
            credential = source.get_credential(name)
            if credential:
                results[source.name] = credential
        return results
    
    def add_file_source(self, file_path: str, priority: int = 30) -> None:
        """
        Add a file-based credential source.
        
        Args:
            file_path: Path to the credentials file
            priority: Priority of the source
        """
        source = FileCredentialSource(file_path, priority=priority)
        self.add_source(source)
    
    def add_rotating_source(
        self, 
        prefix: str = "", 
        priority: int = 50, 
        cooldown: int = 60
    ) -> RotatingCredentialSource:
        """
        Add a rotating credential source.
        
        Args:
            prefix: Prefix for credential names
            priority: Priority of the source
            cooldown: Cooldown period in seconds
            
        Returns:
            The created rotating source
        """
        source = RotatingCredentialSource(
            credential_prefix=prefix,
            priority=priority,
            cooldown_seconds=cooldown
        )
        self.add_source(source)
        return source
    
    def get_errors(self, name: str) -> List[str]:
        """
        Get errors for a credential.
        
        Args:
            name: The name of the credential
            
        Returns:
            List of error messages
        """
        return self._credential_errors.get(name, [])
    
    def clear_errors(self, name: Optional[str] = None) -> None:
        """
        Clear errors for a credential or all credentials.
        
        Args:
            name: The name of the credential, or None for all
        """
        if name is None:
            self._credential_errors.clear()
        elif name in self._credential_errors:
            self._credential_errors[name] = []
    
    def validate_all_required(self, required_credentials: List[str]) -> bool:
        """
        Validate all required credentials.
        
        Args:
            required_credentials: List of required credential names
            
        Returns:
            True if all required credentials are valid, False otherwise
        """
        all_valid = True
        for name in required_credentials:
            value = self.get_credential(name)
            if value is None:
                self._credential_errors[name].append(f"Required credential {name} is missing")
                all_valid = False
                continue
            
            if name in self.validators:
                result = self.validators[name].validate(value)
                if not result.is_valid:
                    all_valid = False
        
        return all_valid


# Default instance for convenience
_default_manager = CredentialManager()

def get_credential_manager() -> CredentialManager:
    """Get the default credential manager."""
    return _default_manager

def get_credential(name: str) -> Optional[str]:
    """Get a credential from the default manager."""
    return _default_manager.get_credential(name)

def set_credential(name: str, value: str) -> None:
    """Set a credential in the default manager."""
    _default_manager.set_credential(name, value)

def validate_credential(name: str, value: Optional[str] = None) -> APIValidationResult:
    """Validate a credential using the default manager."""
    return _default_manager.validate_credential(name, value)

def get_with_fallback(
    names: List[str], 
    validate: bool = True,
    raise_on_missing: bool = False
) -> Optional[str]:
    """Get a credential with fallbacks using the default manager."""
    return _default_manager.get_with_fallback(names, validate, raise_on_missing)