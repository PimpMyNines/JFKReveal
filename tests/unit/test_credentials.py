"""
Tests for the credential management system.
"""
import os
import json
import tempfile
import pytest
from unittest.mock import patch, MagicMock

from src.jfkreveal.utils.credentials import (
    CredentialManager,
    MemoryCredentialSource,
    EnvironmentCredentialSource,
    FileCredentialSource,
    RotatingCredentialSource,
    APIValidationResult,
    CredentialNotFoundError,
    OpenAIValidator
)


class TestMemoryCredentialSource:
    """Tests for the in-memory credential source."""
    
    def test_get_set_credential(self):
        """Test getting and setting credentials in memory."""
        source = MemoryCredentialSource()
        
        # Initially, no credentials
        assert source.get_credential("TEST_KEY") is None
        
        # Set and get a credential
        source.set_credential("TEST_KEY", "test_value")
        assert source.get_credential("TEST_KEY") == "test_value"
        
        # Update a credential
        source.set_credential("TEST_KEY", "new_value")
        assert source.get_credential("TEST_KEY") == "new_value"
    
    def test_has_credential(self):
        """Test checking if a credential exists."""
        source = MemoryCredentialSource()
        
        # Initially, no credentials
        assert not source.has_credential("TEST_KEY")
        
        # Set a credential
        source.set_credential("TEST_KEY", "test_value")
        assert source.has_credential("TEST_KEY")
    
    def test_clear_credential(self):
        """Test clearing a credential."""
        source = MemoryCredentialSource()
        
        # Set a credential
        source.set_credential("TEST_KEY", "test_value")
        assert source.has_credential("TEST_KEY")
        
        # Clear the credential
        assert source.clear_credential("TEST_KEY")
        assert not source.has_credential("TEST_KEY")
        
        # Clearing a non-existent credential returns False
        assert not source.clear_credential("NON_EXISTENT")


class TestEnvironmentCredentialSource:
    """Tests for the environment variable credential source."""
    
    def test_get_credential_from_env(self):
        """Test getting a credential from environment variables."""
        with patch.dict(os.environ, {"TEST_ENV_KEY": "test_env_value"}):
            source = EnvironmentCredentialSource(load_dotenv=False)
            assert source.get_credential("TEST_ENV_KEY") == "test_env_value"
            # Non-existent key
            assert source.get_credential("NON_EXISTENT") is None
    
    def test_set_credential_to_env(self):
        """Test setting a credential in environment variables."""
        source = EnvironmentCredentialSource(load_dotenv=False)
        
        # Set a credential
        source.set_credential("TEST_ENV_SET", "test_env_set_value")
        
        # Check it was set in the environment
        assert os.environ.get("TEST_ENV_SET") == "test_env_set_value"


class TestFileCredentialSource:
    """Tests for the file-based credential source."""
    
    def test_file_source_operations(self):
        """Test file credential source operations."""
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False) as temp:
            temp_path = temp.name
        
        try:
            # Create a new file source
            source = FileCredentialSource(temp_path)
            
            # Set some credentials
            source.set_credential("FILE_KEY1", "file_value1")
            source.set_credential("FILE_KEY2", "file_value2")
            
            # Check they were saved
            with open(temp_path, "r") as f:
                data = json.load(f)
                assert data["FILE_KEY1"] == "file_value1"
                assert data["FILE_KEY2"] == "file_value2"
            
            # Create a new source to read from the file
            source2 = FileCredentialSource(temp_path)
            assert source2.get_credential("FILE_KEY1") == "file_value1"
            assert source2.get_credential("FILE_KEY2") == "file_value2"
            
            # Update a credential
            source2.set_credential("FILE_KEY1", "updated_value")
            
            # Check it was updated
            source3 = FileCredentialSource(temp_path)
            assert source3.get_credential("FILE_KEY1") == "updated_value"
            
        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_file_source_invalid_json(self):
        """Test handling invalid JSON in credential file."""
        # Create a temporary file with invalid JSON
        with tempfile.NamedTemporaryFile(delete=False) as temp:
            temp.write(b"not valid json")
            temp_path = temp.name
        
        try:
            # Create a file source, which should handle the invalid JSON
            source = FileCredentialSource(temp_path)
            
            # Should return None for non-existent keys
            assert source.get_credential("ANY_KEY") is None
            
            # Should be able to set a new credential
            source.set_credential("NEW_KEY", "new_value")
            
            # Check it was saved and the file is now valid JSON
            with open(temp_path, "r") as f:
                data = json.load(f)
                assert data["NEW_KEY"] == "new_value"
            
        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.unlink(temp_path)


class TestRotatingCredentialSource:
    """Tests for the rotating credential source."""
    
    def test_rotation_basic(self):
        """Test basic credential rotation."""
        source = RotatingCredentialSource(cooldown_seconds=0.1)  # Low cooldown for testing
        
        # Add multiple credentials
        source._source.set_credential("TEST_KEY_1", "value1")
        source._source.set_credential("TEST_KEY_2", "value2")
        
        # Get credentials, which should rotate
        keys_seen = set()
        for _ in range(10):
            key = "TEST_KEY"
            value = source.get_credential(key)
            if value == "value1":
                keys_seen.add("TEST_KEY_1")
            elif value == "value2":
                keys_seen.add("TEST_KEY_2")
        
        # Should have seen both keys
        assert len(keys_seen) == 2
    
    def test_cooldown(self):
        """Test credential cooldown mechanism."""
        source = RotatingCredentialSource(cooldown_seconds=1)
        
        # Add two credentials
        source._source.set_credential("COOLDOWN_KEY_1", "cooldown1")
        source._source.set_credential("COOLDOWN_KEY_2", "cooldown2")
        
        # Get a credential and mark it as used
        key = "COOLDOWN_KEY"
        value1 = source.get_credential(key)
        
        # Mark the credential as used with a long cooldown
        if value1 == "cooldown1":
            source.mark_credential_as_rate_limited("COOLDOWN_KEY_1", 10)
        else:
            source.mark_credential_as_rate_limited("COOLDOWN_KEY_2", 10)
        
        # Get the credential again, should get the other one
        value2 = source.get_credential(key)
        assert value1 != value2
    
    def test_add_credential(self):
        """Test adding a credential to rotation."""
        source = RotatingCredentialSource()
        
        # Add credentials using the add_credential method
        added_key1 = source.add_credential("ROTATE_KEY", "rotate_value1")
        added_key2 = source.add_credential("ROTATE_KEY", "rotate_value2")
        
        # Check they were added with different names
        assert added_key1 != added_key2
        assert source._source.get_credential(added_key1) == "rotate_value1"
        assert source._source.get_credential(added_key2) == "rotate_value2"
        
        # Get the credential, should get one of the values
        value = source.get_credential("ROTATE_KEY")
        assert value in ["rotate_value1", "rotate_value2"]


class TestCredentialManager:
    """Tests for the credential manager."""
    
    def test_basic_operations(self):
        """Test basic operations of the credential manager."""
        manager = CredentialManager()
        
        # Set and get a credential
        manager.set_credential("MANAGER_KEY", "manager_value")
        assert manager.get_credential("MANAGER_KEY") == "manager_value"
    
    def test_sources_priority(self):
        """Test source priority in the credential manager."""
        manager = CredentialManager()
        
        # Add a higher priority source
        high_source = MemoryCredentialSource("high", priority=5)
        high_source.set_credential("PRIORITY_KEY", "high_value")
        manager.add_source(high_source)
        
        # Add a lower priority source
        low_source = MemoryCredentialSource("low", priority=15)
        low_source.set_credential("PRIORITY_KEY", "low_value")
        manager.add_source(low_source)
        
        # The manager should return the value from the higher priority source
        assert manager.get_credential("PRIORITY_KEY") == "high_value"
    
    def test_fallback(self):
        """Test credential fallback mechanism."""
        manager = CredentialManager()
        
        # Set one credential
        manager.set_credential("FALLBACK_KEY1", "fallback_value1")
        
        # Try to get with fallback
        value = manager.get_with_fallback(
            ["NON_EXISTENT", "FALLBACK_KEY1", "FALLBACK_KEY2"],
            validate=False
        )
        
        # Should get the value from the first available credential
        assert value == "fallback_value1"
        
        # Try with no valid credentials
        value = manager.get_with_fallback(
            ["NON_EXISTENT1", "NON_EXISTENT2"],
            validate=False
        )
        
        # Should return None
        assert value is None
        
        # Try with raise_on_missing=True
        with pytest.raises(CredentialNotFoundError):
            manager.get_with_fallback(
                ["NON_EXISTENT1", "NON_EXISTENT2"],
                validate=False,
                raise_on_missing=True
            )
    
    def test_validators(self):
        """Test credential validation."""
        manager = CredentialManager()
        
        # Create a mock validator
        mock_validator = MagicMock()
        mock_validator.validate.return_value = APIValidationResult(True, "Valid")
        
        # Register the validator
        manager.register_validator("VALID_KEY", mock_validator)
        manager.set_credential("VALID_KEY", "valid_value")
        
        # Validate the credential
        result = manager.validate_credential("VALID_KEY")
        assert result.is_valid
        
        # Check the validator was called
        mock_validator.validate.assert_called_once_with("valid_value")
        
        # Test invalid credential
        mock_validator.validate.return_value = APIValidationResult(False, "Invalid")
        result = manager.validate_credential("VALID_KEY")
        assert not result.is_valid
        
        # Check error messages were stored
        assert "Invalid" in manager.get_errors("VALID_KEY")
    
    def test_validate_all_required(self):
        """Test validating all required credentials."""
        manager = CredentialManager()
        
        # Add some credentials
        manager.set_credential("REQUIRED1", "value1")
        manager.set_credential("REQUIRED2", "value2")
        
        # Mock validators that return success
        mock_validator1 = MagicMock()
        mock_validator1.validate.return_value = APIValidationResult(True, "Valid")
        
        mock_validator2 = MagicMock()
        mock_validator2.validate.return_value = APIValidationResult(True, "Valid")
        
        # Register the validators
        manager.register_validator("REQUIRED1", mock_validator1)
        manager.register_validator("REQUIRED2", mock_validator2)
        
        # Validate all required credentials
        result = manager.validate_all_required(["REQUIRED1", "REQUIRED2"])
        assert result
        
        # Make a validator fail
        mock_validator2.validate.return_value = APIValidationResult(False, "Invalid")
        
        # Validate again
        result = manager.validate_all_required(["REQUIRED1", "REQUIRED2"])
        assert not result
        
        # Test missing credential
        result = manager.validate_all_required(["REQUIRED1", "MISSING"])
        assert not result


class TestAPIValidators:
    """Tests for API validators."""
    
    @patch("openai.OpenAI")
    def test_openai_validator(self, mock_openai):
        """Test OpenAI API validator."""
        # Create a validator
        validator = OpenAIValidator()
        
        # Mock successful validation
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        
        # Test valid key
        result = validator.validate("valid_key")
        assert result.is_valid
        
        # Test auth error
        import openai
        from openai import AuthenticationError, RateLimitError
        
        # For newer OpenAI versions (v1+), we need to create a proper APIError
        # Create a mock response and error object for the AuthenticationError
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.headers = {}
        mock_response.json.return_value = {"error": {"message": "Invalid key"}}
        
        # Create the auth error with the required parameters
        auth_error = AuthenticationError(message="Invalid key", response=mock_response, body={"error": {"message": "Invalid key"}})
        mock_client.models.list.side_effect = auth_error
        
        result = validator.validate("invalid_key")
        assert not result.is_valid
        
        # Test rate limit error
        # Create a mock response and error object for the RateLimitError
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.headers = {"x-ratelimit-reset-tokens": "60"}
        mock_response.json.return_value = {"error": {"message": "Rate limited"}}
        
        # Create the rate limit error with the required parameters
        rate_limit_error = RateLimitError(message="Rate limited", response=mock_response, body={"error": {"message": "Rate limited"}})
        mock_client.models.list.side_effect = rate_limit_error
        
        result = validator.validate("rate_limited_key")
        assert result.is_valid
        assert result.rate_limited


def test_create_credential_provider():
    """Test the create_credential_provider factory function."""
    from src.jfkreveal.factories import create_credential_provider
    
    # Basic provider
    provider = create_credential_provider()
    assert provider is not None
    
    # With config file
    with tempfile.NamedTemporaryFile(delete=False) as temp:
        temp.write(b'{"TEST_KEY": "test_value"}')
        temp_path = temp.name
    
    try:
        provider = create_credential_provider(config_file=temp_path)
        assert provider.get_credential("TEST_KEY") == "test_value"
        
    finally:
        # Clean up
        if os.path.exists(temp_path):
            os.unlink(temp_path)
    
    # With required credentials and validation
    with patch("src.jfkreveal.utils.credentials.CredentialManager.get_credential") as mock_get:
        # Set up the mock to return None to simulate missing credential
        mock_get.return_value = None
        
        # Create a provider with required credentials and verify logging
        with patch("src.jfkreveal.factories.logger.warning") as mock_warning:
            provider = create_credential_provider(required_credentials=["REQUIRED_KEY"])
            
            # Verify get_credential was called with the required key
            mock_get.assert_called_with("REQUIRED_KEY")
            
            # Verify warning was logged for missing credential
            mock_warning.assert_called_with("Required credential REQUIRED_KEY is missing")