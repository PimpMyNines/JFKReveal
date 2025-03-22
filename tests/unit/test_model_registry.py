"""
Unit tests for the ModelRegistry class
"""
import os
import pytest
from unittest.mock import patch, MagicMock

from jfkreveal.utils.model_registry import (
    ModelRegistry,
    ModelProvider,
    ModelType,
    ModelInfo
)


class TestModelRegistry:
    """Test the ModelRegistry class"""

    def test_get_model_info(self):
        """Test getting model information"""
        # Test with known model
        model_info = ModelRegistry.get_model_info("gpt-4o")
        assert model_info is not None
        assert model_info.name == "gpt-4o"
        assert model_info.provider == ModelProvider.OPENAI
        assert model_info.model_type == ModelType.CHAT
        assert model_info.context_length == 128000
        
        # Test with unknown model
        model_info = ModelRegistry.get_model_info("nonexistent-model")
        assert model_info is None
    
    def test_list_models(self):
        """Test listing models with filters"""
        # Test all models
        all_models = ModelRegistry.list_models()
        assert len(all_models) > 0
        
        # Test filtering by provider
        openai_models = ModelRegistry.list_models(provider=ModelProvider.OPENAI)
        assert len(openai_models) > 0
        assert all(model.provider == ModelProvider.OPENAI for model in openai_models)
        
        # Test filtering by model type
        embedding_models = ModelRegistry.list_models(model_type=ModelType.EMBEDDING)
        assert len(embedding_models) > 0
        assert all(model.model_type == ModelType.EMBEDDING for model in embedding_models)
        
        # Test filtering by local only
        local_models = ModelRegistry.list_models(local_only=True)
        assert len(local_models) > 0
        assert all(model.local for model in local_models)
        
        # Test combined filters
        local_embeddings = ModelRegistry.list_models(
            model_type=ModelType.EMBEDDING, 
            local_only=True
        )
        assert len(local_embeddings) > 0
        assert all(model.local and model.model_type == ModelType.EMBEDDING for model in local_embeddings)
    
    @patch('jfkreveal.utils.model_registry.OpenAIEmbeddings')
    def test_get_embedding_model_openai(self, mock_openai_embeddings):
        """Test getting OpenAI embedding model"""
        # Setup mock
        mock_embeddings_instance = MagicMock()
        mock_openai_embeddings.return_value = mock_embeddings_instance
        
        # Test with explicit model name and provider
        embeddings = ModelRegistry.get_embedding_model(
            model_name="text-embedding-3-large",
            provider=ModelProvider.OPENAI,
            api_key="test-key"
        )
        
        # Verify OpenAI embeddings were created correctly
        mock_openai_embeddings.assert_called_once_with(
            model="text-embedding-3-large",
            openai_api_key="test-key"
        )
        
        # Verify the correct instance was returned
        assert embeddings == mock_embeddings_instance
    
    @patch('jfkreveal.utils.model_registry.OllamaEmbeddings')
    def test_get_embedding_model_ollama(self, mock_ollama_embeddings):
        """Test getting Ollama embedding model"""
        # Setup mock
        mock_embeddings_instance = MagicMock()
        mock_ollama_embeddings.return_value = mock_embeddings_instance
        
        # Test with explicit model name and provider
        embeddings = ModelRegistry.get_embedding_model(
            model_name="nomic-embed-text",
            provider=ModelProvider.OLLAMA,
            base_url="http://test-ollama:11434"
        )
        
        # Verify Ollama embeddings were created correctly
        mock_ollama_embeddings.assert_called_once_with(
            model="nomic-embed-text",
            base_url="http://test-ollama:11434"
        )
        
        # Verify the correct instance was returned
        assert embeddings == mock_embeddings_instance
    
    @patch('jfkreveal.utils.model_registry.HuggingFaceEmbeddings')
    def test_get_embedding_model_huggingface(self, mock_hf_embeddings):
        """Test getting HuggingFace embedding model"""
        # Setup mock
        mock_embeddings_instance = MagicMock()
        mock_hf_embeddings.return_value = mock_embeddings_instance
        
        # Test with explicit model name and provider
        embeddings = ModelRegistry.get_embedding_model(
            model_name="sentence-transformers/all-mpnet-base-v2",
            provider=ModelProvider.HUGGINGFACE
        )
        
        # Verify HuggingFace embeddings were created correctly
        mock_hf_embeddings.assert_called_once_with(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )
        
        # Verify the correct instance was returned
        assert embeddings == mock_embeddings_instance
    
    @patch.dict(os.environ, {"EMBEDDING_PROVIDER": "openai", "OPENAI_EMBEDDING_MODEL": "text-embedding-3-small", "OPENAI_API_KEY": "your-api-key-here"})
    @patch('jfkreveal.utils.model_registry.OpenAIEmbeddings')
    def test_get_embedding_model_from_env(self, mock_openai_embeddings):
        """Test getting embedding model from environment variables"""
        # Setup mock
        mock_embeddings_instance = MagicMock()
        mock_openai_embeddings.return_value = mock_embeddings_instance
        
        # Test with no explicit model or provider
        embeddings = ModelRegistry.get_embedding_model()
        
        # Verify OpenAI embeddings were created with env vars
        mock_openai_embeddings.assert_called_once_with(
            model="text-embedding-3-small",
            openai_api_key="your-api-key-here"
        )
        
        # Verify the correct instance was returned
        assert embeddings == mock_embeddings_instance
    
    @patch('jfkreveal.utils.model_registry.ChatOpenAI')
    def test_get_chat_model_openai(self, mock_chat_openai):
        """Test getting OpenAI chat model"""
        # Setup mock
        mock_chat_instance = MagicMock()
        mock_chat_openai.return_value = mock_chat_instance
        
        # Test with explicit model name and provider
        chat_model = ModelRegistry.get_chat_model(
            model_name="gpt-4o",
            provider=ModelProvider.OPENAI,
            api_key="test-key",
            temperature=0.5
        )
        
        # Verify OpenAI chat was created correctly
        mock_chat_openai.assert_called_once_with(
            model="gpt-4o",
            openai_api_key="test-key",
            temperature=0.5
        )
        
        # Verify the correct instance was returned
        assert chat_model == mock_chat_instance
    
    @patch('jfkreveal.utils.model_registry.ChatOllama')
    def test_get_chat_model_ollama(self, mock_chat_ollama):
        """Test getting Ollama chat model"""
        # Setup mock
        mock_chat_instance = MagicMock()
        mock_chat_ollama.return_value = mock_chat_instance
        
        # Test with explicit model name and provider
        chat_model = ModelRegistry.get_chat_model(
            model_name="llama3",
            provider=ModelProvider.OLLAMA,
            base_url="http://test-ollama:11434",
            temperature=0.7
        )
        
        # Verify Ollama chat was created correctly
        mock_chat_ollama.assert_called_once_with(
            model="llama3",
            base_url="http://test-ollama:11434",
            temperature=0.7
        )
        
        # Verify the correct instance was returned
        assert chat_model == mock_chat_instance
    
    @patch('subprocess.run')
    def test_download_model(self, mock_subprocess_run):
        """Test downloading a model"""
        # Setup mock for success
        mock_subprocess_run.return_value = MagicMock(returncode=0)
        
        # Test downloading model
        result = ModelRegistry.download_model("llama3", provider=ModelProvider.OLLAMA)
        
        # Verify subprocess was called correctly
        mock_subprocess_run.assert_called_once_with(
            ["ollama", "pull", "llama3"], 
            capture_output=True, 
            text=True, 
            check=False
        )
        
        # Verify result
        assert result is True
        
        # Test with error
        mock_subprocess_run.reset_mock()
        mock_subprocess_run.return_value = MagicMock(returncode=1, stderr="Error message")
        result = ModelRegistry.download_model("nonexistent-model", provider=ModelProvider.OLLAMA)
        
        # Verify result
        assert result is False
        
        # Test with non-Ollama provider
        result = ModelRegistry.download_model("gpt-4o", provider=ModelProvider.OPENAI)
        assert result is False