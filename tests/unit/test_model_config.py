"""
Unit tests for the ModelConfiguration class
"""
import os
import json
import pytest
from unittest.mock import patch, MagicMock, mock_open

from jfkreveal.utils.model_config import (
    ModelConfiguration, 
    ModelProvider,
    ModelType,
    AnalysisTask,
    ReportType
)
from jfkreveal.utils.model_registry import ModelRegistry


class TestModelConfiguration:
    """Test the ModelConfiguration class"""

    @patch('os.path.exists')
    def test_init_default(self, mock_exists):
        """Test initialization with default values"""
        # Configure mock to indicate config file doesn't exist
        mock_exists.return_value = False
        
        # Mock save_config to avoid actual file operations
        with patch.object(ModelConfiguration, 'save_config') as mock_save:
            # Create instance with defaults
            config = ModelConfiguration(prefer_local=True)
            
            # Verify attributes
            assert config.prefer_local is True
            assert config.config_path == "config/models.json"
            
            # Verify config loaded correctly with defaults
            assert "prefer_local" in config.config
            assert config.config["prefer_local"] is True
            assert "models" in config.config
            
            # Verify default models are set
            assert "embedding" in config.config["models"]
            assert AnalysisTask.DOCUMENT_ANALYSIS.value in config.config["models"]
            
            # Verify save_config was called
            mock_save.assert_called_once()
    
    @patch('os.path.exists')
    def test_init_with_file(self, mock_exists):
        """Test initialization with existing config file"""
        # Configure mock to indicate config file exists
        mock_exists.return_value = True
        
        # Sample config data
        sample_config = {
            "prefer_local": False,
            "models": {
                "embedding": {
                    "default": "test-embedding-model",
                    "provider": "openai"
                },
                "document_analysis": {
                    "default": "test-analysis-model",
                    "provider": "openai"
                }
            }
        }
        
        # Mock open and json.load
        m = mock_open(read_data=json.dumps(sample_config))
        with patch('builtins.open', m):
            with patch('json.load', return_value=sample_config):
                # Create instance
                config = ModelConfiguration()
                
                # Verify config loaded from file
                assert config.config["prefer_local"] is False
                assert config.config["models"]["embedding"]["default"] == "test-embedding-model"
                assert config.config["models"]["document_analysis"]["default"] == "test-analysis-model"
    
    @patch('os.path.exists')
    def test_save_config(self, mock_exists):
        """Test saving configuration to file"""
        # Configure mock to indicate config file doesn't exist
        mock_exists.return_value = False
        
        # Create test instance
        config = ModelConfiguration(prefer_local=True)
        
        # Mock open and json.dump for save operation
        m = mock_open()
        with patch('builtins.open', m):
            with patch('json.dump') as mock_json_dump:
                # Call save_config with explicit config
                result = config.save_config({"test": "config"})
                
                # Verify file was opened for writing
                m.assert_called_once_with("config/models.json", 'w')
                
                # Verify json.dump was called with correct args
                mock_json_dump.assert_called_once()
                args, kwargs = mock_json_dump.call_args
                assert args[0] == {"test": "config"}
                
                # Verify result is True
                assert result is True
    
    @patch('subprocess.run')
    def test_detect_available_models(self, mock_subprocess_run):
        """Test detecting available models"""
        # Mock subprocess for Ollama models
        mock_subprocess_run.return_value = MagicMock(
            returncode=0,
            stdout="NAME       ID        SIZE   MODIFIED\nllama3     latest    3.8GB  1 day ago\nmistral    latest    4.1GB  2 days ago\n"
        )
        
        # Patch environment variables for OpenAI
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            # Patch transformers import check
            with patch('importlib.import_module', return_value=None):
                # Create instance
                config = ModelConfiguration()
                
                # Verify available models were detected
                available = config._detect_available_models()
                
                # OpenAI models should be available with API key
                assert len(available[ModelProvider.OPENAI]) > 0
                assert "gpt-4o" in available[ModelProvider.OPENAI]
                
                # Ollama models should be detected from subprocess output
                assert "llama3" in available[ModelProvider.OLLAMA]
                assert "mistral" in available[ModelProvider.OLLAMA]
    
    @patch.object(ModelConfiguration, '_detect_available_models')
    def test_is_model_available(self, mock_detect):
        """Test checking if models are available"""
        # Setup mock to return known available models
        mock_detect.return_value = {
            ModelProvider.OPENAI: ["gpt-4o", "text-embedding-3-large"],
            ModelProvider.OLLAMA: ["llama3", "nomic-embed-text"],
            ModelProvider.HUGGINGFACE: []
        }
        
        # Create instance
        config = ModelConfiguration()
        
        # Test with known models
        assert config.is_model_available("gpt-4o", ModelProvider.OPENAI) is True
        assert config.is_model_available("llama3", ModelProvider.OLLAMA) is True
        
        # Test with unknown models
        assert config.is_model_available("nonexistent", ModelProvider.OPENAI) is False
        assert config.is_model_available("mistral", ModelProvider.OLLAMA) is False
        
        # Test with provider inference
        assert config.is_model_available("gpt-4o") is True
        assert config.is_model_available("llama3") is True
        assert config.is_model_available("nonexistent") is False
    
    @patch.object(ModelConfiguration, 'is_model_available')
    def test_get_model_for_task(self, mock_is_available):
        """Test getting model for a specific task"""
        # Setup mock for availability checks
        mock_is_available.side_effect = lambda model, provider: model == "llama3"
        
        # Create test configuration
        config = ModelConfiguration()
        
        # Setup test config with specific model for document analysis
        config.config["models"] = {
            AnalysisTask.DOCUMENT_ANALYSIS.value: {
                "default": "llama3",
                "provider": "ollama",
                "fallback": "gpt-3.5-turbo"
            }
        }
        
        # Test with configured task
        model_name, provider = config.get_model_for_task(AnalysisTask.DOCUMENT_ANALYSIS)
        assert model_name == "llama3"
        assert provider == ModelProvider.OLLAMA
        
        # Change availability mock to indicate model not available
        mock_is_available.side_effect = lambda model, provider: False
        
        # Test fallback to OpenAI
        model_name, provider = config.get_model_for_task(AnalysisTask.DOCUMENT_ANALYSIS)
        assert model_name == "gpt-3.5-turbo"
        assert provider == ModelProvider.OPENAI
    
    @patch.object(ModelConfiguration, 'is_model_available')
    def test_get_embedding_model(self, mock_is_available):
        """Test getting embedding model"""
        # Setup mock for availability checks
        mock_is_available.side_effect = lambda model, provider: model == "nomic-embed-text"
        
        # Create test configuration
        config = ModelConfiguration()
        
        # Setup test config with specific embedding model
        config.config["models"] = {
            "embedding": {
                "default": "nomic-embed-text",
                "provider": "ollama",
                "fallback": "text-embedding-3-small"
            }
        }
        
        # Test with configured embedding
        model_name, provider = config.get_embedding_model()
        assert model_name == "nomic-embed-text"
        assert provider == ModelProvider.OLLAMA
        
        # Change availability mock to indicate model not available
        mock_is_available.side_effect = lambda model, provider: False
        
        # Test fallback to OpenAI
        model_name, provider = config.get_embedding_model()
        assert model_name == "text-embedding-3-small"
        assert provider == ModelProvider.OPENAI
    
    @patch.object(ModelConfiguration, 'is_model_available')
    def test_get_all_report_models(self, mock_is_available):
        """Test getting all models for reporting"""
        # Setup mock for availability checks
        mock_is_available.side_effect = lambda model, provider: model in ["gpt-4o", "llama3"]
        
        # Create test configuration
        config = ModelConfiguration()
        
        # Setup test config for multi-model reports
        config.config["report_configuration"] = {
            "type": ReportType.MULTI_MODEL_COMPARISON.value,
            "multi_model_enabled": True,
            "models_to_compare": ["gpt-4o", "llama3", "mistral"]
        }
        
        # Test with multi-model configuration
        models = config.get_all_report_models()
        
        # Should return available models from the comparison list
        assert len(models) == 2
        assert ("gpt-4o", ModelProvider.OPENAI) in models
        assert ("llama3", ModelProvider.OLLAMA) in models
        
        # Disable multi-model reports
        config.config["report_configuration"]["multi_model_enabled"] = False
        
        # Configure default model for report generation
        config.config["models"] = {
            AnalysisTask.REPORT_GENERATION.value: {
                "default": "gpt-4o",
                "provider": "openai"
            }
        }
        
        # Test with standard report configuration
        models = config.get_all_report_models()
        
        # Should return single model for report generation
        assert len(models) == 1
        assert models[0] == ("gpt-4o", ModelProvider.OPENAI)
    
    def test_get_report_type(self):
        """Test getting report type"""
        # Create test configuration
        config = ModelConfiguration()
        
        # Test default
        assert config.get_report_type() == ReportType.STANDARD
        
        # Set custom type
        config.config["report_configuration"] = {"type": "multi_model_comparison"}
        assert config.get_report_type() == ReportType.MULTI_MODEL_COMPARISON
    
    @patch.object(ModelConfiguration, 'save_config')
    def test_set_report_type(self, mock_save):
        """Test setting report type"""
        # Create test configuration
        config = ModelConfiguration()
        
        # Test setting with enum
        config.set_report_type(ReportType.MULTI_MODEL_COMPARISON)
        assert config.config["report_configuration"]["type"] == "multi_model_comparison"
        
        # Test setting with string
        config.set_report_type("consolidated")
        assert config.config["report_configuration"]["type"] == "consolidated"
        
        # Verify save_config was called
        assert mock_save.call_count == 2
    
    @patch.object(ModelConfiguration, 'save_config')
    def test_enable_multi_model_reports(self, mock_save):
        """Test enabling multi-model reports"""
        # Create test configuration
        config = ModelConfiguration()
        
        # Test enabling with default models
        config.enable_multi_model_reports(True)
        assert config.config["report_configuration"]["multi_model_enabled"] is True
        
        # Test with custom models
        models = ["gpt-4o", "mistral", "llama3"]
        config.enable_multi_model_reports(True, models)
        assert config.config["report_configuration"]["multi_model_enabled"] is True
        assert config.config["report_configuration"]["models_to_compare"] == models
        
        # Test disabling
        config.enable_multi_model_reports(False)
        assert config.config["report_configuration"]["multi_model_enabled"] is False
        
        # Verify save_config was called
        assert mock_save.call_count == 3
    
    @patch.object(ModelConfiguration, 'save_config')
    def test_set_model_for_task(self, mock_save):
        """Test setting model for a task"""
        # Create test configuration
        config = ModelConfiguration()
        
        # Test setting with enum
        config.set_model_for_task(
            AnalysisTask.DOCUMENT_ANALYSIS,
            "gpt-4o",
            ModelProvider.OPENAI
        )
        assert config.config["models"][AnalysisTask.DOCUMENT_ANALYSIS.value]["default"] == "gpt-4o"
        assert config.config["models"][AnalysisTask.DOCUMENT_ANALYSIS.value]["provider"] == "openai"
        
        # Test setting with strings
        config.set_model_for_task(
            "topic_analysis",
            "llama3",
            "ollama"
        )
        assert config.config["models"]["topic_analysis"]["default"] == "llama3"
        assert config.config["models"]["topic_analysis"]["provider"] == "ollama"
        
        # Verify save_config was called
        assert mock_save.call_count == 2
    
    @patch.object(ModelConfiguration, 'save_config')
    def test_set_embedding_model(self, mock_save):
        """Test setting embedding model"""
        # Create test configuration
        config = ModelConfiguration()
        
        # Test setting with enum
        config.set_embedding_model("text-embedding-3-large", ModelProvider.OPENAI)
        assert config.config["models"]["embedding"]["default"] == "text-embedding-3-large"
        assert config.config["models"]["embedding"]["provider"] == "openai"
        
        # Test setting with string
        config.set_embedding_model("nomic-embed-text", "ollama")
        assert config.config["models"]["embedding"]["default"] == "nomic-embed-text"
        assert config.config["models"]["embedding"]["provider"] == "ollama"
        
        # Verify save_config was called
        assert mock_save.call_count == 2
    
    @patch.object(ModelConfiguration, 'save_config')
    def test_set_prefer_local(self, mock_save):
        """Test setting prefer_local flag"""
        # Create test configuration
        config = ModelConfiguration(prefer_local=False)
        
        # Change to True
        config.set_prefer_local(True)
        assert config.prefer_local is True
        assert config.config["prefer_local"] is True
        
        # Change back to False
        config.set_prefer_local(False)
        assert config.prefer_local is False
        assert config.config["prefer_local"] is False
        
        # Verify save_config was called
        assert mock_save.call_count == 2
    
    @patch.object(ModelConfiguration, '_detect_available_models')
    def test_list_available_models(self, mock_detect):
        """Test listing available models"""
        # Setup mock to return known available models
        mock_detect.return_value = {
            ModelProvider.OPENAI: ["gpt-4o", "text-embedding-3-large"],
            ModelProvider.OLLAMA: ["llama3", "nomic-embed-text"],
            ModelProvider.HUGGINGFACE: []
        }
        
        # Create instance
        config = ModelConfiguration()
        
        # Test listing all available models
        models = config.list_available_models()
        assert len(models) >= 4
        
        # Test filtering by provider
        openai_models = config.list_available_models(provider=ModelProvider.OPENAI)
        assert len(openai_models) == 2
        assert all(model.provider == ModelProvider.OPENAI for model in openai_models)
        
        # Test filtering by model type
        embedding_models = config.list_available_models(model_type=ModelType.EMBEDDING)
        assert len(embedding_models) >= 2
        assert all(model.model_type == ModelType.EMBEDDING for model in embedding_models)