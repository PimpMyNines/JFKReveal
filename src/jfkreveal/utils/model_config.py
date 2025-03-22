"""
Configuration management for AI models used in the JFKReveal project.
"""
import os
import logging
from typing import Dict, Any, Optional, List, Set, Tuple, Union
from enum import Enum
import json
from pathlib import Path

from .model_registry import ModelRegistry, ModelProvider, ModelType, ModelInfo

logger = logging.getLogger(__name__)

class AnalysisTask(str, Enum):
    """Tasks that require AI models."""
    DOCUMENT_ANALYSIS = "document_analysis"
    TOPIC_ANALYSIS = "topic_analysis"
    REPORT_GENERATION = "report_generation"
    EXECUTIVE_SUMMARY = "executive_summary"
    SUSPECTS_ANALYSIS = "suspects_analysis"
    COVERUP_ANALYSIS = "coverup_analysis"
    SEMANTIC_SEARCH = "semantic_search"

class ReportType(str, Enum):
    """Types of reports that can be generated."""
    STANDARD = "standard"
    MULTI_MODEL_COMPARISON = "multi_model_comparison"
    CONSOLIDATED = "consolidated"

class ModelConfiguration:
    """
    Configure and manage AI models for different tasks.
    Provides flexibility to use different models for different tasks,
    with options to prefer local over cloud models.
    """
    
    # Default configuration file path
    DEFAULT_CONFIG_PATH = "config/models.json"
    
    def __init__(self, 
                config_path: Optional[str] = None, 
                prefer_local: bool = True,
                default_models: Optional[Dict[str, Dict[str, str]]] = None):
        """
        Initialize the model configuration.
        
        Args:
            config_path: Path to configuration file
            prefer_local: Whether to prefer local models by default
            default_models: Default model configuration to use if no file is found
        """
        self.prefer_local = prefer_local
        self.config_path = config_path or os.environ.get("MODEL_CONFIG_PATH", self.DEFAULT_CONFIG_PATH)
        self.config = self._load_config(default_models)
        self.available_models = self._detect_available_models()
    
    def _load_config(self, default_models: Optional[Dict[str, Dict[str, str]]] = None) -> Dict[str, Any]:
        """
        Load configuration from file or use defaults.
        
        Args:
            default_models: Default model configuration
            
        Returns:
            Configuration dictionary
        """
        config = {
            "prefer_local": self.prefer_local,
            "models": default_models or {
                # Default embedding models
                "embedding": {
                    "default": ModelRegistry.DEFAULT_EMBEDDING_MODELS[
                        ModelProvider.OLLAMA if self.prefer_local else ModelProvider.OPENAI
                    ],
                    "provider": ModelProvider.OLLAMA.value if self.prefer_local else ModelProvider.OPENAI.value,
                    "fallback": ModelRegistry.DEFAULT_EMBEDDING_MODELS[ModelProvider.OPENAI]
                },
                
                # Default models for each task
                AnalysisTask.DOCUMENT_ANALYSIS.value: {
                    "default": ModelRegistry.DEFAULT_CHAT_MODELS[
                        ModelProvider.OLLAMA if self.prefer_local else ModelProvider.OPENAI
                    ],
                    "provider": ModelProvider.OLLAMA.value if self.prefer_local else ModelProvider.OPENAI.value,
                    "fallback": ModelRegistry.DEFAULT_CHAT_MODELS[ModelProvider.OPENAI]
                },
                AnalysisTask.TOPIC_ANALYSIS.value: {
                    "default": ModelRegistry.DEFAULT_CHAT_MODELS[
                        ModelProvider.OLLAMA if self.prefer_local else ModelProvider.OPENAI
                    ],
                    "provider": ModelProvider.OLLAMA.value if self.prefer_local else ModelProvider.OPENAI.value,
                    "fallback": ModelRegistry.DEFAULT_CHAT_MODELS[ModelProvider.OPENAI]
                },
                AnalysisTask.REPORT_GENERATION.value: {
                    "default": ModelRegistry.DEFAULT_CHAT_MODELS[
                        ModelProvider.OLLAMA if self.prefer_local else ModelProvider.OPENAI
                    ],
                    "provider": ModelProvider.OLLAMA.value if self.prefer_local else ModelProvider.OPENAI.value,
                    "fallback": ModelRegistry.DEFAULT_CHAT_MODELS[ModelProvider.OPENAI]
                },
                AnalysisTask.EXECUTIVE_SUMMARY.value: {
                    "default": "gpt-4o-mini" if not self.prefer_local else "llama3",
                    "provider": ModelProvider.OPENAI.value if not self.prefer_local else ModelProvider.OLLAMA.value,
                    "fallback": "gpt-3.5-turbo"
                },
                AnalysisTask.SUSPECTS_ANALYSIS.value: {
                    "default": "gpt-4o" if not self.prefer_local else "llama3",
                    "provider": ModelProvider.OPENAI.value if not self.prefer_local else ModelProvider.OLLAMA.value,
                    "fallback": "gpt-3.5-turbo"
                },
                AnalysisTask.COVERUP_ANALYSIS.value: {
                    "default": "gpt-4o" if not self.prefer_local else "llama3",
                    "provider": ModelProvider.OPENAI.value if not self.prefer_local else ModelProvider.OLLAMA.value,
                    "fallback": "gpt-3.5-turbo"
                },
            },
            "report_configuration": {
                "type": ReportType.STANDARD.value,
                "multi_model_enabled": False,
                "models_to_compare": [
                    "gpt-4o",
                    "llama3", 
                    "mistral"
                ],
                "consolidated_model": "gpt-4o"
            }
        }
        
        # Try to load from file if it exists
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    loaded_config = json.load(f)
                    # Update our default config with loaded values
                    config.update(loaded_config)
                    logger.info(f"Loaded model configuration from {self.config_path}")
            except Exception as e:
                logger.warning(f"Error loading model configuration from {self.config_path}: {e}")
                logger.info("Using default model configuration")
        else:
            logger.info(f"Configuration file {self.config_path} not found, using defaults")
            # Create the config directory if it doesn't exist
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            # Save the default configuration
            self.save_config(config)
            
        return config
    
    def save_config(self, config: Optional[Dict[str, Any]] = None) -> bool:
        """
        Save the current configuration to file.
        
        Args:
            config: Configuration to save (uses self.config if None)
            
        Returns:
            True if successful, False otherwise
        """
        config_to_save = config or self.config
        try:
            # Create directories if they don't exist
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            
            with open(self.config_path, 'w') as f:
                json.dump(config_to_save, f, indent=2)
            logger.info(f"Saved model configuration to {self.config_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving model configuration to {self.config_path}: {e}")
            return False
    
    def _detect_available_models(self) -> Dict[ModelProvider, List[str]]:
        """
        Detect which models are available on the system.
        
        Returns:
            Dictionary of available models by provider
        """
        available = {
            ModelProvider.OPENAI: [],
            ModelProvider.OLLAMA: [],
            ModelProvider.HUGGINGFACE: []
        }
        
        # Check OpenAI API key
        if os.environ.get("OPENAI_API_KEY"):
            # All OpenAI models should be available if API key is set
            available[ModelProvider.OPENAI] = [
                model.name for model in ModelRegistry.list_models(provider=ModelProvider.OPENAI)
            ]
        
        # Check for Ollama
        try:
            import subprocess
            result = subprocess.run(["ollama", "list"], capture_output=True, text=True, check=False)
            if result.returncode == 0:
                # Parse output to get available models
                lines = result.stdout.strip().split('\n')
                if len(lines) > 1:  # Skip header line
                    for line in lines[1:]:
                        parts = line.split()
                        if parts:
                            model_name = parts[0]
                            available[ModelProvider.OLLAMA].append(model_name)
        except Exception as e:
            logger.warning(f"Error detecting Ollama models: {e}")
        
        # For HuggingFace, assume all models are available if transformers is installed
        try:
            import transformers
            available[ModelProvider.HUGGINGFACE] = [
                model.name for model in ModelRegistry.list_models(provider=ModelProvider.HUGGINGFACE)
            ]
        except ImportError:
            logger.warning("Transformers not installed, HuggingFace models unavailable")
        
        return available
    
    def is_model_available(self, model_name: str, provider: Optional[ModelProvider] = None) -> bool:
        """
        Check if a specific model is available.
        
        Args:
            model_name: Name of the model to check
            provider: Provider of the model (inferred from model name if None)
            
        Returns:
            True if the model is available, False otherwise
        """
        if provider is None:
            # Try to get model info from registry
            model_info = ModelRegistry.get_model_info(model_name)
            if model_info:
                provider = model_info.provider
            else:
                # Try to infer provider from model name
                if "gpt" in model_name.lower() or "text-embedding-" in model_name.lower():
                    provider = ModelProvider.OPENAI
                elif any(name in model_name.lower() for name in ["llama", "mistral", "gemma", "nomic", "mxbai"]):
                    provider = ModelProvider.OLLAMA
                else:
                    provider = ModelProvider.HUGGINGFACE
        
        return model_name in self.available_models.get(provider, [])
    
    def get_model_for_task(self, task: Union[str, AnalysisTask], 
                          fallback_to_openai: bool = True) -> Tuple[str, ModelProvider]:
        """
        Get the configured model for a specific task.
        
        Args:
            task: Task requiring a model
            fallback_to_openai: Whether to fallback to OpenAI if local model unavailable
            
        Returns:
            Tuple of (model_name, provider)
        """
        task_value = task.value if isinstance(task, AnalysisTask) else task
        
        # Get task configuration
        task_config = self.config.get("models", {}).get(task_value, {})
        if not task_config:
            logger.warning(f"No configuration found for task {task_value}, using default chat model")
            provider_name = ModelProvider.OLLAMA.value if self.prefer_local else ModelProvider.OPENAI.value
            provider = ModelProvider(provider_name)
            return ModelRegistry.DEFAULT_CHAT_MODELS[provider], provider
        
        # Get model and provider
        model_name = task_config.get("default")
        provider = ModelProvider(task_config.get("provider"))
        
        # Check if model is available
        if not self.is_model_available(model_name, provider):
            logger.warning(f"Model {model_name} not available for provider {provider}")
            
            if fallback_to_openai and provider != ModelProvider.OPENAI:
                # Fallback to OpenAI
                fallback_model = task_config.get("fallback")
                logger.info(f"Falling back to OpenAI model: {fallback_model}")
                return fallback_model, ModelProvider.OPENAI
            
            # If no fallback or already OpenAI, use default
            logger.warning(f"Using default model for provider {provider}")
            return ModelRegistry.DEFAULT_CHAT_MODELS[provider], provider
        
        return model_name, provider
    
    def get_embedding_model(self, fallback_to_openai: bool = True) -> Tuple[str, ModelProvider]:
        """
        Get the configured embedding model.
        
        Args:
            fallback_to_openai: Whether to fallback to OpenAI if local unavailable
            
        Returns:
            Tuple of (model_name, provider)
        """
        # Get embedding configuration
        embed_config = self.config.get("models", {}).get("embedding", {})
        if not embed_config:
            logger.warning("No embedding configuration found, using default")
            provider = ModelProvider.OLLAMA if self.prefer_local else ModelProvider.OPENAI
            return ModelRegistry.DEFAULT_EMBEDDING_MODELS[provider], provider
        
        # Get model and provider
        model_name = embed_config.get("default")
        provider = ModelProvider(embed_config.get("provider"))
        
        # Check if model is available
        if not self.is_model_available(model_name, provider):
            logger.warning(f"Embedding model {model_name} not available for provider {provider}")
            
            if fallback_to_openai and provider != ModelProvider.OPENAI:
                # Fallback to OpenAI
                fallback_model = embed_config.get("fallback")
                logger.info(f"Falling back to OpenAI embedding model: {fallback_model}")
                return fallback_model, ModelProvider.OPENAI
            
            # If no fallback or already OpenAI, use default
            logger.warning(f"Using default embedding model for provider {provider}")
            return ModelRegistry.DEFAULT_EMBEDDING_MODELS[provider], provider
        
        return model_name, provider
    
    def get_all_report_models(self) -> List[Tuple[str, ModelProvider]]:
        """
        Get all models configured for report generation based on report type.
        
        Returns:
            List of (model_name, provider) tuples
        """
        report_config = self.config.get("report_configuration", {})
        report_type = report_config.get("type", ReportType.STANDARD.value)
        
        if report_type == ReportType.MULTI_MODEL_COMPARISON.value and report_config.get("multi_model_enabled", False):
            # For multi-model reports, get all configured models
            models_to_compare = report_config.get("models_to_compare", [])
            result = []
            
            for model_name in models_to_compare:
                # Try to determine provider
                model_info = ModelRegistry.get_model_info(model_name)
                if model_info:
                    provider = model_info.provider
                else:
                    # Infer provider from name
                    if "gpt" in model_name.lower():
                        provider = ModelProvider.OPENAI
                    elif any(name in model_name.lower() for name in ["llama", "mistral", "gemma"]):
                        provider = ModelProvider.OLLAMA
                    else:
                        provider = ModelProvider.HUGGINGFACE
                
                # Check if model is available
                if self.is_model_available(model_name, provider):
                    result.append((model_name, provider))
                else:
                    logger.warning(f"Model {model_name} not available for multi-model report")
            
            return result
        else:
            # For standard reports, just get the configured model for report generation
            return [self.get_model_for_task(AnalysisTask.REPORT_GENERATION)]
    
    def get_report_type(self) -> ReportType:
        """Get the configured report type."""
        report_config = self.config.get("report_configuration", {})
        report_type_str = report_config.get("type", ReportType.STANDARD.value)
        return ReportType(report_type_str)
    
    def set_report_type(self, report_type: Union[str, ReportType]) -> None:
        """
        Set the report type.
        
        Args:
            report_type: Type of report to generate
        """
        report_type_value = report_type.value if isinstance(report_type, ReportType) else report_type
        
        if "report_configuration" not in self.config:
            self.config["report_configuration"] = {}
        
        self.config["report_configuration"]["type"] = report_type_value
        self.save_config()
    
    def enable_multi_model_reports(self, enabled: bool = True, 
                                 models: Optional[List[str]] = None) -> None:
        """
        Enable or disable multi-model reports.
        
        Args:
            enabled: Whether multi-model reports are enabled
            models: List of models to compare in multi-model reports
        """
        if "report_configuration" not in self.config:
            self.config["report_configuration"] = {}
        
        self.config["report_configuration"]["multi_model_enabled"] = enabled
        
        if models is not None:
            self.config["report_configuration"]["models_to_compare"] = models
        
        self.save_config()
    
    def set_model_for_task(self, task: Union[str, AnalysisTask], 
                          model_name: str, 
                          provider: Union[str, ModelProvider]) -> None:
        """
        Configure which model to use for a specific task.
        
        Args:
            task: Task requiring a model
            model_name: Name of the model to use
            provider: Provider of the model
        """
        task_value = task.value if isinstance(task, AnalysisTask) else task
        provider_value = provider.value if isinstance(provider, ModelProvider) else provider
        
        if "models" not in self.config:
            self.config["models"] = {}
        
        if task_value not in self.config["models"]:
            self.config["models"][task_value] = {}
        
        self.config["models"][task_value]["default"] = model_name
        self.config["models"][task_value]["provider"] = provider_value
        
        self.save_config()
    
    def set_embedding_model(self, model_name: str, 
                          provider: Union[str, ModelProvider]) -> None:
        """
        Configure which embedding model to use.
        
        Args:
            model_name: Name of the model to use
            provider: Provider of the model
        """
        provider_value = provider.value if isinstance(provider, ModelProvider) else provider
        
        if "models" not in self.config:
            self.config["models"] = {}
        
        if "embedding" not in self.config["models"]:
            self.config["models"]["embedding"] = {}
        
        self.config["models"]["embedding"]["default"] = model_name
        self.config["models"]["embedding"]["provider"] = provider_value
        
        self.save_config()
    
    def set_prefer_local(self, prefer_local: bool) -> None:
        """
        Set whether to prefer local models.
        
        Args:
            prefer_local: Whether to prefer local models
        """
        self.prefer_local = prefer_local
        self.config["prefer_local"] = prefer_local
        self.save_config()
    
    def list_available_models(self, provider: Optional[ModelProvider] = None,
                            model_type: Optional[ModelType] = None) -> List[ModelInfo]:
        """
        List models available on the system.
        
        Args:
            provider: Filter by provider
            model_type: Filter by model type
            
        Returns:
            List of available model info objects
        """
        result = []
        
        for provider_enum, model_names in self.available_models.items():
            if provider is not None and provider_enum != provider:
                continue
                
            for model_name in model_names:
                # Try to get model info from registry
                model_info = ModelRegistry.get_model_info(model_name)
                
                # If not in registry, create basic info
                if model_info is None:
                    if model_type is not None:
                        # Skip if we're filtering by type and can't determine it
                        continue
                        
                    # Create basic model info
                    model_info = ModelInfo(
                        name=model_name,
                        provider=provider_enum,
                        model_type=ModelType.CHAT,  # Assume chat model
                        description=f"{model_name} (detected)",
                        context_length=0,
                        local=(provider_enum != ModelProvider.OPENAI)
                    )
                    
                elif model_type is not None and model_info.model_type != model_type:
                    # Skip if filtered by type and doesn't match
                    continue
                
                result.append(model_info)
                
        return result