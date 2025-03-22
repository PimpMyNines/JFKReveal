"""
Model registry for managing different AI models for various tasks.

Supports multiple providers including:
- OpenAI models
- Local Ollama models
- HuggingFace models
"""
import os
import logging
from typing import Dict, Any, Optional, List, Union, Tuple
from enum import Enum
from dataclasses import dataclass

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_community.embeddings import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)

class ModelProvider(str, Enum):
    """Supported model providers."""
    OPENAI = "openai"
    OLLAMA = "ollama" 
    HUGGINGFACE = "huggingface"
    
class ModelType(str, Enum):
    """Types of models supported."""
    EMBEDDING = "embedding"
    CHAT = "chat"
    COMPLETION = "completion"

@dataclass
class ModelInfo:
    """Information about an AI model."""
    name: str
    provider: ModelProvider
    model_type: ModelType
    description: str
    context_length: int
    size_mb: Optional[int] = None
    performance_rating: Optional[float] = None
    cost_per_1k_tokens: Optional[float] = None
    recommended_tasks: Optional[List[str]] = None
    local: bool = False

class ModelRegistry:
    """Registry for managing AI models from different providers."""
    
    # Default embedding models for each provider
    DEFAULT_EMBEDDING_MODELS = {
        ModelProvider.OPENAI: "text-embedding-3-large",
        ModelProvider.OLLAMA: "nomic-embed-text",
        ModelProvider.HUGGINGFACE: "sentence-transformers/all-mpnet-base-v2",
    }
    
    # Default chat models for each provider
    DEFAULT_CHAT_MODELS = {
        ModelProvider.OPENAI: "gpt-4o",
        ModelProvider.OLLAMA: "llama3",
        ModelProvider.HUGGINGFACE: "mistralai/Mistral-7B-Instruct-v0.2",
    }
    
    # Model registry with detailed information
    MODELS = {
        # OpenAI Models
        "gpt-4o": ModelInfo(
            name="gpt-4o",
            provider=ModelProvider.OPENAI,
            model_type=ModelType.CHAT,
            description="GPT-4o (omni) - Powerful multimodal model with 128K context",
            context_length=128000,
            performance_rating=9.5,
            cost_per_1k_tokens=0.01,
            recommended_tasks=["analysis", "report_generation", "summarization"],
            local=False
        ),
        "gpt-4o-mini": ModelInfo(
            name="gpt-4o-mini",
            provider=ModelProvider.OPENAI,
            model_type=ModelType.CHAT,
            description="GPT-4o Mini - Faster, more cost-effective omni model",
            context_length=128000,
            performance_rating=8.7,
            cost_per_1k_tokens=0.003,
            recommended_tasks=["analysis", "report_generation", "summarization"],
            local=False
        ),
        "gpt-3.5-turbo": ModelInfo(
            name="gpt-3.5-turbo",
            provider=ModelProvider.OPENAI,
            model_type=ModelType.CHAT,
            description="GPT-3.5 Turbo - Balanced speed and performance",
            context_length=16384,
            performance_rating=7.5,
            cost_per_1k_tokens=0.001,
            recommended_tasks=["analysis", "report_generation", "summarization"],
            local=False
        ),
        "text-embedding-3-large": ModelInfo(
            name="text-embedding-3-large",
            provider=ModelProvider.OPENAI,
            model_type=ModelType.EMBEDDING,
            description="OpenAI's most capable embedding model (3072 dimensions)",
            context_length=8191,
            performance_rating=9.0,
            cost_per_1k_tokens=0.00013,
            recommended_tasks=["embedding", "vector_search"],
            local=False
        ),
        "text-embedding-3-small": ModelInfo(
            name="text-embedding-3-small",
            provider=ModelProvider.OPENAI,
            model_type=ModelType.EMBEDDING,
            description="Efficient embedding model with good performance (1536 dimensions)",
            context_length=8191,
            performance_rating=8.0,
            cost_per_1k_tokens=0.00002,
            recommended_tasks=["embedding", "vector_search"],
            local=False
        ),
        
        # Ollama Models
        "llama3": ModelInfo(
            name="llama3",
            provider=ModelProvider.OLLAMA,
            model_type=ModelType.CHAT,
            description="Meta's Llama 3 (8B) - Powerful open-source model",
            context_length=8192,
            size_mb=4700,
            performance_rating=8.0,
            recommended_tasks=["analysis", "report_generation", "summarization"],
            local=True
        ),
        "mistral": ModelInfo(
            name="mistral",
            provider=ModelProvider.OLLAMA,
            model_type=ModelType.CHAT,
            description="Mistral 7B - High quality open-source 7B parameter model",
            context_length=8192,
            size_mb=4100,
            performance_rating=7.5,
            recommended_tasks=["analysis", "report_generation", "summarization"],
            local=True
        ),
        "gemma": ModelInfo(
            name="gemma",
            provider=ModelProvider.OLLAMA,
            model_type=ModelType.CHAT,
            description="Google's Gemma (7B) - Lightweight yet capable model",
            context_length=8192,
            size_mb=4100,
            performance_rating=7.2,
            recommended_tasks=["analysis", "summarization"],
            local=True
        ),
        "nomic-embed-text": ModelInfo(
            name="nomic-embed-text",
            provider=ModelProvider.OLLAMA,
            model_type=ModelType.EMBEDDING,
            description="High-performing open embedding model (8K context, 137M parameters)",
            context_length=8192,
            size_mb=274,
            performance_rating=7.8,
            recommended_tasks=["embedding", "vector_search"],
            local=True
        ),
        "mxbai-embed-large": ModelInfo(
            name="mxbai-embed-large",
            provider=ModelProvider.OLLAMA,
            model_type=ModelType.EMBEDDING,
            description="State-of-the-art large embedding model (334M parameters)",
            context_length=8192,
            size_mb=680,
            performance_rating=8.2,
            recommended_tasks=["embedding", "vector_search"],
            local=True
        ),
        "all-minilm": ModelInfo(
            name="all-minilm",
            provider=ModelProvider.OLLAMA,
            model_type=ModelType.EMBEDDING,
            description="Lightweight embedding model (22M parameters)",
            context_length=8192,
            size_mb=50,
            performance_rating=6.5,
            recommended_tasks=["embedding", "vector_search"],
            local=True
        ),
        
        # HuggingFace Models
        "sentence-transformers/all-mpnet-base-v2": ModelInfo(
            name="sentence-transformers/all-mpnet-base-v2",
            provider=ModelProvider.HUGGINGFACE,
            model_type=ModelType.EMBEDDING,
            description="Versatile sentence embedding model with strong performance",
            context_length=384,
            size_mb=420,
            performance_rating=8.0,
            recommended_tasks=["embedding", "vector_search"],
            local=True
        ),
        "mistralai/Mistral-7B-Instruct-v0.2": ModelInfo(
            name="mistralai/Mistral-7B-Instruct-v0.2",
            provider=ModelProvider.HUGGINGFACE,
            model_type=ModelType.CHAT,
            description="Instruction-tuned Mistral 7B model",
            context_length=8192,
            size_mb=4100,
            performance_rating=7.5,
            recommended_tasks=["analysis", "report_generation", "summarization"],
            local=True
        ),
    }
    
    @classmethod
    def get_model_info(cls, model_name: str) -> Optional[ModelInfo]:
        """Get information about a specific model."""
        return cls.MODELS.get(model_name)
    
    @classmethod
    def list_models(cls, provider: Optional[ModelProvider] = None, 
                   model_type: Optional[ModelType] = None,
                   local_only: bool = False) -> List[ModelInfo]:
        """
        List available models filtered by provider and/or type.
        
        Args:
            provider: Filter by provider (OpenAI, Ollama, etc.)
            model_type: Filter by model type (embedding, chat, etc.)
            local_only: Only show models that can run locally
            
        Returns:
            List of model info objects
        """
        filtered_models = []
        
        for model_info in cls.MODELS.values():
            # Apply filters
            if provider is not None and model_info.provider != provider:
                continue
            if model_type is not None and model_info.model_type != model_type:
                continue
            if local_only and not model_info.local:
                continue
                
            filtered_models.append(model_info)
            
        return filtered_models
    
    @classmethod
    def get_embedding_model(cls, 
                           model_name: Optional[str] = None, 
                           provider: Optional[ModelProvider] = None,
                           prefer_local: bool = False,
                           **kwargs) -> Any:
        """
        Get an embedding model instance.
        
        Args:
            model_name: Name of the model to use
            provider: Provider to use for embeddings
            prefer_local: Prefer local models over cloud models
            **kwargs: Additional kwargs to pass to the model constructor
            
        Returns:
            Embedding model instance
        """
        # If both model_name and provider are None, use environment variables or defaults
        if model_name is None and provider is None:
            # Check for environment variables
            env_provider = os.environ.get("EMBEDDING_PROVIDER", "").lower()
            if env_provider == "openai":
                provider = ModelProvider.OPENAI
                model_name = os.environ.get("OPENAI_EMBEDDING_MODEL", cls.DEFAULT_EMBEDDING_MODELS[provider])
            elif env_provider == "ollama":
                provider = ModelProvider.OLLAMA
                model_name = os.environ.get("OLLAMA_EMBEDDING_MODEL", cls.DEFAULT_EMBEDDING_MODELS[provider])
            elif env_provider == "huggingface":
                provider = ModelProvider.HUGGINGFACE
                model_name = os.environ.get("HF_EMBEDDING_MODEL", cls.DEFAULT_EMBEDDING_MODELS[provider])
            else:
                # Default to Ollama if prefer_local is True, otherwise OpenAI
                provider = ModelProvider.OLLAMA if prefer_local else ModelProvider.OPENAI
                model_name = cls.DEFAULT_EMBEDDING_MODELS[provider]
        
        # If only provider is specified, use default model for that provider
        elif model_name is None and provider is not None:
            model_name = cls.DEFAULT_EMBEDDING_MODELS[provider]
            
        # If only model name is specified, infer the provider
        elif model_name is not None and provider is None:
            model_info = cls.get_model_info(model_name)
            if model_info is not None:
                provider = model_info.provider
            else:
                # Try to infer provider from model name
                if "gpt" in model_name.lower() or "text-embedding" in model_name.lower():
                    provider = ModelProvider.OPENAI
                elif any(name in model_name.lower() for name in ["llama", "mistral", "gemma", "nomic", "mxbai"]):
                    provider = ModelProvider.OLLAMA
                else:
                    provider = ModelProvider.HUGGINGFACE
        
        # Create the embedding model based on provider
        if provider == ModelProvider.OPENAI:
            api_key = kwargs.pop("api_key", os.environ.get("OPENAI_API_KEY"))
            return OpenAIEmbeddings(
                model=model_name, 
                openai_api_key=api_key,
                **kwargs
            )
        elif provider == ModelProvider.OLLAMA:
            base_url = kwargs.pop("base_url", os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"))
            return OllamaEmbeddings(
                model=model_name,
                base_url=base_url,
                **kwargs
            )
        elif provider == ModelProvider.HUGGINGFACE:
            return HuggingFaceEmbeddings(
                model_name=model_name,
                **kwargs
            )
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    @classmethod
    def get_chat_model(cls,
                      model_name: Optional[str] = None,
                      provider: Optional[ModelProvider] = None,
                      prefer_local: bool = False,
                      **kwargs) -> Any:
        """
        Get a chat model instance.
        
        Args:
            model_name: Name of the model to use
            provider: Provider to use for chat
            prefer_local: Prefer local models over cloud models
            **kwargs: Additional kwargs to pass to the model constructor
            
        Returns:
            Chat model instance
        """
        # Determine provider and model name using similar logic to get_embedding_model
        if model_name is None and provider is None:
            env_provider = os.environ.get("CHAT_PROVIDER", "").lower()
            if env_provider == "openai":
                provider = ModelProvider.OPENAI
                model_name = os.environ.get("OPENAI_CHAT_MODEL", cls.DEFAULT_CHAT_MODELS[provider])
            elif env_provider == "ollama":
                provider = ModelProvider.OLLAMA
                model_name = os.environ.get("OLLAMA_CHAT_MODEL", cls.DEFAULT_CHAT_MODELS[provider])
            elif env_provider == "huggingface":
                provider = ModelProvider.HUGGINGFACE
                model_name = os.environ.get("HF_CHAT_MODEL", cls.DEFAULT_CHAT_MODELS[provider])
            else:
                provider = ModelProvider.OLLAMA if prefer_local else ModelProvider.OPENAI
                model_name = cls.DEFAULT_CHAT_MODELS[provider]
        
        elif model_name is None and provider is not None:
            model_name = cls.DEFAULT_CHAT_MODELS[provider]
            
        elif model_name is not None and provider is None:
            model_info = cls.get_model_info(model_name)
            if model_info is not None:
                provider = model_info.provider
            else:
                if "gpt" in model_name.lower():
                    provider = ModelProvider.OPENAI
                elif any(name in model_name.lower() for name in ["llama", "mistral", "gemma"]):
                    provider = ModelProvider.OLLAMA
                else:
                    provider = ModelProvider.HUGGINGFACE
        
        # Create the chat model based on provider
        if provider == ModelProvider.OPENAI:
            api_key = kwargs.pop("api_key", os.environ.get("OPENAI_API_KEY"))
            return ChatOpenAI(
                model=model_name, 
                openai_api_key=api_key,
                **kwargs
            )
        elif provider == ModelProvider.OLLAMA:
            base_url = kwargs.pop("base_url", os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"))
            return ChatOllama(
                model=model_name,
                base_url=base_url,
                **kwargs
            )
        elif provider == ModelProvider.HUGGINGFACE:
            # HuggingFacePipeline requires more setup, simplified for this example
            return HuggingFacePipeline.from_model_id(
                model_id=model_name,
                task="text-generation",
                **kwargs
            )
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    @classmethod
    def download_model(cls, model_name: str, provider: ModelProvider = ModelProvider.OLLAMA) -> bool:
        """
        Download a model for local use.
        Currently only supports Ollama models.
        
        Args:
            model_name: Name of the model to download
            provider: Provider of the model
            
        Returns:
            True if successful, False otherwise
        """
        if provider != ModelProvider.OLLAMA:
            logger.warning(f"Model download only supported for Ollama provider, not {provider}")
            return False
        
        try:
            import subprocess
            result = subprocess.run(["ollama", "pull", model_name], capture_output=True, text=True, check=False)
            if result.returncode == 0:
                logger.info(f"Successfully downloaded model: {model_name}")
                return True
            else:
                logger.error(f"Failed to download model {model_name}: {result.stderr}")
                return False
        except Exception as e:
            logger.error(f"Error downloading model {model_name}: {e}")
            return False