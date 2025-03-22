"""
Factory functions for creating dependencies.

This module provides factory functions for creating and configuring the dependencies
used in the JFKReveal application. These factories help with dependency injection
and make the dependencies more explicit.
"""
import os
import logging
from typing import Optional, Dict, Any, Type, TypeVar, List

from .interfaces import (
    ITextCleaner, 
    IVectorStore, 
    IDocumentProcessor, 
    IDocumentScraper, 
    IDocumentAnalyzer, 
    IFindingsReport,
    ILLMProvider,
    ICredentialProvider
)
from .database.text_cleaner import TextCleaner
from .database.vector_store import VectorStore
from .database.document_processor import DocumentProcessor
from .scrapers.archives_gov import ArchivesGovScraper
from .analysis.document_analyzer import DocumentAnalyzer
from .summarization.findings_report import FindingsReport

logger = logging.getLogger(__name__)

T = TypeVar('T')

"""Legacy DefaultCredentialProvider has been replaced by the CredentialManager in utils.credentials"""


class LangChainOpenAIProvider:
    """LLM provider that uses LangChain with OpenAI."""
    
    def __init__(
        self,
        model_name: str = "gpt-4o",
        temperature: float = 0.0,
        max_retries: int = 5,
        api_key: Optional[str] = None,
        credential_provider: Optional[ICredentialProvider] = None,
        api_type: str = "openai",  # "openai", "azure", or "anthropic"
        fallback_model: Optional[str] = None
    ):
        """
        Initialize the LLM provider.
        
        Args:
            model_name: The name of the model to use
            temperature: The temperature for LLM generation
            max_retries: The maximum number of retries for API calls
            api_key: The API key
            credential_provider: The credential provider to use for getting the API key
            api_type: The type of API to use (openai, azure, or anthropic)
            fallback_model: Fallback model to use if the primary model is unavailable
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_retries = max_retries
        self.api_type = api_type
        self.fallback_model = fallback_model
        self.credential_provider = credential_provider
        self.api_key = api_key
        
        # Import what we need based on API type
        if api_type == "openai":
            from langchain_openai import ChatOpenAI
            self.llm_class = ChatOpenAI
        elif api_type == "azure":
            from langchain_openai import AzureChatOpenAI
            self.llm_class = AzureChatOpenAI
        elif api_type == "anthropic":
            from langchain_anthropic import ChatAnthropic
            self.llm_class = ChatAnthropic
        else:
            raise ValueError(f"Unsupported API type: {api_type}")
        
        # Initialize LLM
        self.llm = self._initialize_llm(model_name)
    
    def _initialize_llm(self, model_name: str):
        """Initialize the LLM with the appropriate credentials."""
        # Determine which credentials to use based on API type
        if self.api_key:
            # Use provided API key directly
            api_key = self.api_key
        elif self.credential_provider:
            if self.api_type == "openai":
                # Try to get OpenAI API key with fallbacks
                api_key = self.credential_provider.get_with_fallback(
                    ["OPENAI_API_KEY", "OPENAI_KEY", "OPENAI_SECRET"],
                    validate=True,
                    raise_on_missing=False
                )
            elif self.api_type == "azure":
                # Get Azure OpenAI key
                api_key = self.credential_provider.get_credential("AZURE_OPENAI_API_KEY")
            elif self.api_type == "anthropic":
                # Get Anthropic key
                api_key = self.credential_provider.get_credential("ANTHROPIC_API_KEY")
            else:
                api_key = None
        else:
            # No credential provider, try environment
            if self.api_type == "openai":
                api_key = os.environ.get("OPENAI_API_KEY")
            elif self.api_type == "azure":
                api_key = os.environ.get("AZURE_OPENAI_API_KEY")
            elif self.api_type == "anthropic":
                api_key = os.environ.get("ANTHROPIC_API_KEY")
            else:
                api_key = None
        
        # If no API key found, log error and return a dummy LLM that will raise an error
        if not api_key:
            logger.error(f"No API key found for {self.api_type}")
            # For OpenAI, we can initialize with an empty key but it will fail on usage
            api_key = ""
        
        # Initialize the appropriate LLM
        if self.api_type == "openai":
            return self.llm_class(
                model=model_name,
                temperature=self.temperature,
                api_key=api_key,
                max_retries=self.max_retries,
            )
        elif self.api_type == "azure":
            # For Azure, we need additional configuration
            azure_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
            if not azure_endpoint and self.credential_provider:
                azure_endpoint = self.credential_provider.get_credential("AZURE_OPENAI_ENDPOINT")
            
            deployment_name = os.environ.get("AZURE_OPENAI_DEPLOYMENT", model_name)
            if not deployment_name and self.credential_provider:
                deployment_name = self.credential_provider.get_credential("AZURE_OPENAI_DEPLOYMENT")
            
            api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2023-05-15")
            
            return self.llm_class(
                deployment_name=deployment_name or model_name,
                model=model_name,
                azure_endpoint=azure_endpoint,
                api_version=api_version,
                api_key=api_key,
                temperature=self.temperature,
                max_retries=self.max_retries,
            )
        elif self.api_type == "anthropic":
            return self.llm_class(
                model=model_name,
                temperature=self.temperature,
                api_key=api_key,
                max_retries=self.max_retries,
            )
    
    def _handle_api_error(self, error):
        """Handle API errors with fallbacks."""
        logger.error(f"API error: {error}")
        
        # If we have a fallback model, try using it
        if self.fallback_model and self.fallback_model != self.model_name:
            logger.info(f"Trying fallback model: {self.fallback_model}")
            try:
                # Reinitialize with fallback model
                return self._initialize_llm(self.fallback_model)
            except Exception as e:
                logger.error(f"Fallback model failed: {e}")
        
        # If we have a credential provider, check for other credentials
        if self.credential_provider and hasattr(self.credential_provider, "get_all_providers"):
            logger.info("Trying alternative credentials")
            try:
                # Try to find alternative credentials
                if self.api_type == "openai":
                    creds = self.credential_provider.get_all_providers("OPENAI_API_KEY")
                    if len(creds) > 1:
                        # We have multiple credentials, mark the current one as rate limited
                        if hasattr(self.credential_provider, "mark_credential_as_rate_limited"):
                            self.credential_provider.mark_credential_as_rate_limited("OPENAI_API_KEY", 60)
                        # Reinitialize with new credentials
                        return self._initialize_llm(self.model_name)
            except Exception as e:
                logger.error(f"Alternative credentials failed: {e}")
        
        # If all fallbacks fail, re-raise the error
        raise error
    
    def generate_text(self, prompt: str) -> str:
        """
        Generate text using the LLM.
        
        Args:
            prompt: The prompt to send to the LLM
            
        Returns:
            Generated text
        """
        try:
            response = self.llm.invoke(prompt)
            if hasattr(response, "content"):
                return response.content
            return str(response)
        except Exception as e:
            # Try to handle the error with fallbacks
            try:
                self.llm = self._handle_api_error(e)
                # Retry with new LLM
                response = self.llm.invoke(prompt)
                if hasattr(response, "content"):
                    return response.content
                return str(response)
            except Exception as final_e:
                logger.error(f"All fallbacks failed: {final_e}")
                raise
    
    def generate_structured_output(self, prompt: str, output_class: Type[T]) -> T:
        """
        Generate structured output using the LLM.
        
        Args:
            prompt: The prompt to send to the LLM
            output_class: The class to use for structured output
            
        Returns:
            Generated structured output
        """
        try:
            llm_with_structured_output = self.llm.with_structured_output(
                output_class,
                method="function_calling"
            )
            
            return llm_with_structured_output.invoke(prompt)
        except Exception as e:
            # Try to handle the error with fallbacks
            try:
                self.llm = self._handle_api_error(e)
                # Retry with new LLM
                llm_with_structured_output = self.llm.with_structured_output(
                    output_class,
                    method="function_calling"
                )
                return llm_with_structured_output.invoke(prompt)
            except Exception as final_e:
                logger.error(f"All fallbacks failed: {final_e}")
                raise


def create_credential_provider(
    config_file: Optional[str] = None, 
    add_rotating: bool = True,
    required_credentials: Optional[List[str]] = None
) -> ICredentialProvider:
    """
    Create a credential provider instance with advanced features.
    
    Args:
        config_file: Optional path to a credentials config file
        add_rotating: Whether to add a rotating credential source
        required_credentials: List of credentials that must be present
        
    Returns:
        A credential provider instance
    """
    from .utils.credentials import CredentialManager
    
    # Create credential manager
    manager = CredentialManager()
    
    # Add file source if provided
    if config_file:
        file_path = os.path.expanduser(config_file)
        manager.add_file_source(file_path)
    
    # Add rotating source if requested
    if add_rotating:
        manager.add_rotating_source(
            prefix="",
            priority=50,
            cooldown=60
        )
    
    # Validate required credentials if provided
    if required_credentials:
        logger.info(f"Validating required credentials: {', '.join(required_credentials)}")
        for cred in required_credentials:
            value = manager.get_credential(cred)
            if value is None:
                logger.warning(f"Required credential {cred} is missing")
    
    return manager


def create_llm_provider(
    model_name: str = "gpt-4o",
    temperature: float = 0.0,
    max_retries: int = 5,
    api_key: Optional[str] = None,
    credential_provider: Optional[ICredentialProvider] = None,
    api_type: str = "openai",
    fallback_model: Optional[str] = None
) -> ILLMProvider:
    """
    Create an LLM provider instance.
    
    Args:
        model_name: The name of the model to use
        temperature: The temperature for LLM generation
        max_retries: The maximum number of retries for API calls
        api_key: The API key
        credential_provider: The credential provider to use for getting the API key
        api_type: The type of API to use (openai, azure, or anthropic)
        fallback_model: Fallback model to use if the primary model is unavailable
        
    Returns:
        An LLM provider instance
    """
    # Set default fallback models based on API type if not provided
    if fallback_model is None:
        if api_type == "openai":
            # If using a high-end model, fall back to lower tier
            if model_name in ["gpt-4o", "gpt-4", "gpt-4-turbo"]:
                fallback_model = "gpt-3.5-turbo"
            # Otherwise, no fallback
        elif api_type == "anthropic":
            # If using Claude 3 Opus, fall back to Haiku
            if "opus" in model_name:
                fallback_model = "claude-3-haiku-20240307"
    
    return LangChainOpenAIProvider(
        model_name=model_name,
        temperature=temperature,
        max_retries=max_retries,
        api_key=api_key,
        credential_provider=credential_provider,
        api_type=api_type,
        fallback_model=fallback_model
    )


def create_text_cleaner() -> ITextCleaner:
    """Create a text cleaner instance."""
    return TextCleaner()


def create_vector_store(
    persist_directory: str = "data/vectordb",
    embedding_model: Optional[str] = None,
    embedding_provider: Optional[str] = None,
    openai_api_key: Optional[str] = None,
    ollama_base_url: Optional[str] = None,
    credential_provider: Optional[ICredentialProvider] = None
) -> IVectorStore:
    """
    Create a vector store instance.
    
    Args:
        persist_directory: Directory to persist the vector database
        embedding_model: Name of the embedding model to use
        embedding_provider: Provider to use for embeddings ('ollama' or 'openai')
        openai_api_key: OpenAI API key
        ollama_base_url: Base URL for Ollama API
        credential_provider: The credential provider to use for getting the API key
        
    Returns:
        A vector store instance
    """
    # Get API key from credential provider if not provided
    if openai_api_key is None and credential_provider is not None:
        openai_api_key = credential_provider.get_credential("OPENAI_API_KEY")
    
    # Get embedding model from environment if not provided
    if embedding_model is None and embedding_provider == "openai":
        embedding_model = os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    elif embedding_model is None and embedding_provider == "ollama":
        embedding_model = os.environ.get("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
    
    # Get Ollama base URL from environment if not provided
    if ollama_base_url is None:
        ollama_base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    
    return VectorStore(
        persist_directory=persist_directory,
        embedding_model=embedding_model,
        embedding_provider=embedding_provider,
        openai_api_key=openai_api_key,
        ollama_base_url=ollama_base_url
    )


def create_document_processor(
    input_dir: str = "data/raw",
    output_dir: str = "data/processed",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    max_workers: int = 20,
    skip_existing: bool = True,
    vector_store: Optional[IVectorStore] = None,
    clean_text: bool = True,
    use_ocr: bool = True,
    ocr_resolution_scale: float = 2.0,
    ocr_language: str = "eng",
    text_cleaner: Optional[ITextCleaner] = None
) -> IDocumentProcessor:
    """
    Create a document processor instance.
    
    Args:
        input_dir: Directory containing PDF files
        output_dir: Directory to save processed chunks
        chunk_size: Target size of text chunks
        chunk_overlap: Overlap between chunks
        max_workers: Maximum number of worker processes
        skip_existing: Whether to skip documents that were already processed
        vector_store: Optional vector store for immediate embedding after processing
        clean_text: Whether to clean text before chunking
        use_ocr: Whether to use OCR for scanned pages without text
        ocr_resolution_scale: Scale factor for OCR resolution
        ocr_language: Language for OCR processing
        text_cleaner: Optional text cleaner instance to use
        
    Returns:
        A document processor instance
    """
    return DocumentProcessor(
        input_dir=input_dir,
        output_dir=output_dir,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        max_workers=max_workers,
        skip_existing=skip_existing,
        vector_store=vector_store,
        clean_text=clean_text,
        use_ocr=use_ocr,
        ocr_resolution_scale=ocr_resolution_scale,
        ocr_language=ocr_language
    )


def create_document_scraper(
    output_dir: str = "data/raw",
    max_documents: Optional[int] = None,
    num_workers: int = 10,
    document_types: Optional[list] = None
) -> IDocumentScraper:
    """
    Create a document scraper instance.
    
    Args:
        output_dir: Directory to save downloaded documents
        max_documents: Maximum number of documents to download (None for all)
        num_workers: Number of parallel download workers
        document_types: List of document types to download (None for all)
        
    Returns:
        A document scraper instance
    """
    # Create a config for the scraper
    from jfkreveal.scrapers.archives_gov import ScraperConfig
    config = ScraperConfig(
        delay=1.0,
        max_retries=5,
        backoff_factor=0.5,
        jitter=0.25,
        timeout=30
    )
    
    # ArchivesGovScraper only accepts output_dir and config parameters
    scraper = ArchivesGovScraper(
        output_dir=output_dir,
        config=config
    )
    
    # Store additional parameters as instance variables
    # that might be used by the implementation
    scraper.max_documents = max_documents
    scraper.num_workers = num_workers
    scraper.document_types = document_types
    
    return scraper


def create_document_analyzer(
    vector_store: IVectorStore,
    output_dir: str = "data/analysis",
    model_name: str = "gpt-4o",
    openai_api_key: Optional[str] = None,
    temperature: float = 0.0,
    max_retries: int = 5,
    llm_provider: Optional[ILLMProvider] = None,
    credential_provider: Optional[ICredentialProvider] = None
) -> IDocumentAnalyzer:
    """
    Create a document analyzer instance.
    
    Args:
        vector_store: Vector store instance for searching documents
        output_dir: Directory to save analysis results
        model_name: OpenAI model to use for analysis
        openai_api_key: OpenAI API key
        temperature: Temperature for LLM generation
        max_retries: Maximum number of retries for API calls
        llm_provider: Optional LLM provider to use
        credential_provider: Optional credential provider to use
        
    Returns:
        A document analyzer instance
    """
    # Get API key from credential provider if not provided
    if openai_api_key is None and credential_provider is not None:
        openai_api_key = credential_provider.get_credential("OPENAI_API_KEY")
    
    return DocumentAnalyzer(
        vector_store=vector_store,
        output_dir=output_dir,
        model_name=model_name,
        openai_api_key=openai_api_key,
        temperature=temperature,
        max_retries=max_retries
    )


def create_findings_report(
    analysis_dir: str = "data/analysis",
    output_dir: str = "data/reports",
    raw_docs_dir: str = "data/raw",
    model_name: str = "gpt-4o",
    openai_api_key: Optional[str] = None,
    temperature: float = 0.1,
    max_retries: int = 5,
    pdf_base_url: str = "https://www.archives.gov/files/research/jfk/releases/2025/0318/",
    llm_provider: Optional[ILLMProvider] = None,
    credential_provider: Optional[ICredentialProvider] = None
) -> IFindingsReport:
    """
    Create a findings report instance.
    
    Args:
        analysis_dir: Directory containing analysis files
        output_dir: Directory to save reports
        raw_docs_dir: Directory containing raw PDF documents
        model_name: OpenAI model to use
        openai_api_key: OpenAI API key
        temperature: Temperature for LLM generation
        max_retries: Maximum number of retries for API calls
        pdf_base_url: Base URL for PDF documents for generating links
        llm_provider: Optional LLM provider to use
        credential_provider: Optional credential provider to use
        
    Returns:
        A findings report instance
    """
    # Get API key from credential provider if not provided
    if openai_api_key is None and credential_provider is not None:
        openai_api_key = credential_provider.get_credential("OPENAI_API_KEY")
    
    return FindingsReport(
        analysis_dir=analysis_dir,
        output_dir=output_dir,
        raw_docs_dir=raw_docs_dir,
        model_name=model_name,
        openai_api_key=openai_api_key,
        temperature=temperature,
        max_retries=max_retries,
        pdf_base_url=pdf_base_url
    )