"""
JFKReveal - Main entry point for the JFK documents analysis pipeline.
"""
import os
import sys
import logging
import argparse
import dotenv
import traceback
import contextlib
import json
import io
import csv
from contextlib import contextmanager
from typing import Optional, List, Dict, Any, Generator, Tuple, Union

# Load environment variables from .env file if it exists
dotenv.load_dotenv()

from .utils.logger import setup_logging, get_logger
from .utils.dependency_injection import DependencyContainer
from .utils.exceptions import (
    JFKRevealError, APIError, FileError, NetworkError, ProcessingError, 
    VectorDBError, AnalysisError, ReportingError, PipelineError,
    CircuitBreakerError, RateLimitError, CredentialError, ModelUnavailableError,
    PipelineExecutionError, DownloadError
)
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
from .factories import (
    create_credential_provider,
    create_llm_provider,
    create_text_cleaner,
    create_vector_store,
    create_document_processor,
    create_document_scraper,
    create_document_analyzer,
    create_findings_report
)

# Get the logger for this module
logger = get_logger("main")

# Maximum number of failures before triggering circuit breaker
MAX_FAILURES = 3
# Mapping of failure counts by component
_component_failures = {}


@contextmanager
def pipeline_step(step_name: str, component: str) -> Generator[None, None, None]:
    """
    Context manager for pipeline steps with error handling and circuit breaker pattern.
    
    Args:
        step_name: Name of the pipeline step
        component: Component handling the step
        
    Yields:
        None
        
    Raises:
        CircuitBreakerError: If component has failed too many times
        PipelineExecutionError: If step execution fails
    """
    # Check if circuit breaker is triggered for this component
    if _component_failures.get(component, 0) >= MAX_FAILURES:
        msg = f"Circuit breaker triggered for {component} ({_component_failures[component]} failures)"
        logger.error(msg)
        raise CircuitBreakerError(component=component, failure_count=_component_failures[component])
    
    logger.info(f"Starting pipeline step: {step_name} (component: {component})")
    try:
        yield
        logger.info(f"Completed pipeline step: {step_name}")
        # Reset failure count on success
        if component in _component_failures:
            _component_failures[component] = 0
    except JFKRevealError as e:
        # Increment failure count
        _component_failures[component] = _component_failures.get(component, 0) + 1
        logger.error(f"Error in pipeline step {step_name} (component: {component}): {str(e)}")
        # Re-raise specific exceptions with added context
        raise PipelineExecutionError(step=step_name, component=component) from e
    except Exception as e:
        # Increment failure count
        _component_failures[component] = _component_failures.get(component, 0) + 1
        logger.error(f"Unexpected error in pipeline step {step_name} (component: {component}): {str(e)}")
        logger.debug(f"Stack trace: {traceback.format_exc()}")
        # Wrap generic exceptions in our pipeline error
        raise PipelineExecutionError(step=step_name, component=component) from e
    

class JFKReveal:
    """Main class for JFK document analysis pipeline."""
    
    def __init__(
        self,
        base_dir: str = ".",
        openai_api_key: Optional[str] = None,
        clean_text: bool = True,
        use_ocr: bool = True,
        ocr_resolution_scale: float = 2.0,
        ocr_language: str = "eng",
        credential_provider: Optional[ICredentialProvider] = None,
        document_scraper: Optional[IDocumentScraper] = None,
        document_processor: Optional[IDocumentProcessor] = None,
        vector_store: Optional[IVectorStore] = None,
        document_analyzer: Optional[IDocumentAnalyzer] = None,
        findings_report: Optional[IFindingsReport] = None
    ):
        """
        Initialize the JFK document analysis pipeline.
        
        Args:
            base_dir: Base directory for data
            openai_api_key: OpenAI API key (uses environment variable if not provided)
            clean_text: Whether to clean OCR text before chunking and embedding
            use_ocr: Whether to apply OCR to scanned pages with no text
            ocr_resolution_scale: Scale factor for OCR resolution (higher = better quality)
            ocr_language: Language for OCR processing
            credential_provider: Optional credential provider to use
            document_scraper: Optional document scraper to use
            document_processor: Optional document processor to use
            vector_store: Optional vector store to use
            document_analyzer: Optional document analyzer to use
            findings_report: Optional findings report to use
        """
        self.base_dir = base_dir
        self.openai_api_key = openai_api_key
        self.clean_text = clean_text
        self.use_ocr = use_ocr
        self.ocr_resolution_scale = ocr_resolution_scale
        self.ocr_language = ocr_language
        
        # Create data directories
        os.makedirs(os.path.join(base_dir, "data/raw"), exist_ok=True)
        os.makedirs(os.path.join(base_dir, "data/processed"), exist_ok=True)
        os.makedirs(os.path.join(base_dir, "data/vectordb"), exist_ok=True)
        os.makedirs(os.path.join(base_dir, "data/analysis"), exist_ok=True)
        os.makedirs(os.path.join(base_dir, "data/reports"), exist_ok=True)
        
        # Initialize dependencies if not provided
        self.credential_provider = credential_provider or create_credential_provider()
        
        # If API key is provided, set it in the credential provider
        if openai_api_key:
            self.credential_provider.set_credential("OPENAI_API_KEY", openai_api_key)
        
        # Store dependencies
        self.document_scraper = document_scraper
        self.document_processor = document_processor
        self.vector_store = vector_store
        self.document_analyzer = document_analyzer
        self.findings_report = findings_report
    
    def scrape_documents(self) -> List[str]:
        """
        Scrape documents from the National Archives website.
        
        Returns:
            List of paths to downloaded files
            
        Raises:
            NetworkError: On network connectivity issues
            DownloadError: On download failures
            PipelineExecutionError: On pipeline execution failures
        """
        with pipeline_step("document_scraping", "document_scraper"):
            # Lazy-initialize document scraper if needed
            if self.document_scraper is None:
                self.document_scraper = create_document_scraper(
                    output_dir=os.path.join(self.base_dir, "data/raw")
                )
            
            try:
                # The scrape_all method returns both the list of file paths and document objects
                downloaded_files, documents = self.document_scraper.scrape_all()
                
                # Log results with more detail
                if downloaded_files:
                    logger.info(f"Completed document scraping: downloaded {len(downloaded_files)} files")
                    for i, file_path in enumerate(downloaded_files[:5]):  # Log first 5 files
                        logger.debug(f"Downloaded file {i+1}: {os.path.basename(file_path)}")
                    if len(downloaded_files) > 5:
                        logger.debug(f"... and {len(downloaded_files) - 5} more files")
                else:
                    logger.warning("Document scraping completed but no files were downloaded")
                
                # Return just the file paths to maintain compatibility with existing code
                return downloaded_files
            except Exception as e:
                # Wrap exceptions in appropriate error types
                if "connection" in str(e).lower() or "timeout" in str(e).lower():
                    raise NetworkError(f"Network error during document scraping: {str(e)}") from e
                elif "download" in str(e).lower() or "404" in str(e):
                    raise DownloadError(f"Download failed during document scraping: {str(e)}") from e
                else:
                    raise  # Let the context manager handle other exceptions
    
    def process_documents(
        self, 
        max_workers: int = 20, 
        skip_existing: bool = True, 
        vector_store = None
    ) -> List[str]:
        """
        Process PDF documents and extract text.
        
        Args:
            max_workers: Number of documents to process in parallel (default 20)
            skip_existing: Whether to skip already processed documents (default True)
            vector_store: Optional vector store for immediate embedding
            
        Returns:
            List of paths to processed files
            
        Raises:
            ProcessingError: On document processing failures
            OCRError: On OCR processing failures 
            FileError: On file system issues
            PipelineExecutionError: On pipeline execution failures
        """
        with pipeline_step("document_processing", "document_processor"):
            # Lazy-initialize document processor if needed
            if self.document_processor is None:
                self.document_processor = create_document_processor(
                    input_dir=os.path.join(self.base_dir, "data/raw"),
                    output_dir=os.path.join(self.base_dir, "data/processed"),
                    max_workers=max_workers,
                    skip_existing=skip_existing,
                    vector_store=vector_store or self.vector_store,
                    clean_text=self.clean_text,
                    use_ocr=self.use_ocr,
                    ocr_resolution_scale=self.ocr_resolution_scale,
                    ocr_language=self.ocr_language
                )
            
            try:
                # Check if input directory has files to process
                pdf_files = []
                for root, _, files in os.walk(os.path.join(self.base_dir, "data/raw")):
                    for file in files:
                        if file.lower().endswith('.pdf'):
                            pdf_files.append(os.path.join(root, file))
                
                if not pdf_files:
                    logger.warning("No PDF files found in input directory")
                    return []
                
                logger.info(f"Found {len(pdf_files)} PDF files to process")
                
                # Process documents
                processed_files = self.document_processor.process_all_documents()
                
                # Log results with more detail
                success_rate = len(processed_files) / len(pdf_files) if pdf_files else 0
                logger.info(f"Completed document processing: processed {len(processed_files)}/{len(pdf_files)} files ({success_rate:.1%} success rate)")
                
                # Log detailed stats about OCR if available
                ocr_files = [f for f in processed_files if ".json" in f and os.path.exists(f)]
                if ocr_files and self.use_ocr:
                    try:
                        import json
                        ocr_count = 0
                        total_pages = 0
                        total_ocr_pages = 0
                        
                        # Sample up to 10 files for OCR stats
                        for file_path in ocr_files[:10]:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                data = json.load(f)
                                if data and isinstance(data, list) and data[0].get("metadata"):
                                    metadata = data[0]["metadata"]
                                    if metadata.get("ocr_applied"):
                                        ocr_count += 1
                                        total_pages += metadata.get("page_count", 0)
                                        total_ocr_pages += metadata.get("ocr_pages", 0)
                        
                        if ocr_count > 0:
                            logger.info(f"OCR applied to {ocr_count}/{min(10, len(ocr_files))} sampled files")
                            if total_pages > 0:
                                ocr_percentage = (total_ocr_pages / total_pages) * 100
                                logger.info(f"OCR applied to {total_ocr_pages}/{total_pages} pages ({ocr_percentage:.1f}%) in sampled files")
                    except Exception as e:
                        logger.debug(f"Error calculating OCR stats: {str(e)}")
                
                return processed_files
                
            except Exception as e:
                # Categorize and handle different error types
                error_msg = str(e).lower()
                if "ocr" in error_msg or "tesseract" in error_msg:
                    raise ProcessingError(f"OCR error during document processing: {str(e)}") from e
                elif any(x in error_msg for x in ["permission", "access", "read", "write"]):
                    raise FileError(f"File access error during document processing: {str(e)}") from e
                elif "disk" in error_msg or "space" in error_msg:
                    raise FileError(f"Disk space error during document processing: {str(e)}") from e
                else:
                    raise ProcessingError(f"Error during document processing: {str(e)}") from e
    
    def get_processed_documents(self):
        """
        Get a list of already processed documents without processing any new ones.
        
        Returns:
            List of paths to processed documents
        """
        logger.info("Getting already processed documents")
        
        # Lazy-initialize document processor if needed
        if self.document_processor is None:
            self.document_processor = create_document_processor(
                input_dir=os.path.join(self.base_dir, "data/raw"),
                output_dir=os.path.join(self.base_dir, "data/processed")
            )
        
        processed_files = self.document_processor.get_processed_documents()
        
        logger.info(f"Found {len(processed_files)} already processed documents")
        return processed_files
    
    def build_vector_database(self) -> Optional[IVectorStore]:
        """
        Build the vector database from processed documents.
        
        Returns:
            Vector store instance or None if initialization failed
            
        Raises:
            VectorDBError: On vector database initialization failures
            EmbeddingError: On embedding API failures
            APIError: On API-related errors
            CredentialError: On credential validation failures
            PipelineExecutionError: On pipeline execution failures
        """
        with pipeline_step("vector_database_build", "vector_store"):
            try:
                # Lazy-initialize vector store if needed
                if self.vector_store is None:
                    # Validate API credentials before attempting to create the vector store
                    embedding_model = os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")
                    
                    # Get API key and validate
                    api_key = self.credential_provider.get_credential("OPENAI_API_KEY")
                    if not api_key:
                        raise CredentialError("Missing OpenAI API key for vector embeddings")
                    
                    logger.info(f"Initializing vector store with embedding model: {embedding_model}")
                    
                    # Create vector store with enhanced error detection
                    try:
                        self.vector_store = create_vector_store(
                            persist_directory=os.path.join(self.base_dir, "data/vectordb"),
                            credential_provider=self.credential_provider
                        )
                    except Exception as e:
                        error_msg = str(e).lower()
                        if "openai" in error_msg and "api" in error_msg:
                            if "rate" in error_msg or "limit" in error_msg:
                                raise RateLimitError("OpenAI API rate limit exceeded during vector store initialization") from e
                            elif "key" in error_msg or "auth" in error_msg or "token" in error_msg:
                                raise CredentialError("Invalid OpenAI API credentials for vector embeddings") from e
                            else:
                                raise APIError(f"OpenAI API error during vector store initialization: {str(e)}") from e
                        else:
                            raise VectorDBError(f"Vector database initialization error: {str(e)}") from e
                
                # Verify that vector store is properly initialized
                if self.vector_store is None:
                    raise VectorDBError("Vector store initialization failed for unknown reason")
                
                # Log success
                logger.info(f"Vector database initialized successfully")
                return self.vector_store
                
            except JFKRevealError:
                # Let the context manager handle our custom exceptions
                raise
            except Exception as e:
                logger.error(f"Unexpected error building vector database: {str(e)}")
                logger.debug(f"Stack trace: {traceback.format_exc()}")
                raise VectorDBError(f"Vector database initialization failed: {str(e)}") from e
    
    def add_all_documents_to_vector_store(self, vector_store: IVectorStore):
        """Add all processed documents to the vector store."""
        logger.info("Adding all documents to vector store")
        
        total_chunks = vector_store.add_all_documents(
            processed_dir=os.path.join(self.base_dir, "data/processed")
        )
        
        logger.info(f"Completed vector database build, added {total_chunks} chunks")
        return total_chunks
    
    def analyze_documents(self, vector_store: IVectorStore):
        """
        Analyze documents and generate topic analyses.
        
        Args:
            vector_store: Vector store containing document embeddings
            
        Returns:
            List of topic analyses
            
        Raises:
            AnalysisError: On analysis failures
            LLMResponseError: On LLM response parsing errors
            APIError: On API-related errors including rate limits
            ModelUnavailableError: If the specified model is unavailable
            PipelineExecutionError: On pipeline execution failures
        """
        with pipeline_step("document_analysis", "document_analyzer"):
            try:
                # Get model name from environment variables if set
                model_name = os.environ.get("OPENAI_ANALYSIS_MODEL", "gpt-4o")  # Using gpt-4o as default
                fallback_model = os.environ.get("OPENAI_FALLBACK_MODEL", "gpt-3.5-turbo")
                
                logger.info(f"Starting document analysis using model: {model_name} (fallback: {fallback_model})")
                
                # Validate API credentials before attempting analysis
                api_key = self.credential_provider.get_credential("OPENAI_API_KEY")
                if not api_key:
                    raise CredentialError("Missing OpenAI API key for document analysis")
                
                # Lazy-initialize document analyzer if needed
                if self.document_analyzer is None:
                    self.document_analyzer = create_document_analyzer(
                        vector_store=vector_store,
                        output_dir=os.path.join(self.base_dir, "data/analysis"),
                        model_name=model_name,
                        credential_provider=self.credential_provider,
                        temperature=0.0,
                        max_retries=5
                    )
                
                # Check for existing analysis results
                analysis_dir = os.path.join(self.base_dir, "data/analysis")
                existing_analyses = [f for f in os.listdir(analysis_dir) if f.endswith('.json')] if os.path.exists(analysis_dir) else []
                
                if existing_analyses:
                    logger.info(f"Found {len(existing_analyses)} existing analysis files")
                
                # Track progress and success/failure counts
                total_topics = 0
                successful_topics = 0
                partial_topics = 0
                failed_topics = 0
                
                # Run the analysis with enhanced error handling
                try:
                    topic_analyses = self.document_analyzer.analyze_key_topics()
                    total_topics = len(topic_analyses)
                    
                    # Count successes, partials, and failures
                    for analysis in topic_analyses:
                        if analysis.error:
                            if len(analysis.document_analyses) > 0:
                                partial_topics += 1
                            else:
                                failed_topics += 1
                        else:
                            successful_topics += 1
                            
                except Exception as e:
                    error_msg = str(e).lower()
                    
                    # Handle different types of API errors
                    if "rate" in error_msg or "limit" in error_msg or "capacity" in error_msg:
                        raise RateLimitError(f"API rate limit exceeded during document analysis: {str(e)}") from e
                    elif "model" in error_msg and ("not found" in error_msg or "unavailable" in error_msg):
                        # Try with fallback model if specified
                        if fallback_model and fallback_model != model_name:
                            logger.warning(f"Model {model_name} unavailable, trying fallback model: {fallback_model}")
                            # Create a new analyzer with the fallback model
                            self.document_analyzer = create_document_analyzer(
                                vector_store=vector_store,
                                output_dir=os.path.join(self.base_dir, "data/analysis"),
                                model_name=fallback_model,
                                credential_provider=self.credential_provider,
                                temperature=0.0,
                                max_retries=5
                            )
                            # Retry with fallback model
                            topic_analyses = self.document_analyzer.analyze_key_topics()
                            total_topics = len(topic_analyses)
                        else:
                            raise ModelUnavailableError(model_name=model_name) from e
                    elif "context" in error_msg and "length" in error_msg:
                        raise AnalysisError(f"Context length exceeded during document analysis: {str(e)}") from e
                    elif "key" in error_msg or "auth" in error_msg or "token" in error_msg:
                        raise CredentialError(f"Invalid API credentials during document analysis: {str(e)}") from e
                    else:
                        raise AnalysisError(f"Error during document analysis: {str(e)}") from e
                
                # Log results with more detail
                if successful_topics > 0 or partial_topics > 0:
                    logger.info(f"Completed document analysis: {successful_topics} successful topics, {partial_topics} partial topics, {failed_topics} failed topics")
                    
                    # Log each topic with its status
                    for analysis in topic_analyses:
                        status = "SUCCESS" if not analysis.error else "PARTIAL" if len(analysis.document_analyses) > 0 else "FAILED"
                        logger.debug(f"Topic '{analysis.topic}': {status} ({len(analysis.document_analyses)} documents analyzed)")
                    
                    return topic_analyses
                else:
                    # If all topics failed, raise an error
                    raise AnalysisError(f"All {total_topics} topic analyses failed")
                    
            except JFKRevealError:
                # Let the context manager handle our custom exceptions
                raise
            except Exception as e:
                logger.error(f"Unexpected error during document analysis: {str(e)}")
                logger.debug(f"Stack trace: {traceback.format_exc()}")
                raise AnalysisError(f"Document analysis failed: {str(e)}") from e
    
    def generate_report(self):
        """
        Generate final report from analysis results.
        
        Returns:
            Report object with paths to generated report files
            
        Raises:
            ReportingError: On report generation failures
            TemplateRenderingError: On template rendering errors 
            APIError: On API-related errors
            CredentialError: On credential validation failures
            PipelineExecutionError: On pipeline execution failures
        """
        with pipeline_step("report_generation", "findings_report"):
            try:
                # Validate that analysis directory exists and has data
                analysis_dir = os.path.join(self.base_dir, "data/analysis")
                
                if not os.path.exists(analysis_dir):
                    raise ReportingError(f"Analysis directory not found: {analysis_dir}")
                
                # Check for analysis files
                analysis_files = [f for f in os.listdir(analysis_dir) if f.endswith('.json')]
                if not analysis_files:
                    raise ReportingError(f"No analysis files found in {analysis_dir}")
                
                logger.info(f"Found {len(analysis_files)} analysis files for report generation")
                
                # Validate API credentials if needed for report generation
                if os.environ.get("OPENAI_ANALYSIS_MODEL"):
                    api_key = self.credential_provider.get_credential("OPENAI_API_KEY")
                    if not api_key:
                        raise CredentialError("Missing OpenAI API key for report generation")
                
                # Lazy-initialize findings report if needed
                if self.findings_report is None:
                    self.findings_report = create_findings_report(
                        analysis_dir=analysis_dir,
                        output_dir=os.path.join(self.base_dir, "data/reports"),
                        raw_docs_dir=os.path.join(self.base_dir, "data/raw"),
                        credential_provider=self.credential_provider
                    )
                
                # Generate the report
                report = self.findings_report.generate_full_report()
                
                # Validate that report files were created
                report_dir = os.path.join(self.base_dir, "data/reports")
                report_files = [f for f in os.listdir(report_dir) if f.endswith('.html') or f.endswith('.md')]
                
                if not report_files:
                    raise ReportingError("No report files were generated")
                
                # Log generated files
                logger.info(f"Generated {len(report_files)} report files:")
                for file in report_files[:5]:  # Log first 5 files
                    logger.debug(f"Generated report file: {file}")
                if len(report_files) > 5:
                    logger.debug(f"... and {len(report_files) - 5} more report files")
                
                logger.info("Completed report generation successfully")
                return report
                
            except JFKRevealError:
                # Let the context manager handle our custom exceptions
                raise
            except Exception as e:
                error_msg = str(e).lower()
                
                # Categorize the error type
                if "template" in error_msg or "render" in error_msg:
                    raise TemplateRenderingError(f"Template rendering failed: {str(e)}") from e
                elif "openai" in error_msg or "api" in error_msg:
                    if "rate" in error_msg or "limit" in error_msg:
                        raise RateLimitError(f"API rate limit exceeded: {str(e)}") from e
                    elif "key" in error_msg or "auth" in error_msg:
                        raise CredentialError(f"Invalid API credentials: {str(e)}") from e
                    else:
                        raise APIError(f"API error: {str(e)}") from e
                else:
                    raise ReportingError(f"Report generation failed: {str(e)}") from e
    
    def run_pipeline(
        self, 
        skip_scraping=False, 
        skip_processing=False, 
        skip_analysis=False,
        skip_vectordb=False,
        skip_report=False,
        use_existing_processed=False,
        use_existing_analysis=False,
        max_workers=20,
        process_batch_size=50,
        fail_fast=False,
        retry_count=3,
        retry_delay=5,
        backoff_factor=2.0,
        cache_downloads=True
    ) -> str:
        """
        Run the complete document analysis pipeline with robust error handling.
        
        Args:
            skip_scraping: Skip document scraping
            skip_processing: Skip document processing
            skip_analysis: Skip document analysis
            skip_vectordb: Skip vector database creation
            skip_report: Skip report generation
            use_existing_processed: Use existing processed documents without processing new ones
            use_existing_analysis: Use existing analysis results without reanalyzing
            max_workers: Number of documents to process in parallel
            process_batch_size: Number of documents to process in a single batch
            fail_fast: If True, stop pipeline on first error; if False, try to continue
            retry_count: Number of retries for failed operations
            retry_delay: Delay between retries in seconds
            backoff_factor: Exponential backoff factor for retries
            cache_downloads: Cache downloaded documents
            
        Returns:
            Path to the generated report file
            
        Raises:
            PipelineExecutionError: On pipeline execution failures (only in fail_fast mode)
        """
        pipeline_errors = []
        logger.info("Starting JFK documents analysis pipeline")
        
        # Configure retry settings globally
        global MAX_FAILURES
        MAX_FAILURES = retry_count
        
        # Step 1: Scrape documents
        if not skip_scraping:
            try:
                # If document scraper was created already, update its settings
                if self.document_scraper is not None:
                    self.document_scraper.set_retry_config(
                        max_retries=retry_count,
                        retry_delay=retry_delay,
                        backoff_factor=backoff_factor
                    )
                    self.document_scraper.use_cache = cache_downloads
                
                self.scrape_documents()
            except Exception as e:
                error_msg = f"Document scraping failed: {str(e)}"
                logger.error(error_msg)
                pipeline_errors.append(error_msg)
                if fail_fast:
                    raise PipelineExecutionError(
                        step="document_scraping", 
                        component="document_scraper"
                    ) from e
                # Continue with other steps if not fail_fast
        else:
            logger.info("Skipping document scraping")
        
        # Initialize vector store early for immediate embedding
        vector_store = None
        if not skip_analysis and not skip_vectordb:
            try:
                vector_store = self.build_vector_database()
                if vector_store is None:
                    error_msg = "Failed to initialize vector store, reverting to default processing without embedding"
                    logger.warning(error_msg)
                    pipeline_errors.append(error_msg)
                    skip_analysis = True
            except Exception as e:
                error_msg = f"Vector database initialization failed: {str(e)}"
                logger.error(error_msg)
                pipeline_errors.append(error_msg)
                skip_analysis = True
                if fail_fast:
                    raise PipelineExecutionError(
                        step="vector_database_build", 
                        component="vector_store"
                    ) from e
                # Continue with other steps if not fail_fast
        else:
            if skip_vectordb:
                logger.info("Skipping vector database creation as requested")
                skip_analysis = True
        
        # Step 2: Process documents or use existing processed documents
        if not skip_processing:
            try:
                if use_existing_processed:
                    logger.info("Using existing processed documents")
                    processed_files = self.get_processed_documents()
                    
                    # If we have a vector store, ensure all existing documents are added
                    if vector_store is not None:
                        try:
                            logger.info("Adding existing processed documents to vector store")
                            
                            # Process in batches to avoid memory issues
                            batches = [processed_files[i:i + process_batch_size] 
                                      for i in range(0, len(processed_files), process_batch_size)]
                            
                            for batch_idx, batch in enumerate(batches):
                                logger.info(f"Processing batch {batch_idx+1}/{len(batches)} ({len(batch)} documents)")
                                for file_path in batch:
                                    vector_store.add_documents_from_file(file_path)
                                    
                        except Exception as e:
                            error_msg = f"Failed to add documents to vector store: {str(e)}"
                            logger.error(error_msg)
                            pipeline_errors.append(error_msg)
                            if fail_fast:
                                raise PipelineExecutionError(
                                    step="add_documents_to_vector_store", 
                                    component="vector_store"
                                ) from e
                else:
                    # Update document processor settings if needed
                    if self.document_processor is not None:
                        # Set batch size and other configuration
                        self.document_processor.batch_size = process_batch_size
                        
                    # Process documents with immediate embedding if vector store is available
                    self.process_documents(max_workers=max_workers, vector_store=vector_store)
            except Exception as e:
                error_msg = f"Document processing failed: {str(e)}"
                logger.error(error_msg)
                pipeline_errors.append(error_msg)
                if fail_fast:
                    raise PipelineExecutionError(
                        step="document_processing", 
                        component="document_processor"
                    ) from e
        else:
            logger.info("Skipping document processing")
            
            # If we're skipping processing but not analysis, we need to ensure the vector store has all documents
            if not skip_analysis and vector_store is not None and not skip_vectordb:
                try:
                    self.add_all_documents_to_vector_store(vector_store)
                except Exception as e:
                    error_msg = f"Failed to add documents to vector store: {str(e)}"
                    logger.error(error_msg)
                    pipeline_errors.append(error_msg)
                    if fail_fast:
                        raise PipelineExecutionError(
                            step="add_all_documents_to_vector_store", 
                            component="vector_store"
                        ) from e
        
        # Step 3: Analyze documents if vector store was created successfully
        if not skip_analysis and vector_store is not None:
            try:
                # Check if we should use existing analysis
                if use_existing_analysis:
                    logger.info("Using existing analysis results")
                    # Verify that analysis results exist
                    analysis_dir = os.path.join(self.base_dir, "data/analysis")
                    analysis_files = [f for f in os.listdir(analysis_dir) if f.endswith('.json')] if os.path.exists(analysis_dir) else []
                    
                    if not analysis_files:
                        logger.warning("No existing analysis files found, running analysis instead")
                        # Update analyzer settings
                        if self.document_analyzer is not None:
                            self.document_analyzer.max_retries = retry_count
                        self.analyze_documents(vector_store)
                else:
                    # Update analyzer settings
                    if self.document_analyzer is not None:
                        self.document_analyzer.max_retries = retry_count
                    self.analyze_documents(vector_store)
            except Exception as e:
                error_msg = f"Document analysis failed: {str(e)}"
                logger.error(error_msg)
                pipeline_errors.append(error_msg)
                if fail_fast:
                    raise PipelineExecutionError(
                        step="document_analysis", 
                        component="document_analyzer"
                    ) from e
                # If analysis fails but we want to continue, we'll generate a partial report
        else:
            if skip_analysis:
                logger.info("Skipping document analysis as requested")
            else:
                logger.info("Skipping document analysis due to missing vector database")
        
        # Step 4: Generate report (with partial results if there were errors)
        report_path = ""
        if not skip_report and not skip_analysis:
            try:
                report = self.generate_report()
                # Print final report location
                report_path = os.path.join(self.base_dir, "data/reports/full_report.html")
                logger.info(f"Final report available at: {report_path}")
            except Exception as e:
                error_msg = f"Report generation failed: {str(e)}"
                logger.error(error_msg)
                pipeline_errors.append(error_msg)
                
                # Create a partial report with error information
                try:
                    report_path = os.path.join(self.base_dir, "data/reports/partial_report.html")
                    with open(report_path, "w") as f:
                        f.write(f"""<html>
                        <head><title>JFK Document Analysis - Partial Report</title></head>
                        <body>
                            <h1>JFK Document Analysis Report</h1>
                            <h2>Warning: Some Pipeline Steps Failed</h2>
                            <p>This is a partial report due to errors in the pipeline.</p>
                            <h3>Pipeline Errors:</h3>
                            <ul>
                                {"".join(f"<li>{error}</li>" for error in pipeline_errors)}
                            </ul>
                            <p>Please check the logs for more details on these errors.</p>
                            <hr>
                            <h3>Partial Results</h3>
                            <p>Some analysis results may be available in the data/analysis directory.</p>
                        </body>
                        </html>""")
                    logger.info(f"Created partial report with error information at: {report_path}")
                except Exception as write_error:
                    logger.error(f"Failed to create partial report: {str(write_error)}")
                
                if fail_fast:
                    raise PipelineExecutionError(
                        step="report_generation", 
                        component="findings_report"
                    ) from e
        else:
            if skip_report:
                logger.info("Skipping report generation as requested")
            elif skip_analysis:
                # Create a dummy report or use an existing one
                report_path = os.path.join(self.base_dir, "data/reports/dummy_report.html")
                with open(report_path, "w") as f:
                    f.write("<html><body><h1>JFK Document Analysis Report</h1><p>Analysis phase was skipped.</p></body></html>")
                logger.info(f"Created dummy report at: {report_path}")
        
        # Log summary of pipeline execution
        if pipeline_errors:
            logger.warning(f"Completed JFK documents analysis pipeline with {len(pipeline_errors)} errors")
            logger.warning("Some steps failed but pipeline completed in partial mode")
        else:
            logger.info("Completed JFK documents analysis pipeline successfully")
        
        return report_path


def setup_container(args=None) -> DependencyContainer:
    """
    Set up the dependency injection container with all the required dependencies.
    
    Args:
        args: Command-line arguments
        
    Returns:
        The dependency container with all dependencies registered
    """
    from .utils.dependency_injection import DependencyContainer
    
    container = DependencyContainer()
    
    # Register credential provider with options from args if provided
    if args:
        credential_provider = create_credential_provider(
            config_file=args.credentials_file,
            add_rotating=not args.no_rotating_credentials,
            required_credentials=["OPENAI_API_KEY"]
        )
    else:
        credential_provider = create_credential_provider(required_credentials=["OPENAI_API_KEY"])
    
    container.register_singleton(ICredentialProvider, credential_provider)
    
    # Register text cleaner
    text_cleaner = create_text_cleaner()
    container.register_singleton(ITextCleaner, text_cleaner)
    
    # Register factories for other dependencies
    container.register_factory(IVectorStore, lambda: create_vector_store(
        credential_provider=container.resolve(ICredentialProvider)
    ))
    
    container.register_factory(IDocumentProcessor, lambda: create_document_processor(
        text_cleaner=container.resolve(ITextCleaner)
    ))
    
    container.register_factory(IDocumentScraper, create_document_scraper)
    
    # Create LLM provider with appropriate settings
    if args:
        api_type = args.api_type
        fallback_model = args.fallback_model
    else:
        api_type = "openai"
        fallback_model = None
    
    # Register LLM provider factory
    container.register_factory(ILLMProvider, lambda: create_llm_provider(
        model_name=os.environ.get("OPENAI_ANALYSIS_MODEL", "gpt-4o"),
        credential_provider=container.resolve(ICredentialProvider),
        api_type=api_type,
        fallback_model=fallback_model
    ))
    
    container.register_factory(IDocumentAnalyzer, lambda: create_document_analyzer(
        vector_store=container.resolve(IVectorStore),
        credential_provider=container.resolve(ICredentialProvider),
        llm_provider=container.resolve(ILLMProvider)
    ))
    
    container.register_factory(IFindingsReport, lambda: create_findings_report(
        credential_provider=container.resolve(ICredentialProvider),
        llm_provider=container.resolve(ILLMProvider)
    ))
    
    return container


def create_app(args) -> JFKReveal:
    """
    Create the JFKReveal application with all dependencies injected.
    
    Args:
        args: Command-line arguments
        
    Returns:
        The JFKReveal application
    """
    # Set up the dependency container
    container = setup_container(args)
    
    # Get credential provider
    credential_provider = container.resolve(ICredentialProvider)
    
    # If API keys are provided, set them in the credential provider
    if args.openai_api_key:
        credential_provider.set_credential("OPENAI_API_KEY", args.openai_api_key)
    
    if args.azure_openai_key:
        credential_provider.set_credential("AZURE_OPENAI_API_KEY", args.azure_openai_key)
        
    if args.azure_openai_endpoint:
        credential_provider.set_credential("AZURE_OPENAI_ENDPOINT", args.azure_openai_endpoint)
        
    if args.anthropic_api_key:
        credential_provider.set_credential("ANTHROPIC_API_KEY", args.anthropic_api_key)
    
    # Validate credentials before starting if requested
    if args.validate_credentials:
        logger.info("Validating API credentials before starting")
        # Check appropriate credentials based on API type
        if args.api_type == "openai":
            result = credential_provider.validate_credential("OPENAI_API_KEY")
            if not result.is_valid:
                logger.error(f"OpenAI API key validation failed: {result.message}")
                if args.strict_validation:
                    sys.exit(1)
        elif args.api_type == "azure":
            result = credential_provider.validate_credential("AZURE_OPENAI_API_KEY")
            if not result.is_valid:
                logger.error(f"Azure OpenAI API key validation failed: {result.message}")
                if args.strict_validation:
                    sys.exit(1)
        elif args.api_type == "anthropic":
            result = credential_provider.validate_credential("ANTHROPIC_API_KEY")
            if not result.is_valid:
                logger.error(f"Anthropic API key validation failed: {result.message}")
                if args.strict_validation:
                    sys.exit(1)
    
    # Create the application with injected dependencies
    return JFKReveal(
        base_dir=args.base_dir,
        openai_api_key=args.openai_api_key,
        clean_text=not args.no_clean_text,
        use_ocr=not args.no_ocr,
        ocr_resolution_scale=args.ocr_resolution,
        ocr_language=args.ocr_language,
        credential_provider=credential_provider
    )


def print_colored(message, color=None):
    """Print a message with ANSI color codes if color output is supported."""
    try:
        import colorama
        colorama.init()
        
        colors = {
            'red': colorama.Fore.RED,
            'green': colorama.Fore.GREEN,
            'yellow': colorama.Fore.YELLOW,
            'blue': colorama.Fore.BLUE,
            'magenta': colorama.Fore.MAGENTA,
            'cyan': colorama.Fore.CYAN,
            'white': colorama.Fore.WHITE,
            'bold': colorama.Style.BRIGHT,
            'reset': colorama.Style.RESET_ALL
        }
        
        if color and color in colors:
            print(f"{colors[color]}{message}{colors['reset']}")
        else:
            print(message)
            
    except ImportError:
        # If colorama isn't available, just print normally
        print(message)


def show_progress(steps, current_step, message):
    """Display a progress indicator for the pipeline steps."""
    try:
        import tqdm
        
        if not hasattr(show_progress, "progress_bar"):
            show_progress.progress_bar = tqdm.tqdm(total=steps, desc="Pipeline Progress", 
                                                 bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")
            
        show_progress.progress_bar.update(1)
        show_progress.progress_bar.set_description(f"Pipeline: {message}")
        
        if current_step == steps:
            show_progress.progress_bar.close()
            delattr(show_progress, "progress_bar")
            
    except ImportError:
        # Fall back to simple progress display if tqdm isn't installed
        print(f"[{current_step}/{steps}] {message}")


def print_examples():
    """Print command-line usage examples with color formatting."""
    examples = [
        "# Run the full pipeline",
        "python -m jfkreveal",
        "",
        "# Skip scraping and use existing processed files",
        "python -m jfkreveal --skip-scraping --use-existing-processed",
        "",
        "# Run only the analysis step with specific model",
        "python -m jfkreveal --skip-scraping --skip-processing --model gpt-4o",
        "",
        "# Process documents in batches to manage memory usage",
        "python -m jfkreveal --skip-scraping --process-batch-size 25 --max-workers 10",
        "",
        "# Skip vector database but still process documents (useful for data preparation)",
        "python -m jfkreveal --skip-vectordb --skip-analysis",
        "",
        "# Reuse existing analysis results but generate a new report",
        "python -m jfkreveal --skip-scraping --skip-processing --use-existing-analysis",
        "",
        "# Specific control over caching and retries",
        "python -m jfkreveal --retry-count 5 --retry-delay 10 --backoff-factor 1.5 --no-cache-downloads",
        "",
        "# Create an analysis specifically on a subject",
        "python -m jfkreveal --skip-scraping --skip-processing run-analysis --topic \"CIA involvement\"",
        "",
        "# Generate a full report from existing analysis data",
        "python -m jfkreveal --skip-scraping --skip-processing --skip-analysis generate-report",
        "",
        "# Launch the interactive dashboard",
        "python -m jfkreveal view-dashboard",
        "",
        "# Search for specific documents",
        "python -m jfkreveal search \"Lee Harvey Oswald CIA connection\"",
    ]
    
    print_colored("\nUsage Examples:", "cyan")
    for line in examples:
        if line.startswith("#"):
            print_colored(line, "green")
        else:
            print(line)
    print()


def add_pipeline_args(parser):
    """Add pipeline control arguments to the argument parser."""
    pipeline_group = parser.add_argument_group("Pipeline Control")
    pipeline_group.add_argument("--skip-scraping", action="store_true",
                    help="Skip document scraping")
    pipeline_group.add_argument("--skip-processing", action="store_true",
                    help="Skip document processing")
    pipeline_group.add_argument("--skip-analysis", action="store_true",
                    help="Skip document analysis and report generation")
    pipeline_group.add_argument("--skip-vectordb", action="store_true",
                    help="Skip vector database creation")
    pipeline_group.add_argument("--skip-report", action="store_true",
                    help="Skip report generation after analysis")
    pipeline_group.add_argument("--use-existing-processed", action="store_true",
                    help="Use existing processed documents without processing new ones")
    pipeline_group.add_argument("--use-existing-analysis", action="store_true",
                    help="Use existing analysis results without reanalyzing")
    pipeline_group.add_argument("--max-workers", type=int, default=20,
                    help="Number of documents to process in parallel (default: 20)")
    pipeline_group.add_argument("--process-batch-size", type=int, default=50,
                    help="Number of documents to process in a single batch (default: 50)")
    pipeline_group.add_argument("--fail-fast", action="store_true",
                    help="Stop pipeline on first error instead of continuing with partial results")
    pipeline_group.add_argument("--retry-count", type=int, default=3,
                    help="Number of retries for failed operations (default: 3)")
    pipeline_group.add_argument("--retry-delay", type=int, default=5,
                    help="Delay between retries in seconds (default: 5)")
    pipeline_group.add_argument("--backoff-factor", type=float, default=2.0,
                    help="Exponential backoff factor for retries (default: 2.0)")
    pipeline_group.add_argument("--progress-bar", action="store_true", default=True,
                    help="Show progress bar during pipeline execution (default: True)")
    pipeline_group.add_argument("--no-progress-bar", action="store_false", dest="progress_bar",
                    help="Hide progress bar during pipeline execution")
    pipeline_group.add_argument("--cache-downloads", action="store_true", default=True,
                    help="Cache downloaded documents (default: True)")
    pipeline_group.add_argument("--no-cache-downloads", action="store_false", dest="cache_downloads",
                    help="Disable caching of downloaded documents")


def add_ocr_args(parser):
    """Add OCR-related arguments to the argument parser."""
    ocr_group = parser.add_argument_group("OCR Configuration")
    ocr_group.add_argument("--no-clean-text", action="store_true",
                    help="Disable text cleaning for OCR documents")
    ocr_group.add_argument("--no-ocr", action="store_true",
                    help="Disable OCR for scanned documents")
    ocr_group.add_argument("--ocr-resolution", type=float, default=2.0,
                    help="OCR resolution scale factor (higher = better quality but slower, default: 2.0)")
    ocr_group.add_argument("--ocr-language", type=str, default="eng",
                    help="Language for OCR (default: eng)")
    ocr_group.add_argument("--ocr-config", type=str,
                    help="Custom Tesseract configuration string")
    ocr_group.add_argument("--ocr-page-segmentation-mode", type=int, choices=range(0, 14), default=3,
                    help="Tesseract page segmentation mode (0-13, default: 3 - automatic page segmentation)")
    ocr_group.add_argument("--ocr-threads", type=int, default=1,
                    help="Number of threads to use for OCR processing (default: 1)")


def add_logging_args(parser):
    """Add logging-related arguments to the argument parser."""
    logging_group = parser.add_argument_group("Logging Configuration")
    logging_group.add_argument("--log-level", type=str, default="INFO",
                    choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                    help="Logging level (default: INFO)")
    logging_group.add_argument("--log-file", type=str, default="jfkreveal.log",
                    help="Log file path (default: jfkreveal.log)")
    logging_group.add_argument("--no-console-log", action="store_true",
                    help="Disable console logging")
    logging_group.add_argument("--log-format", type=str, default="standard",
                    choices=["standard", "detailed", "minimal"],
                    help="Log message format (default: standard)")
    logging_group.add_argument("--log-to-file", action="store_true", default=True,
                    help="Enable logging to file (default: True)")
    logging_group.add_argument("--no-log-to-file", action="store_false", dest="log_to_file",
                    help="Disable logging to file")


def add_credential_args(parser):
    """Add credential-related arguments to the argument parser."""
    credential_group = parser.add_argument_group("API Credentials")
    credential_group.add_argument("--openai-api-key", type=str,
                    help="OpenAI API key (uses env var OPENAI_API_KEY if not provided)")
    credential_group.add_argument("--azure-openai-key", type=str,
                    help="Azure OpenAI API key (uses env var AZURE_OPENAI_API_KEY if not provided)")
    credential_group.add_argument("--azure-openai-endpoint", type=str,
                    help="Azure OpenAI endpoint (uses env var AZURE_OPENAI_ENDPOINT if not provided)")
    credential_group.add_argument("--anthropic-api-key", type=str,
                    help="Anthropic API key (uses env var ANTHROPIC_API_KEY if not provided)")
    credential_group.add_argument("--credentials-file", type=str,
                    help="Path to JSON file with API credentials")
    credential_group.add_argument("--no-rotating-credentials", action="store_true",
                    help="Disable credential rotation for rate limiting")
    credential_group.add_argument("--validate-credentials", action="store_true",
                    help="Validate API credentials before starting")
    credential_group.add_argument("--strict-validation", action="store_true",
                    help="Exit if credential validation fails")


def add_llm_args(parser):
    """Add LLM-related arguments to the argument parser."""
    llm_group = parser.add_argument_group("LLM Configuration")
    llm_group.add_argument("--api-type", type=str, default="openai",
                    choices=["openai", "azure", "anthropic"],
                    help="API type to use (default: openai)")
    llm_group.add_argument("--fallback-model", type=str,
                    help="Fallback model to use if primary model fails")
    llm_group.add_argument("--model", type=str,
                    help="Model name to use (overrides OPENAI_ANALYSIS_MODEL env var)")
    llm_group.add_argument("--temperature", type=float, default=0.0,
                    help="Temperature for LLM generation (0.0-2.0, default: 0.0)")
    llm_group.add_argument("--max-tokens", type=int,
                    help="Maximum tokens for LLM generation")
    llm_group.add_argument("--embedding-model", type=str,
                    help="Model to use for embeddings (overrides OPENAI_EMBEDDING_MODEL env var)")
    llm_group.add_argument("--ollama-url", type=str,
                    help="URL for Ollama API if using local models")


def add_storage_args(parser):
    """Add storage-related arguments to the argument parser."""
    storage_group = parser.add_argument_group("Storage Configuration")
    storage_group.add_argument("--base-dir", type=str, default=".",
                    help="Base directory for data storage")
    storage_group.add_argument("--raw-dir", type=str,
                    help="Directory for raw downloaded documents (default: data/raw)")
    storage_group.add_argument("--processed-dir", type=str,
                    help="Directory for processed documents (default: data/processed)")
    storage_group.add_argument("--vectordb-dir", type=str,
                    help="Directory for vector database (default: data/vectordb)")
    storage_group.add_argument("--analysis-dir", type=str,
                    help="Directory for analysis results (default: data/analysis)")
    storage_group.add_argument("--reports-dir", type=str,
                    help="Directory for generated reports (default: data/reports)")


def add_run_analysis_args(parser):
    """Add arguments for the run-analysis command."""
    parser.add_argument("--topic", type=str, required=True,
                help="Topic to analyze (e.g. 'Lee Harvey Oswald', 'CIA involvement')")
    parser.add_argument("--output-file", type=str,
                help="Output file for analysis results (default: topic name with underscores)")
    parser.add_argument("--max-documents", type=int, default=50,
                help="Maximum number of documents to analyze per topic (default: 50)")
    parser.add_argument("--max-retries", type=int, default=5,
                help="Maximum number of retries for LLM calls (default: 5)")


def add_search_args(parser):
    """Add arguments for the search command."""
    parser.add_argument("query", type=str,
                help="Search query string")
    parser.add_argument("--limit", type=int, default=10,
                help="Maximum number of results to return (default: 10)")
    parser.add_argument("--format", type=str, default="text",
                choices=["text", "json", "html", "csv"],
                help="Output format (default: text)")
    parser.add_argument("--output-file", type=str,
                help="Output file for search results (default: stdout)")
    parser.add_argument("--search-type", type=str, default="hybrid",
                choices=["vector", "bm25", "hybrid"],
                help="Search algorithm to use (default: hybrid)")
    parser.add_argument("--rerank", action="store_true",
                help="Rerank search results for improved relevance")


def add_generate_report_args(parser):
    """Add arguments for the generate-report command."""
    parser.add_argument("--report-types", type=str, nargs="+",
                default=["executive_summary", "full_report", "detailed_findings", "suspects_analysis", "coverup_analysis"],
                help="Types of reports to generate (default: all types)")
    parser.add_argument("--template-dir", type=str,
                help="Directory containing report templates (default: built-in templates)")
    parser.add_argument("--custom-css", type=str,
                help="Path to custom CSS file for HTML reports")
    parser.add_argument("--include-evidence", action="store_true", default=True,
                help="Include supporting evidence in reports (default: True)")
    parser.add_argument("--no-include-evidence", action="store_false", dest="include_evidence",
                help="Exclude supporting evidence from reports")
    parser.add_argument("--include-sources", action="store_true", default=True,
                help="Include source document references in reports (default: True)")
    parser.add_argument("--no-include-sources", action="store_false", dest="include_sources",
                help="Exclude source document references from reports")


def add_dashboard_args(parser):
    """Add arguments for the view-dashboard command."""
    parser.add_argument("--host", type=str, default="127.0.0.1",
                help="Host to serve dashboard on (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8050,
                help="Port to serve dashboard on (default: 8050)")
    parser.add_argument("--debug", action="store_true",
                help="Run dashboard in debug mode")
    parser.add_argument("--theme", type=str, default="light",
                choices=["light", "dark"],
                help="Dashboard theme (default: light)")
    parser.add_argument("--data-dir", type=str,
                help="Data directory for dashboard (default: data/analysis)")


def main():
    """Command-line entry point."""
    # Create the primary parser with subcommands
    parser = argparse.ArgumentParser(
        description="JFKReveal - Analyze declassified JFK assassination documents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Run with --examples to see usage examples"
    )
    
    # Add global arguments that apply to all commands
    parser.add_argument("--examples", action="store_true",
                      help="Show usage examples and exit")
    parser.add_argument("--version", action="store_true",
                      help="Show version information and exit")
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", title="Commands")
    
    # Default command (no subcommand specified) - runs the full pipeline
    add_pipeline_args(parser)
    add_ocr_args(parser)
    add_logging_args(parser)
    add_credential_args(parser)
    add_llm_args(parser)
    add_storage_args(parser)
    
    # Command: run-analysis
    run_analysis_parser = subparsers.add_parser("run-analysis", 
                                              help="Analyze documents on a specific topic")
    add_run_analysis_args(run_analysis_parser)
    add_pipeline_args(run_analysis_parser)
    add_logging_args(run_analysis_parser)
    add_credential_args(run_analysis_parser)
    add_llm_args(run_analysis_parser)
    add_storage_args(run_analysis_parser)
    
    # Command: search
    search_parser = subparsers.add_parser("search", 
                                        help="Search documents using semantic search")
    add_search_args(search_parser)
    add_logging_args(search_parser)
    add_credential_args(search_parser)
    add_llm_args(search_parser)
    add_storage_args(search_parser)
    
    # Command: generate-report
    report_parser = subparsers.add_parser("generate-report", 
                                        help="Generate reports from analysis results")
    add_generate_report_args(report_parser)
    add_logging_args(report_parser)
    add_credential_args(report_parser)
    add_llm_args(report_parser)
    add_storage_args(report_parser)
    
    # Command: view-dashboard
    dashboard_parser = subparsers.add_parser("view-dashboard", 
                                           help="Launch interactive visualization dashboard")
    add_dashboard_args(dashboard_parser)
    add_logging_args(dashboard_parser)
    add_storage_args(dashboard_parser)
    
    # Parse the arguments
    args = parser.parse_args()
    
    # Handle special arguments
    if args.examples:
        print_examples()
        return
    
    if args.version:
        import pkg_resources
        try:
            version = pkg_resources.get_distribution("jfkreveal").version
            print_colored(f"JFKReveal v{version}", "green")
        except pkg_resources.DistributionNotFound:
            print_colored("JFKReveal (development version)", "yellow")
        return
    
    # Configure logging with the specified options
    log_level = getattr(logging, args.log_level)
    log_format = args.log_format if hasattr(args, "log_format") else "standard"
    
    # Handle log file parameter
    log_file = None
    if hasattr(args, "log_to_file") and args.log_to_file:
        log_file = args.log_file if hasattr(args, "log_file") else "jfkreveal.log"
    
    # Configure console logging
    console_log = True
    if hasattr(args, "no_console_log") and args.no_console_log:
        console_log = False
    
    setup_logging(
        level=log_level,
        log_file=log_file,
        console=console_log,
        format_type=log_format
    )
    
    # Log the startup message with level info
    logger.info(f"JFKReveal starting with log level: {args.log_level}")
    
    # Log debug information about arguments
    logger.debug(f"Command-line arguments: {args}")
    
    # Process command-specific environment variables
    if hasattr(args, "model") and args.model:
        os.environ["OPENAI_ANALYSIS_MODEL"] = args.model
        logger.info(f"Using model: {args.model}")
    
    if hasattr(args, "embedding_model") and args.embedding_model:
        os.environ["OPENAI_EMBEDDING_MODEL"] = args.embedding_model
        logger.info(f"Using embedding model: {args.embedding_model}")
    
    if hasattr(args, "ollama_url") and args.ollama_url:
        os.environ["OLLAMA_BASE_URL"] = args.ollama_url
        logger.info(f"Using Ollama URL: {args.ollama_url}")
    
    # Handle different commands
    if args.command == "run-analysis":
        # Import here to avoid circular imports
        from .analysis.document_analyzer import DocumentAnalyzer
        
        print_colored(f"Running analysis on topic: '{args.topic}'", "cyan")
        
        # Create the application with injected dependencies
        jfk_reveal = create_app(args)
        vector_store = jfk_reveal.build_vector_database()
        
        # Create document analyzer
        analyzer = DocumentAnalyzer(
            vector_store=vector_store,
            output_dir=args.analysis_dir or os.path.join(args.base_dir, "data/analysis"),
            model_name=args.model or os.environ.get("OPENAI_ANALYSIS_MODEL", "gpt-4o"),
            max_retries=args.max_retries
        )
        
        # Run analysis on the specified topic
        result = analyzer.analyze_topic(args.topic, max_documents=args.max_documents)
        
        # Output file
        output_file = args.output_file or f"{args.topic.replace(' ', '_').lower()}_analysis.json"
        output_path = os.path.join(
            args.analysis_dir or os.path.join(args.base_dir, "data/analysis"),
            output_file
        )
        
        # Save result
        import json
        with open(output_path, 'w') as f:
            json.dump(result.dict(), f, indent=2)
        
        print_colored(f"\nAnalysis complete! Results saved to: {output_path}", "green")
        return
    
    elif args.command == "search":
        # Import here to avoid circular imports
        from .search.semantic_search import SemanticSearchEngine
        
        print_colored(f"Searching for: '{args.query}'", "cyan")
        
        # Create the application with injected dependencies
        jfk_reveal = create_app(args)
        vector_store = jfk_reveal.build_vector_database()
        
        # Create search engine
        search_engine = SemanticSearchEngine(
            vector_store=vector_store,
            use_reranker=args.rerank,
            search_type=args.search_type
        )
        
        # Perform search
        results = search_engine.search(args.query, limit=args.limit)
        
        # Handle different output formats
        if args.format == "json":
            import json
            output = json.dumps([r.dict() for r in results], indent=2)
        elif args.format == "csv":
            import csv
            import io
            output = io.StringIO()
            writer = csv.writer(output)
            writer.writerow(["Document", "Page", "Relevance", "Content"])
            for r in results:
                writer.writerow([r.document_id, r.page_number, r.score, r.content[:100] + "..."])
            output = output.getvalue()
        elif args.format == "html":
            output = "<html><body><h1>Search Results</h1><table>"
            output += "<tr><th>Document</th><th>Page</th><th>Relevance</th><th>Content</th></tr>"
            for r in results:
                output += f"<tr><td>{r.document_id}</td><td>{r.page_number}</td><td>{r.score:.2f}</td>"
                output += f"<td>{r.content[:100]}...</td></tr>"
            output += "</table></body></html>"
        else:  # text format
            output = f"Search Results for: '{args.query}'\n"
            output += f"Using search type: {args.search_type}\n\n"
            for i, r in enumerate(results, 1):
                output += f"{i}. Document: {r.document_id} (Page {r.page_number}) - Relevance: {r.score:.2f}\n"
                output += f"   {r.content[:100]}...\n\n"
        
        # Output results
        if args.output_file:
            with open(args.output_file, 'w') as f:
                f.write(output)
            print_colored(f"\nSearch results saved to: {args.output_file}", "green")
        else:
            print(output)
        
        return
    
    elif args.command == "generate-report":
        print_colored("Generating reports from analysis data...", "cyan")
        
        # Create the application with injected dependencies
        jfk_reveal = create_app(args)
        
        # Generate reports
        with show_progress(len(args.report_types), 0, "Initializing report generation") if args.progress_bar else contextlib.nullcontext():
            report = jfk_reveal.generate_report()
            
            # Show generated report files
            report_files = []
            reports_dir = args.reports_dir or os.path.join(args.base_dir, "data/reports")
            for i, report_type in enumerate(args.report_types, 1):
                file_path = os.path.join(reports_dir, f"{report_type}.html")
                if os.path.exists(file_path):
                    report_files.append(file_path)
                if args.progress_bar:
                    show_progress(len(args.report_types), i, f"Generated {report_type}")
        
        if report_files:
            print_colored("\nGenerated reports:", "green")
            for file_path in report_files:
                print(f"- {file_path}")
            print_colored(f"\nMain report available at: {os.path.join(reports_dir, 'full_report.html')}", "green")
        else:
            print_colored("\nNo reports were generated. Check logs for errors.", "red")
        
        return
    
    elif args.command == "view-dashboard":
        # Import here to avoid circular imports
        from .visualization.dashboard import JFKDashboard
        
        print_colored("Launching interactive dashboard...", "cyan")
        
        # Create and run dashboard
        dashboard = JFKDashboard(
            data_dir=args.data_dir or os.path.join(args.base_dir, "data/analysis"),
            theme=args.theme
        )
        
        # Run the dashboard
        dashboard.run(host=args.host, port=args.port, debug=args.debug)
        return
    
    # Default command: run the pipeline
    print_colored("Starting JFK documents analysis pipeline...", "cyan")
    
    # Create the application with injected dependencies
    jfk_reveal = create_app(args)
    
    # Calculate number of pipeline steps
    total_steps = 4  # Default: scrape, process, analyze, report
    if args.skip_scraping:
        total_steps -= 1
    if args.skip_processing:
        total_steps -= 1
    if args.skip_analysis or args.skip_vectordb:
        total_steps -= 1
    if args.skip_report:
        total_steps -= 1
    
    current_step = 0
    pipeline_context = show_progress(total_steps, current_step, "Starting pipeline") if args.progress_bar else contextlib.nullcontext()
    
    # Run pipeline with progress bar if enabled
    with pipeline_context:
        report_path = jfk_reveal.run_pipeline(
            skip_scraping=args.skip_scraping,
            skip_processing=args.skip_processing,
            skip_analysis=args.skip_analysis,
            skip_vectordb=args.skip_vectordb if hasattr(args, 'skip_vectordb') else False,
            skip_report=args.skip_report if hasattr(args, 'skip_report') else False,
            use_existing_processed=args.use_existing_processed,
            use_existing_analysis=args.use_existing_analysis if hasattr(args, 'use_existing_analysis') else False,
            max_workers=args.max_workers,
            process_batch_size=args.process_batch_size if hasattr(args, 'process_batch_size') else 50,
            fail_fast=args.fail_fast,
            retry_count=args.retry_count if hasattr(args, 'retry_count') else 3,
            retry_delay=args.retry_delay if hasattr(args, 'retry_delay') else 5,
            backoff_factor=args.backoff_factor if hasattr(args, 'backoff_factor') else 2.0,
            cache_downloads=args.cache_downloads if hasattr(args, 'cache_downloads') else True
        )
    
    # Final report message
    if report_path:
        logger.info(f"Analysis complete! Final report available at: {report_path}")
        print_colored(f"\nAnalysis complete! Final report available at: {report_path}", "green")
    else:
        logger.info("Pipeline execution completed successfully")
        print_colored("\nPipeline execution completed successfully", "green")


if __name__ == "__main__":
    main()