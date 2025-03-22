"""
End-to-End tests for the JFKReveal application
"""
import os
import pytest
import tempfile
import time
import shutil
import logging
from unittest.mock import patch, MagicMock

import openai
try:
    from langchain_community.llms.openai import OpenAI
except ImportError:
    from langchain.llms.openai import OpenAI
from jfkreveal.main import JFKReveal

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Constants for rate limiting
MAX_RETRIES = 5
RETRY_DELAY = 5  # seconds
RATE_LIMIT_ERRORS = ["rate_limit", "rate_limit_exceeded", "capacity", "maximum_context_length_exceeded"]


@pytest.fixture
def e2e_data_dir():
    """Create a temporary directory for E2E tests with real data."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create directory structure
        data_dir = os.path.join(temp_dir, "data")
        raw_dir = os.path.join(data_dir, "raw")
        processed_dir = os.path.join(data_dir, "processed")
        vector_dir = os.path.join(data_dir, "vectordb")
        analysis_dir = os.path.join(data_dir, "analysis")
        reports_dir = os.path.join(data_dir, "reports")
        
        # Create directories
        os.makedirs(raw_dir, exist_ok=True)
        os.makedirs(processed_dir, exist_ok=True)
        os.makedirs(vector_dir, exist_ok=True)
        os.makedirs(analysis_dir, exist_ok=True)
        os.makedirs(reports_dir, exist_ok=True)
        
        yield temp_dir


@pytest.fixture
def api_key_manager():
    """
    Manages API keys for tests, with fallbacks to environment variables.
    
    This fixture ensures we have proper credentials for tests or skip
    tests that require credentials if they're not available.
    """
    # Original state of environment variables
    original_openai_key = os.environ.get("OPENAI_API_KEY")
    original_embedding_model = os.environ.get("OPENAI_EMBEDDING_MODEL")
    original_analysis_model = os.environ.get("OPENAI_ANALYSIS_MODEL")
    
    # Check for CI/CD environment
    in_ci = os.environ.get("CI") == "true"
    
    # Set test API credentials
    if in_ci:
        # In CI, use environment secrets
        if not original_openai_key:
            # If still no API key available, we'll use mock mode
            logger.warning("No OpenAI API key found in CI environment, tests will use mock mode")
            os.environ["OPENAI_API_KEY"] = "sk-dummy-key-for-testing"
            os.environ["USE_MOCK_API"] = "1"
    else:
        # Local development - if not set, use dummy key and enable mock mode
        if not original_openai_key:
            os.environ["OPENAI_API_KEY"] = "sk-dummy-key-for-testing"
            os.environ["USE_MOCK_API"] = "1"
    
    # Always set reasonable defaults for models
    if not original_embedding_model:
        os.environ["OPENAI_EMBEDDING_MODEL"] = "text-embedding-3-large"
    if not original_analysis_model:
        os.environ["OPENAI_ANALYSIS_MODEL"] = "gpt-3.5-turbo"
    
    # Return API status for tests to use
    has_real_credentials = os.environ.get("OPENAI_API_KEY", "").startswith("sk-") and \
                          not os.environ.get("USE_MOCK_API")
    
    yield {
        "has_real_credentials": has_real_credentials,
        "use_mock_api": os.environ.get("USE_MOCK_API") == "1",
        "openai_api_key": os.environ.get("OPENAI_API_KEY"),
        "in_ci": in_ci
    }
    
    # Restore original state
    if original_openai_key:
        os.environ["OPENAI_API_KEY"] = original_openai_key
    else:
        os.environ.pop("OPENAI_API_KEY", None)
        
    if original_embedding_model:
        os.environ["OPENAI_EMBEDDING_MODEL"] = original_embedding_model
    else:
        os.environ.pop("OPENAI_EMBEDDING_MODEL", None)
        
    if original_analysis_model:
        os.environ["OPENAI_ANALYSIS_MODEL"] = original_analysis_model
    else:
        os.environ.pop("OPENAI_ANALYSIS_MODEL", None)
        
    os.environ.pop("USE_MOCK_API", None)


@pytest.fixture
def sample_pdf_file(e2e_data_dir):
    """Create a real sample PDF file for end-to-end testing."""
    # Set up the path for the test PDF file
    test_dir = os.path.dirname(os.path.abspath(__file__))
    test_pdf_path = os.path.join(test_dir, "test_data", "sample_document.pdf")
    
    # If test_data directory doesn't exist, create it
    os.makedirs(os.path.join(test_dir, "test_data"), exist_ok=True)
    
    # Create a simple PDF file if it doesn't exist
    if not os.path.exists(test_pdf_path):
        try:
            import fitz  # PyMuPDF
            
            # Create a new PDF document
            doc = fitz.open()
            page = doc.new_page()
            
            # Add text content with JFK-related information
            text = """
            CONFIDENTIAL
            
            November 22, 1963
            
            Lee Harvey Oswald was observed at the Texas School Book Depository at
            approximately 12:30 PM. Multiple witnesses reported hearing shots from
            the grassy knoll area, which contradicts the official findings.
            
            Further analysis of ballistic evidence suggests the possibility of
            additional shooters, though this remains unproven.
            
            The Warren Commission's investigation concluded that Oswald acted alone,
            but numerous inconsistencies have been noted by independent researchers.
            
            CIA and FBI files related to the assassination remain partially classified.
            """
            
            # Insert text into the PDF
            page.insert_text((50, 50), text, fontsize=11)
            
            # Save the PDF
            doc.save(test_pdf_path)
            doc.close()
            
        except ImportError:
            # If PyMuPDF is not available, create a simple text file
            with open(test_pdf_path.replace(".pdf", ".txt"), "w") as f:
                f.write("Sample JFK document for testing")
            # Skip the actual PDF creation in this case
    
    # Copy the test PDF to the e2e raw data directory
    raw_dir = os.path.join(e2e_data_dir, "data", "raw")
    target_path = os.path.join(raw_dir, "sample_document.pdf")
    
    if os.path.exists(test_pdf_path):
        shutil.copy(test_pdf_path, target_path)
    else:
        # Create a placeholder file if PDF creation failed
        with open(target_path, "w") as f:
            f.write("PDF placeholder")
    
    return target_path


class ApiRateLimitError(Exception):
    """Exception raised when API rate limits are encountered."""
    pass


def retry_on_rate_limit(func):
    """
    Decorator to handle API rate limit errors with exponential backoff.
    
    Args:
        func: The function to wrap with retry logic
        
    Returns:
        The wrapped function with retry logic
    """
    import functools
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        retry_count = 0
        while retry_count < MAX_RETRIES:
            try:
                return func(*args, **kwargs)
            except (openai.RateLimitError, openai.APIError, ApiRateLimitError) as e:
                error_str = str(e).lower()
                
                # Check if this is a rate limit related error
                is_rate_limit = any(err_type in error_str for err_type in RATE_LIMIT_ERRORS)
                
                if not is_rate_limit or retry_count >= MAX_RETRIES - 1:
                    logger.error(f"API error not recoverable or max retries exceeded: {e}")
                    raise
                
                # Calculate exponential backoff delay (with jitter)
                backoff_delay = RETRY_DELAY * (2 ** retry_count) * (0.5 + 0.5 * (1.0 - 0.5))
                logger.warning(f"Rate limit hit, retrying in {backoff_delay:.1f}s (attempt {retry_count+1}/{MAX_RETRIES})")
                
                # Sleep with backoff delay
                time.sleep(backoff_delay)
                retry_count += 1
                
    return wrapper


@pytest.mark.e2e
class TestEndToEnd:
    """End-to-end tests for the JFKReveal application"""

    @pytest.fixture(autouse=True)
    def setup_mocks(self, api_key_manager):
        """Setup API mocks if we're not using real credentials."""
        self.api_status = api_key_manager
        self.mock_patches = []
        
        # Set up mock for API if using mock mode
        if self.api_status["use_mock_api"]:
            # Mock OpenAI client
            openai_patch = patch('openai.OpenAI')
            self.mock_openai = openai_patch.start()
            self.mock_patches.append(openai_patch)
            
            mock_client = MagicMock()
            mock_client.embeddings.create.return_value = MagicMock(
                data=[MagicMock(embedding=[0.1] * 1536)]
            )
            self.mock_openai.return_value = mock_client
            
            # Mock LangChain OpenAI
            langchain_patch = patch('langchain.llms.openai.OpenAI')
            self.mock_langchain = langchain_patch.start()
            self.mock_patches.append(langchain_patch)
            
            # Mock LangChain client response
            mock_langchain_instance = MagicMock()
            mock_langchain_instance.invoke.return_value = (
                "Mock LLM response for testing purposes. "
                "This is a placeholder for actual AI-generated content."
            )
            self.mock_langchain.return_value = mock_langchain_instance
            
            # Mock LangChain Embeddings
            embeddings_patch = patch('langchain.embeddings.openai.OpenAIEmbeddings')
            self.mock_embeddings = embeddings_patch.start()
            self.mock_patches.append(embeddings_patch)
            
            # Return mock embedding vectors
            mock_embeddings_instance = MagicMock()
            mock_embeddings_instance.embed_documents.return_value = [[0.1] * 1536]
            mock_embeddings_instance.embed_query.return_value = [0.1] * 1536
            self.mock_embeddings.return_value = mock_embeddings_instance
        
        yield
        
        # Clean up patches
        for patch_obj in self.mock_patches:
            patch_obj.stop()

    def test_basic_processing_pipeline(self, e2e_data_dir, sample_pdf_file):
        """Test the basic document processing pipeline without analysis."""
        # Create JFKReveal instance
        jfk = JFKReveal(base_dir=e2e_data_dir)
        
        # Skip the scraping phase since we already have a PDF
        processed_files = jfk.process_documents(skip_existing=False)
        
        # Verify document was processed
        assert len(processed_files) == 1
        processed_file = processed_files[0]
        assert os.path.exists(processed_file)
        
        # Verify the structure of the output file
        with open(processed_file, 'r') as f:
            import json
            data = json.load(f)
            
            # Basic validation of the processed document
            assert isinstance(data, list)
            assert len(data) > 0
            assert "text" in data[0]
            assert "metadata" in data[0]
            assert "document_id" in data[0]["metadata"]
    
    @retry_on_rate_limit
    def test_vector_database_creation(self, e2e_data_dir, sample_pdf_file):
        """Test creation of the vector database with real document."""
        # Skip if we're in CI and not using real credentials
        if self.api_status["in_ci"] and not self.api_status["has_real_credentials"]:
            pytest.skip("Skipping in CI environment without real credentials")
            
        # Process a document first
        jfk = JFKReveal(base_dir=e2e_data_dir)
        jfk.process_documents(skip_existing=False)
        
        # Build the vector database
        vector_store = jfk.build_vector_database()
        
        # Verify vector store was created
        assert vector_store is not None
        
        # Add documents to vector store
        chunks = jfk.add_all_documents_to_vector_store(vector_store)
        
        # Verify documents were added
        assert chunks > 0
        
        # Test simple search query
        results = vector_store.similarity_search("Oswald")
        
        # Verify search returns results
        assert len(results) > 0

    @pytest.mark.skipif(os.environ.get("CI") == "true", 
                       reason="Skipping full analysis test in CI environment")
    @retry_on_rate_limit
    def test_full_pipeline_with_real_credentials(self, e2e_data_dir, sample_pdf_file):
        """
        Test the full pipeline including analysis with real OpenAI API.
        
        This test only runs when:
        1. We have real credentials available
        2. We're not in a CI environment
        3. The RUN_E2E_FULL environment variable is set
        """
        # Skip if we don't have real credentials or RUN_E2E_FULL is not set
        if not self.api_status["has_real_credentials"]:
            pytest.skip("Skipping test that requires real OpenAI API key")
        if not os.environ.get("RUN_E2E_FULL"):
            pytest.skip("Skipping full pipeline test, set RUN_E2E_FULL=1 to run")
            
        # Create JFKReveal instance with real API key
        jfk = JFKReveal(
            base_dir=e2e_data_dir,
            openai_api_key=self.api_status["openai_api_key"]
        )
        
        # Run the pipeline with our sample document
        # Skip scraping since we already have our test document
        report_path = jfk.run_pipeline(
            skip_scraping=True,
            skip_processing=False,
            skip_analysis=False
        )
        
        # Verify the pipeline completed and produced a report
        assert report_path is not None
        assert os.path.exists(report_path)
        
    def test_full_pipeline_with_mocked_api(self, e2e_data_dir, sample_pdf_file):
        """Test the full pipeline with mocked API for CI and testing purposes."""
        # Only run this test if we're using mock API mode
        if not self.api_status["use_mock_api"]:
            pytest.skip("This test is only for mock API mode")
        
        # Apply additional mocking for analysis components
        with patch('jfkreveal.analysis.document_analyzer.DocumentAnalyzer.analyze_key_topics') as mock_analyze:
            # Mock the analyze method to return a predefined analysis
            mock_analyze.return_value = {
                "topic": "JFK Assassination",
                "summary": "Mock summary of JFK assassination for testing",
                "key_findings": ["Finding 1", "Finding 2"],
                "credibility": 0.8,
                "alternative_theories": [
                    {"theory": "Theory 1", "evidence": ["Evidence 1"], "credibility": 0.5},
                    {"theory": "Theory 2", "evidence": ["Evidence 2"], "credibility": 0.3}
                ]
            }
            
            # Create JFKReveal instance
            jfk = JFKReveal(base_dir=e2e_data_dir)
            
            # Run the pipeline with mocked components
            report_path = jfk.run_pipeline(
                skip_scraping=True,
                skip_processing=False,
                skip_analysis=False
            )
            
            # Verify the pipeline completed and produced a report
            assert report_path is not None
            assert os.path.exists(report_path)