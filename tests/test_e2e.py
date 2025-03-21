"""
End-to-End tests for the JFKReveal application
"""
import os
import pytest
import tempfile
import shutil
from unittest.mock import patch

from jfkreveal.main import JFKReveal


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


@pytest.mark.e2e
class TestEndToEnd:
    """End-to-end tests for the JFKReveal application"""

    @patch('openai.OpenAI')  # Mock OpenAI to avoid real API calls
    def test_basic_processing_pipeline(self, mock_openai, e2e_data_dir, sample_pdf_file):
        """Test the basic document processing pipeline without analysis."""
        # Set environment variables for testing
        os.environ["OPENAI_API_KEY"] = "sk-dummy-key-for-testing"
        os.environ["OPENAI_EMBEDDING_MODEL"] = "text-embedding-3-large"
        
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
    
    @patch.multiple('jfkreveal.main.JFKReveal', 
                   scrape_documents=lambda self: ["not_needed_for_this_test"],
                   analyze_documents=lambda self, vector_store: ["topic1"])
    @pytest.mark.skipif(not os.environ.get("RUN_SLOW_TESTS"), 
                        reason="Skipping slow test, set RUN_SLOW_TESTS=1 to run")
    def test_vector_database_creation(self, e2e_data_dir, sample_pdf_file):
        """Test creation of the vector database with real document."""
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
    
    @pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), 
                        reason="Skipping test that requires OpenAI API key")
    def test_full_pipeline_with_analysis(self, e2e_data_dir, sample_pdf_file):
        """
        Test the full pipeline including analysis with real OpenAI API.
        
        Note: This test requires a valid OPENAI_API_KEY environment variable.
        It is skipped by default to avoid unexpected API charges.
        """
        # Create JFKReveal instance
        jfk = JFKReveal(base_dir=e2e_data_dir)
        
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