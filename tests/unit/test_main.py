"""
Unit tests for the main JFKReveal class
"""
import os
import json
import pytest
import contextlib
from unittest.mock import patch, MagicMock, ANY
from unittest.mock import mock_open as unittest_mock_open

from jfkreveal.main import JFKReveal


class TestJFKReveal:
    """Test the main JFKReveal class"""

    def test_init(self, temp_data_dir):
        """Test initialization of JFKReveal"""
        # Test default initialization
        jfk = JFKReveal(base_dir=temp_data_dir["root"])
        
        # Verify directories were created
        assert os.path.exists(os.path.join(temp_data_dir["root"], "data/raw"))
        assert os.path.exists(os.path.join(temp_data_dir["root"], "data/processed"))
        assert os.path.exists(os.path.join(temp_data_dir["root"], "data/vectordb"))
        assert os.path.exists(os.path.join(temp_data_dir["root"], "data/analysis"))
        assert os.path.exists(os.path.join(temp_data_dir["root"], "data/reports"))
        
        # Verify attributes
        assert jfk.base_dir == temp_data_dir["root"]
        assert jfk.clean_text is True

    def test_scrape_documents(self, temp_data_dir):
        """Test document scraping functionality with simplified mocking"""
        # Create instance of JFKReveal
        jfk = JFKReveal(base_dir=temp_data_dir["root"])
        
        # Create a mock for direct replacement of the document_scraper
        mock_scraper = MagicMock()
        mock_scraper.scrape_all.return_value = (["file1.pdf", "file2.pdf"], ["doc1", "doc2"])
        
        # Directly set the instance property
        jfk.document_scraper = mock_scraper
        
        # Call the method
        result = jfk.scrape_documents()
        
        # Verify scrape_all was called
        mock_scraper.scrape_all.assert_called_once()
        
        # Verify result
        assert result == ["file1.pdf", "file2.pdf"]

    def test_process_documents(self, temp_data_dir):
        """Test document processing functionality with simplified mocking"""
        import tempfile
        
        # Create instance of JFKReveal
        jfk = JFKReveal(base_dir=temp_data_dir["root"])
        
        # Create dummy PDF files in the input directory
        os.makedirs(os.path.join(temp_data_dir["root"], "data/raw"), exist_ok=True)
        with open(os.path.join(temp_data_dir["root"], "data/raw/test1.pdf"), "w") as f:
            f.write("Dummy PDF content")
        with open(os.path.join(temp_data_dir["root"], "data/raw/test2.pdf"), "w") as f:
            f.write("Another dummy PDF content")
        
        # Create a mock for direct replacement of the document_processor
        mock_processor = MagicMock()
        mock_processor.process_all_documents.return_value = ["file1.json", "file2.json"]
        
        # Directly set the instance property
        jfk.document_processor = mock_processor
        
        # Call the method
        result = jfk.process_documents(max_workers=5, skip_existing=True)
        
        # Verify process_all_documents was called
        mock_processor.process_all_documents.assert_called_once()
        
        # Verify result
        assert result == ["file1.json", "file2.json"]

    def test_get_processed_documents(self, temp_data_dir):
        """Test get_processed_documents method with simplified mocking"""
        # Create instance of JFKReveal
        jfk = JFKReveal(base_dir=temp_data_dir["root"])
        
        # Create a mock for direct replacement of the document_processor
        mock_processor = MagicMock()
        mock_processor.get_processed_documents.return_value = ["file1.json", "file2.json"]
        
        # Directly set the instance property
        jfk.document_processor = mock_processor
        
        # Call the method
        result = jfk.get_processed_documents()
        
        # Verify get_processed_documents was called
        mock_processor.get_processed_documents.assert_called_once()
        
        # Verify result
        assert result == ["file1.json", "file2.json"]

    def test_build_vector_database(self, temp_data_dir):
        """Test building vector database with simplified mocking"""
        # Create instance of JFKReveal
        jfk = JFKReveal(base_dir=temp_data_dir["root"], openai_api_key="test_key")
        
        # Create mocks for credentials and vector store
        mock_credential_provider = MagicMock()
        mock_credential_provider.get_credential.return_value = "test_key"
        
        # Create a mock vector store
        mock_vector_store = MagicMock()
        
        # Directly set instance properties
        jfk.credential_provider = mock_credential_provider
        
        # Mock create_vector_store function to return our mock
        with patch('jfkreveal.main.create_vector_store', return_value=mock_vector_store):
            # Call the method
            result = jfk.build_vector_database()
        
        # Verify credential was retrieved
        mock_credential_provider.get_credential.assert_called_with("OPENAI_API_KEY")
        
        # Verify result is the mock instance
        assert result == mock_vector_store

    def test_build_vector_database_error(self, temp_data_dir):
        """Test error handling in build_vector_database with simplified mocking"""
        # Create instance of JFKReveal
        jfk = JFKReveal(base_dir=temp_data_dir["root"])
        
        # Create mocks for credentials
        mock_credential_provider = MagicMock()
        mock_credential_provider.get_credential.return_value = "test_key"
        
        # Directly set instance properties
        jfk.credential_provider = mock_credential_provider
        
        # Create a custom context manager that can be used in place of pipeline_step
        @contextlib.contextmanager
        def mock_pipeline_step(step_name, component):
            try:
                yield
            except Exception:
                # Catch the exception but do not re-raise
                pass
        
        # Mock the pipeline_step with our custom context manager
        with patch('jfkreveal.main.pipeline_step', mock_pipeline_step):
            # Mock create_vector_store to raise an exception 
            with patch('jfkreveal.main.create_vector_store', side_effect=Exception("API error")):
                # Call the method
                result = jfk.build_vector_database()
        
        # Verify result is None on error
        assert result is None

    def test_add_all_documents_to_vector_store(self, temp_data_dir):
        """Test adding all documents to vector store"""
        # Create mock vector store
        mock_vector_store = MagicMock()
        mock_vector_store.add_all_documents.return_value = 10
        
        # Create instance and call method
        jfk = JFKReveal(base_dir=temp_data_dir["root"])
        result = jfk.add_all_documents_to_vector_store(mock_vector_store)
        
        # Verify method call
        mock_vector_store.add_all_documents.assert_called_once_with(
            processed_dir=os.path.join(temp_data_dir["root"], "data/processed")
        )
        
        # Verify result
        assert result == 10

    def test_analyze_documents(self, temp_data_dir, sample_topic_analysis):
        """Test document analysis functionality with comprehensive mocking"""
        # Save original environment variable if it exists
        original_model = os.environ.get("OPENAI_ANALYSIS_MODEL")
        
        try:
            # Set environment variable for test
            os.environ["OPENAI_ANALYSIS_MODEL"] = "test-model"
            
            # Create instance of JFKReveal
            jfk = JFKReveal(base_dir=temp_data_dir["root"], openai_api_key="test_key")
            
            # Create mocks
            mock_credential_provider = MagicMock()
            mock_credential_provider.get_credential.return_value = "test_key"
            
            # Create a mock document analyzer with proper return value (TopicAnalysis objects)
            mock_document_analyzer = MagicMock()
            mock_document_analyzer.analyze_key_topics.return_value = [sample_topic_analysis]
            
            mock_vector_store = MagicMock()
            
            # Create dummy analysis file
            os.makedirs(os.path.join(temp_data_dir["root"], "data/analysis"), exist_ok=True)
            with open(os.path.join(temp_data_dir["root"], "data/analysis/dummy.json"), "w") as f:
                f.write("{}")
            
            # Directly set instance properties and mock create_document_analyzer
            jfk.credential_provider = mock_credential_provider
            jfk.document_analyzer = mock_document_analyzer
            
            # Create a simple no-op context manager for pipeline_step
            @contextlib.contextmanager
            def mock_pipeline_step(step_name, component):
                yield
                
            # Setup additional mocks to prevent side effects
            with patch('jfkreveal.main.pipeline_step', mock_pipeline_step), \
                 patch('jfkreveal.main.create_document_analyzer', return_value=mock_document_analyzer), \
                 patch('jfkreveal.main.os.listdir', return_value=["dummy.json"]):
                result = jfk.analyze_documents(mock_vector_store)
            
            # Verify method call
            mock_document_analyzer.analyze_key_topics.assert_called_once()
            
            # Verify result is the list of TopicAnalysis objects
            assert result == [sample_topic_analysis]
            
        finally:
            # Restore original environment variable
            if original_model:
                os.environ["OPENAI_ANALYSIS_MODEL"] = original_model
            elif "OPENAI_ANALYSIS_MODEL" in os.environ:
                del os.environ["OPENAI_ANALYSIS_MODEL"]

    def test_generate_report(self, temp_data_dir):
        """Test report generation functionality with comprehensive mocking"""
        # Create test directory structure
        analysis_dir = os.path.join(temp_data_dir["root"], "data/analysis")
        os.makedirs(analysis_dir, exist_ok=True)
        
        # Create a dummy analysis file to ensure directory check passes
        with open(os.path.join(analysis_dir, "dummy_analysis.json"), "w") as f:
            f.write("{}")
        
        # Create instance of JFKReveal
        jfk = JFKReveal(base_dir=temp_data_dir["root"], openai_api_key="test_key")
        
        # Create mocks for credentials and findings report
        mock_credential_provider = MagicMock()
        mock_credential_provider.get_credential.return_value = "test_key"
        
        mock_findings_report = MagicMock()
        mock_findings_report.generate_full_report.return_value = True
        
        # Directly set instance properties
        jfk.credential_provider = mock_credential_provider
        jfk.findings_report = mock_findings_report
        
        # Create a simple no-op context manager for pipeline_step
        @contextlib.contextmanager
        def mock_pipeline_step(step_name, component):
            yield
        
        # Create mocks for all file system operations
        analysis_files = ["dummy_analysis.json"]
        report_files = ["full_report.html", "executive_summary.html"]
        
        # Call the method with our mocks
        with patch('jfkreveal.main.pipeline_step', mock_pipeline_step), \
             patch('jfkreveal.main.create_findings_report', return_value=mock_findings_report), \
             patch('jfkreveal.main.os.path.exists', return_value=True), \
             patch('jfkreveal.main.os.listdir', side_effect=[analysis_files, report_files]):
            result = jfk.generate_report()
        
        # Verify method call
        mock_findings_report.generate_full_report.assert_called_once()
        
        # Verify result
        assert result is True

    def test_run_pipeline_complete(self, temp_data_dir, sample_topic_analysis):
        """Test running the complete pipeline by patching JFKReveal methods"""
        # Create all necessary directories
        os.makedirs(os.path.join(temp_data_dir["root"], "data/raw"), exist_ok=True)
        os.makedirs(os.path.join(temp_data_dir["root"], "data/processed"), exist_ok=True)
        os.makedirs(os.path.join(temp_data_dir["root"], "data/vectordb"), exist_ok=True)
        os.makedirs(os.path.join(temp_data_dir["root"], "data/analysis"), exist_ok=True)
        os.makedirs(os.path.join(temp_data_dir["root"], "data/reports"), exist_ok=True)
        
        # Create a test instance of JFKReveal
        jfk = JFKReveal(base_dir=temp_data_dir["root"])
        
        # The expected report path based on temp_data_dir
        expected_report_path = os.path.join(temp_data_dir["root"], "data/reports/full_report.html")
        
        mock_vector_store = MagicMock()
        
        # Using the patch.object approach for better method isolation
        with patch.object(JFKReveal, 'scrape_documents', return_value=["file1.pdf"]) as mock_scrape, \
             patch.object(JFKReveal, 'build_vector_database', return_value=mock_vector_store) as mock_build_db, \
             patch.object(JFKReveal, 'process_documents', return_value=["file1.json"]) as mock_process, \
             patch.object(JFKReveal, 'add_all_documents_to_vector_store', return_value=10) as mock_add_docs, \
             patch.object(JFKReveal, 'analyze_documents', return_value=[sample_topic_analysis]) as mock_analyze, \
             patch.object(JFKReveal, 'generate_report', return_value=expected_report_path) as mock_report:
            
            # Call method with all steps enabled
            result = jfk.run_pipeline(
                skip_scraping=False,
                skip_processing=False,
                skip_analysis=False,
                skip_vectordb=False,
                skip_report=False,
                max_workers=20
            )
            
            # Verify method calls
            mock_scrape.assert_called_once()
            mock_build_db.assert_called_once()
            # When skip_processing=False, process_documents is called with vector_store
            mock_process.assert_called_once_with(max_workers=20, vector_store=mock_vector_store)
            # In the complete pipeline with no skipping, add_all_documents_to_vector_store is NOT called directly
            # because the documents are added during process_documents
            mock_add_docs.assert_not_called()
            mock_analyze.assert_called_once_with(mock_vector_store)
            mock_report.assert_called_once()
            
            # Verify result
            assert result == expected_report_path

    def test_run_pipeline_use_existing(self, temp_data_dir):
        """Test running the pipeline with existing processed documents using specific inspection"""
        # Create instance of JFKReveal
        jfk = JFKReveal(base_dir=temp_data_dir["root"])
        
        # Setup mock returns
        mock_vector_store = MagicMock()
        processed_documents = ["file1.json", "file2.json"]
        
        # The expected report path based on temp_data_dir
        expected_report_path = os.path.join(temp_data_dir["root"], "data/reports/full_report.html")
        
        # Mock all methods and their return values
        jfk.scrape_documents = MagicMock(return_value=["file1.pdf"])
        jfk.get_processed_documents = MagicMock(return_value=processed_documents)
        jfk.build_vector_database = MagicMock(return_value=mock_vector_store)
        jfk.process_documents = MagicMock(return_value=processed_documents)
        jfk.add_all_documents_to_vector_store = MagicMock(return_value=10)
        jfk.analyze_documents = MagicMock(return_value=["topic1"])
        jfk.generate_report = MagicMock(return_value=expected_report_path)
        
        # Create test directories
        os.makedirs(os.path.join(temp_data_dir["root"], "data/raw"), exist_ok=True)
        os.makedirs(os.path.join(temp_data_dir["root"], "data/processed"), exist_ok=True)
        os.makedirs(os.path.join(temp_data_dir["root"], "data/reports"), exist_ok=True)
        
        # Mock vector_store.add_documents_from_file to ensure it's callable
        mock_vector_store.add_documents_from_file = MagicMock()
        
        # Call method directly with use_existing_processed=True
        result = jfk.run_pipeline(use_existing_processed=True)
        
        # Check that get_processed_documents was called
        jfk.get_processed_documents.assert_called_once()
        
        # Check that vector_store.add_documents_from_file was called for each document
        assert mock_vector_store.add_documents_from_file.call_count == len(processed_documents)
        
        # Check that build_vector_database was called
        jfk.build_vector_database.assert_called_once()
        
        # Check analyze_documents and generate_report were called
        jfk.analyze_documents.assert_called_once_with(mock_vector_store)  
        jfk.generate_report.assert_called_once()
        
        # Verify the result is the expected report path
        assert result == expected_report_path

    def test_run_pipeline_vector_store_failure(self, temp_data_dir):
        """Test pipeline when vector store initialization fails"""
        # Create instance of JFKReveal
        jfk = JFKReveal(base_dir=temp_data_dir["root"])
        
        # Mock open function for the dummy report
        mock_file = unittest_mock_open()
        
        # Mock all methods in the pipeline
        jfk.scrape_documents = MagicMock(return_value=["file1.pdf"])
        # Return None to simulate vector store initialization failure
        jfk.build_vector_database = MagicMock(return_value=None)
        jfk.process_documents = MagicMock(return_value=["file1.json"])
        jfk.add_all_documents_to_vector_store = MagicMock(return_value=10)
        jfk.analyze_documents = MagicMock(return_value=["topic1"])
        # Mock generate_report as MagicMock (it was originally a real function)
        jfk.generate_report = MagicMock(return_value=True)
        
        # In the actual implementation, when vector_store is None, 
        # it skips analysis and generates a dummy report file directly
        # rather than calling generate_report
        
        # Call the method with a patched open
        with patch('builtins.open', mock_file):
            result = jfk.run_pipeline()
        
        # Verify scraping and processing still occur
        jfk.scrape_documents.assert_called_once()
        jfk.build_vector_database.assert_called_once()
        # In the implementation, skip_existing parameter is not passed, so we shouldn't assert it
        jfk.process_documents.assert_called_once_with(
            max_workers=20, 
            vector_store=None
        )
        
        # Analysis should be skipped
        jfk.analyze_documents.assert_not_called()
        
        # generate_report should NOT be called because we're creating a dummy file
        jfk.generate_report.assert_not_called()
        
        # Verify the dummy report path
        expected_path = os.path.join(temp_data_dir["root"], "data/reports/dummy_report.html")
        assert result == expected_path
        
        # Verify that the open was called with the correct path
        mock_file.assert_called_once_with(expected_path, "w") 