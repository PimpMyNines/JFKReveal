"""
Unit tests for the main JFKReveal class
"""
import os
import json
import pytest
from unittest.mock import patch, MagicMock

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

    @patch('jfkreveal.scrapers.archives_gov.ArchivesGovScraper')
    def test_scrape_documents(self, mock_scraper, temp_data_dir):
        """Test document scraping functionality"""
        # Setup mock
        mock_instance = MagicMock()
        mock_instance.scrape_all.return_value = ["file1.pdf", "file2.pdf"]
        mock_scraper.return_value = mock_instance
        
        # Create instance and call method
        jfk = JFKReveal(base_dir=temp_data_dir["root"])
        result = jfk.scrape_documents()
        
        # Verify correct initialization and calls
        mock_scraper.assert_called_once_with(
            output_dir=os.path.join(temp_data_dir["root"], "data/raw")
        )
        mock_instance.scrape_all.assert_called_once()
        
        # Verify result
        assert result == ["file1.pdf", "file2.pdf"]

    @patch('jfkreveal.database.document_processor.DocumentProcessor')
    def test_process_documents(self, mock_processor, temp_data_dir):
        """Test document processing functionality"""
        # Setup mock
        mock_instance = MagicMock()
        mock_instance.process_all_documents.return_value = ["file1.json", "file2.json"]
        mock_processor.return_value = mock_instance
        
        # Create instance and call method
        jfk = JFKReveal(base_dir=temp_data_dir["root"])
        result = jfk.process_documents(max_workers=5, skip_existing=True)
        
        # Verify correct initialization
        mock_processor.assert_called_once_with(
            input_dir=os.path.join(temp_data_dir["root"], "data/raw"),
            output_dir=os.path.join(temp_data_dir["root"], "data/processed"),
            max_workers=5,
            skip_existing=True,
            vector_store=None,
            clean_text=True
        )
        
        # Verify method call
        mock_instance.process_all_documents.assert_called_once()
        
        # Verify result
        assert result == ["file1.json", "file2.json"]

    @patch('jfkreveal.database.document_processor.DocumentProcessor')
    def test_get_processed_documents(self, mock_processor, temp_data_dir):
        """Test get_processed_documents method"""
        # Setup mock
        mock_instance = MagicMock()
        mock_instance.get_processed_documents.return_value = ["file1.json", "file2.json"]
        mock_processor.return_value = mock_instance
        
        # Create instance and call method
        jfk = JFKReveal(base_dir=temp_data_dir["root"])
        result = jfk.get_processed_documents()
        
        # Verify correct initialization
        mock_processor.assert_called_once_with(
            input_dir=os.path.join(temp_data_dir["root"], "data/raw"),
            output_dir=os.path.join(temp_data_dir["root"], "data/processed")
        )
        
        # Verify method call
        mock_instance.get_processed_documents.assert_called_once()
        
        # Verify result
        assert result == ["file1.json", "file2.json"]

    @patch('jfkreveal.database.vector_store.VectorStore')
    def test_build_vector_database(self, mock_vector_store, temp_data_dir):
        """Test building vector database"""
        # Setup mock
        mock_instance = MagicMock()
        mock_vector_store.return_value = mock_instance
        
        # Create instance and call method
        jfk = JFKReveal(base_dir=temp_data_dir["root"], openai_api_key="test_key")
        result = jfk.build_vector_database()
        
        # Verify correct initialization
        mock_vector_store.assert_called_once_with(
            persist_directory=os.path.join(temp_data_dir["root"], "data/vectordb"),
            openai_api_key="test_key"
        )
        
        # Verify result is the mock instance
        assert result == mock_instance

    @patch('jfkreveal.database.vector_store.VectorStore')
    def test_build_vector_database_error(self, mock_vector_store, temp_data_dir):
        """Test error handling in build_vector_database"""
        # Setup mock to raise an exception
        mock_vector_store.side_effect = Exception("API error")
        
        # Create instance and call method
        jfk = JFKReveal(base_dir=temp_data_dir["root"])
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

    @patch('jfkreveal.analysis.document_analyzer.DocumentAnalyzer')
    def test_analyze_documents(self, mock_analyzer, temp_data_dir):
        """Test document analysis functionality"""
        # Setup mock
        mock_instance = MagicMock()
        mock_instance.analyze_key_topics.return_value = ["topic1", "topic2"]
        mock_analyzer.return_value = mock_instance
        
        # Create mock vector store
        mock_vector_store = MagicMock()
        
        # Set environment variable for test
        os.environ["OPENAI_ANALYSIS_MODEL"] = "test-model"
        
        # Create instance and call method
        jfk = JFKReveal(base_dir=temp_data_dir["root"], openai_api_key="test_key")
        result = jfk.analyze_documents(mock_vector_store)
        
        # Verify correct initialization
        mock_analyzer.assert_called_once_with(
            vector_store=mock_vector_store,
            output_dir=os.path.join(temp_data_dir["root"], "data/analysis"),
            model_name="test-model",
            openai_api_key="test_key",
            temperature=0.0,
            max_retries=5
        )
        
        # Verify method call
        mock_instance.analyze_key_topics.assert_called_once()
        
        # Verify result
        assert result == ["topic1", "topic2"]

    @patch('jfkreveal.summarization.findings_report.FindingsReport')
    def test_generate_report(self, mock_report, temp_data_dir):
        """Test report generation functionality"""
        # Setup mock
        mock_instance = MagicMock()
        mock_instance.generate_full_report.return_value = "report.html"
        mock_report.return_value = mock_instance
        
        # Create instance and call method
        jfk = JFKReveal(base_dir=temp_data_dir["root"], openai_api_key="test_key")
        result = jfk.generate_report()
        
        # Verify correct initialization
        mock_report.assert_called_once_with(
            analysis_dir=os.path.join(temp_data_dir["root"], "data/analysis"),
            output_dir=os.path.join(temp_data_dir["root"], "data/reports"),
            openai_api_key="test_key"
        )
        
        # Verify method call
        mock_instance.generate_full_report.assert_called_once()
        
        # Verify result
        assert result == "report.html"

    @patch.multiple('jfkreveal.main.JFKReveal', 
                   scrape_documents=MagicMock(return_value=["file1.pdf"]),
                   build_vector_database=MagicMock(),
                   process_documents=MagicMock(),
                   analyze_documents=MagicMock(),
                   generate_report=MagicMock(return_value="report.html"))
    def test_run_pipeline_complete(self, temp_data_dir):
        """Test running the complete pipeline"""
        # Mock returning a valid vector store
        jfk = JFKReveal(base_dir=temp_data_dir["root"])
        jfk.build_vector_database = MagicMock(return_value=MagicMock())
        
        # Call the method
        result = jfk.run_pipeline()
        
        # Verify methods were called in order
        jfk.scrape_documents.assert_called_once()
        jfk.build_vector_database.assert_called_once()
        jfk.process_documents.assert_called_once_with(max_workers=20, vector_store=jfk.build_vector_database.return_value)
        jfk.analyze_documents.assert_called_once()
        jfk.generate_report.assert_called_once()
        
        # Verify result
        assert result == "report.html"

    @patch.multiple('jfkreveal.main.JFKReveal', 
                   scrape_documents=MagicMock(return_value=["file1.pdf"]),
                   get_processed_documents=MagicMock(return_value=["file1.json"]),
                   build_vector_database=MagicMock(),
                   process_documents=MagicMock(),
                   analyze_documents=MagicMock(),
                   generate_report=MagicMock(return_value="report.html"))
    def test_run_pipeline_use_existing(self, temp_data_dir):
        """Test running the pipeline with existing processed documents"""
        # Mock returning a valid vector store
        jfk = JFKReveal(base_dir=temp_data_dir["root"])
        mock_vector_store = MagicMock()
        jfk.build_vector_database = MagicMock(return_value=mock_vector_store)
        
        # Call the method with use_existing_processed=True
        result = jfk.run_pipeline(use_existing_processed=True)
        
        # Verify existing documents are used
        jfk.scrape_documents.assert_called_once()
        jfk.build_vector_database.assert_called_once()
        jfk.get_processed_documents.assert_called_once()
        jfk.process_documents.assert_not_called()
        
        # Vector store add_documents_from_file should be called for each file
        assert mock_vector_store.add_documents_from_file.call_count == 1
        
        # Analysis and reporting should proceed
        jfk.analyze_documents.assert_called_once()
        jfk.generate_report.assert_called_once()
        
        # Verify result
        assert result == "report.html"

    @patch.multiple('jfkreveal.main.JFKReveal', 
                   scrape_documents=MagicMock(),
                   build_vector_database=MagicMock(return_value=None),
                   process_documents=MagicMock(),
                   analyze_documents=MagicMock(),
                   generate_report=MagicMock())
    def test_run_pipeline_vector_store_failure(self, temp_data_dir):
        """Test pipeline when vector store initialization fails"""
        jfk = JFKReveal(base_dir=temp_data_dir["root"])
        
        # Call the method
        result = jfk.run_pipeline()
        
        # Verify scraping and processing still occur
        jfk.scrape_documents.assert_called_once()
        jfk.build_vector_database.assert_called_once()
        jfk.process_documents.assert_called_once()
        
        # Analysis should be skipped
        jfk.analyze_documents.assert_not_called()
        jfk.generate_report.assert_not_called()
        
        # A dummy report should be created
        assert "dummy_report.html" in result 