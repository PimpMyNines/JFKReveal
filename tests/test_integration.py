"""
Integration tests for the JFKReveal application
"""
import os
import json
import pytest
from unittest.mock import patch, MagicMock

from jfkreveal.main import JFKReveal
from jfkreveal.database.document_processor import DocumentProcessor
from jfkreveal.database.vector_store import VectorStore
from jfkreveal.analysis.document_analyzer import DocumentAnalyzer


class TestIntegration:
    """Integration tests for the JFKReveal application"""

    def test_document_processor_to_vector_store(self, temp_data_dir, sample_text):
        """Test integration between DocumentProcessor and VectorStore"""
        # Create PDF file
        pdf_path = os.path.join(temp_data_dir["raw"], "test_integration.pdf")
        with open(pdf_path, "w") as f:
            f.write("PDF content - will be mocked")
        
        # Initialize components
        processor = DocumentProcessor(
            input_dir=temp_data_dir["raw"],
            output_dir=temp_data_dir["processed"],
            chunk_size=100,
            chunk_overlap=10
        )
        
        vector_store = VectorStore(
            persist_directory=temp_data_dir["vector"],
            embedding_provider="fake"  # Use fake embeddings for testing
        )
        
        # Mock PDF extraction to return our sample text
        with patch.object(processor, 'extract_text_from_pdf') as mock_extract:
            mock_extract.return_value = (sample_text, {"filename": "test_integration.pdf"})
            
            # Process document
            processed_file = processor.process_document(pdf_path)
            
            # Verify file was processed
            assert processed_file is not None
            assert os.path.exists(processed_file)
            
            # Add to vector store
            num_chunks = vector_store.add_documents_from_file(processed_file)
            
            # Verify chunks were added
            assert num_chunks > 0
            
            # Test search functionality
            results = vector_store.similarity_search("Oswald")
            
            # Verify search returned results
            assert len(results) > 0
            assert any("Oswald" in result["text"] for result in results)

    def test_processor_with_immediate_embedding(self, temp_data_dir, sample_text):
        """Test document processor with immediate embedding into vector store"""
        # Create PDF file
        pdf_path = os.path.join(temp_data_dir["raw"], "test_immediate.pdf")
        with open(pdf_path, "w") as f:
            f.write("PDF content - will be mocked")
        
        # Initialize vector store
        vector_store = VectorStore(
            persist_directory=temp_data_dir["vector"],
            embedding_provider="fake"  # Use fake embeddings for testing
        )
        
        # Initialize processor with vector store
        processor = DocumentProcessor(
            input_dir=temp_data_dir["raw"],
            output_dir=temp_data_dir["processed"],
            chunk_size=100,
            chunk_overlap=10,
            vector_store=vector_store
        )
        
        # Mock vector store add_documents_from_file
        with patch.object(vector_store, 'add_documents_from_file') as mock_add:
            mock_add.return_value = 3  # Pretend 3 chunks were added
            
            # Mock PDF extraction
            with patch.object(processor, 'extract_text_from_pdf') as mock_extract:
                mock_extract.return_value = (sample_text, {"filename": "test_immediate.pdf"})
                
                # Process document
                processed_file = processor.process_document(pdf_path)
                
                # Verify file was processed
                assert processed_file is not None
                
                # Verify vector store was called immediately
                mock_add.assert_called_once_with(processed_file)
                
                # Verify document was marked as embedded
                assert processor.check_if_embedded(processed_file)

    def test_pipeline_integration(self, temp_data_dir, sample_text, sample_document_chunks):
        """Test the complete pipeline integration"""
        # Create JFKReveal instance
        jfk = JFKReveal(base_dir=temp_data_dir["root"])
        
        # Mock the scraper to return a list of PDF files
        with patch.object(jfk, 'scrape_documents') as mock_scrape:
            mock_scrape.return_value = [
                os.path.join(temp_data_dir["raw"], "doc1.pdf"),
                os.path.join(temp_data_dir["raw"], "doc2.pdf")
            ]
            
            # Create test PDF files
            for pdf_path in mock_scrape.return_value:
                with open(pdf_path, "w") as f:
                    f.write("Test PDF content")
            
            # Mock document processor to return processed files
            with patch.object(DocumentProcessor, 'process_document') as mock_process:
                processed_files = [
                    pdf_path.replace(".pdf", ".json").replace(temp_data_dir["raw"], temp_data_dir["processed"])
                    for pdf_path in mock_scrape.return_value
                ]
                mock_process.side_effect = processed_files
                
                # Create processed files with test content
                for file_path in processed_files:
                    os.makedirs(os.path.dirname(file_path), exist_ok=True)
                    with open(file_path, "w", encoding="utf-8") as f:
                        json.dump(sample_document_chunks, f)
                
                # Mock vector store to avoid actual embedding
                with patch.object(VectorStore, 'similarity_search') as mock_search:
                    mock_search.return_value = [
                        {
                            "text": chunk["text"],
                            "metadata": chunk["metadata"],
                            "score": 0.95 - i * 0.1
                        }
                        for i, chunk in enumerate(sample_document_chunks)
                    ]
                    
                    # Mock document analyzer to avoid OpenAI API calls
                    with patch.object(DocumentAnalyzer, 'analyze_key_topics') as mock_analyze:
                        mock_analyze.return_value = ["topic1", "topic2"]
                        
                        # Run the pipeline
                        result = jfk.run_pipeline(skip_scraping=False)
                        
                        # Verify pipeline completed
                        assert result is not None
                        assert os.path.exists(result) or "dummy_report" in result
                        
                        # Verify key methods were called
                        mock_scrape.assert_called_once()
                        assert mock_process.call_count == 2  # Two documents
                        mock_analyze.assert_called_once()

    @patch('langchain_openai.ChatOpenAI')
    def test_document_analysis_chain(self, mock_chat_openai, temp_data_dir, vector_store, sample_document_chunks):
        """Test the document analysis chain from vector store to analyzer"""
        # Mock LLM to avoid API calls
        mock_llm = MagicMock()
        mock_chat_openai.return_value = mock_llm
        
        # Mock structured output method
        mock_chain = MagicMock()
        mock_llm.with_structured_output.return_value = mock_chain
        
        # Mock vector store search to return our sample chunks
        vector_store.similarity_search = MagicMock(return_value=[
            {
                "text": chunk["text"],
                "metadata": chunk["metadata"],
                "score": 0.95
            }
            for chunk in sample_document_chunks
        ])
        
        # Create analyzer
        analyzer = DocumentAnalyzer(
            vector_store=vector_store,
            output_dir=temp_data_dir["root"] + "/analysis"
        )
        
        # Use a real DocumentAnalyzer but mock the LLM responses
        with patch.object(analyzer, 'analyze_document_chunk') as mock_analyze_chunk:
            # Return fake analyzed documents
            mock_analyze_chunk.return_value = MagicMock(error=None)
            
            # Search and analyze a topic
            result = analyzer.search_and_analyze_topic("Test Topic", num_results=3)
            
            # Verify vector store was queried
            vector_store.similarity_search.assert_called_once_with("Test Topic", k=3)
            
            # Verify all chunks were analyzed
            assert mock_analyze_chunk.call_count == len(sample_document_chunks)
            
            # Verify result structure
            assert result.topic == "Test Topic"
            assert result.num_documents == len(sample_document_chunks)
            assert result.error is None 