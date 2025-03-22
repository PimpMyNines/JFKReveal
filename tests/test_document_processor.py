"""
Unit tests for the DocumentProcessor class
"""
import os
import json
import pytest
from unittest.mock import patch, MagicMock, mock_open

from jfkreveal.database.document_processor import DocumentProcessor


class TestDocumentProcessor:
    """Test the DocumentProcessor class"""

    def test_init(self, temp_data_dir):
        """Test initialization of DocumentProcessor"""
        # Test with default parameters
        processor = DocumentProcessor(
            input_dir=temp_data_dir["raw"],
            output_dir=temp_data_dir["processed"]
        )
        
        # Verify attributes
        assert processor.input_dir == temp_data_dir["raw"]
        assert processor.output_dir == temp_data_dir["processed"]
        assert processor.chunk_size == 1000
        assert processor.chunk_overlap == 200
        assert processor.max_workers == 20
        assert processor.skip_existing is True
        assert processor.clean_text is True
        assert processor.vector_store is None
        
        # Test with custom parameters
        processor = DocumentProcessor(
            input_dir=temp_data_dir["raw"],
            output_dir=temp_data_dir["processed"],
            chunk_size=500,
            chunk_overlap=50,
            max_workers=10,
            skip_existing=False,
            vector_store=MagicMock(),
            clean_text=False
        )
        
        # Verify custom attributes
        assert processor.chunk_size == 500
        assert processor.chunk_overlap == 50
        assert processor.max_workers == 10
        assert processor.skip_existing is False
        assert processor.clean_text is False
        assert processor.vector_store is not None

    @patch('fitz.open')
    def test_extract_text_from_pdf(self, mock_fitz_open, temp_data_dir):
        """Test extracting text from PDF"""
        # Setup mock PDF document
        mock_doc = MagicMock()
        mock_doc.metadata = {
            "title": "Test Document",
            "author": "Test Author",
            "subject": "Test Subject",
            "keywords": "test, keywords",
            "creator": "Test Creator",
            "producer": "Test Producer",
            "creationDate": "2023-01-01",
            "modDate": "2023-01-02"
        }
        
        # Setup mock pages
        mock_page1 = MagicMock()
        mock_page1.get_text.return_value = "Page 1 content"
        mock_page2 = MagicMock()
        mock_page2.get_text.return_value = "Page 2 content"
        mock_doc.__len__.return_value = 2
        mock_doc.__iter__.return_value = iter([mock_page1, mock_page2])
        
        # Configure mock to return our mock document
        mock_fitz_open.return_value = mock_doc
        
        # Create processor with clean_text=False to test raw text extraction
        processor = DocumentProcessor(
            input_dir=temp_data_dir["raw"],
            output_dir=temp_data_dir["processed"],
            clean_text=False
        )
        
        # Call the method
        pdf_path = os.path.join(temp_data_dir["raw"], "test.pdf")
        text, metadata = processor.extract_text_from_pdf(pdf_path)
        
        # Verify extracted text
        assert "[Page 1] Page 1 content" in text
        assert "[Page 2] Page 2 content" in text
        
        # Verify metadata
        assert metadata["filename"] == "test.pdf"
        assert metadata["filepath"] == pdf_path
        assert metadata["page_count"] == 2
        assert metadata["title"] == "Test Document"
        assert metadata["author"] == "Test Author"
        assert metadata["subject"] == "Test Subject"
        assert metadata["keywords"] == "test, keywords"
        assert metadata["creator"] == "Test Creator"
        assert metadata["producer"] == "Test Producer"
        assert metadata["creation_date"] == "2023-01-01"
        assert metadata["modification_date"] == "2023-01-02"
        assert "document_id" in metadata

    @patch('fitz.open')
    @patch('jfkreveal.database.document_processor.clean_pdf_text')
    def test_extract_text_from_pdf_with_cleaning(self, mock_clean_pdf, mock_fitz_open, temp_data_dir):
        """Test extracting text from PDF with cleaning"""
        # Setup mock PDF document
        mock_doc = MagicMock()
        mock_doc.metadata = {"title": "Test Document"}
        
        # Setup mock pages
        mock_page = MagicMock()
        mock_page.get_text.return_value = "Page content"
        mock_doc.__len__.return_value = 1
        mock_doc.__iter__.return_value = iter([mock_page])
        
        # Configure mocks
        mock_fitz_open.return_value = mock_doc
        mock_clean_pdf.return_value = "Cleaned page content"
        
        # Create processor with clean_text=True
        processor = DocumentProcessor(
            input_dir=temp_data_dir["raw"],
            output_dir=temp_data_dir["processed"],
            clean_text=True
        )
        
        # Call the method
        pdf_path = os.path.join(temp_data_dir["raw"], "test.pdf")
        text, metadata = processor.extract_text_from_pdf(pdf_path)
        
        # Verify cleaning was called
        mock_clean_pdf.assert_called_once()
        
        # Verify cleaned text is returned
        assert text == "Cleaned page content"
        
        # Verify cleaned flag in metadata
        assert metadata["cleaned"] is True

    @patch('fitz.open')
    def test_extract_text_from_pdf_error(self, mock_fitz_open, temp_data_dir):
        """Test error handling when extracting text from PDF"""
        # Configure mock to raise an exception
        mock_fitz_open.side_effect = Exception("PDF error")
        
        processor = DocumentProcessor(
            input_dir=temp_data_dir["raw"],
            output_dir=temp_data_dir["processed"]
        )
        
        # Call the method
        pdf_path = os.path.join(temp_data_dir["raw"], "test.pdf")
        text, metadata = processor.extract_text_from_pdf(pdf_path)
        
        # Verify empty text is returned on error
        assert text == ""
        
        # Verify error is captured in metadata
        assert metadata["filename"] == "test.pdf"
        assert "error" in metadata
        assert "PDF error" in metadata["error"]

    def test_chunk_document(self, temp_data_dir):
        """Test chunking document text"""
        # Create test text with page markers
        text = (
            "[Page 1] This is the first page of content that should be split into chunks "
            "based on the specified chunk size and overlap. "
            "[Page 2] This is the second page with additional content that will form "
            "another chunk or be part of the previous chunk depending on size."
        )
        
        # Setup metadata
        metadata = {
            "filename": "test.pdf",
            "document_id": "test123"
        }
        
        # Create processor with small chunk size to ensure multiple chunks
        processor = DocumentProcessor(
            input_dir=temp_data_dir["raw"],
            output_dir=temp_data_dir["processed"],
            chunk_size=100,
            chunk_overlap=10,
            clean_text=False
        )
        
        # Call the method
        chunks = processor.chunk_document(text, metadata)
        
        # Verify chunks were created
        assert len(chunks) > 1
        
        # Verify chunk content and metadata
        for i, chunk in enumerate(chunks):
            # Check that text exists in chunk
            assert isinstance(chunk["text"], str)
            assert len(chunk["text"]) > 0
            
            # Check metadata
            assert chunk["metadata"]["document_id"] == "test123"
            assert chunk["metadata"]["chunk_id"] == f"test123-{i}"
            assert chunk["metadata"]["chunk_index"] == i
            assert chunk["metadata"]["total_chunks"] == len(chunks)
            
            # Check page numbers extraction
            if "[Page 1]" in chunk["text"]:
                assert "1" in chunk["metadata"]["pages"]
            if "[Page 2]" in chunk["text"]:
                assert "2" in chunk["metadata"]["pages"]

    @patch('jfkreveal.database.document_processor.clean_document_chunks')
    def test_chunk_document_with_cleaning(self, mock_clean_chunks, temp_data_dir):
        """Test chunking with chunk-level cleaning"""
        # Create test text
        text = "This is test content"
        
        # Setup metadata (without cleaned flag)
        metadata = {
            "filename": "test.pdf",
            "document_id": "test123"
        }
        
        # Setup mock return value
        mock_cleaned_chunks = [
            {
                "text": "Cleaned chunk",
                "metadata": {
                    "document_id": "test123",
                    "chunk_id": "test123-0",
                    "cleaned": True
                }
            }
        ]
        mock_clean_chunks.return_value = mock_cleaned_chunks
        
        # Create processor with clean_text=True
        processor = DocumentProcessor(
            input_dir=temp_data_dir["raw"],
            output_dir=temp_data_dir["processed"],
            chunk_size=100,
            chunk_overlap=10,
            clean_text=True
        )
        
        # Call the method
        chunks = processor.chunk_document(text, metadata)
        
        # Verify clean_document_chunks was called
        mock_clean_chunks.assert_called_once()
        
        # Verify cleaned chunks are returned
        assert chunks == mock_cleaned_chunks

    @patch('jfkreveal.database.document_processor.DocumentProcessor.extract_text_from_pdf')
    @patch('jfkreveal.database.document_processor.DocumentProcessor.chunk_document')
    def test_process_document(self, mock_chunk_document, mock_extract_text, temp_data_dir):
        """Test processing a single document"""
        # Setup mocks
        mock_extract_text.return_value = ("Document text", {"filename": "test.pdf", "document_id": "test123"})
        mock_chunk_document.return_value = [
            {
                "text": "Chunk 1",
                "metadata": {"document_id": "test123", "chunk_id": "test123-0"}
            },
            {
                "text": "Chunk 2",
                "metadata": {"document_id": "test123", "chunk_id": "test123-1"}
            }
        ]
        
        # Create processor
        processor = DocumentProcessor(
            input_dir=temp_data_dir["raw"],
            output_dir=temp_data_dir["processed"],
            skip_existing=False
        )
        
        # Mock file open and json dump
        m = mock_open()
        with patch('builtins.open', m):
            # Call the method
            pdf_path = os.path.join(temp_data_dir["raw"], "test.pdf")
            result = processor.process_document(pdf_path)
        
        # Verify extract_text_from_pdf and chunk_document were called
        mock_extract_text.assert_called_once_with(pdf_path)
        mock_chunk_document.assert_called_once()
        
        # Verify file was opened for writing
        expected_output_path = os.path.join(temp_data_dir["processed"], "test.json")
        m.assert_called_once_with(expected_output_path, 'w', encoding='utf-8')
        
        # Verify result is the output path
        assert result == expected_output_path

    @patch('os.path.exists')
    def test_process_document_skip_existing(self, mock_exists, temp_data_dir):
        """Test skipping already processed documents"""
        # Configure mock to indicate file exists
        mock_exists.return_value = True
        
        # Create processor with skip_existing=True
        processor = DocumentProcessor(
            input_dir=temp_data_dir["raw"],
            output_dir=temp_data_dir["processed"],
            skip_existing=True
        )
        
        # Call the method
        pdf_path = os.path.join(temp_data_dir["raw"], "test.pdf")
        result = processor.process_document(pdf_path)
        
        # Verify exists was called with correct path
        expected_output_path = os.path.join(temp_data_dir["processed"], "test.json")
        mock_exists.assert_called_with(expected_output_path)
        
        # Verify result is the existing output path
        assert result == expected_output_path

    @patch('jfkreveal.database.document_processor.DocumentProcessor.extract_text_from_pdf')
    def test_process_document_no_text(self, mock_extract_text, temp_data_dir):
        """Test handling when no text is extracted"""
        # Setup mock to return empty text
        mock_extract_text.return_value = ("", {"filename": "test.pdf", "error": "No text found"})
        
        # Create processor
        processor = DocumentProcessor(
            input_dir=temp_data_dir["raw"],
            output_dir=temp_data_dir["processed"]
        )
        
        # Call the method
        pdf_path = os.path.join(temp_data_dir["raw"], "test.pdf")
        result = processor.process_document(pdf_path)
        
        # Verify result is None when no text is extracted
        assert result is None

    @patch('jfkreveal.database.document_processor.DocumentProcessor.process_document')
    @patch('jfkreveal.utils.parallel_processor.process_documents_parallel')
    def test_process_all_documents(self, mock_parallel_processor, mock_process_document, temp_data_dir):
        """Test processing all documents in parallel"""
        # Setup mock for parallel processing to simply call the function directly
        mock_parallel_processor.side_effect = lambda func, paths, max_workers, desc: [func(path) for path in paths]
        
        # Setup process_document to return output paths
        mock_process_document.side_effect = lambda path: path.replace(".pdf", ".json").replace(temp_data_dir["raw"], temp_data_dir["processed"])
        
        # Create some test PDF files
        pdf_paths = [
            os.path.join(temp_data_dir["raw"], "doc1.pdf"),
            os.path.join(temp_data_dir["raw"], "doc2.pdf"),
            os.path.join(temp_data_dir["raw"], "doc3.pdf")
        ]
        for path in pdf_paths:
            with open(path, 'w') as f:
                f.write("Test PDF content")
        
        # Create processor
        processor = DocumentProcessor(
            input_dir=temp_data_dir["raw"],
            output_dir=temp_data_dir["processed"],
            max_workers=2
        )
        
        # Call the method
        with patch('glob.glob', return_value=pdf_paths):
            results = processor.process_all_documents()
        
        # Verify parallel_processor was called with correct arguments
        mock_parallel_processor.assert_called_once()
        
        # Verify process_document was called for each PDF
        assert mock_process_document.call_count == 3
        
        # Verify results contain output paths
        assert len(results) == 3
        for i, path in enumerate(pdf_paths):
            expected_output = path.replace(".pdf", ".json").replace(temp_data_dir["raw"], temp_data_dir["processed"])
            assert expected_output in results

    def test_check_if_embedded(self, temp_data_dir):
        """Test checking if a document has been embedded"""
        # Create a test file
        file_path = os.path.join(temp_data_dir["processed"], "test.json")
        
        # Create processor
        processor = DocumentProcessor(
            input_dir=temp_data_dir["raw"],
            output_dir=temp_data_dir["processed"]
        )
        
        # Test when marker file doesn't exist
        with patch('os.path.exists', return_value=False):
            result = processor.check_if_embedded(file_path)
            assert result is False
        
        # Test when marker file exists
        with patch('os.path.exists', return_value=True):
            result = processor.check_if_embedded(file_path)
            assert result is True

    def test_mark_as_embedded(self, temp_data_dir):
        """Test marking a document as embedded"""
        # Create a test file path
        file_path = os.path.join(temp_data_dir["processed"], "test.json")
        
        # Create processor
        processor = DocumentProcessor(
            input_dir=temp_data_dir["raw"],
            output_dir=temp_data_dir["processed"]
        )
        
        # Test marking a file
        m = mock_open()
        with patch('builtins.open', m):
            processor.mark_as_embedded(file_path)
        
        # Verify marker file was created
        marker_path = f"{file_path}.embedded"
        m.assert_called_once_with(marker_path, 'w')
        
        # Verify correct content was written
        handle = m()
        handle.write.assert_called_once_with("1")

    def test_get_processed_documents(self, temp_data_dir):
        """Test getting list of processed documents"""
        # Create processor
        processor = DocumentProcessor(
            input_dir=temp_data_dir["raw"],
            output_dir=temp_data_dir["processed"]
        )
        
        # Create some test JSON files
        json_paths = [
            os.path.join(temp_data_dir["processed"], "doc1.json"),
            os.path.join(temp_data_dir["processed"], "doc2.json")
        ]
        for path in json_paths:
            with open(path, 'w') as f:
                f.write('{"test": "data"}')
        
        # Call the method
        results = processor.get_processed_documents()
        
        # Verify results contain all JSON files
        assert len(results) == 2
        for path in json_paths:
            assert path in results 