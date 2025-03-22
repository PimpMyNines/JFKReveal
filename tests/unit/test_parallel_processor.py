"""
Unit tests for the parallel processing utilities
"""
import os
import logging
import pytest
from unittest.mock import patch, MagicMock, call
from concurrent.futures import ProcessPoolExecutor

from jfkreveal.utils.parallel_processor import process_documents_parallel, integrate_with_document_processor
from tests.unit.test_helpers import process_doc, identity_processor


# Helper function for testing, not a test itself
def process_serially(document_paths, processing_function):
    """Serial implementation for testing."""
    return [processing_function(doc) for doc in document_paths]


class TestParallelProcessor:
    """Test the parallel processing utilities"""

    @patch('jfkreveal.utils.parallel_processor.ProcessPoolExecutor')
    def test_process_documents_parallel_mocked(self, mock_pool):
        """Test parallel processing with mocks"""
        # Setup mock ProcessPoolExecutor
        mock_instance = MagicMock()
        mock_pool.return_value.__enter__.return_value = mock_instance
        
        # Set up mock return from map function
        expected_results = ["processed_doc1.pdf", "processed_doc2.pdf", "processed_doc3.pdf"]
        mock_instance.map.return_value = expected_results
        
        # Input data
        document_paths = ["doc1.pdf", "doc2.pdf", "doc3.pdf"]
        
        # Run the function
        results = process_documents_parallel(
            document_paths,
            process_doc,
            max_workers=2,
            chunk_size=5
        )
        
        # Verify ProcessPoolExecutor was created
        mock_pool.assert_called_once_with(max_workers=2)
        
        # Verify map was called with correct arguments
        mock_instance.map.assert_called_once()
        args, kwargs = mock_instance.map.call_args
        assert args[0] == process_doc
        assert list(args[1]) == document_paths
        assert kwargs.get('chunksize') == 5
        
        # Verify the results are correct
        assert results == expected_results
    
    @patch('jfkreveal.utils.parallel_processor.ProcessPoolExecutor')
    @patch('os.cpu_count')
    def test_process_documents_parallel_default_workers_mocked(self, mock_cpu_count, mock_pool):
        """Test default worker count"""
        # Setup CPU count
        mock_cpu_count.return_value = 8
        
        # Setup mock ProcessPoolExecutor
        mock_instance = MagicMock()
        mock_pool.return_value.__enter__.return_value = mock_instance
        
        # Set up mock return from map function
        mock_instance.map.return_value = ["doc1.pdf"]
        
        # Input data
        document_paths = ["doc1.pdf"]
        
        # Run the function
        results = process_documents_parallel(document_paths, identity_processor)
        
        # Verify CPU count was queried
        mock_cpu_count.assert_called_once()
        
        # Verify ProcessPoolExecutor was created with CPU count
        mock_pool.assert_called_once_with(max_workers=8)
        
        # Verify the results are correct
        assert results == ["doc1.pdf"]
    
    def test_integrate_with_document_processor(self):
        """Test enhancing a processor class with parallel processing"""
        # Create mock class
        class MockProcessor:
            def process_document(self, document):
                return f"processed_{document}"
                
            def process_batch(self, documents):
                return [self.process_document(doc) for doc in documents]
        
        # Store original method for comparison
        original_process_batch = MockProcessor.process_batch
        
        # Apply the integration
        enhanced_class = integrate_with_document_processor(MockProcessor)
        
        # Verify the class was enhanced (process_batch was replaced)
        assert enhanced_class.process_batch != original_process_batch
        
        # Create an instance
        processor = enhanced_class()
        
        # Test with small batch (should use original method)
        with patch.object(processor, 'process_document') as mock_process_document:
            processor.process_batch(["doc1", "doc2"])
            assert mock_process_document.call_count == 2
        
        # Test with large batch (should use parallel processing)
        documents = [f"doc{i}" for i in range(20)]
        with patch('jfkreveal.utils.parallel_processor.process_documents_parallel') as mock_parallel:
            processor.process_batch(documents)
            mock_parallel.assert_called_once_with(
                documents,
                processor.process_document,
                max_workers=None
            )