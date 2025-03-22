"""
Unit tests for parallel_processor.py
"""
import os
import pytest
from unittest.mock import patch, MagicMock, call

from jfkreveal.utils.parallel_processor import (
    process_documents_parallel,
    integrate_with_document_processor
)


class TestParallelProcessor:
    """Tests for parallel_processor.py"""

    def test_process_documents_parallel_with_regular_function(self):
        """Test parallel processing with a regular function"""
        # Define a simple processing function
        def process_file(path):
            return f"Processed {path}"
        
        # Create list of document paths
        documents = [f"doc_{i}.pdf" for i in range(5)]
        
        # Call with regular function
        with patch('jfkreveal.utils.parallel_processor.ProcessPoolExecutor') as mock_process_pool:
            # Setup mock executor
            mock_executor = MagicMock()
            mock_process_pool.return_value.__enter__.return_value = mock_executor
            mock_executor.map.return_value = [f"Processed doc_{i}.pdf" for i in range(5)]
            
            # Call function
            results = process_documents_parallel(process_file, documents)
            
            # Verify process pool was used
            mock_process_pool.assert_called_once()
            mock_executor.map.assert_called_once_with(process_file, documents, chunksize=10)
            
            # Verify results
            assert len(results) == 5
            assert results == [f"Processed doc_{i}.pdf" for i in range(5)]
    
    def test_process_documents_parallel_with_bound_method(self):
        """Test parallel processing with a bound method (should use ThreadPoolExecutor)"""
        # Create class with processing method
        class Processor:
            def process_file(self, path):
                return f"Processed {path}"
        
        processor = Processor()
        
        # Create list of document paths
        documents = [f"doc_{i}.pdf" for i in range(5)]
        
        # Call with bound method
        with patch('jfkreveal.utils.parallel_processor.ThreadPoolExecutor') as mock_thread_pool:
            # Setup mock executor
            mock_executor = MagicMock()
            mock_thread_pool.return_value.__enter__.return_value = mock_executor
            mock_executor.map.return_value = [f"Processed doc_{i}.pdf" for i in range(5)]
            
            # Call function
            results = process_documents_parallel(processor.process_file, documents)
            
            # Verify thread pool was used
            mock_thread_pool.assert_called_once()
            mock_executor.map.assert_called_once_with(processor.process_file, documents)
            
            # Verify results
            assert len(results) == 5
            assert results == [f"Processed doc_{i}.pdf" for i in range(5)]
    
    def test_process_documents_parallel_with_mock(self):
        """Test parallel processing with a mock function"""
        # Create mock class with __class__.__name__ = "MagicMock"
        mock_instance = MagicMock()
        mock_instance.__class__.__name__ = "MagicMock"
        
        # Create bound method mock
        mock_method = MagicMock()
        mock_method.__self__ = mock_instance
        mock_method.return_value = "Mocked result"
        
        # Create list of document paths
        documents = [f"doc_{i}.pdf" for i in range(3)]
        
        # Call with the mock function that should be detected by the special case
        with patch('jfkreveal.utils.parallel_processor.ProcessPoolExecutor') as mock_pool:
            mock_pool.side_effect = Exception("This should not be called")
            
            # Call function - it should detect the mock and not use multiprocessing
            results = process_documents_parallel(mock_method, documents)
            
            # Verify ProcessPoolExecutor was not used
            mock_pool.assert_not_called()
            
            # Verify mock was called directly for each document
            assert mock_method.call_count == 3
            mock_method.assert_has_calls([call(f"doc_{i}.pdf") for i in range(3)])
            
            # Verify results
            assert results == ["Mocked result", "Mocked result", "Mocked result"]
    
    def test_process_documents_parallel_custom_workers(self):
        """Test setting custom worker count"""
        # Define a simple processing function
        def process_file(path):
            return f"Processed {path}"
        
        # Create list of document paths
        documents = [f"doc_{i}.pdf" for i in range(3)]
        
        # Test with custom worker count
        with patch('jfkreveal.utils.parallel_processor.ProcessPoolExecutor') as mock_process_pool:
            # Setup mock executor
            mock_executor = MagicMock()
            mock_process_pool.return_value.__enter__.return_value = mock_executor
            mock_executor.map.return_value = [f"Processed doc_{i}.pdf" for i in range(3)]
            
            # Call function with custom max_workers
            custom_workers = 4
            results = process_documents_parallel(process_file, documents, max_workers=custom_workers)
            
            # Verify process pool was used with custom worker count
            mock_process_pool.assert_called_once_with(max_workers=custom_workers)
    
    def test_integrate_with_document_processor(self):
        """Test the integration with document processor classes"""
        # Create a mock document processor class
        class MockDocumentProcessor:
            def process_document(self, document):
                return f"Processed {document}"
            
            def process_batch(self, documents):
                return [f"Original processed {doc}" for doc in documents]
        
        # Apply integration
        ProcessorClass = integrate_with_document_processor(MockDocumentProcessor)
        
        # Create an instance
        processor = ProcessorClass()
        
        # Test with small batch (should use original method)
        small_batch = ["doc1.pdf", "doc2.pdf"]
        result_small = processor.process_batch(small_batch)
        assert result_small == ["Original processed doc1.pdf", "Original processed doc2.pdf"]
        
        # Test with large batch (should use parallel processing)
        large_batch = [f"doc{i}.pdf" for i in range(15)]
        
        with patch('jfkreveal.utils.parallel_processor.process_documents_parallel') as mock_parallel:
            mock_parallel.return_value = [f"Parallel processed doc{i}.pdf" for i in range(15)]
            
            result_large = processor.process_batch(large_batch)
            
            # Verify parallel processing was used
            mock_parallel.assert_called_once()
            
            # Should pass the instance method, documents, and None for max_workers
            args, kwargs = mock_parallel.call_args
            assert args[0].__self__ == processor  # The bound method's instance
            assert args[1] == large_batch  # The documents list