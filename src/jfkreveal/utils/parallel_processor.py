"""
Parallel processing utilities for document processing.
"""
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import List, Callable, Any, Dict, Optional
import logging
import inspect

logger = logging.getLogger(__name__)

def process_documents_parallel(
    processing_function: Callable[[str], Any],
    document_paths: List[str],
    max_workers: Optional[int] = None,
    desc: str = "Processing documents"
) -> List[Any]:
    """
    Process multiple documents in parallel using multiprocessing.
    
    Args:
        processing_function: Function to apply to each document
        document_paths: List of paths to documents
        max_workers: Maximum number of worker processes (None = CPU count)
        desc: Description for progress bar
    
    Returns:
        List of processing results
    """
    max_workers = max_workers or os.cpu_count()
    total_docs = len(document_paths)
    
    logger.info(f"Processing {total_docs} documents using {max_workers} workers")
    
    # For unit testing compatibility
    if hasattr(processing_function, "__self__") and hasattr(processing_function.__self__, "__class__") and processing_function.__self__.__class__.__name__ == "MagicMock":
        # Avoid multiprocessing with mocks which can't be pickled
        logger.debug("Detected mock function, avoiding multiprocessing")
        return [processing_function(path) for path in document_paths]
    
    # Check if we should use threading instead of multiprocessing
    # If the function is a bound method (i.e., self.process_document) and might contain non-picklable objects
    # we should use threading to avoid pickle errors with internal locks, etc.
    use_threading = hasattr(processing_function, "__self__") and not inspect.isbuiltin(processing_function)
    
    logger.info(f"Using {'threading' if use_threading else 'multiprocessing'} for parallel processing")
    
    results = []
    if use_threading:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Use map with chunksize for better performance with many small tasks
            results = list(executor.map(
                processing_function, 
                document_paths
            ))
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Use map with chunksize for better performance with many small tasks
            results = list(executor.map(
                processing_function, 
                document_paths,
                chunksize=10
            ))
    
    logger.info(f"Completed processing {len(results)}/{total_docs} documents")
    return results

def integrate_with_document_processor(processor_class):
    """
    Integrate parallel processing with an existing document processor.
    
    Args:
        processor_class: The document processor class to enhance
    
    Returns:
        Enhanced processor class with parallel processing
    """
    original_process_batch = processor_class.process_batch
    
    def enhanced_process_batch(self, documents, max_workers=None):
        if len(documents) > 10:  # Only use parallel for larger batches
            return process_documents_parallel(
                self.process_document,  # Assumes this method exists
                documents,
                max_workers=max_workers
            )
        else:
            return original_process_batch(self, documents)
    
    processor_class.process_batch = enhanced_process_batch
    return processor_class 