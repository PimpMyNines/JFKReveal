"""
Parallel processing utilities for document processing.
"""
import os
from concurrent.futures import ProcessPoolExecutor
from typing import List, Callable, Any, Dict, Optional
import logging

logger = logging.getLogger(__name__)

def process_documents_parallel(
    document_paths: List[str],
    processing_function: Callable[[str], Any],
    max_workers: Optional[int] = None,
    chunk_size: int = 10
) -> List[Any]:
    """
    Process multiple documents in parallel using multiprocessing.
    
    Args:
        document_paths: List of paths to documents
        processing_function: Function to apply to each document
        max_workers: Maximum number of worker processes (None = CPU count)
        chunk_size: Number of items to send to each worker at once
    
    Returns:
        List of processing results
    """
    max_workers = max_workers or os.cpu_count()
    total_docs = len(document_paths)
    
    logger.info(f"Processing {total_docs} documents using {max_workers} workers")
    
    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Use map with chunksize for better performance with many small tasks
        results = list(executor.map(
            processing_function, 
            document_paths,
            chunksize=chunk_size
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
                documents,
                self.process_document,  # Assumes this method exists
                max_workers=max_workers
            )
        else:
            return original_process_batch(self, documents)
    
    processor_class.process_batch = enhanced_process_batch
    return processor_class 