"""
Document processor for parsing PDFs and converting them to vector embeddings.
"""
import os
import re
import logging
import json
from typing import List, Dict, Any, Optional, Tuple
import hashlib
import concurrent.futures

import fitz  # PyMuPDF
import nltk
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm import tqdm

from .text_cleaner import clean_pdf_text, clean_document_chunks
from ..utils.parallel_processor import process_documents_parallel

logger = logging.getLogger(__name__)

# Download NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class DocumentProcessor:
    """Process PDF documents for vectorization."""
    
    def __init__(
        self, 
        input_dir: str = "data/raw", 
        output_dir: str = "data/processed",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        max_workers: int = 20,  # Default to 20 parallel workers
        skip_existing: bool = True,  # Skip already processed documents by default
        vector_store = None,  # Optional vector store for immediate embedding
        clean_text: bool = True  # Whether to clean text before chunking
    ):
        """
        Initialize the document processor.
        
        Args:
            input_dir: Directory containing PDF files
            output_dir: Directory to save processed chunks
            chunk_size: Target size of text chunks
            chunk_overlap: Overlap between chunks
            max_workers: Maximum number of worker processes (default 20)
            skip_existing: Whether to skip documents that were already processed
            vector_store: Optional vector store for immediate embedding after processing
            clean_text: Whether to clean text before chunking
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_workers = max_workers
        self.skip_existing = skip_existing
        self.vector_store = vector_store
        self.clean_text = clean_text
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False
        )
    
    def extract_text_from_pdf(self, pdf_path: str) -> Tuple[str, Dict[str, Any]]:
        """
        Extract text and metadata from a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Tuple of (text content, metadata dict)
        """
        try:
            # Open the PDF
            doc = fitz.open(pdf_path)
            
            # Extract metadata
            metadata = {
                "filename": os.path.basename(pdf_path),
                "filepath": pdf_path,
                "page_count": len(doc),
                "title": doc.metadata.get("title", ""),
                "author": doc.metadata.get("author", ""),
                "subject": doc.metadata.get("subject", ""),
                "keywords": doc.metadata.get("keywords", ""),
                "creator": doc.metadata.get("creator", ""),
                "producer": doc.metadata.get("producer", ""),
                "creation_date": doc.metadata.get("creationDate", ""),
                "modification_date": doc.metadata.get("modDate", ""),
                "document_id": hashlib.md5(os.path.basename(pdf_path).encode()).hexdigest()
            }
            
            # Extract text from each page with page numbers
            full_text = []
            for page_num, page in enumerate(doc):
                text = page.get_text()
                if text.strip():  # Only add non-empty pages
                    full_text.append(f"[Page {page_num + 1}] {text}")
            
            raw_text = "\n\n".join(full_text)
            
            # Clean the text if enabled
            if self.clean_text:
                logger.debug(f"Cleaning text for PDF: {pdf_path}")
                cleaned_text = clean_pdf_text(raw_text)
                metadata["cleaned"] = True
                return cleaned_text, metadata
            
            return raw_text, metadata
            
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {e}")
            return "", {"filename": os.path.basename(pdf_path), "error": str(e)}
    
    def chunk_document(
        self, 
        text: str, 
        metadata: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Split document text into chunks with metadata.
        
        Args:
            text: Document text
            metadata: Document metadata
            
        Returns:
            List of chunk objects with text and metadata
        """
        # Split the text into chunks
        chunks = self.text_splitter.split_text(text)
        
        # Add metadata and chunk info to each chunk
        result = []
        for i, chunk_text in enumerate(chunks):
            # Extract page numbers that appear in this chunk
            page_numbers = re.findall(r'\[Page (\d+)\]', chunk_text)
            
            # Create chunk with metadata
            chunk = {
                "text": chunk_text,
                "metadata": {
                    **metadata,
                    "chunk_id": f"{metadata['document_id']}-{i}",
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "pages": page_numbers
                }
            }
            result.append(chunk)
        
        # Clean individual chunks if needed and not already cleaned at document level
        if self.clean_text and not metadata.get("cleaned", False):
            logger.debug(f"Cleaning chunks for document: {metadata.get('filename', 'unknown')}")
            result = clean_document_chunks(result)
            
        return result
    
    def process_document(self, pdf_path: str) -> Optional[str]:
        """
        Process a single PDF document and save chunks.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Path to the saved chunks file, or None if processing failed
        """
        filename = os.path.basename(pdf_path)
        output_path = os.path.join(
            self.output_dir, 
            f"{os.path.splitext(filename)[0]}.json"
        )
        
        # Skip if already processed and skip_existing is True
        if self.skip_existing and os.path.exists(output_path):
            logger.info(f"Document already processed: {output_path}")
            # If we have a vector store and file exists, check if we need to add it
            if self.vector_store and not self.check_if_embedded(output_path):
                logger.info(f"Adding previously processed document to vector store: {output_path}")
                self.vector_store.add_documents_from_file(output_path)
            return output_path
        
        logger.info(f"Processing document: {pdf_path}")
        
        # Extract text and metadata
        text, metadata = self.extract_text_from_pdf(pdf_path)
        
        if not text:
            logger.warning(f"No text extracted from {pdf_path}")
            return None
        
        # Chunk the document
        chunks = self.chunk_document(text, metadata)
        
        # Save chunks to file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
            
        logger.info(f"Saved {len(chunks)} chunks to {output_path}")
        
        # If vector store is provided, add the document immediately
        if self.vector_store:
            logger.info(f"Adding document to vector store immediately: {output_path}")
            self.vector_store.add_documents_from_file(output_path)
            self.mark_as_embedded(output_path)
            
        return output_path
    
    def check_if_embedded(self, file_path: str) -> bool:
        """
        Check if a document has already been embedded in the vector store.
        
        Args:
            file_path: Path to the processed document file
            
        Returns:
            True if the document has been embedded, False otherwise
        """
        # Create a simple marker file to track embedded documents
        marker_path = f"{file_path}.embedded"
        return os.path.exists(marker_path)
    
    def mark_as_embedded(self, file_path: str) -> None:
        """
        Mark a document as embedded in the vector store.
        
        Args:
            file_path: Path to the processed document file
        """
        # Create a simple marker file to track embedded documents
        marker_path = f"{file_path}.embedded"
        with open(marker_path, 'w') as f:
            f.write("1")
    
    def process_all_documents(self) -> List[str]:
        """
        Process all PDF documents in the input directory in parallel.
        
        Returns:
            List of paths to processed files
        """
        # List all PDF files in input directory
        pdf_files = []
        for root, _, files in os.walk(self.input_dir):
            for file in files:
                if file.lower().endswith('.pdf'):
                    pdf_files.append(os.path.join(root, file))
                    
        logger.info(f"Found {len(pdf_files)} PDF files to process")
        
        # Use parallel processing for better performance
        results = process_documents_parallel(
            document_paths=pdf_files,
            processing_function=self.process_document,
            max_workers=self.max_workers
        )
        
        # Filter out None results from failed processing
        processed_files = [path for path in results if path is not None]
            
        logger.info(f"Successfully processed {len(processed_files)}/{len(pdf_files)} files")
        
        return processed_files
        
    def get_processed_documents(self) -> List[str]:
        """
        Get a list of all processed document files without processing any new ones.
        This is useful when skipping OCR and going directly to vector embedding.
        
        Returns:
            List of paths to all processed chunk files
        """
        processed_files = []
        for root, _, files in os.walk(self.output_dir):
            for file in files:
                if file.lower().endswith('.json'):
                    processed_files.append(os.path.join(root, file))
                    
        logger.info(f"Found {len(processed_files)} already processed documents")
        return processed_files