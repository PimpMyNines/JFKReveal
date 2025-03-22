"""
Document processor for parsing PDFs and converting them to vector embeddings.

This module handles both text-based PDFs and scanned (image-based) PDFs.
For image-based PDFs, OCR is applied using Tesseract via pytesseract.
"""
import os
import re
import logging
import json
import io
import hashlib
from typing import List, Dict, Any, Optional, Tuple, Union
import concurrent.futures

import fitz  # PyMuPDF
import nltk
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm import tqdm

# Optional imports for OCR
try:
    import pytesseract
    from PIL import Image
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("pytesseract or Pillow not installed. OCR functionality will be disabled.")

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
        batch_size: int = 50,  # Number of documents to process in a single batch
        skip_existing: bool = True,  # Skip already processed documents by default
        vector_store = None,  # Optional vector store for immediate embedding
        clean_text: bool = True,  # Whether to clean text before chunking
        use_ocr: bool = True,  # Whether to use OCR for pages without text
        ocr_resolution_scale: float = 2.0,  # Scale factor for OCR resolution (higher = better quality but slower)
        ocr_language: str = "eng"  # Language for OCR
    ):
        """
        Initialize the document processor.
        
        Args:
            input_dir: Directory containing PDF files
            output_dir: Directory to save processed chunks
            chunk_size: Target size of text chunks
            chunk_overlap: Overlap between chunks
            max_workers: Maximum number of worker processes (default 20)
            batch_size: Number of documents to process in a single batch (default 50)
            skip_existing: Whether to skip documents that were already processed
            vector_store: Optional vector store for immediate embedding after processing
            clean_text: Whether to clean text before chunking
            use_ocr: Whether to use OCR for scanned pages without text
            ocr_resolution_scale: Scale factor for OCR resolution (higher = better but slower)
            ocr_language: Language for OCR processing
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_workers = max_workers
        self.batch_size = batch_size
        self.skip_existing = skip_existing
        self.vector_store = vector_store
        self.clean_text = clean_text
        
        # OCR configuration
        self.use_ocr = use_ocr and OCR_AVAILABLE
        self.ocr_resolution_scale = ocr_resolution_scale
        self.ocr_language = ocr_language
        
        if self.use_ocr and not OCR_AVAILABLE:
            logger.warning("OCR dependencies not available. OCR functionality is disabled.")
            self.use_ocr = False
        
        # Ensure output directory exists
        from ..utils.file_utils import ensure_directory_exists
        ensure_directory_exists(output_dir)
        
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
            
            # Import file_utils here to avoid circular imports
            from ..utils.file_utils import get_document_id
            
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
                "document_id": get_document_id(pdf_path),
                "ocr_applied": False  # Track if OCR was applied
            }
            
            # Extract text from each page with page numbers
            full_text = []
            ocr_page_count = 0
            total_page_count = len(doc)
            
            for page_num, page in enumerate(doc):
                # Try normal text extraction first
                text = page.get_text().strip()
                
                # If page has no text but OCR is enabled, check if it has images
                if not text and self.use_ocr:
                    # Check if page has images
                    image_list = page.get_images()
                    
                    if image_list:
                        logger.info(f"Applying OCR to page {page_num + 1} of {pdf_path}")
                        
                        # Render page to an image for OCR at higher resolution
                        scale_matrix = fitz.Matrix(self.ocr_resolution_scale, self.ocr_resolution_scale)
                        pix = page.get_pixmap(matrix=scale_matrix)
                        
                        # Convert to PIL Image and apply OCR
                        img = Image.open(io.BytesIO(pix.tobytes()))
                        
                        # Apply OCR with specified language
                        ocr_text = pytesseract.image_to_string(
                            img, 
                            lang=self.ocr_language,
                            config='--psm 1'  # Automatic page segmentation with OSD
                        )
                        
                        text = ocr_text.strip()
                        ocr_page_count += 1
                        metadata["ocr_applied"] = True
                
                if text:  # Only add non-empty pages
                    full_text.append(f"[Page {page_num + 1}] {text}")
            
            # Close the document to free resources
            doc.close()
            
            if metadata["ocr_applied"]:
                metadata["ocr_pages"] = ocr_page_count
                metadata["ocr_percentage"] = round((ocr_page_count / total_page_count) * 100, 2)
                logger.info(f"Applied OCR to {ocr_page_count}/{total_page_count} pages ({metadata['ocr_percentage']}%) in {pdf_path}")
            
            raw_text = "\n\n".join(full_text)
            
            # If we got no text at all, log a warning
            if not raw_text:
                logger.warning(f"No text extracted from {pdf_path} (OCR {'enabled' if self.use_ocr else 'disabled'})")
                return "", {"filename": os.path.basename(pdf_path), "error": "No text extracted"}
            
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
        # Import file_utils here to avoid circular imports
        from ..utils.file_utils import get_output_path, check_if_embedded
        
        # Get the output path for this document
        output_path = get_output_path(pdf_path, self.output_dir, "json")
        
        # Skip if already processed and skip_existing is True
        if self.skip_existing and os.path.exists(output_path):
            logger.info(f"Document already processed: {output_path}")
            # If we have a vector store and file exists, check if we need to add it
            if self.vector_store and not check_if_embedded(output_path):
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
        # Import file_utils here to avoid circular imports
        from ..utils.file_utils import check_if_embedded
        return check_if_embedded(file_path)
    
    def mark_as_embedded(self, file_path: str) -> None:
        """
        Mark a document as embedded in the vector store.
        
        Args:
            file_path: Path to the processed document file
        """
        # Import file_utils here to avoid circular imports
        from ..utils.file_utils import mark_as_embedded
        mark_as_embedded(file_path)
    
    def process_all_documents(self) -> List[str]:
        """
        Process all PDF documents in the input directory in parallel.
        
        Returns:
            List of paths to processed files
        """
        # Import file_utils here to avoid circular imports
        from ..utils.file_utils import list_pdf_files
        
        # List all PDF files in input directory
        pdf_files = list_pdf_files(self.input_dir, recursive=True)
                    
        logger.info(f"Found {len(pdf_files)} PDF files to process")
        
        # Process in batches if needed to manage memory usage
        if self.batch_size and self.batch_size < len(pdf_files):
            logger.info(f"Processing documents in batches of {self.batch_size}")
            
            # Split into batches
            batches = [pdf_files[i:i + self.batch_size] for i in range(0, len(pdf_files), self.batch_size)]
            processed_files = []
            
            for batch_idx, batch in enumerate(batches):
                logger.info(f"Processing batch {batch_idx+1}/{len(batches)} ({len(batch)} documents)")
                
                # Process this batch in parallel
                batch_results = process_documents_parallel(
                    processing_function=self.process_document,
                    document_paths=batch,
                    max_workers=self.max_workers,
                    desc=f"Batch {batch_idx+1}/{len(batches)}"
                )
                
                # Filter out None results from failed processing
                batch_processed = [path for path in batch_results if path is not None]
                processed_files.extend(batch_processed)
                
                logger.info(f"Completed batch {batch_idx+1}: processed {len(batch_processed)}/{len(batch)} documents")
        else:
            # Use parallel processing for better performance (all documents at once)
            results = process_documents_parallel(
                processing_function=self.process_document,
                document_paths=pdf_files,
                max_workers=self.max_workers,
                desc="Processing PDF documents"
            )
            
            # Filter out None results from failed processing
            processed_files = [path for path in results if path is not None]
        
        logger.info(f"Successfully processed {len(processed_files)}/{len(pdf_files)} files")
        
        return processed_files
    
    def process_all_documents_sequential(self) -> List[str]:
        """
        Process all PDF documents in the input directory sequentially (no parallel processing).
        This method is primarily for testing purposes where parallel processing might cause issues.
        
        Returns:
            List of paths to processed files
        """
        # Import file_utils here to avoid circular imports
        from ..utils.file_utils import list_pdf_files
        
        # List all PDF files in input directory
        pdf_files = list_pdf_files(self.input_dir, recursive=True)
                    
        logger.info(f"Found {len(pdf_files)} PDF files to process")
        
        # Process files sequentially
        results = [self.process_document(path) for path in pdf_files]
        
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
        # Import file_utils here to avoid circular imports
        from ..utils.file_utils import list_json_files
        
        processed_files = list_json_files(self.output_dir, recursive=True)
                    
        logger.info(f"Found {len(processed_files)} already processed documents")
        return processed_files