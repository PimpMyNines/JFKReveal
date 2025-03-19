"""
Document processor for parsing PDFs and converting them to vector embeddings.
"""
import os
import re
import logging
import json
from typing import List, Dict, Any, Optional, Tuple
import hashlib

import fitz  # PyMuPDF
import nltk
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm import tqdm

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
        chunk_overlap: int = 200
    ):
        """
        Initialize the document processor.
        
        Args:
            input_dir: Directory containing PDF files
            output_dir: Directory to save processed chunks
            chunk_size: Target size of text chunks
            chunk_overlap: Overlap between chunks
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
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
            
            return "\n\n".join(full_text), metadata
            
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
        
        # Skip if already processed
        if os.path.exists(output_path):
            logger.info(f"Document already processed: {output_path}")
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
        return output_path
    
    def process_all_documents(self) -> List[str]:
        """
        Process all PDF documents in the input directory.
        
        Returns:
            List of paths to all processed chunk files
        """
        # Find all PDF files
        pdf_files = []
        for root, _, files in os.walk(self.input_dir):
            for file in files:
                if file.lower().endswith('.pdf'):
                    pdf_files.append(os.path.join(root, file))
                    
        logger.info(f"Found {len(pdf_files)} PDF files to process")
        
        # Process each file
        processed_files = []
        for pdf_path in tqdm(pdf_files, desc="Processing PDFs"):
            output_path = self.process_document(pdf_path)
            if output_path:
                processed_files.append(output_path)
                
        logger.info(f"Processed {len(processed_files)} documents")
        return processed_files