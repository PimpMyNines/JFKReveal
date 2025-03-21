"""
Text cleaning module for OCR-extracted text from PDF documents.

This module provides functions to clean and normalize text extracted from scanned PDF documents
before embedding. It handles common OCR issues such as:
- Removal of scanning artifacts
- Fixing broken words and line breaks
- Normalizing whitespace and punctuation
- Detecting and fixing common OCR errors
- Removing headers, footers, and page numbers
"""
import re
import logging
from typing import List, Dict, Any, Optional, Tuple

import nltk
from nltk.tokenize import sent_tokenize
import spacy

logger = logging.getLogger(__name__)

# Load spaCy model for improved text processing
try:
    nlp = spacy.load("en_core_web_sm")
    SPACY_AVAILABLE = True
except (ImportError, OSError):
    logger.warning("spaCy model 'en_core_web_sm' not available. Some advanced cleaning features will be disabled.")
    SPACY_AVAILABLE = False

# Download NLTK resources if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)


class TextCleaner:
    """Clean and normalize OCR text from scanned documents."""
    
    def __init__(
        self,
        remove_headers_footers: bool = True,
        fix_line_breaks: bool = True,
        normalize_whitespace: bool = True,
        fix_ocr_errors: bool = True,
        use_spacy: bool = SPACY_AVAILABLE,
        preserve_page_boundaries: bool = True,
    ):
        """
        Initialize the text cleaner.
        
        Args:
            remove_headers_footers: Whether to attempt to remove headers and footers
            fix_line_breaks: Whether to fix broken line breaks in paragraphs
            normalize_whitespace: Whether to normalize whitespace
            fix_ocr_errors: Whether to attempt to fix common OCR errors
            use_spacy: Whether to use spaCy for advanced NLP-based cleaning
            preserve_page_boundaries: Whether to preserve page boundary markers
        """
        self.remove_headers_footers = remove_headers_footers
        self.fix_line_breaks = fix_line_breaks
        self.normalize_whitespace = normalize_whitespace
        self.fix_ocr_errors = fix_ocr_errors
        self.use_spacy = use_spacy and SPACY_AVAILABLE
        self.preserve_page_boundaries = preserve_page_boundaries
        
        # Common OCR error patterns to fix
        self.ocr_error_patterns = [
            # Fix common OCR errors
            (r'([a-z])\.([a-z])', r'\1\2'),  # Remove periods incorrectly inserted in words
            (r'([a-z]),([a-z])', r'\1\2'),   # Remove commas incorrectly inserted in words
            (r'([a-z]);([a-z])', r'\1\2'),   # Remove semicolons incorrectly inserted in words
            # Fix broken words
            (r'(\w+)-\s*\n\s*(\w+)', r'\1\2'),  # Words broken across lines
            # Fix common character substitutions
            (r'0', 'o', re.IGNORECASE),  # Zero instead of 'o'
            (r'l', 'i', re.IGNORECASE),  # Lowercase L instead of 'i'
            # More patterns can be added as needed
        ]
        
        # Patterns for identifying headers and footers
        self.header_footer_patterns = [
            r'^.*CONFIDENTIAL.*$',
            r'^.*CLASSIFIED.*$',
            r'^.*SECRET.*$',
            r'^.*Page \d+( of \d+)?.*$',
            r'^\d+$',  # Isolated page numbers
            r'^\s*\[\s*Page\s+\d+\s*\]\s*$',  # Page markers in square brackets
        ]
    
    def clean_text(self, text: str) -> str:
        """
        Apply all selected cleaning steps to the text.
        
        Args:
            text: Text to clean
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Store original page boundary markers if needed
        if self.preserve_page_boundaries:
            page_boundaries = re.findall(r'\[Page \d+\]', text)
        
        # Apply cleaning steps in sequence
        if self.remove_headers_footers:
            text = self.remove_headers_and_footers(text)
        
        if self.fix_ocr_errors:
            text = self.fix_common_ocr_errors(text)
            
        if self.fix_line_breaks:
            text = self.fix_paragraph_breaks(text)
            
        if self.normalize_whitespace:
            text = self.normalize_all_whitespace(text)
        
        if self.use_spacy:
            text = self.apply_spacy_cleaning(text)
        
        # Reinsert page boundary markers if they were removed
        if self.preserve_page_boundaries and page_boundaries:
            # This is a simplified approach - in a real implementation,
            # you would need to track where page boundaries should be inserted
            for marker in page_boundaries:
                if marker not in text:
                    # For now, we'll just ensure the markers exist somewhere in the text
                    text = f"{marker}\n{text}"
        
        return text
    
    def remove_headers_and_footers(self, text: str) -> str:
        """
        Remove headers and footers from the text.
        
        Args:
            text: Text to process
            
        Returns:
            Text with headers and footers removed
        """
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            should_remove = False
            for pattern in self.header_footer_patterns:
                if re.match(pattern, line.strip(), re.IGNORECASE):
                    should_remove = True
                    break
            
            if not should_remove:
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def fix_common_ocr_errors(self, text: str) -> str:
        """
        Fix common OCR errors in the text.
        
        Args:
            text: Text to process
            
        Returns:
            Text with OCR errors fixed
        """
        cleaned_text = text
        
        for pattern, replacement in self.ocr_error_patterns:
            cleaned_text = re.sub(pattern, replacement, cleaned_text)
            
        return cleaned_text
    
    def fix_paragraph_breaks(self, text: str) -> str:
        """
        Fix broken paragraphs and line breaks.
        
        Args:
            text: Text to process
            
        Returns:
            Text with fixed paragraphs
        """
        # Replace single line breaks that don't end with punctuation
        # but keep paragraph breaks (double line breaks)
        text = re.sub(r'([a-zA-Z0-9,;:])\n(?!\n|$)([a-zA-Z0-9])', r'\1 \2', text)
        
        # Preserve paragraph structure
        paragraphs = text.split('\n\n')
        cleaned_paragraphs = []
        
        for para in paragraphs:
            # Join lines within paragraphs
            para = re.sub(r'\n(?!\n|$)', ' ', para)
            cleaned_paragraphs.append(para)
        
        return '\n\n'.join(cleaned_paragraphs)
    
    def normalize_all_whitespace(self, text: str) -> str:
        """
        Normalize all whitespace in the text.
        
        Args:
            text: Text to process
            
        Returns:
            Text with normalized whitespace
        """
        # Replace multiple spaces with a single space
        text = re.sub(r' {2,}', ' ', text)
        
        # Normalize line breaks (preserve paragraph breaks)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Remove spaces at the beginning and end of lines
        text = re.sub(r'^ +| +$', '', text, flags=re.MULTILINE)
        
        return text.strip()
    
    def apply_spacy_cleaning(self, text: str) -> str:
        """
        Apply spaCy-based advanced cleaning.
        
        Args:
            text: Text to process
            
        Returns:
            Cleaned text
        """
        if not self.use_spacy:
            return text
            
        # Process text with spaCy
        doc = nlp(text)
        
        # Reconstruct text with corrected spacing and sentence boundaries
        sentences = []
        for sent in doc.sents:
            # Clean up the sentence
            clean_sent = str(sent).strip()
            if clean_sent:
                sentences.append(clean_sent)
        
        # Join sentences with proper spacing
        return ' '.join(sentences)


def clean_document_chunks(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Clean text in document chunks before vectorization.
    
    Args:
        chunks: List of document chunks with text and metadata
        
    Returns:
        Cleaned document chunks
    """
    cleaner = TextCleaner()
    cleaned_chunks = []
    
    for chunk in chunks:
        # Create a copy of the chunk to avoid modifying the original
        clean_chunk = chunk.copy()
        
        # Clean the text
        clean_chunk['text'] = cleaner.clean_text(chunk['text'])
        
        # Add an indicator that this text has been cleaned
        if 'metadata' in clean_chunk:
            clean_chunk['metadata'] = clean_chunk['metadata'].copy()
            clean_chunk['metadata']['cleaned'] = True
        
        cleaned_chunks.append(clean_chunk)
    
    return cleaned_chunks


def clean_pdf_text(text: str) -> str:
    """
    Clean raw text extracted from a PDF.
    
    Args:
        text: Raw text extracted from PDF
        
    Returns:
        Cleaned text
    """
    cleaner = TextCleaner()
    return cleaner.clean_text(text) 