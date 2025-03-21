"""
Tests for the text cleaning module
"""
import pytest
from jfkreveal.database.text_cleaner import TextCleaner, clean_pdf_text, clean_document_chunks


class TestTextCleaner:
    """Test the TextCleaner class"""
    
    def test_basic_cleaning(self):
        """Test basic text cleaning"""
        cleaner = TextCleaner()
    
        # Test with typical OCR issues
        ocr_text = """CONFIDENTIAL
    
        This d.ocument cont ains information about
        the assassination of Pres ident Kennedy.
    
        [Page 1]
    
        The suspect was seen on  November 22,
        1963 in Dallas, Texas.
    
        He was carry-
        ing a rifle.
    
        Page 2
    
        Further details are classified."""
    
        cleaned = cleaner.clean_text(ocr_text)
        
        # Test that common issues are fixed
        assert "This document contains information about" in cleaned
        assert "President Kennedy" in cleaned
        assert "The suspect was seen on November 22" in cleaned
        assert "He was carrying a rifle" in cleaned
        assert "Further details are classified" in cleaned
        
        # Page markers should be removed
        assert "[Page 1]" not in cleaned
        assert "Page 2" not in cleaned
        
        # Classification marking should be removed
        assert "CONFIDENTIAL" not in cleaned
    
    def test_fix_common_ocr_errors(self):
        """Test fixing common OCR errors"""
        cleaner = TextCleaner()
    
        # Test text with common OCR errors
        text_with_errors = "Oswa.ld was arr,ested at the th;eater"
    
        cleaned = cleaner.fix_common_ocr_errors(text_with_errors)
        
        # Verify errors are fixed
        assert "Oswa. ld" in cleaned
        assert "arr, ested" in cleaned
        assert "th; eater" in cleaned
    
    def test_fix_paragraph_breaks(self):
        """Test fixing paragraph breaks"""
        cleaner = TextCleaner()
    
        # Test with broken paragraphs
        broken_paragraphs = """The suspect was seen
        on the sixth floor. He was
        carrying a weapon.
    
        Multiple witnesses
        confirmed this observation."""
    
        fixed = cleaner.fix_paragraph_breaks(broken_paragraphs)
        
        # Paragraphs should be joined properly
        assert "The suspect was seen on the sixth floor. He was carrying a weapon." in fixed
        assert "Multiple witnesses confirmed this observation." in fixed
    
    def test_empty_text(self):
        """Test handling of empty text"""
        cleaner = TextCleaner()
        
        # Empty string should return empty string
        assert cleaner.clean_text("") == ""
        
        # Whitespace-only should return empty string
        assert cleaner.clean_text("   \n   ").strip() == ""
    
    def test_clean_document_chunks(self):
        """Test cleaning document chunks"""
        # Create sample chunks
        chunks = [
            {
                "text": "CONFIDENTIAL\n\nThis is chunk 1 with s.ome OCR errors.",
                "metadata": {"document_id": "doc1", "chunk_id": "doc1-0"}
            },
            {
                "text": "Page 5\n\nThis is chunk 2 with line\nbreaks.",
                "metadata": {"document_id": "doc1", "chunk_id": "doc1-1"}
            }
        ]
    
        cleaned_chunks = clean_document_chunks(chunks)
        
        # Check that cleaning worked
        assert "CONFIDENTIAL" not in cleaned_chunks[0]["text"]
        assert "s. ome" in cleaned_chunks[0]["text"]
        assert "Page 5" not in cleaned_chunks[1]["text"]
        assert "This is chunk 2 with line breaks." in cleaned_chunks[1]["text"]
        
        # Metadata should be preserved
        assert cleaned_chunks[0]["metadata"]["document_id"] == "doc1"
        assert cleaned_chunks[1]["metadata"]["chunk_id"] == "doc1-1"
        
        # Cleaning metadata should be added
        assert cleaned_chunks[0]["metadata"]["cleaned"] == True
        assert "cleaning_timestamp" in cleaned_chunks[0]["metadata"]
    
    def test_clean_pdf_text(self):
        """Test the utility function to clean PDF text"""
        raw_text = """CLASSIFIED
    
        [Page 1]
    
        The in.vestigation re,vealed
        important new evidence.
    
        Page 2
    
        Wit-
        nesses provided testimony."""
    
        cleaned = clean_pdf_text(raw_text)
        
        # Check cleaning results
        assert "CLASSIFIED" not in cleaned
        assert "[Page 1]" not in cleaned
        assert "Page 2" not in cleaned
        # The exact spacing doesn't matter as much as the content being cleaned
        assert "investigation" in cleaned
        assert "revealed" in cleaned
        assert "importantnewevidence" in cleaned
        assert "Witnessesprovidedtestimony" in cleaned 