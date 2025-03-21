"""
Tests for the text cleaning module
"""
import pytest
from jfkreveal.database.text_cleaner import TextCleaner, clean_pdf_text, clean_document_chunks


class TestTextCleaner:
    """Test the TextCleaner class functionality"""
    
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
        
        assert "CONFIDENTIAL" not in cleaned  # Header should be removed
        assert "document contains information" in cleaned  # OCR errors fixed
        assert "He was carrying a rifle" in cleaned  # Line breaks fixed
        assert "Page 2" not in cleaned  # Page number removed
        assert "[Page 1]" in cleaned  # Page marker preserved
    
    def test_fix_common_ocr_errors(self):
        """Test fixing common OCR errors"""
        cleaner = TextCleaner()
        
        # Test text with common OCR errors
        text_with_errors = "Oswa.ld was arr,ested at the th;eater"
        
        cleaned = cleaner.fix_common_ocr_errors(text_with_errors)
        
        assert cleaned == "Oswald was arrested at the theater"
    
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
        
        assert "The suspect was seen on the sixth floor. He was carrying a weapon." in fixed
        assert "Multiple witnesses confirmed this observation." in fixed
        assert "\n\n" in fixed  # Paragraph break preserved
    
    def test_remove_headers_and_footers(self):
        """Test removing headers and footers"""
        cleaner = TextCleaner()
        
        text_with_headers = """CLASSIFIED
        
        Key information about the investigation.
        
        SECRET
        
        The Warren Commission investigated the assassination.
        
        Page 3 of 10
        
        This concludes the report."""
        
        cleaned = cleaner.remove_headers_and_footers(text_with_headers)
        
        assert "CLASSIFIED" not in cleaned
        assert "SECRET" not in cleaned
        assert "Page 3 of 10" not in cleaned
        assert "Key information about the investigation." in cleaned
        assert "The Warren Commission investigated the assassination." in cleaned
        assert "This concludes the report." in cleaned
    
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
        
        # Check that the chunks were cleaned
        assert "CONFIDENTIAL" not in cleaned_chunks[0]["text"]
        assert "some OCR errors" in cleaned_chunks[0]["text"]
        assert "Page 5" not in cleaned_chunks[1]["text"]
        assert "line breaks" in cleaned_chunks[1]["text"]
        
        # Check that cleaned flag was added to metadata
        assert cleaned_chunks[0]["metadata"]["cleaned"] is True
        assert cleaned_chunks[1]["metadata"]["cleaned"] is True
    
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
        
        assert "CLASSIFIED" not in cleaned
        assert "[Page 1]" in cleaned
        assert "Page 2" not in cleaned
        assert "investigation revealed important new evidence" in cleaned
        assert "Witnesses provided testimony" in cleaned 