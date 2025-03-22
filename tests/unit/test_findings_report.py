"""
Unit tests for the FindingsReport class
"""
import os
import pytest
from unittest.mock import patch, MagicMock, mock_open

from jfkreveal.summarization.findings_report import FindingsReport

class TestFindingsReport:
    """Test the FindingsReport class"""

    def test_init(self, temp_data_dir):
        """Test initialization of FindingsReport"""
        # Create report instance
        report = FindingsReport(
            analysis_dir=temp_data_dir["root"] + "/analysis",
            output_dir=temp_data_dir["root"] + "/reports"
        )
        
        # Verify attributes
        assert report.analysis_dir == temp_data_dir["root"] + "/analysis"
        assert report.output_dir == temp_data_dir["root"] + "/reports"
        assert report.model_name == "gpt-4o"
        assert report.temperature == 0.1
        assert report.max_retries == 5
        
        # Verify output directory was created
        assert os.path.exists(temp_data_dir["root"] + "/reports")

    @patch('jfkreveal.summarization.findings_report.FindingsReport._save_report_file')
    @patch('jfkreveal.summarization.findings_report.ChatOpenAI')
    def test_generate_executive_summary(self, mock_chat_openai, mock_save_report, temp_data_dir):
        """Test generating executive summary"""
        # Setup mock LLM
        mock_llm = MagicMock()
        mock_chat_openai.return_value = mock_llm
        
        # Mock LLM response
        mock_response = MagicMock()
        mock_response.content = "Executive Summary Content"
        mock_llm.invoke.return_value = mock_response
        
        # Create report instance
        report = FindingsReport(
            analysis_dir=temp_data_dir["root"] + "/analysis",
            output_dir=temp_data_dir["root"] + "/reports"
        )
        
        # Mock loading analyses
        mock_analyses = [
            {
                "topic": "Test Topic 1",
                "summary": {
                    "key_findings": ["Finding 1", "Finding 2"],
                    "potential_evidence": ["Evidence 1", "Evidence 2"],
                    "credibility": "High"
                }
            }
        ]
        report.load_analyses = MagicMock(return_value=mock_analyses)
        
        # Call method
        result = report.generate_executive_summary(mock_analyses)
        
        # Verify result
        assert "Executive Summary Content" in result
        
        # Verify LLM was called with correct prompt
        mock_llm.invoke.assert_called_once()
        
        # Verify report was saved
        mock_save_report.assert_called_once()

    @patch('jfkreveal.summarization.findings_report.FindingsReport._save_report_file')
    @patch('jfkreveal.summarization.findings_report.ChatOpenAI')
    def test_generate_detailed_findings(self, mock_chat_openai, mock_save_report, temp_data_dir):
        """Test generating detailed findings"""
        # Setup mock LLM
        mock_llm = MagicMock()
        mock_chat_openai.return_value = mock_llm
        
        # Create a mock response for structured output fallback
        mock_response = MagicMock()
        mock_response.content = "Detailed Findings Content"
        
        # Mock the regular invoke method for fallback
        mock_llm.invoke.return_value = mock_response
        
        # Create report instance
        report = FindingsReport(
            analysis_dir=temp_data_dir["root"] + "/analysis",
            output_dir=temp_data_dir["root"] + "/reports"
        )
        
        # Replace report's LLM with our mock
        report.llm = mock_llm
        
        # Mock loading analyses
        mock_analyses = [
            {
                "topic": "Test Topic 1",
                "documents": [
                    {"document_id": "doc1", "title": "Document 1"}
                ],
                "entities": [
                    {"name": "Person 1", "type": "PERSON"}
                ],
                "summary": {
                    "key_findings": ["Finding 1", "Finding 2"],
                    "potential_evidence": ["Evidence 1", "Evidence 2"]
                }
            }
        ]
        report.load_analyses = MagicMock(return_value=mock_analyses)
        
        # Set up structured output to fail so it falls back to regular invoke
        mock_structured_llm = MagicMock()
        mock_structured_llm.invoke.side_effect = Exception("Failed structured output")
        mock_llm.with_structured_output.return_value = mock_structured_llm
        
        # Call method
        result = report.generate_detailed_findings(mock_analyses)
        
        # Verify that structured output was attempted but failed
        mock_llm.with_structured_output.assert_called_once()
        
        # Verify that regular invoke was used as fallback
        mock_llm.invoke.assert_called_once()
        
        # Verify result contains the fallback content
        assert "Detailed Findings Content" in result
        
        # Verify report was saved
        mock_save_report.assert_called_once()

    def test_build_document_urls(self, temp_data_dir):
        """Test building document URLs"""
        # Create some PDF files
        raw_dir = os.path.join(temp_data_dir["root"], "raw")
        os.makedirs(raw_dir, exist_ok=True)
        
        # Create empty PDF files
        pdf_files = ["doc1.pdf", "doc2.pdf", "doc3.pdf"]
        for pdf_file in pdf_files:
            with open(os.path.join(raw_dir, pdf_file), 'w') as f:
                f.write("Dummy PDF")
        
        # Create report instance
        report = FindingsReport(
            analysis_dir=temp_data_dir["root"] + "/analysis",
            output_dir=temp_data_dir["root"] + "/reports",
            raw_docs_dir=raw_dir
        )
        
        # Call method
        document_urls = report._build_document_urls()
        
        # Verify URLs were built correctly
        assert len(document_urls) == 3
        assert "doc1" in document_urls
        assert document_urls["doc1"].endswith("doc1.pdf")
        assert document_urls["doc2"].endswith("doc2.pdf")
        assert document_urls["doc3"].endswith("doc3.pdf")

    @patch('builtins.open', new_callable=mock_open)
    def test_save_report_file(self, mock_file, temp_data_dir):
        """Test saving report file"""
        # Create report instance
        report = FindingsReport(
            analysis_dir=temp_data_dir["root"] + "/analysis",
            output_dir=temp_data_dir["root"] + "/reports"
        )
        
        # Test content
        content = "# Test Report\n\nThis is a test report."
        
        # Call method
        report._save_report_file(content, "test_report.md")
        
        # Verify file was opened for writing
        mock_file.assert_called_once_with(
            os.path.join(temp_data_dir["root"] + "/reports", "test_report.md"),
            'w',
            encoding='utf-8'
        )
        
        # Verify content was written
        mock_file.return_value.write.assert_called_once_with(content)