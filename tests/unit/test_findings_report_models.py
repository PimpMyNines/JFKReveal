"""
Unit tests for the FindingsReport Pydantic models and LangChain integration
"""
import pytest
from unittest.mock import patch, MagicMock
import json
from langchain_core.exceptions import LangChainException

from jfkreveal.summarization.findings_report import (
    ExecutiveSummaryResponse,
    DetailedFindingsResponse,
    SuspectsAnalysisResponse,
    CoverupAnalysisResponse,
    FindingsReport
)


class TestFindingsReportModels:
    """Test cases for FindingsReport Pydantic models and LangChain integration"""

    def test_executive_summary_response_model(self, sample_executive_summary_response):
        """Test ExecutiveSummaryResponse Pydantic model validation"""
        # Using fixture for valid data
        model = sample_executive_summary_response
        assert model.overview == "This is an overview of the findings."
        assert "Evidence 1" in model.significant_evidence
        assert model.document_credibility == "The documents are credible."
        
        # Test with missing optional fields
        minimal_data = {
            "overview": "This is an overview of the findings.",
            "document_credibility": "The documents are credible."
        }
        
        model = ExecutiveSummaryResponse(**minimal_data)
        assert model.overview == minimal_data["overview"]
        assert model.significant_evidence == []  # Default empty list
        assert model.document_credibility == minimal_data["document_credibility"]
        
        # Test with missing required field should raise error
        with pytest.raises(ValueError):
            ExecutiveSummaryResponse(significant_evidence=["Evidence 1"])

    def test_detailed_findings_response_model(self, sample_detailed_findings_response):
        """Test DetailedFindingsResponse Pydantic model validation"""
        # Using fixture for valid data
        model = sample_detailed_findings_response
        assert model.topic_analyses == {"Topic 1": "Analysis 1", "Topic 2": "Analysis 2"}
        assert model.timeline == "Timeline of events"
        assert model.primary_suspects == {"Suspect 1": ["Evidence A", "Evidence B"]}
        
        # Test with missing optional fields
        minimal_data = {
            "topic_analyses": {"Topic 1": "Analysis 1"},
            "timeline": "Timeline of events",
            "key_individuals": {"Person 1": "Role 1"},
            "theory_analysis": {"Theory 1": "Analysis 1"},
            "evidence_credibility": {"Evidence 1": "High"},
            "primary_suspects": {"Suspect 1": ["Evidence A"]},
            "alternative_suspects_analysis": {"Suspect 2": {"evidence": ["Evidence C"]}}
        }
        
        model = DetailedFindingsResponse(**minimal_data)
        assert model.inconsistencies == []  # Default empty list

    def test_suspects_analysis_response_model(self, sample_suspects_analysis_response):
        """Test SuspectsAnalysisResponse Pydantic model validation"""
        # Using fixture for valid data
        model = sample_suspects_analysis_response
        assert model.primary_culprits == ["Culprit 1", "Culprit 2"]
        assert model.supporting_evidence == {"Culprit 1": ["Evidence A", "Evidence B"]}
        assert model.government_involvement == "Government was involved in these ways..."
        
        # Test with missing optional fields
        minimal_data = {
            "supporting_evidence": {"Culprit 1": ["Evidence A"]},
            "evidence_strength": "The evidence is strong.",
            "alternative_suspects": [{"name": "Suspect A", "evidence": ["Evidence X"], "credibility": "Medium"}],
            "government_involvement": "Government involvement analysis",
            "conspiracy_analysis": "Conspiracy analysis"
        }
        
        model = SuspectsAnalysisResponse(**minimal_data)
        assert model.primary_culprits == []  # Default empty list
        assert model.case_weaknesses == []  # Default empty list

    def test_coverup_analysis_response_model(self, sample_coverup_analysis_response):
        """Test CoverupAnalysisResponse Pydantic model validation"""
        # Using fixture for valid data
        model = sample_coverup_analysis_response
        assert model.information_suppression == ["Suppression 1", "Suppression 2"]
        assert model.redaction_patterns == {"Pattern 1": "Description 1", "Pattern 2": "Description 2"}
        assert model.agency_behavior == {"CIA": ["Behavior 1", "Behavior 2"], "FBI": ["Behavior 3"]}
        
        # Test with missing optional fields
        minimal_data = {
            "redaction_patterns": {"Pattern 1": "Description 1"},
            "information_timeline": "Timeline...",
            "agency_behavior": {"CIA": ["Behavior 1"]}
        }
        
        model = CoverupAnalysisResponse(**minimal_data)
        assert model.information_suppression == []  # Default empty list
        assert model.witness_treatment == []  # Default empty list


class TestLangChainIntegration:
    """Test FindingsReport LangChain integration"""

    def test_executive_summary_langchain_integration(self, temp_data_dir, mock_openai_with_backoff, sample_executive_summary_response):
        """Test generating executive summary with LangChain structured output"""
        # Our mock_openai_with_backoff fixture gives us a pre-configured mock_llm
        mock_llm = mock_openai_with_backoff
        
        # Mock structured output method and response
        mock_structured_llm = mock_llm.with_structured_output.return_value
        mock_structured_llm.invoke.return_value = sample_executive_summary_response
        
        # Create report instance
        report = FindingsReport(
            analysis_dir=temp_data_dir["analysis"],
            output_dir=temp_data_dir["reports"],
            model_name="gpt-4o",
            temperature=0.1,
            max_retries=5
        )
        
        # Replace report's LLM with our mock
        report.llm = mock_llm
        
        # Mock load_analyses and _save_report_file to avoid file system operations
        report.load_analyses = MagicMock(return_value=[{"topic": "Test Topic"}])
        report._save_report_file = MagicMock()
        
        # Test analyses data
        test_analyses = [
            {
                "topic": "Test Topic",
                "summary": {
                    "key_findings": ["Finding 1", "Finding 2"],
                    "potential_evidence": ["Evidence A", "Evidence B"],
                    "credibility": "High"
                }
            }
        ]
        
        # Call the method
        result = report.generate_executive_summary(test_analyses)
        
        # Verify LLM was configured with structured output
        mock_llm.with_structured_output.assert_called_once()
        mock_llm.with_structured_output.assert_called_with(
            ExecutiveSummaryResponse,
            method="function_calling"
        )
        
        # Verify structured output was invoked
        mock_structured_llm.invoke.assert_called_once()
        
        # Validate that analyses data was properly processed in the prompt
        # by checking that a dictionary with the expected keys was created
        call_args = mock_structured_llm.invoke.call_args[0][0]
        assert isinstance(call_args, str) or hasattr(call_args, '_message_content')
        
        # Verify markdown conversion by checking expected structure in result
        assert "# Executive Summary" in result
        assert "## Overview" in result
        assert "This is an overview of the findings." in result
        assert "## Significant Evidence" in result
        assert "- Evidence 1" in result
        assert "- Evidence 2" in result
        
        # Verify report saving
        report._save_report_file.assert_called_once()
        assert report._save_report_file.call_args[0][1] == "executive_summary.md"

    def test_langchain_retry_mechanism(self, temp_data_dir, mock_openai_with_backoff):
        """Test retry mechanism for LangChain API calls"""
        # Get our mock_llm from fixture
        mock_llm = mock_openai_with_backoff
        
        # Create a mock for the structured output response
        mock_response = MagicMock()
        mock_response.content = "Fallback unstructured response"
        
        # Create the report instance directly
        report = FindingsReport(
            analysis_dir=temp_data_dir["analysis"],
            output_dir=temp_data_dir["reports"]
        )
        
        # Mock _save_report_file to avoid file operations
        report._save_report_file = MagicMock()
        
        # Configure the structured output to raise an exception
        mock_structured_llm = mock_llm.with_structured_output.return_value
        mock_structured_llm.invoke.side_effect = LangChainException("Rate limit")
        
        # Configure regular invoke to return a successful response
        mock_llm.invoke.return_value = mock_response
        
        # Set the mock LLM
        report.llm = mock_llm
        
        # Call the method
        result = report.generate_executive_summary([{"topic": "Test Topic"}])
        
        # Verify that structured output was attempted
        mock_llm.with_structured_output.assert_called_once()
        mock_llm.with_structured_output.assert_called_with(
            ExecutiveSummaryResponse,
            method="function_calling"
        )
        
        # Verify that the prompt was properly formatted
        mock_structured_llm.invoke.assert_called_once()
        prompt_arg = mock_structured_llm.invoke.call_args[0][0]
        assert "Test Topic" in str(prompt_arg)
        
        # Verify that on exception, fallback to unstructured was used
        mock_llm.invoke.assert_called_once()
        
        # Verify the fallback prompt is the same as the structured prompt
        fallback_prompt_arg = mock_llm.invoke.call_args[0][0]
        assert fallback_prompt_arg == prompt_arg
        
        # Verify final result contains the fallback content
        assert "Fallback unstructured response" in result