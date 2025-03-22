"""
Unit tests for the FindingsReport Pydantic models and LangChain integration
"""
import pytest
from unittest.mock import patch, MagicMock
import json

from jfkreveal.summarization.findings_report import (
    ExecutiveSummaryResponse,
    DetailedFindingsResponse,
    SuspectsAnalysisResponse,
    CoverupAnalysisResponse,
    FindingsReport
)
from langchain_core.exceptions import LangChainException


class TestFindingsReportModels:
    """Test cases for FindingsReport Pydantic models and LangChain integration"""

    def test_executive_summary_response_model(self):
        """Test ExecutiveSummaryResponse Pydantic model validation"""
        # Test valid data
        valid_data = {
            "overview": "This is an overview of the findings.",
            "significant_evidence": ["Evidence 1", "Evidence 2"],
            "potential_government_involvement": ["Involvement 1", "Involvement 2"],
            "credible_theories": ["Theory 1", "Theory 2"],
            "likely_culprits": ["Culprit 1", "Culprit 2"],
            "alternative_suspects": ["Suspect 1", "Suspect 2"],
            "redaction_patterns": ["Pattern 1", "Pattern 2"],
            "document_credibility": "The documents are credible."
        }
        
        model = ExecutiveSummaryResponse(**valid_data)
        assert model.overview == valid_data["overview"]
        assert model.significant_evidence == valid_data["significant_evidence"]
        assert model.document_credibility == valid_data["document_credibility"]
        
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

    def test_detailed_findings_response_model(self):
        """Test DetailedFindingsResponse Pydantic model validation"""
        # Test valid data
        valid_data = {
            "topic_analyses": {"Topic 1": "Analysis 1", "Topic 2": "Analysis 2"},
            "timeline": "Timeline of events",
            "key_individuals": {"Person 1": "Role 1", "Person 2": "Role 2"},
            "theory_analysis": {"Theory 1": "Analysis 1", "Theory 2": "Analysis 2"},
            "inconsistencies": ["Inconsistency 1", "Inconsistency 2"],
            "information_withholding": ["Withholding 1", "Withholding 2"],
            "evidence_credibility": {"Evidence 1": "High", "Evidence 2": "Low"},
            "likely_scenarios": ["Scenario 1", "Scenario 2"],
            "primary_suspects": {"Suspect 1": ["Evidence A", "Evidence B"]},
            "alternative_suspects_analysis": {
                "Suspect 2": {"evidence": ["Evidence C"], "credibility": "Medium"}
            }
        }
        
        model = DetailedFindingsResponse(**valid_data)
        assert model.topic_analyses == valid_data["topic_analyses"]
        assert model.timeline == valid_data["timeline"]
        assert model.primary_suspects == valid_data["primary_suspects"]
        
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

    def test_suspects_analysis_response_model(self):
        """Test SuspectsAnalysisResponse Pydantic model validation"""
        # Test valid data
        valid_data = {
            "primary_culprits": ["Culprit 1", "Culprit 2"],
            "supporting_evidence": {"Culprit 1": ["Evidence A", "Evidence B"]},
            "evidence_strength": "The evidence is strong and convincing.",
            "case_weaknesses": ["Weakness 1", "Weakness 2"],
            "alternative_suspects": [
                {"name": "Suspect A", "evidence": ["Evidence X"], "credibility": "High"},
                {"name": "Suspect B", "evidence": ["Evidence Y"], "credibility": "Medium"}
            ],
            "collaborations": ["Collaboration 1", "Collaboration 2"],
            "government_involvement": "Government was involved in these ways...",
            "conspiracy_analysis": "Analysis of conspiracy theories..."
        }
        
        model = SuspectsAnalysisResponse(**valid_data)
        assert model.primary_culprits == valid_data["primary_culprits"]
        assert model.supporting_evidence == valid_data["supporting_evidence"]
        assert model.government_involvement == valid_data["government_involvement"]
        
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

    def test_coverup_analysis_response_model(self):
        """Test CoverupAnalysisResponse Pydantic model validation"""
        # Test valid data
        valid_data = {
            "information_suppression": ["Suppression 1", "Suppression 2"],
            "redaction_patterns": {"Pattern 1": "Description 1", "Pattern 2": "Description 2"},
            "narrative_inconsistencies": ["Inconsistency 1", "Inconsistency 2"],
            "information_timeline": "Timeline of information releases...",
            "agency_behavior": {"CIA": ["Behavior 1", "Behavior 2"], "FBI": ["Behavior 3"]},
            "evidence_destruction": ["Destruction 1", "Destruction 2"],
            "witness_treatment": ["Treatment 1", "Treatment 2"],
            "document_handling": ["Handling 1", "Handling 2"],
            "coverup_motives": ["Motive 1", "Motive 2"],
            "beneficiaries": ["Beneficiary 1", "Beneficiary 2"]
        }
        
        model = CoverupAnalysisResponse(**valid_data)
        assert model.information_suppression == valid_data["information_suppression"]
        assert model.redaction_patterns == valid_data["redaction_patterns"]
        assert model.agency_behavior == valid_data["agency_behavior"]
        
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

    @pytest.fixture
    def temp_data_dir(self, tmp_path):
        """Create a temporary directory structure for testing"""
        # Create directory structure
        analysis_dir = tmp_path / "analysis"
        reports_dir = tmp_path / "reports"
        analysis_dir.mkdir()
        reports_dir.mkdir()
        
        # Return paths for use in tests
        return {
            "root": str(tmp_path),
            "analysis": str(analysis_dir),
            "reports": str(reports_dir)
        }

    @patch('langchain_openai.ChatOpenAI')
    def test_executive_summary_langchain_integration(self, mock_chat_openai, temp_data_dir):
        """Test generating executive summary with LangChain structured output"""
        # Mock LLM and response
        mock_llm = MagicMock()
        mock_chat_openai.return_value = mock_llm
        
        # Mock structured output method
        mock_structured_llm = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured_llm
        
        # Create example response
        example_response = ExecutiveSummaryResponse(
            overview="Test overview",
            significant_evidence=["Evidence 1", "Evidence 2"],
            potential_government_involvement=["Involvement 1"],
            credible_theories=["Theory 1"],
            likely_culprits=["Culprit 1"],
            alternative_suspects=["Suspect 1"],
            redaction_patterns=["Pattern 1"],
            document_credibility="Document credibility assessment"
        )
        
        # Mock LLM response
        mock_structured_llm.invoke.return_value = example_response
        
        # Create report instance
        report = FindingsReport(
            analysis_dir=temp_data_dir["root"] + "/analysis",
            output_dir=temp_data_dir["root"] + "/reports"
        )
        
        # Replace report's LLM with our mock
        report.llm = mock_llm
        
        # Mock load_analyses and _save_report_file to avoid file system operations
        report.load_analyses = MagicMock(return_value=[{"topic": "Test Topic"}])
        report._save_report_file = MagicMock()
        
        # Call the method
        result = report.generate_executive_summary([{"topic": "Test Topic"}])
        
        # Verify LLM was configured with structured output
        mock_llm.with_structured_output.assert_called_once()
        
        # Verify structured output was used
        mock_structured_llm.invoke.assert_called_once()
        
        # Verify markdown conversion
        assert "# Executive Summary" in result
        assert "## Overview" in result
        assert "Test overview" in result

    def test_langchain_retry_mechanism(self, temp_data_dir):
        """Test retry mechanism for LangChain API calls"""
        # Create a mock for the structured output response
        mock_response = MagicMock()
        mock_response.content = "Fallback unstructured response"
        
        # Create a mock LLM that simulates the retry behavior
        mock_llm = MagicMock()
        
        # Create the report instance directly
        report = FindingsReport(
            analysis_dir=temp_data_dir["root"] + "/analysis",
            output_dir=temp_data_dir["root"] + "/reports"
        )
        
        # We'll manually call the fallback logic since the retry decorator
        # is hard to mock in a unit test
        
        # Mock _save_report_file to avoid file operations
        report._save_report_file = MagicMock()
        
        # Mock the with_structured_output method to raise an exception
        mock_structured_llm = MagicMock()
        mock_structured_llm.invoke.side_effect = LangChainException("Rate limit")
        mock_llm.with_structured_output.return_value = mock_structured_llm
        
        # Mock the regular invoke to return a successful response
        mock_llm.invoke.return_value = mock_response
        
        # Set the mock LLM
        report.llm = mock_llm
        
        # Call the method
        result = report.generate_executive_summary([{"topic": "Test Topic"}])
        
        # Verify that structured output was attempted
        mock_llm.with_structured_output.assert_called_once()
        
        # Verify that on exception, fallback to unstructured was used
        mock_llm.invoke.assert_called_once()
        
        # Verify final result contains the fallback content
        assert "Fallback unstructured response" in result 