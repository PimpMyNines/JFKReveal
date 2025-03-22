"""
Test fixtures for JFKReveal project.
"""
import os
import tempfile
import json
import shutil
import pytest
from typing import List, Dict, Any, Optional
from unittest.mock import MagicMock, patch

# Project imports
from jfkreveal.database.text_cleaner import TextCleaner
from jfkreveal.database.document_processor import DocumentProcessor
from jfkreveal.database.vector_store import VectorStore
from jfkreveal.analysis.document_analyzer import (
    DocumentAnalysisItem, 
    DocumentAnalysisResult, 
    AnalyzedDocument, 
    TopicSummary, 
    TopicAnalysis
)
from jfkreveal.summarization.findings_report import (
    ExecutiveSummaryResponse,
    DetailedFindingsResponse,
    SuspectsAnalysisResponse,
    CoverupAnalysisResponse
)

# ------------------------------------------------------------------------------
# Sample data fixtures
# ------------------------------------------------------------------------------

@pytest.fixture
def sample_text():
    """Sample text for testing."""
    return """CONFIDENTIAL
    
    This document contains information related to the events of November 22, 1963.
    
    [Page 1]
    
    Lee Harvey Oswald was observed at the Texas School Book Depository.
    
    The investigation revealed that multiple witnesses reported hearing shots from
    the grassy knoll area, contradicting the official findings.
    
    [Page 2]
    
    Further analysis of ballistic evidence suggests the possibility of
    additional shooters, though this remains unproven.
    
    CLASSIFIED
    """


@pytest.fixture
def sample_document_chunks():
    """Sample document chunks for testing."""
    return [
        {
            "text": "Lee Harvey Oswald was observed at the Texas School Book Depository.",
            "metadata": {
                "document_id": "doc_001",
                "chunk_id": "doc_001-1",
                "pages": ["1"]
            }
        },
        {
            "text": "Multiple witnesses reported hearing shots from the grassy knoll area.",
            "metadata": {
                "document_id": "doc_001",
                "chunk_id": "doc_001-2",
                "pages": ["1", "2"]
            }
        },
        {
            "text": "Further analysis of ballistic evidence suggests additional shooters.",
            "metadata": {
                "document_id": "doc_001",
                "chunk_id": "doc_001-3",
                "pages": ["2"]
            }
        }
    ]

# ------------------------------------------------------------------------------
# Directory and file fixtures
# ------------------------------------------------------------------------------

@pytest.fixture
def temp_dir():
    """Temporary directory for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def temp_data_dir(temp_dir):
    """Creates a temporary data directory structure for testing."""
    # Create directory structure
    raw_dir = os.path.join(temp_dir, "raw")
    processed_dir = os.path.join(temp_dir, "processed")
    vector_dir = os.path.join(temp_dir, "vectordb")
    analysis_dir = os.path.join(temp_dir, "analysis")
    reports_dir = os.path.join(temp_dir, "reports")
    
    # Create all required directories
    dirs = [raw_dir, processed_dir, vector_dir, analysis_dir, reports_dir]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    
    return {
        "root": temp_dir,
        "raw": raw_dir,
        "processed": processed_dir,
        "vector": vector_dir,
        "analysis": analysis_dir,
        "reports": reports_dir
    }


@pytest.fixture
def sample_processed_file(temp_data_dir, sample_document_chunks):
    """Create a sample processed file for testing."""
    file_path = os.path.join(temp_data_dir["processed"], "sample_doc.json")
    
    with open(file_path, "w") as f:
        json.dump(sample_document_chunks, f)
    
    return file_path

# ------------------------------------------------------------------------------
# Component fixtures
# ------------------------------------------------------------------------------

@pytest.fixture
def text_cleaner():
    """TextCleaner instance for testing."""
    return TextCleaner()


@pytest.fixture
def document_processor(temp_data_dir):
    """DocumentProcessor instance for testing."""
    return DocumentProcessor(
        input_dir=temp_data_dir["raw"],
        output_dir=temp_data_dir["processed"],
        chunk_size=500,
        chunk_overlap=50,
        max_workers=2,
        skip_existing=True,
        clean_text=True
    )


@pytest.fixture
def vector_store(temp_data_dir):
    """VectorStore instance for testing with fake embeddings."""
    return VectorStore(
        persist_directory=temp_data_dir["vector"],
        embedding_provider="fake"  # Use fake embeddings for testing
    )

# ------------------------------------------------------------------------------
# LangChain and LLM mocking fixtures
# ------------------------------------------------------------------------------

@pytest.fixture
def mock_llm():
    """Mock LangChain LLM for testing."""
    mock = MagicMock()
    
    # Create a structured output mock that can be chained
    mock_structured_output = MagicMock()
    mock.with_structured_output.return_value = mock_structured_output
    
    # Create an invoke mock for the structured output
    mock_structured_output.invoke.return_value = MagicMock()
    
    # Create a regular invoke for the base LLM
    mock.invoke.return_value = MagicMock()
    mock.invoke.return_value.content = "Mock LLM response"
    
    return mock


@pytest.fixture
def mock_retry():
    """Mock tenacity.retry decorator for testing."""
    mock = MagicMock()
    # Setup mock retry behavior to pass through to the function directly
    mock.side_effect = lambda *args, **kwargs: lambda func: func
    return mock


@pytest.fixture
def mock_openai_with_backoff(mock_llm, mock_retry):
    """Fixture that provides a patched decorator for retry mechanisms."""
    with patch('langchain_openai.ChatOpenAI') as mock_chat_openai, \
         patch('tenacity.retry') as patched_retry:
        mock_chat_openai.return_value = mock_llm
        patched_retry.side_effect = mock_retry.side_effect
        yield mock_llm

# ------------------------------------------------------------------------------
# Model response fixtures
# ------------------------------------------------------------------------------

@pytest.fixture
def sample_document_analysis_result():
    """Sample DocumentAnalysisResult for testing."""
    return DocumentAnalysisResult(
        key_individuals=[
            DocumentAnalysisItem(
                information="Lee Harvey Oswald",
                quote="Lee Harvey Oswald was observed at the Book Depository",
                page="1"
            ),
            DocumentAnalysisItem(
                information="Jack Ruby",
                quote="Jack Ruby was known to have connections",
                page="2"
            )
        ],
        government_agencies=[
            DocumentAnalysisItem(
                information="CIA",
                quote="CIA was monitoring Oswald",
                page="2"
            ),
            DocumentAnalysisItem(
                information="FBI",
                quote="FBI investigation concluded",
                page="3"
            )
        ],
        suspicious_activities=[
            DocumentAnalysisItem(
                information="Suspicious phone call",
                quote="A phone call was made minutes before",
                page="4"
            )
        ]
    )


@pytest.fixture
def sample_analyzed_document(sample_document_analysis_result):
    """Sample AnalyzedDocument for testing."""
    return AnalyzedDocument(
        text="Sample document text for analysis testing.",
        metadata={
            "document_id": "doc123",
            "chunk_id": "doc123-chunk1",
            "filename": "jfk_document.pdf",
            "pages": ["1", "2"]
        },
        analysis=sample_document_analysis_result
    )


@pytest.fixture
def sample_topic_summary():
    """Sample TopicSummary for testing."""
    return TopicSummary(
        key_findings=["Oswald was involved", "Evidence of multiple shooters"],
        consistent_information=["Oswald was at the Book Depository"],
        contradictions=["Reports on number of shots fired"],
        potential_evidence=["Missing bullet fragments"],
        missing_information=["CIA files still classified"],
        assassination_theories=["Grassy knoll theory"],
        credibility="medium",
        document_references={
            "Oswald sighting": ["doc1-1"],
            "Multiple shooters": ["doc2-1"]
        }
    )


@pytest.fixture
def sample_topic_analysis(sample_topic_summary, sample_analyzed_document):
    """Sample TopicAnalysis for testing."""
    return TopicAnalysis(
        topic="JFK Assassination",
        summary=sample_topic_summary,
        document_analyses=[sample_analyzed_document, sample_analyzed_document],
        num_documents=2
    )


@pytest.fixture
def sample_executive_summary_response():
    """Sample ExecutiveSummaryResponse for testing."""
    return ExecutiveSummaryResponse(
        overview="This is an overview of the findings.",
        significant_evidence=["Evidence 1", "Evidence 2"],
        potential_government_involvement=["Involvement 1", "Involvement 2"],
        credible_theories=["Theory 1", "Theory 2"],
        likely_culprits=["Culprit 1", "Culprit 2"],
        alternative_suspects=["Suspect 1", "Suspect 2"],
        redaction_patterns=["Pattern 1", "Pattern 2"],
        document_credibility="The documents are credible."
    )


@pytest.fixture
def sample_detailed_findings_response():
    """Sample DetailedFindingsResponse for testing."""
    return DetailedFindingsResponse(
        topic_analyses={"Topic 1": "Analysis 1", "Topic 2": "Analysis 2"},
        timeline="Timeline of events",
        key_individuals={"Person 1": "Role 1", "Person 2": "Role 2"},
        theory_analysis={"Theory 1": "Analysis 1", "Theory 2": "Analysis 2"},
        inconsistencies=["Inconsistency 1", "Inconsistency 2"],
        information_withholding=["Withholding 1", "Withholding 2"],
        evidence_credibility={"Evidence 1": "High", "Evidence 2": "Low"},
        likely_scenarios=["Scenario 1", "Scenario 2"],
        primary_suspects={"Suspect 1": ["Evidence A", "Evidence B"]},
        alternative_suspects_analysis={
            "Suspect 2": {"evidence": ["Evidence C"], "credibility": "Medium"}
        }
    )


@pytest.fixture
def sample_suspects_analysis_response():
    """Sample SuspectsAnalysisResponse for testing."""
    return SuspectsAnalysisResponse(
        primary_culprits=["Culprit 1", "Culprit 2"],
        supporting_evidence={"Culprit 1": ["Evidence A", "Evidence B"]},
        evidence_strength="The evidence is strong and convincing.",
        case_weaknesses=["Weakness 1", "Weakness 2"],
        alternative_suspects=[
            {"name": "Suspect A", "evidence": ["Evidence X"], "credibility": "High"},
            {"name": "Suspect B", "evidence": ["Evidence Y"], "credibility": "Medium"}
        ],
        collaborations=["Collaboration 1", "Collaboration 2"],
        government_involvement="Government was involved in these ways...",
        conspiracy_analysis="Analysis of conspiracy theories..."
    )


@pytest.fixture
def sample_coverup_analysis_response():
    """Sample CoverupAnalysisResponse for testing."""
    return CoverupAnalysisResponse(
        information_suppression=["Suppression 1", "Suppression 2"],
        redaction_patterns={"Pattern 1": "Description 1", "Pattern 2": "Description 2"},
        narrative_inconsistencies=["Inconsistency 1", "Inconsistency 2"],
        information_timeline="Timeline of information releases...",
        agency_behavior={"CIA": ["Behavior 1", "Behavior 2"], "FBI": ["Behavior 3"]},
        evidence_destruction=["Destruction 1", "Destruction 2"],
        witness_treatment=["Treatment 1", "Treatment 2"],
        document_handling=["Handling 1", "Handling 2"],
        coverup_motives=["Motive 1", "Motive 2"],
        beneficiaries=["Beneficiary 1", "Beneficiary 2"]
    )

# ------------------------------------------------------------------------------
# HTTP and API mocking fixtures
# ------------------------------------------------------------------------------

@pytest.fixture
def mock_http_response():
    """Mock HTTP response for testing."""
    mock = MagicMock()
    mock.status_code = 200
    mock.text = "<html><body>Test content</body></html>"
    mock.content = b"Test binary content"
    mock.headers = {"Content-Type": "text/html", "Content-Length": "100"}
    mock.raise_for_status = MagicMock()
    return mock


@pytest.fixture
def mock_pdf_response():
    """Mock HTTP response with PDF content for testing."""
    mock = MagicMock()
    mock.status_code = 200
    # A minimal valid PDF header
    mock.content = b"%PDF-1.5\nTest PDF content"
    mock.headers = {"Content-Type": "application/pdf", "Content-Length": "25"}
    mock.raise_for_status = MagicMock()
    mock.iter_content.return_value = [mock.content]
    return mock 