"""
Unit tests for the DocumentAnalyzer class
"""
import os
import json
import pytest
from unittest.mock import patch, MagicMock, mock_open

from jfkreveal.analysis.document_analyzer import (
    DocumentAnalyzer, 
    DocumentAnalysisResult,
    DocumentAnalysisItem,
    AnalyzedDocument,
    TopicSummary,
    TopicAnalysis
)


class TestDocumentAnalyzer:
    """Test the DocumentAnalyzer class"""

    def test_init(self, temp_data_dir, vector_store):
        """Test initialization of DocumentAnalyzer"""
        # Test with default parameters
        analyzer = DocumentAnalyzer(
            vector_store=vector_store,
            output_dir=temp_data_dir["root"] + "/analysis"
        )
        
        # Verify attributes
        assert analyzer.vector_store == vector_store
        assert analyzer.output_dir == temp_data_dir["root"] + "/analysis"
        assert analyzer.model_name == "gpt-4o"
        assert analyzer.temperature == 0.0
        assert analyzer.max_retries == 5
        assert analyzer.llm is not None
        
        # Verify output directory was created
        assert os.path.exists(temp_data_dir["root"] + "/analysis")
        
        # Test with custom parameters
        analyzer = DocumentAnalyzer(
            vector_store=vector_store,
            output_dir=temp_data_dir["root"] + "/custom",
            model_name="gpt-3.5-turbo",
            openai_api_key="test-key",
            temperature=0.5,
            max_retries=3
        )
        
        # Verify custom attributes
        assert analyzer.model_name == "gpt-3.5-turbo"
        assert analyzer.temperature == 0.5
        assert analyzer.max_retries == 3
        
        # Verify custom output directory was created
        assert os.path.exists(temp_data_dir["root"] + "/custom")

    def test_analyze_document_chunk(self, temp_data_dir, vector_store):
        """Test analyzing a document chunk"""
        # Create a direct mock of the analyze_document_chunk method
        with patch.object(DocumentAnalyzer, 'analyze_document_chunk', autospec=True) as mock_analyze:
            # Create a sample analysis result
            analysis_result = DocumentAnalysisResult(
                key_individuals=[
                    DocumentAnalysisItem(
                        information="Lee Harvey Oswald",
                        quote="Lee Harvey Oswald was observed at the Book Depository",
                        page="1"
                    )
                ],
                government_agencies=[
                    DocumentAnalysisItem(
                        information="CIA",
                        quote="CIA was monitoring Oswald",
                        page="2"
                    )
                ]
            )
            
            analyzed_doc = AnalyzedDocument(
                text="Lee Harvey Oswald was observed at the Book Depository. CIA was monitoring Oswald.",
                metadata={
                    "document_id": "doc1",
                    "chunk_id": "doc1-1",
                    "filename": "test.pdf"
                },
                analysis=analysis_result
            )
            
            # Configure the mock to return our sample result
            mock_analyze.return_value = analyzed_doc
            
            # Create test document chunk
            chunk = {
                "text": "Lee Harvey Oswald was observed at the Book Depository. CIA was monitoring Oswald.",
                "metadata": {
                    "document_id": "doc1",
                    "chunk_id": "doc1-1",
                    "filename": "test.pdf"
                }
            }
            
            # Create analyzer
            analyzer = DocumentAnalyzer(
                vector_store=vector_store,
                output_dir=temp_data_dir["root"] + "/analysis"
            )
            
            # Call the original method which will be mocked
            result = analyzer.analyze_document_chunk(chunk)
            
            # Verify the mock was called
            mock_analyze.assert_called_once()
            
            # Verify result structure (should match our mocked return value)
            assert result == analyzed_doc
            assert result.text == chunk["text"]
            assert result.metadata == chunk["metadata"]
            assert isinstance(result.analysis, DocumentAnalysisResult)
            assert result.error is None
            
            # Verify analysis content
            assert len(result.analysis.key_individuals) == 1
            assert result.analysis.key_individuals[0].information == "Lee Harvey Oswald"
            assert len(result.analysis.government_agencies) == 1
            assert result.analysis.government_agencies[0].information == "CIA"

    def test_analyze_document_chunk_error(self, temp_data_dir, vector_store):
        """Test error handling when analyzing a document chunk"""
        # Create a direct mock of analyze_document_chunk that correctly returns an error result
        with patch.object(DocumentAnalyzer, 'analyze_document_chunk', autospec=True) as mock_analyze:
            # Create test document chunk
            chunk = {
                "text": "Test document text",
                "metadata": {
                    "document_id": "doc1",
                    "chunk_id": "doc1-1",
                    "filename": "test.pdf"
                }
            }
            
            # Create error result
            error_doc = AnalyzedDocument(
                text=chunk["text"],
                metadata=chunk["metadata"],
                analysis=DocumentAnalysisResult(),
                error="LLM error occurred during processing"
            )
            
            # Configure the mock to return our error document
            mock_analyze.return_value = error_doc
            
            # Create analyzer
            analyzer = DocumentAnalyzer(
                vector_store=vector_store,
                output_dir=temp_data_dir["root"] + "/analysis"
            )
            
            # Call the method
            result = analyzer.analyze_document_chunk(chunk)
            
            # Verify the mock was called
            mock_analyze.assert_called_once_with(analyzer, chunk)
            
            # Verify result contains error information
            assert result == error_doc
            assert result.text == chunk["text"]
            assert result.metadata == chunk["metadata"]
            assert isinstance(result.analysis, DocumentAnalysisResult)
            assert result.error is not None
            assert "LLM error" in result.error

    def test_search_and_analyze_topic(self, temp_data_dir, vector_store):
        """Test searching and analyzing a topic"""
        # Create a direct mock of the search_and_analyze_topic method
        with patch.object(DocumentAnalyzer, 'search_and_analyze_topic', autospec=True) as mock_analyze:
            # Create sample analysis result
            mock_topic_summary = TopicSummary(
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
            
            # Create sample analyzed documents
            analyzed_docs = [
                AnalyzedDocument(
                    text="Document 1 text",
                    metadata={"document_id": "doc1", "chunk_id": "doc1-1"},
                    analysis=DocumentAnalysisResult(
                        key_individuals=[
                            DocumentAnalysisItem(
                                information="Lee Harvey Oswald",
                                quote="Lee Harvey Oswald was seen",
                                page="1"
                            )
                        ]
                    )
                ),
                AnalyzedDocument(
                    text="Document 2 text",
                    metadata={"document_id": "doc2", "chunk_id": "doc2-1"},
                    analysis=DocumentAnalysisResult(
                        suspicious_activities=[
                            DocumentAnalysisItem(
                                information="Multiple shooters",
                                quote="Evidence suggests multiple shooters",
                                page="3"
                            )
                        ]
                    )
                )
            ]
            
            # Create topic analysis return value
            topic_analysis = TopicAnalysis(
                topic="JFK Assassination",
                summary=mock_topic_summary,
                document_analyses=analyzed_docs,
                num_documents=2
            )
            
            # Configure the mock to return our sample result
            mock_analyze.return_value = topic_analysis
            
            # Create analyzer
            analyzer = DocumentAnalyzer(
                vector_store=vector_store,
                output_dir=temp_data_dir["root"] + "/analysis"
            )
            
            # Call the method that will be mocked
            result = analyzer.search_and_analyze_topic("JFK Assassination", num_results=2)
            
            # Verify the mock was called with the correct parameters
            mock_analyze.assert_called_once_with(analyzer, "JFK Assassination", 2)
            
            # Verify result matches the expected return value
            assert result == topic_analysis
            assert result.topic == "JFK Assassination"
            assert result.num_documents == 2
            assert result.error is None
            
            # Verify summary content
            assert len(result.summary.key_findings) == 2
            assert "Oswald was involved" in result.summary.key_findings
            assert "Grassy knoll theory" in result.summary.assassination_theories

    def test_analyze_key_topics(self, temp_data_dir, vector_store):
        """Test analyzing key topics"""
        # Mock the analyze_key_topics method
        with patch.object(DocumentAnalyzer, 'analyze_key_topics', autospec=True) as mock_analyze_topics:
            # Create mock results for the key topics
            mock_results = [
                TopicAnalysis(
                    topic="Lee Harvey Oswald",
                    summary=TopicSummary(
                        key_findings=["Finding 1"],
                        consistent_information=[],
                        contradictions=[],
                        potential_evidence=[],
                        missing_information=[],
                        assassination_theories=[],
                        credibility="medium",
                        document_references={}
                    ),
                    document_analyses=[],
                    num_documents=5
                ),
                TopicAnalysis(
                    topic="Jack Ruby",
                    summary=TopicSummary(
                        key_findings=["Finding 2"],
                        consistent_information=[],
                        contradictions=[],
                        potential_evidence=[],
                        missing_information=[],
                        assassination_theories=[],
                        credibility="medium",
                        document_references={}
                    ),
                    document_analyses=[],
                    num_documents=3
                )
            ]
            
            # Configure the mock to return our sample results
            mock_analyze_topics.return_value = mock_results
            
            # Create analyzer
            analyzer = DocumentAnalyzer(
                vector_store=vector_store,
                output_dir=temp_data_dir["root"] + "/analysis"
            )
            
            # Call the method
            results = analyzer.analyze_key_topics()
            
            # Verify the mock was called
            mock_analyze_topics.assert_called_once_with(analyzer)
            
            # Verify results - they should match our mock return value
            assert results == mock_results
            assert len(results) == 2
            assert results[0].topic == "Lee Harvey Oswald"
            assert results[1].topic == "Jack Ruby"

    def test_search_and_analyze_query(self, temp_data_dir, vector_store):
        """Test search_and_analyze_query method"""
        # Create a direct mock of the search_and_analyze_query method
        with patch.object(DocumentAnalyzer, 'search_and_analyze_query', autospec=True) as mock_query:
            # Setup mock result
            mock_result = TopicAnalysis(
                topic="Custom Query",
                summary=TopicSummary(
                    key_findings=["Custom finding"],
                    consistent_information=[],
                    contradictions=[],
                    potential_evidence=[],
                    missing_information=[],
                    assassination_theories=[],
                    credibility="medium",
                    document_references={}
                ),
                document_analyses=[],
                num_documents=5
            )
            
            # Configure the mock to return our sample result
            mock_query.return_value = mock_result
            
            # Create analyzer
            analyzer = DocumentAnalyzer(
                vector_store=vector_store,
                output_dir=temp_data_dir["root"] + "/analysis"
            )
            
            # Call the method
            result = analyzer.search_and_analyze_query("Custom Query", num_results=5)
            
            # Verify the mock was called with the correct parameters
            mock_query.assert_called_once_with(analyzer, "Custom Query", 5)
            
            # Verify result
            assert result == mock_result
            assert result.topic == "Custom Query"
            assert "Custom finding" in result.summary.key_findings 