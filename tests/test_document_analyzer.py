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

    @patch('langchain_openai.ChatOpenAI')
    def test_analyze_document_chunk(self, mock_chat_openai, temp_data_dir, vector_store):
        """Test analyzing a document chunk"""
        # Setup mock for LLM response
        mock_llm_instance = MagicMock()
        mock_chat_openai.return_value = mock_llm_instance
        
        # Mock the LLM chain response
        mock_chain = MagicMock()
        mock_llm_instance.with_structured_output.return_value = mock_chain
        
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
        mock_chain.invoke.return_value = analysis_result
        
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
        
        # Call the method
        result = analyzer.analyze_document_chunk(chunk)
        
        # Verify LLM was called with correct setup
        mock_llm_instance.with_structured_output.assert_called_once_with(
            DocumentAnalysisResult,
            method="function_calling"
        )
        
        # Verify invoke was called with the text
        mock_chain.invoke.assert_called_once()
        invoke_args = mock_chain.invoke.call_args[0][0]
        assert chunk["text"] in invoke_args["text"]
        
        # Verify result structure
        assert isinstance(result, AnalyzedDocument)
        assert result.text == chunk["text"]
        assert result.metadata == chunk["metadata"]
        assert isinstance(result.analysis, DocumentAnalysisResult)
        assert result.error is None
        
        # Verify analysis content
        assert len(result.analysis.key_individuals) == 1
        assert result.analysis.key_individuals[0].information == "Lee Harvey Oswald"
        assert len(result.analysis.government_agencies) == 1
        assert result.analysis.government_agencies[0].information == "CIA"

    @patch('langchain_openai.ChatOpenAI')
    def test_analyze_document_chunk_error(self, mock_chat_openai, temp_data_dir, vector_store):
        """Test error handling when analyzing a document chunk"""
        # Setup mock to raise an exception
        mock_llm_instance = MagicMock()
        mock_chat_openai.return_value = mock_llm_instance
        
        mock_chain = MagicMock()
        mock_llm_instance.with_structured_output.return_value = mock_chain
        mock_chain.invoke.side_effect = Exception("LLM error")
        
        # Create test document chunk
        chunk = {
            "text": "Test document text",
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
        
        # Call the method
        result = analyzer.analyze_document_chunk(chunk)
        
        # Verify result contains error information
        assert isinstance(result, AnalyzedDocument)
        assert result.text == chunk["text"]
        assert result.metadata == chunk["metadata"]
        assert isinstance(result.analysis, DocumentAnalysisResult)
        assert result.error is not None
        assert "LLM error" in result.error

    @patch('jfkreveal.analysis.document_analyzer.DocumentAnalyzer.analyze_document_chunk')
    def test_search_and_analyze_topic(self, mock_analyze_chunk, temp_data_dir, vector_store):
        """Test searching and analyzing a topic"""
        # Setup mock for vector store search
        search_results = [
            {
                "text": "Document 1 text",
                "metadata": {"document_id": "doc1", "chunk_id": "doc1-1"},
                "score": 0.95
            },
            {
                "text": "Document 2 text",
                "metadata": {"document_id": "doc2", "chunk_id": "doc2-1"},
                "score": 0.85
            }
        ]
        vector_store.similarity_search = MagicMock(return_value=search_results)
        
        # Setup mock for document analysis
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
        mock_analyze_chunk.side_effect = analyzed_docs
        
        # Create mock for LLM chain to summarize
        with patch('langchain_openai.ChatOpenAI') as mock_chat_openai:
            mock_llm_instance = MagicMock()
            mock_chat_openai.return_value = mock_llm_instance
            
            mock_chain = MagicMock()
            mock_llm_instance.with_structured_output.return_value = mock_chain
            
            # Create a sample summary
            mock_summary = TopicSummary(
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
            mock_chain.invoke.return_value = mock_summary
            
            # Create analyzer
            analyzer = DocumentAnalyzer(
                vector_store=vector_store,
                output_dir=temp_data_dir["root"] + "/analysis"
            )
            
            # Call the method
            result = analyzer.search_and_analyze_topic("JFK Assassination", num_results=2)
            
            # Verify vector_store.similarity_search was called with correct arguments
            vector_store.similarity_search.assert_called_once_with("JFK Assassination", k=2)
            
            # Verify analyze_document_chunk was called for each result
            assert mock_analyze_chunk.call_count == 2
            
            # Verify LLM was called to summarize
            mock_llm_instance.with_structured_output.assert_called_with(
                TopicSummary,
                method="function_calling"
            )
            
            # Verify result structure
            assert isinstance(result, TopicAnalysis)
            assert result.topic == "JFK Assassination"
            assert result.num_documents == 2
            assert result.error is None
            
            # Verify documents are included
            assert len(result.document_analyses) == 2
            assert result.document_analyses[0] == analyzed_docs[0]
            assert result.document_analyses[1] == analyzed_docs[1]
            
            # Verify summary is included
            assert result.summary == mock_summary
            assert "Oswald was involved" in result.summary.key_findings
            assert "medium" == result.summary.credibility

    @patch('jfkreveal.analysis.document_analyzer.DocumentAnalyzer.search_and_analyze_topic')
    def test_analyze_key_topics(self, mock_search_analyze, temp_data_dir, vector_store):
        """Test analyzing key topics"""
        # Setup mock for search_and_analyze_topic
        mock_topic_analysis = TopicAnalysis(
            topic="Test Topic",
            summary=TopicSummary(
                key_findings=["Finding 1", "Finding 2"],
                credibility="high"
            ),
            document_analyses=[
                AnalyzedDocument(
                    text="Document text",
                    metadata={"document_id": "doc1"},
                    analysis=DocumentAnalysisResult()
                )
            ],
            num_documents=1
        )
        mock_search_analyze.return_value = mock_topic_analysis
        
        # Mock JSON serialization
        m = mock_open()
        
        # Create analyzer
        analyzer = DocumentAnalyzer(
            vector_store=vector_store,
            output_dir=temp_data_dir["root"] + "/analysis"
        )
        
        # Patch builtins.open to use the mock
        with patch('builtins.open', m):
            # Call the method
            results = analyzer.analyze_key_topics()
        
        # Verify search_and_analyze_topic was called for each category
        call_count = len(analyzer.ANALYSIS_CATEGORIES)
        assert mock_search_analyze.call_count == call_count
        
        # Verify results
        assert len(results) == call_count
        assert all(isinstance(result, TopicAnalysis) for result in results)
        
        # Verify files were saved
        assert m.call_count == call_count
        
    @patch('jfkreveal.analysis.document_analyzer.DocumentAnalyzer.search_and_analyze_topic')
    def test_search_and_analyze_query(self, mock_search_analyze, temp_data_dir, vector_store):
        """Test analyzing a custom query"""
        # Setup mock for search_and_analyze_topic
        mock_topic_analysis = TopicAnalysis(
            topic="Custom Query",
            summary=TopicSummary(
                key_findings=["Finding 1", "Finding 2"],
                credibility="high"
            ),
            document_analyses=[
                AnalyzedDocument(
                    text="Document text",
                    metadata={"document_id": "doc1"},
                    analysis=DocumentAnalysisResult()
                )
            ],
            num_documents=1
        )
        mock_search_analyze.return_value = mock_topic_analysis
        
        # Create analyzer
        analyzer = DocumentAnalyzer(
            vector_store=vector_store,
            output_dir=temp_data_dir["root"] + "/analysis"
        )
        
        # Call the method
        result = analyzer.search_and_analyze_query("Custom query about JFK", num_results=5)
        
        # Verify search_and_analyze_topic was called with correct arguments
        mock_search_analyze.assert_called_once_with("Custom query about JFK", num_results=5)
        
        # Verify result is the same as search_and_analyze_topic
        assert result == mock_topic_analysis 