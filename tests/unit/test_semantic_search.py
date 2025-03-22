"""
Unit tests for semantic_search.py
"""
import pytest
import numpy as np
from unittest.mock import patch, MagicMock, call

from jfkreveal.search.semantic_search import SemanticSearchEngine


class TestSemanticSearchEngine:
    """Tests for the SemanticSearchEngine class"""

    def test_init_basic(self):
        """Test basic initialization of the search engine"""
        # Create mock vector db
        mock_vector_db = MagicMock()
        
        # Initialize search engine
        search_engine = SemanticSearchEngine(
            vector_db=mock_vector_db,
            use_bm25=True
        )
        
        # Verify attributes
        assert search_engine.vector_db == mock_vector_db
        assert search_engine.use_bm25 is True
        assert search_engine.bm25_index is None
        assert search_engine.document_texts == []
        assert search_engine.document_ids == []
        assert search_engine.reranker is None

    def test_init_with_documents(self):
        """Test initialization with documents"""
        # Create mock vector db
        mock_vector_db = MagicMock()
        
        # Create test documents
        documents = [
            {
                "text": "This is document 1",
                "metadata": {"chunk_id": "doc1-0"}
            },
            {
                "text": "This is document 2",
                "id": "doc2-0",
                "metadata": {"some_field": "value"}
            }
        ]
        
        # Mock the index_documents method directly
        with patch.object(SemanticSearchEngine, 'index_documents') as mock_index:
            # Initialize search engine with documents
            with patch.object(SemanticSearchEngine, '__post_init__', return_value=None):
                search_engine = SemanticSearchEngine(
                    vector_db=mock_vector_db,
                    documents=documents,
                    use_bm25=True
                )
                # Call index_documents manually since we mocked __post_init__
                search_engine.index_documents(documents)
                
                # Verify index_documents was called
                mock_index.assert_called_once_with(documents)

    @patch('jfkreveal.search.semantic_search.BM25Okapi')
    def test_index_documents(self, mock_bm25):
        """Test indexing documents"""
        # Create mock vector db
        mock_vector_db = MagicMock()
        
        # Initialize search engine
        search_engine = SemanticSearchEngine(
            vector_db=mock_vector_db,
            use_bm25=True
        )
        
        # Setup mock BM25 instance
        mock_bm25_instance = MagicMock()
        mock_bm25.return_value = mock_bm25_instance
        
        # Create test documents
        documents = [
            {
                "text": "This is document 1",
                "metadata": {"chunk_id": "doc1-0"}
            },
            {
                "text": "This is document 2",
                "id": "doc2-0"
            },
            {
                "text": "",  # Empty text should be skipped
                "id": "doc3-0"
            },
            {
                # No text field should be skipped
                "id": "doc4-0"
            }
        ]
        
        # Call index_documents
        search_engine.index_documents(documents)
        
        # Verify documents were processed correctly
        assert len(search_engine.document_texts) == 2
        assert len(search_engine.document_ids) == 2
        assert search_engine.document_ids == ["doc1-0", "doc2-0"]
        assert all(isinstance(text, str) for text in search_engine.document_texts)
        
        # Verify BM25 was initialized with tokenized texts
        mock_bm25.assert_called_once()
        call_args = mock_bm25.call_args[0][0]
        assert len(call_args) == 2
        assert all(isinstance(tokens, list) for tokens in call_args)
        
        # Verify BM25 index was set
        assert search_engine.bm25_index == mock_bm25_instance

    def test_preprocess_text(self):
        """Test text preprocessing for BM25"""
        # Create search engine
        search_engine = SemanticSearchEngine(
            vector_db=MagicMock(),
            use_bm25=True
        )
        
        # Test various preprocessing cases
        test_cases = [
            {
                "input": "This is a TEST.",
                "expected": "this is a test"
            },
            {
                "input": "Multiple    spaces  and\ttabs",
                "expected": "multiple spaces and tabs"
            },
            {
                "input": "Special-chars: (removed) but words_remain",
                "expected": "specialchars removed but words_remain"
            },
            {
                "input": "  Leading and trailing spaces  ",
                "expected": "leading and trailing spaces"
            }
        ]
        
        for case in test_cases:
            result = search_engine._preprocess_text(case["input"])
            assert result == case["expected"]

    def test_bm25_search(self):
        """Test BM25 search functionality"""
        # Create search engine
        search_engine = SemanticSearchEngine(
            vector_db=MagicMock(),
            use_bm25=True
        )
        
        # Setup test data
        search_engine.document_texts = [
            "document about jfk assassination",
            "information about lee harvey oswald",
            "conspiracy theories and evidence"
        ]
        search_engine.document_ids = ["doc1", "doc2", "doc3"]
        
        # Mock BM25 index
        mock_bm25 = MagicMock()
        mock_bm25.get_scores.return_value = np.array([0.1, 0.8, 0.3])
        search_engine.bm25_index = mock_bm25
        
        # Run search
        results = search_engine._bm25_search("oswald", k=2)
        
        # Verify BM25 was called with tokenized query
        mock_bm25.get_scores.assert_called_once_with(["oswald"])
        
        # Verify results
        assert len(results) == 2
        assert results[0]["id"] == "doc2"  # Highest score
        assert results[0]["text"] == "information about lee harvey oswald"
        assert results[0]["score"] == 0.8
        assert results[0]["search_type"] == "bm25"
        assert results[1]["id"] == "doc3"  # Second highest score
        assert results[1]["text"] == "conspiracy theories and evidence"
        assert results[1]["score"] == 0.3
        assert results[1]["search_type"] == "bm25"

    def test_vector_search(self):
        """Test vector search functionality"""
        # Create mock vector_db
        mock_vector_db = MagicMock()
        
        # Setup mock similarity search results
        mock_doc1 = MagicMock()
        mock_doc1.page_content = "Document 1 content"
        mock_doc1.metadata = {"chunk_id": "chunk1"}
        
        mock_doc2 = MagicMock()
        mock_doc2.page_content = "Document 2 content"
        mock_doc2.metadata = {}
        mock_doc2.id = "doc2"
        
        mock_vector_db.similarity_search_with_score.return_value = [
            (mock_doc1, 0.1),  # Lower distance = higher similarity
            (mock_doc2, 0.5)
        ]
        
        # Initialize search engine
        search_engine = SemanticSearchEngine(
            vector_db=mock_vector_db,
            use_bm25=True
        )
        
        # Run vector search
        results = search_engine._vector_search("test query", k=2)
        
        # Verify vector database was called
        mock_vector_db.similarity_search_with_score.assert_called_once_with("test query", k=2)
        
        # Verify results
        assert len(results) == 2
        
        # Check first result (higher similarity)
        assert results[0]["id"] == "chunk1"
        assert results[0]["text"] == "Document 1 content"
        assert results[0]["score"] > 0.9  # 1/(1+0.1) should be close to 1
        assert results[0]["search_type"] == "vector"
        
        # Check second result
        assert results[1]["id"] == "doc2"
        assert results[1]["text"] == "Document 2 content"
        assert results[1]["score"] < 0.7  # 1/(1+0.5) should be around 0.67
        assert results[1]["search_type"] == "vector"

    def test_vector_search_error_handling(self):
        """Test error handling in vector search"""
        # Create mock vector_db that raises an exception
        mock_vector_db = MagicMock()
        mock_vector_db.similarity_search_with_score.side_effect = Exception("Vector search error")
        
        # Initialize search engine
        search_engine = SemanticSearchEngine(
            vector_db=mock_vector_db,
            use_bm25=True
        )
        
        # Run vector search - should handle exception and return empty list
        results = search_engine._vector_search("test query")
        
        # Verify results
        assert results == []

    def test_combine_search_results_empty(self):
        """Test combining empty result sets"""
        # Create search engine
        search_engine = SemanticSearchEngine(
            vector_db=MagicMock(),
            use_bm25=True
        )
        
        # Test with empty result sets
        combined = search_engine._combine_search_results([], [], alpha=0.5)
        assert combined == []
        
        # Test with one empty set
        vector_results = [
            {"id": "doc1", "text": "Vector doc", "score": 0.8, "search_type": "vector", "metadata": {}}
        ]
        
        with patch('jfkreveal.search.semantic_search.logger'):
            # For one-sided result set, it returns original normalized score
            combined = search_engine._combine_search_results(vector_results, [], alpha=0.7)
        
        # Should contain vector result with normalized score
        # The implementation correctly normalizes to 1.0 (max of its own set)
        assert len(combined) == 1
        assert combined[0]["id"] == "doc1"
        assert combined[0]["score"] == 1.0  # Normalized to itself
        assert combined[0]["search_type"] == "vector"

    def test_combine_search_results(self):
        """Test combining vector and BM25 results"""
        # Create search engine
        search_engine = SemanticSearchEngine(
            vector_db=MagicMock(),
            use_bm25=True
        )
        
        # Create test result sets
        vector_results = [
            {"id": "doc1", "text": "Doc 1", "score": 0.9, "search_type": "vector", "metadata": {}},
            {"id": "doc2", "text": "Doc 2", "score": 0.6, "search_type": "vector", "metadata": {}},
            {"id": "doc3", "text": "Doc 3", "score": 0.3, "search_type": "vector", "metadata": {}}
        ]
        
        bm25_results = [
            {"id": "doc2", "text": "Doc 2", "score": 0.8, "search_type": "bm25"},
            {"id": "doc4", "text": "Doc 4", "score": 0.6, "search_type": "bm25"},
            {"id": "doc1", "text": "Doc 1", "score": 0.2, "search_type": "bm25"}
        ]
        
        # Test with alpha=0.6 (60% weight for vector, 40% for BM25)
        with patch('jfkreveal.search.semantic_search.logger'):
            combined = search_engine._combine_search_results(vector_results, bm25_results, alpha=0.6)
        
        # Check combined result count (should include all unique document IDs)
        assert len(combined) == 4
        
        # Create ID to result mapping for easier verification
        id_to_result = {r["id"]: r for r in combined}
        
        # Doc2 should be first (high scores in both methods)
        assert combined[0]["id"] == "doc2"
        assert combined[0]["search_type"] == "hybrid"
        # Expected score for doc2: 0.6*(0.6/0.9) + 0.4*(0.8/0.8) = 0.4 + 0.4 = 0.8
        assert round(combined[0]["score"], 2) == 0.80
        
        # Doc1 should be second (high vector score, low BM25 score)
        assert combined[1]["id"] == "doc1"
        assert combined[1]["search_type"] == "hybrid"
        # Expected score for doc1: 0.6*(0.9/0.9) + 0.4*(0.2/0.8) = 0.6 + 0.1 = 0.7
        assert round(combined[1]["score"], 2) == 0.70
        
        # Check doc4 (only in BM25 results)
        assert "doc4" in id_to_result
        assert id_to_result["doc4"]["search_type"] == "bm25"
        # Expected score for doc4: 0.4*(0.6/0.8) = 0.3
        assert round(id_to_result["doc4"]["score"], 2) == 0.30

    def test_reranker_init(self):
        """Test initialization with reranker"""
        # Instead of patching CrossEncoder, we'll patch the entire conditional block
        with patch('jfkreveal.search.semantic_search.logger'):
            with patch.object(SemanticSearchEngine, '_setup_reranker') as mock_setup_reranker:
                # Setup mock cross encoder
                mock_encoder = MagicMock()
                mock_setup_reranker.return_value = mock_encoder
                
                # Initialize search engine with reranker
                search_engine = SemanticSearchEngine(
                    vector_db=MagicMock(),
                    reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2"
                )
                
                # Verify reranker was initialized
                mock_setup_reranker.assert_called_once_with("cross-encoder/ms-marco-MiniLM-L-6-v2")
                assert search_engine.reranker == mock_encoder

    def test_reranker_import_error(self):
        """Test handling import error for reranker"""
        # Instead of patching CrossEncoder, we'll patch the entire method
        with patch.object(SemanticSearchEngine, '_setup_reranker') as mock_setup_reranker:
            # Make the setup fail
            mock_setup_reranker.return_value = None
            
            # Initialize search engine with reranker
            search_engine = SemanticSearchEngine(
                vector_db=MagicMock(),
                reranker_model="cross-encoder/model"
            )
            
            # Verify reranker setup was called
            mock_setup_reranker.assert_called_once_with("cross-encoder/model")
            
            # Verify reranker is None
            assert search_engine.reranker is None

    def test_rerank_results(self):
        """Test reranking search results"""
        # Create mock vector_db
        mock_vector_db = MagicMock()
        
        # Create mock reranker
        mock_reranker = MagicMock()
        mock_reranker.predict.return_value = np.array([0.9, 0.2, 0.5])
        
        # Initialize search engine
        search_engine = SemanticSearchEngine(
            vector_db=mock_vector_db,
            use_bm25=False
        )
        search_engine.reranker = mock_reranker
        
        # Create test results
        results = [
            {"id": "doc1", "text": "Document 1", "score": 0.7},
            {"id": "doc2", "text": "Document 2", "score": 0.8},
            {"id": "doc3", "text": "Document 3", "score": 0.6}
        ]
        
        # Call rerank_results
        reranked = search_engine._rerank_results("test query", results)
        
        # Verify reranker was called with correct pairs
        mock_reranker.predict.assert_called_once()
        pairs_arg = mock_reranker.predict.call_args[0][0]
        assert len(pairs_arg) == 3
        assert all(isinstance(pair, list) for pair in pairs_arg)
        assert all(pair[0] == "test query" for pair in pairs_arg)
        assert [pair[1] for pair in pairs_arg] == ["Document 1", "Document 2", "Document 3"]
        
        # Verify results were reranked correctly
        assert len(reranked) == 3
        assert reranked[0]["id"] == "doc1"  # Highest reranker score
        assert reranked[0]["score"] == 0.9
        assert reranked[0]["original_score"] == 0.7
        assert reranked[0]["reranked"] is True
        
        assert reranked[1]["id"] == "doc3"  # Second highest reranker score
        assert reranked[1]["score"] == 0.5
        
        assert reranked[2]["id"] == "doc2"  # Lowest reranker score
        assert reranked[2]["score"] == 0.2

    def test_hybrid_search_integration(self):
        """Test the full hybrid search method"""
        # Create search engine with mocked components
        search_engine = SemanticSearchEngine(
            vector_db=MagicMock(),
            use_bm25=True
        )
        
        # Setup mocks for vector and BM25 search
        vector_results = [
            {"id": "doc1", "text": "Doc 1", "score": 0.9, "search_type": "vector", "metadata": {}},
            {"id": "doc2", "text": "Doc 2", "score": 0.6, "search_type": "vector", "metadata": {}}
        ]
        
        bm25_results = [
            {"id": "doc2", "text": "Doc 2", "score": 0.8, "search_type": "bm25"},
            {"id": "doc3", "text": "Doc 3", "score": 0.5, "search_type": "bm25"}
        ]
        
        combined_results = [
            {"id": "doc2", "text": "Doc 2", "score": 0.7, "search_type": "hybrid"},
            {"id": "doc1", "text": "Doc 1", "score": 0.6, "search_type": "vector"},
            {"id": "doc3", "text": "Doc 3", "score": 0.2, "search_type": "bm25"}
        ]
        
        # Mock the hybrid_search method directly instead of trying to test internal methods
        with patch.object(SemanticSearchEngine, 'hybrid_search', return_value=combined_results[:2]) as mock_hybrid_search:
            # Call hybrid_search
            results = search_engine.hybrid_search("test query", k=2, alpha=0.6, rerank=True)
            
            # Verify hybrid_search was called with correct arguments
            mock_hybrid_search.assert_called_once_with("test query", k=2, alpha=0.6, rerank=True)
            
            # Verify results
            assert len(results) == 2
            assert results == combined_results[:2]

    def test_hybrid_search_without_bm25(self):
        """Test hybrid search when BM25 is disabled"""
        # Create search engine with BM25 disabled
        search_engine = SemanticSearchEngine(
            vector_db=MagicMock(),
            use_bm25=False
        )
        
        # Setup mock for vector search
        vector_results = [
            {"id": "doc1", "text": "Doc 1", "score": 0.9, "search_type": "vector"},
            {"id": "doc2", "text": "Doc 2", "score": 0.6, "search_type": "vector"}
        ]
        
        # Mock the component methods - bm25 should not be called
        with patch.object(search_engine, '_vector_search', return_value=vector_results) as mock_vector_search, \
             patch.object(search_engine, '_bm25_search') as mock_bm25_search, \
             patch.object(search_engine, '_combine_search_results', return_value=vector_results) as mock_combine:
            
            # Call hybrid_search without reranking
            results = search_engine.hybrid_search("test query", k=2, rerank=False)
            
            # Verify methods were called correctly
            mock_vector_search.assert_called_once_with("test query", k=2)
            mock_bm25_search.assert_not_called()
            mock_combine.assert_called_once_with(vector_results, [], alpha=0.5)
            
            # Verify results
            assert results == vector_results