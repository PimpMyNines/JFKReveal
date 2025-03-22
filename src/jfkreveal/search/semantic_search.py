"""
Enhanced semantic search with hybrid retrieval and reranking.
"""
import logging
from typing import List, Dict, Any, Optional, Union, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from rank_bm25 import BM25Okapi
import os
from tqdm import tqdm

logger = logging.getLogger(__name__)

class SemanticSearchEngine:
    """Advanced semantic search with hybrid retrieval and re-ranking."""
    
    def __init__(
        self, 
        vector_db,
        documents: Optional[List[Dict[str, Any]]] = None,
        reranker_model: Optional[str] = None,
        use_bm25: bool = True
    ):
        """
        Initialize the semantic search engine.
        
        Args:
            vector_db: Vector database for embeddings-based search
            documents: Optional list of document chunks to index
            reranker_model: Optional model for reranking results
            use_bm25: Whether to use BM25 for hybrid search
        """
        self.vector_db = vector_db
        self.use_bm25 = use_bm25
        self.bm25_index = None
        self.document_texts = []
        self.document_ids = []
        
        # Set up reranking if specified
        self.reranker = None
        if reranker_model:
            try:
                from sentence_transformers import CrossEncoder
                self.reranker = CrossEncoder(reranker_model)
                logger.info(f"Initialized reranker with model: {reranker_model}")
            except ImportError:
                logger.warning("sentence-transformers not installed. Reranking disabled.")
            except Exception as e:
                logger.warning(f"Failed to load reranker model: {e}")
        
        # Index documents if provided
        if documents:
            self.index_documents(documents)
    
    def index_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        Index documents for search.
        
        Args:
            documents: List of document chunks with text and metadata
        """
        if not documents:
            logger.warning("No documents provided for indexing")
            return
        
        logger.info(f"Indexing {len(documents)} documents")
        
        # Extract text and IDs for BM25
        texts = []
        ids = []
        
        for doc in documents:
            if "text" in doc and doc["text"]:
                # Preprocess text for BM25
                text = self._preprocess_text(doc["text"])
                texts.append(text)
                
                # Get document ID from metadata
                doc_id = None
                if "metadata" in doc and "chunk_id" in doc["metadata"]:
                    doc_id = doc["metadata"]["chunk_id"]
                elif "id" in doc:
                    doc_id = doc["id"]
                else:
                    doc_id = f"doc_{len(ids)}"
                
                ids.append(doc_id)
        
        # Create BM25 index if using hybrid search
        if self.use_bm25 and texts:
            # Tokenize texts
            tokenized_texts = [text.split() for text in texts]
            self.bm25_index = BM25Okapi(tokenized_texts)
            self.document_texts = texts
            self.document_ids = ids
            logger.info(f"Created BM25 index with {len(texts)} documents")
    
    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess text for BM25 indexing.
        
        Args:
            text: Document text
            
        Returns:
            Preprocessed text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but keep spaces
        text = re.sub(r'[^\w\s]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _bm25_search(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        """
        Search using BM25 algorithm.
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of search results with scores
        """
        if not self.bm25_index or not self.document_texts:
            logger.warning("BM25 index not initialized")
            return []
        
        # Preprocess query
        query = self._preprocess_text(query)
        query_tokens = query.split()
        
        # Get scores from BM25
        scores = self.bm25_index.get_scores(query_tokens)
        
        # Sort documents by score
        results_with_scores = sorted(
            zip(self.document_ids, self.document_texts, scores),
            key=lambda x: x[2],
            reverse=True
        )
        
        # Convert to result format
        results = []
        for doc_id, text, score in results_with_scores[:k]:
            results.append({
                "id": doc_id,
                "text": text,
                "score": float(score),
                "search_type": "bm25"
            })
        
        return results
    
    def _vector_search(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        """
        Search using vector embeddings.
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of search results with scores
        """
        try:
            # Use vector database to get results
            vector_results = self.vector_db.similarity_search_with_score(query, k=k)
            
            # Convert to result format
            results = []
            for doc, score in vector_results:
                # Normalize score (convert distance to similarity)
                similarity = 1.0 / (1.0 + score)
                
                results.append({
                    "id": doc.metadata.get("chunk_id", str(doc.id)),
                    "text": doc.page_content,
                    "score": similarity,
                    "metadata": doc.metadata,
                    "search_type": "vector"
                })
            
            return results
        except Exception as e:
            logger.error(f"Vector search error: {e}")
            return []
    
    def _combine_search_results(
        self, 
        vector_results: List[Dict[str, Any]], 
        bm25_results: List[Dict[str, Any]], 
        alpha: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Combine results from vector and BM25 search.
        
        Args:
            vector_results: Results from vector search
            bm25_results: Results from BM25 search
            alpha: Weight for vector results (1-alpha for BM25)
            
        Returns:
            Combined search results
        """
        # Create a map of document IDs to results
        combined_map = {}
        
        # Normalize scores within each result set
        if vector_results:
            max_vector_score = max(r["score"] for r in vector_results)
            for r in vector_results:
                r["score_normalized"] = r["score"] / max_vector_score if max_vector_score > 0 else 0
                combined_map[r["id"]] = r
        
        if bm25_results:
            max_bm25_score = max(r["score"] for r in bm25_results)
            for r in bm25_results:
                r["score_normalized"] = r["score"] / max_bm25_score if max_bm25_score > 0 else 0
                
                if r["id"] in combined_map:
                    # Document exists in both result sets, combine scores
                    combined_map[r["id"]]["score_normalized"] = (
                        alpha * combined_map[r["id"]]["score_normalized"] + 
                        (1 - alpha) * r["score_normalized"]
                    )
                    combined_map[r["id"]]["search_type"] = "hybrid"
                else:
                    # Document only in BM25 results
                    r["score_normalized"] = (1 - alpha) * r["score_normalized"]
                    combined_map[r["id"]] = r
        
        # Convert back to list and sort by normalized score
        combined_results = list(combined_map.values())
        combined_results.sort(key=lambda x: x["score_normalized"], reverse=True)
        
        # Update actual scores based on normalized scores
        for r in combined_results:
            r["score"] = r["score_normalized"]
            del r["score_normalized"]
        
        return combined_results
    
    def _rerank_results(
        self, 
        query: str, 
        results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Rerank search results using a cross-encoder.
        
        Args:
            query: Original search query
            results: Search results to rerank
            
        Returns:
            Reranked search results
        """
        if not self.reranker or not results:
            return results
        
        # Prepare passages for reranking
        passages = [r["text"] for r in results]
        
        # Create query-passage pairs
        pairs = [[query, passage] for passage in passages]
        
        # Get scores from reranker
        try:
            scores = self.reranker.predict(pairs)
            
            # Update results with new scores
            for i, score in enumerate(scores):
                results[i]["original_score"] = results[i]["score"]
                results[i]["score"] = float(score)
                results[i]["reranked"] = True
            
            # Sort by new scores
            results.sort(key=lambda x: x["score"], reverse=True)
            
            return results
        except Exception as e:
            logger.error(f"Reranking error: {e}")
            return results
    
    def hybrid_search(
        self, 
        query: str, 
        k: int = 10, 
        alpha: float = 0.5,
        rerank: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining vector search with BM25.
        
        Args:
            query: Search query
            k: Number of results to return
            alpha: Weight for vector results (1-alpha for BM25)
            rerank: Whether to rerank results
            
        Returns:
            List of search results
        """
        # Expand to get more candidates for reranking
        candidate_k = k * 3 if rerank and self.reranker else k
        
        # Get vector search results
        vector_results = self._vector_search(query, k=candidate_k)
        
        # Get BM25 results if enabled
        bm25_results = []
        if self.use_bm25 and self.bm25_index:
            bm25_results = self._bm25_search(query, k=candidate_k)
        
        # Combine results
        combined_results = self._combine_search_results(
            vector_results, bm25_results, alpha=alpha
        )
        
        # Apply reranking if requested and available
        if rerank and self.reranker:
            combined_results = self._rerank_results(query, combined_results)
        
        # Return top k results
        return combined_results[:k] 