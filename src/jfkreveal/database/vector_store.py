"""
Vector database for storing and retrieving document embeddings.
"""
import os
import json
import logging
from typing import List, Dict, Any, Optional, Union, Tuple

from tqdm import tqdm
import numpy as np
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_core.exceptions import LangChainException
from langchain_core.embeddings import FakeEmbeddings
from langchain_community.vectorstores.utils import filter_complex_metadata
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Using Ollama embeddings with OpenAI as fallback

logger = logging.getLogger(__name__)

class VectorStore:
    """Vector store for document chunks using ChromaDB and embeddings (Ollama or OpenAI)."""
    
    def __init__(
        self,
        persist_directory: str = "data/vectordb",
        embedding_model: Optional[str] = None,
        embedding_provider: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        ollama_base_url: Optional[str] = None,
    ):
        """
        Initialize the vector store.
        
        Args:
            persist_directory: Directory to persist the vector database
            embedding_model: Name of the embedding model to use
            embedding_provider: Provider to use for embeddings ('ollama' or 'openai')
            openai_api_key: OpenAI API key (uses environment variable if not provided)
            ollama_base_url: Base URL for Ollama API (defaults to http://localhost:11434)
        """
        self.persist_directory = persist_directory
        
        # Get embedding provider from parameter or env var (defaults to 'ollama')
        if embedding_provider is None:
            embedding_provider = os.environ.get("EMBEDDING_PROVIDER", "ollama")
        
        # Get embedding model or use default based on provider
        if embedding_model is None:
            if embedding_provider == "ollama":
                embedding_model = os.environ.get("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
            else:
                embedding_model = os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")
        
        # Get Ollama base URL
        if ollama_base_url is None:
            ollama_base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        
        # Try to initialize embeddings based on provider
        try:
            if embedding_provider.lower() == "ollama":
                logger.info(f"Initializing Ollama embeddings with model: {embedding_model}")
                self.embedding_function = OllamaEmbeddings(
                    model=embedding_model,
                    base_url=ollama_base_url
                )
                logger.info(f"Successfully initialized Ollama embeddings with model: {embedding_model}")
            else:
                logger.info(f"Initializing OpenAI embeddings with model: {embedding_model}")
                self.embedding_function = OpenAIEmbeddings(
                    model=embedding_model,
                    openai_api_key=openai_api_key
                )
                logger.info(f"Successfully initialized OpenAI embeddings with model: {embedding_model}")
        except Exception as e:
            logger.error(f"Error initializing embeddings with {embedding_provider}: {e}")
            logger.warning("Falling back to FakeEmbeddings for testing purposes")
            self.embedding_function = FakeEmbeddings(size=1536)  # Default dimension
            logger.info("Successfully initialized FakeEmbeddings")
        
        # Create or load vector store
        self.vector_store = Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embedding_function,
        )
    
    def add_documents_from_file(self, file_path: str) -> int:
        """
        Add document chunks from a processed JSON file.
        
        Args:
            file_path: Path to JSON file with processed chunks
            
        Returns:
            Number of chunks added to the vector store
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                chunks = json.load(f)
                
            # Handle case where chunks might not be in the expected format
            if not isinstance(chunks, list):
                logger.warning(f"Invalid chunks format in {file_path}: expected list but got {type(chunks)}")
                return 0
                
            texts = []
            metadatas = []
            ids = []
            
            for chunk_idx, chunk in enumerate(chunks):
                # Validate chunk has required fields
                if not isinstance(chunk, dict) or 'text' not in chunk or 'metadata' not in chunk:
                    logger.warning(f"Invalid chunk format at index {chunk_idx} in {file_path}")
                    continue
                    
                if not isinstance(chunk['text'], str) or not isinstance(chunk['metadata'], dict):
                    logger.warning(f"Invalid text/metadata format at index {chunk_idx} in {file_path}")
                    continue
                    
                texts.append(chunk['text'])
                # Filter complex metadata to avoid ChromaDB errors
                clean_metadata = filter_complex_metadata(chunk['metadata'])
                metadatas.append(clean_metadata)
                # Extract chunk_id safely
                chunk_id = str(chunk['metadata'].get('chunk_id', f"chunk_{file_path}_{chunk_idx}"))
                ids.append(chunk_id)
            
            if not texts:
                logger.warning(f"No valid chunks found in {file_path}")
                return 0
                
            logger.info(f"Adding {len(texts)} chunks to vector store from {file_path}")
            
            # Add chunks to vector store
            self.vector_store.add_texts(
                texts=texts,
                metadatas=metadatas,
                ids=ids
            )
            
            # Explicitly persist changes after adding documents
            logger.debug(f"Persisting changes to vector store at {self.persist_directory}")
            self.vector_store.persist()
            
            return len(texts)
            
        except Exception as e:
            logger.error(f"Error adding chunks from {file_path}: {e}")
            return 0
    
    def add_all_documents(self, processed_dir: str = "data/processed") -> int:
        """
        Add all processed document chunks to the vector store.
        
        Args:
            processed_dir: Directory containing processed chunk files
            
        Returns:
            Total number of chunks added
        """
        # Find all JSON files
        json_files = []
        for root, _, files in os.walk(processed_dir):
            for file in files:
                if file.lower().endswith('.json'):
                    json_files.append(os.path.join(root, file))
                    
        logger.info(f"Found {len(json_files)} chunk files to add")
        
        # Add each file
        total_chunks = 0
        for file_path in tqdm(json_files, desc="Adding to vector store"):
            num_chunks = self.add_documents_from_file(file_path)
            total_chunks += num_chunks
                
        logger.info(f"Added total of {total_chunks} chunks to vector store")
        return total_chunks
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        retry=retry_if_exception_type(LangChainException)
    )
    def similarity_search(
        self, 
        query: str, 
        k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Perform a similarity search against the vector store.
        
        Args:
            query: The query text
            k: Number of results to return
            
        Returns:
            List of document chunks with scores
        """
        try:
            results = self.vector_store.similarity_search_with_score(query, k=k)
            
            # Format results for easier use
            formatted_results = []
            for doc, score in results:
                formatted_results.append({
                    "text": doc.page_content,
                    "metadata": doc.metadata,
                    "score": float(score)
                })
                
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error performing similarity search for '{query}': {e}")
            # Re-raise to trigger retry if needed
            raise