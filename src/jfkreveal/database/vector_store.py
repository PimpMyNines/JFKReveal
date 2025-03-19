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
from langchain_core.exceptions import LangChainException
from langchain_core.embeddings import FakeEmbeddings
from langchain_community.vectorstores.utils import filter_complex_metadata
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Using OpenAI embeddings only

logger = logging.getLogger(__name__)

class VectorStore:
    """Vector store for document chunks using ChromaDB and OpenAI embeddings."""
    
    def __init__(
        self,
        persist_directory: str = "data/vectordb",
        embedding_model: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        xai_api_key: Optional[str] = None,
    ):
        """
        Initialize the vector store.
        
        Args:
            persist_directory: Directory to persist the vector database
            embedding_model: OpenAI embedding model to use
            openai_api_key: OpenAI API key (uses environment variable if not provided)
            xai_api_key: X AI API key (for future X AI embedding models)
        """
        self.persist_directory = persist_directory
        
        # Get embedding model from parameter, env var, or use default
        if embedding_model is None:
            embedding_model = os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")
        
        # Use OpenAI embeddings with the specified model
        try:
            self.embedding_function = OpenAIEmbeddings(
                model=embedding_model,
                openai_api_key=openai_api_key
            )
            logger.info(f"Successfully initialized OpenAI embeddings with model: {embedding_model}")
        except Exception as e:
            logger.error(f"Error initializing OpenAI embeddings: {e}")
            logger.warning("Falling back to FakeEmbeddings for testing purposes")
            self.embedding_function = FakeEmbeddings(size=1536)  # Dimension for OpenAI embeddings
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
                
            texts = []
            metadatas = []
            ids = []
            
            for chunk in chunks:
                texts.append(chunk['text'])
                # Filter complex metadata to avoid ChromaDB errors
                clean_metadata = filter_complex_metadata(chunk['metadata'])
                metadatas.append(clean_metadata)
                # Extract chunk_id safely
                chunk_id = str(chunk['metadata'].get('chunk_id', f"chunk_{len(ids)}"))
                ids.append(chunk_id)
            
            logger.info(f"Adding {len(texts)} chunks to vector store from {file_path}")
            
            # Add chunks to vector store
            self.vector_store.add_texts(
                texts=texts,
                metadatas=metadatas,
                ids=ids
            )
            
            # Note: persist() method may not be needed with newer versions of Chroma
            # Changes are automatically persisted
            
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