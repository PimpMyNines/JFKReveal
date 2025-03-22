"""
Unit tests for the VectorStore class
"""
import os
import json
import pytest
from unittest.mock import patch, MagicMock, mock_open

from jfkreveal.database.vector_store import VectorStore


class TestVectorStore:
    """Test the VectorStore class"""

    @patch('langchain_ollama.OllamaEmbeddings')
    @patch('langchain_chroma.Chroma')
    def test_init_ollama(self, mock_chroma, mock_ollama_embeddings, temp_data_dir):
        """Test initialization with Ollama embeddings"""
        # Setup mocks
        mock_embeddings_instance = MagicMock()
        mock_ollama_embeddings.return_value = mock_embeddings_instance
        
        mock_chroma_instance = MagicMock()
        mock_chroma.return_value = mock_chroma_instance
        
        # Create instance with Ollama provider
        vector_store = VectorStore(
            persist_directory=temp_data_dir["vector"],
            embedding_provider="ollama",
            embedding_model="test-model",
            ollama_base_url="http://test-ollama:11434"
        )
        
        # Verify Ollama embeddings were initialized correctly
        mock_ollama_embeddings.assert_called_once_with(
            model="test-model",
            base_url="http://test-ollama:11434"
        )
        
        # Verify Chroma was initialized correctly
        mock_chroma.assert_called_once_with(
            persist_directory=temp_data_dir["vector"],
            embedding_function=mock_embeddings_instance
        )
        
        # Verify instance attributes
        assert vector_store.embedding_function == mock_embeddings_instance
        assert vector_store.vector_store == mock_chroma_instance
        assert vector_store.persist_directory == temp_data_dir["vector"]

    @patch('langchain_openai.OpenAIEmbeddings')
    @patch('langchain_chroma.Chroma')
    def test_init_openai(self, mock_chroma, mock_openai_embeddings, temp_data_dir):
        """Test initialization with OpenAI embeddings"""
        # Setup mocks
        mock_embeddings_instance = MagicMock()
        mock_openai_embeddings.return_value = mock_embeddings_instance
        
        mock_chroma_instance = MagicMock()
        mock_chroma.return_value = mock_chroma_instance
        
        # Create instance with OpenAI provider
        vector_store = VectorStore(
            persist_directory=temp_data_dir["vector"],
            embedding_provider="openai",
            embedding_model="text-embedding-3-large",
            openai_api_key="test-key"
        )
        
        # Verify OpenAI embeddings were initialized correctly
        mock_openai_embeddings.assert_called_once_with(
            model="text-embedding-3-large",
            openai_api_key="test-key"
        )
        
        # Verify Chroma was initialized correctly
        mock_chroma.assert_called_once_with(
            persist_directory=temp_data_dir["vector"],
            embedding_function=mock_embeddings_instance
        )
        
        # Verify instance attributes
        assert vector_store.embedding_function == mock_embeddings_instance
        assert vector_store.vector_store == mock_chroma_instance

    @patch('langchain_ollama.OllamaEmbeddings')
    @patch('langchain_core.embeddings.FakeEmbeddings')
    @patch('langchain_chroma.Chroma')
    def test_init_error_fallback(self, mock_chroma, mock_fake_embeddings, mock_ollama_embeddings, temp_data_dir):
        """Test fallback to FakeEmbeddings on initialization error"""
        # Setup mocks
        mock_ollama_embeddings.side_effect = Exception("Ollama error")
        
        mock_fake_instance = MagicMock()
        mock_fake_embeddings.return_value = mock_fake_instance
        
        mock_chroma_instance = MagicMock()
        mock_chroma.return_value = mock_chroma_instance
        
        # Create instance (should fall back to FakeEmbeddings)
        vector_store = VectorStore(
            persist_directory=temp_data_dir["vector"],
            embedding_provider="ollama"
        )
        
        # Verify FakeEmbeddings were initialized as fallback
        mock_fake_embeddings.assert_called_once_with(size=1536)
        
        # Verify Chroma was initialized with fake embeddings
        mock_chroma.assert_called_once_with(
            persist_directory=temp_data_dir["vector"],
            embedding_function=mock_fake_instance
        )
        
        # Verify instance attributes
        assert vector_store.embedding_function == mock_fake_instance
        assert vector_store.vector_store == mock_chroma_instance

    @patch('langchain_openai.OpenAIEmbeddings')
    @patch('langchain_chroma.Chroma')
    def test_init_default_values(self, mock_chroma, mock_openai_embeddings, temp_data_dir):
        """Test initialization with default values from environment variables"""
        # Setup environment variables
        os.environ["EMBEDDING_PROVIDER"] = "openai"
        os.environ["OPENAI_EMBEDDING_MODEL"] = "env-model"
        
        # Setup mocks
        mock_embeddings_instance = MagicMock()
        mock_openai_embeddings.return_value = mock_embeddings_instance
        
        mock_chroma_instance = MagicMock()
        mock_chroma.return_value = mock_chroma_instance
        
        # Create instance with minimal parameters
        vector_store = VectorStore(
            persist_directory=temp_data_dir["vector"]
        )
        
        # Verify OpenAI embeddings were initialized with environment values
        mock_openai_embeddings.assert_called_once_with(
            model="env-model",
            openai_api_key=None
        )
        
        # Verify Chroma was initialized correctly
        mock_chroma.assert_called_once_with(
            persist_directory=temp_data_dir["vector"],
            embedding_function=mock_embeddings_instance
        )
        
        # Clean up environment
        del os.environ["EMBEDDING_PROVIDER"]
        del os.environ["OPENAI_EMBEDDING_MODEL"]

    def test_add_documents_from_file(self, temp_data_dir, sample_document_chunks):
        """Test adding documents from a file"""
        # Create a sample file
        file_path = os.path.join(temp_data_dir["processed"], "test_doc.json")
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(sample_document_chunks, f)
        
        # Create a mock vector store
        mock_vector_store = MagicMock()
        
        # Create instance with the mock
        vector_store = VectorStore(persist_directory=temp_data_dir["vector"])
        vector_store.vector_store = mock_vector_store
        
        # Call the method
        result = vector_store.add_documents_from_file(file_path)
        
        # Verify vector_store.add_texts was called with correct arguments
        mock_vector_store.add_texts.assert_called_once()
        args = mock_vector_store.add_texts.call_args[1]
        
        # Check texts argument
        assert len(args["texts"]) == 3
        assert "Lee Harvey Oswald" in args["texts"][0]
        assert "Multiple witnesses" in args["texts"][1]
        assert "ballistic evidence" in args["texts"][2]
        
        # Check ids argument
        assert len(args["ids"]) == 3
        assert args["ids"][0] == "doc_001-1"
        assert args["ids"][1] == "doc_001-2"
        assert args["ids"][2] == "doc_001-3"
        
        # Check metadatas argument
        assert len(args["metadatas"]) == 3
        assert args["metadatas"][0]["document_id"] == "doc_001"
        assert args["metadatas"][1]["chunk_id"] == "doc_001-2"
        assert "pages" in args["metadatas"][2]
        
        # Verify persist was called
        mock_vector_store.persist.assert_called_once()
        
        # Verify result is correct
        assert result == 3

    def test_add_documents_from_file_error(self, temp_data_dir):
        """Test error handling when adding documents from a file"""
        # Create a mock vector store that raises an exception
        mock_vector_store = MagicMock()
        mock_vector_store.add_texts.side_effect = Exception("Vector store error")
        
        # Create instance with the mock
        vector_store = VectorStore(persist_directory=temp_data_dir["vector"])
        vector_store.vector_store = mock_vector_store
        
        # Call the method with a non-existent file
        file_path = os.path.join(temp_data_dir["processed"], "nonexistent.json")
        result = vector_store.add_documents_from_file(file_path)
        
        # Verify result is 0 on error
        assert result == 0
        
        # Verify persist was not called
        mock_vector_store.persist.assert_not_called()

    @patch('os.walk')
    def test_add_all_documents(self, mock_walk, temp_data_dir, sample_document_chunks):
        """Test adding all documents from a directory"""
        # Setup mock walk to return sample file paths
        json_files = [
            os.path.join(temp_data_dir["processed"], "doc1.json"),
            os.path.join(temp_data_dir["processed"], "doc2.json")
        ]
        mock_walk.return_value = [
            (temp_data_dir["processed"], [], ["doc1.json", "doc2.json", "not_json.txt"])
        ]
        
        # Create sample files
        for file_path in json_files:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(sample_document_chunks, f)
        
        # Setup instance with mock for add_documents_from_file
        vector_store = VectorStore(persist_directory=temp_data_dir["vector"])
        vector_store.add_documents_from_file = MagicMock(return_value=3)
        
        # Call the method
        result = vector_store.add_all_documents(temp_data_dir["processed"])
        
        # Verify add_documents_from_file was called for each JSON file
        assert vector_store.add_documents_from_file.call_count == 2
        
        # Verify result is the total number of chunks added
        assert result == 6  # 2 files * 3 chunks per file

    @patch('langchain_chroma.Chroma')
    def test_similarity_search(self, mock_chroma, temp_data_dir):
        """Test similarity search"""
        # Setup mock for similarity_search_with_score
        mock_doc1 = MagicMock()
        mock_doc1.page_content = "Document 1 content"
        mock_doc1.metadata = {"document_id": "doc1", "chunk_id": "doc1-1"}
        
        mock_doc2 = MagicMock()
        mock_doc2.page_content = "Document 2 content"
        mock_doc2.metadata = {"document_id": "doc2", "chunk_id": "doc2-1"}
        
        mock_results = [
            (mock_doc1, 0.95),
            (mock_doc2, 0.85)
        ]
        
        mock_store = MagicMock()
        mock_store.similarity_search_with_score.return_value = mock_results
        mock_chroma.return_value = mock_store
        
        # Create instance
        vector_store = VectorStore(persist_directory=temp_data_dir["vector"])
        
        # Call the method
        results = vector_store.similarity_search("test query", k=2)
        
        # Verify similarity_search_with_score was called with correct arguments
        mock_store.similarity_search_with_score.assert_called_once_with("test query", k=2)
        
        # Verify results are formatted correctly
        assert len(results) == 2
        
        assert results[0]["text"] == "Document 1 content"
        assert results[0]["metadata"]["document_id"] == "doc1"
        assert results[0]["metadata"]["chunk_id"] == "doc1-1"
        assert results[0]["score"] == 0.95
        
        assert results[1]["text"] == "Document 2 content"
        assert results[1]["metadata"]["document_id"] == "doc2"
        assert results[1]["metadata"]["chunk_id"] == "doc2-1"
        assert results[1]["score"] == 0.85

    @patch('langchain_chroma.Chroma')
    def test_similarity_search_error(self, mock_chroma, temp_data_dir):
        """Test error handling in similarity search"""
        # Setup mock to raise an exception
        mock_store = MagicMock()
        mock_store.similarity_search_with_score.side_effect = Exception("Search error")
        mock_chroma.return_value = mock_store
        
        # Create instance
        vector_store = VectorStore(persist_directory=temp_data_dir["vector"])
        
        # Call the method, should raise the exception due to retry decorator
        with pytest.raises(Exception) as excinfo:
            vector_store.similarity_search("test query")
        
        # Verify exception message
        assert "Search error" in str(excinfo.value) 