"""
Test fixtures for JFKReveal project.
"""
import os
import tempfile
import json
import shutil
import pytest
from typing import List, Dict, Any

from jfkreveal.database.text_cleaner import TextCleaner
from jfkreveal.database.document_processor import DocumentProcessor
from jfkreveal.database.vector_store import VectorStore


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
    
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(vector_dir, exist_ok=True)
    
    return {
        "root": temp_dir,
        "raw": raw_dir,
        "processed": processed_dir,
        "vector": vector_dir
    }


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


@pytest.fixture
def sample_processed_file(temp_data_dir, sample_document_chunks):
    """Create a sample processed file for testing."""
    file_path = os.path.join(temp_data_dir["processed"], "sample_doc.json")
    
    with open(file_path, "w") as f:
        json.dump(sample_document_chunks, f)
    
    return file_path 