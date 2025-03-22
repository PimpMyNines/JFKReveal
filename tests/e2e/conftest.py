"""
Test fixtures for JFKReveal project end-to-end tests.
"""
import os
import sys
import tempfile
import json
import shutil
import pytest
from typing import List, Dict, Any

# Add the src directory to the Python path
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from jfkreveal.database.text_cleaner import TextCleaner
from jfkreveal.database.document_processor import DocumentProcessor
from jfkreveal.database.vector_store import VectorStore


@pytest.fixture
def sample_text():
    """Sample text for testing."""
    return """CONFIDENTIAL
    
    This document contains information related to the events of November 22, 1963.
    Lee Harvey Oswald was seen at the Texas School Book Depository.
    
    Contact: Agent Smith
    Phone: 555-123-4567
    
    TOP SECRET
    """


@pytest.fixture
def sample_document_chunks():
    """Sample document chunks for testing."""
    return [
        {
            "text": "This document contains information related to the events of November 22, 1963.",
            "metadata": {
                "source": "document1.pdf",
                "page": 1,
                "chunk_id": "document1_1",
            }
        },
        {
            "text": "Lee Harvey Oswald was seen at the Texas School Book Depository.",
            "metadata": {
                "source": "document1.pdf",
                "page": 2,
                "chunk_id": "document1_2",
            }
        },
        {
            "text": "Contact: Agent Smith\nPhone: 555-123-4567",
            "metadata": {
                "source": "document2.pdf",
                "page": 1,
                "chunk_id": "document2_1",
            }
        }
    ]


@pytest.fixture
def temp_dir():
    """Create a temporary directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def temp_data_dir(temp_dir):
    """Create a temporary data directory structure."""
    data_dir = os.path.join(temp_dir, "data")
    raw_dir = os.path.join(data_dir, "raw")
    processed_dir = os.path.join(data_dir, "processed")
    vector_dir = os.path.join(data_dir, "vectordb")
    reports_dir = os.path.join(data_dir, "reports")
    
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(vector_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    
    return {
        "data": data_dir,
        "raw": raw_dir,
        "processed": processed_dir,
        "vector": vector_dir,
        "reports": reports_dir,
    }


@pytest.fixture
def text_cleaner():
    """Initialize a TextCleaner instance."""
    return TextCleaner()


@pytest.fixture
def document_processor(temp_data_dir):
    """Initialize a DocumentProcessor instance."""
    return DocumentProcessor(
        data_dir=temp_data_dir["data"],
        raw_dir=temp_data_dir["raw"],
        processed_dir=temp_data_dir["processed"],
    )


@pytest.fixture
def vector_store(temp_data_dir):
    """Initialize a VectorStore instance."""
    return VectorStore(
        vector_db_path=temp_data_dir["vector"],
        provider="openai",
        embedding_model="text-embedding-ada-002",
    )


@pytest.fixture
def sample_processed_file(temp_data_dir, sample_document_chunks):
    """Create a sample processed file."""
    processed_file = os.path.join(temp_data_dir["processed"], "document1.json")
    with open(processed_file, "w") as f:
        json.dump(sample_document_chunks, f)
    yield processed_file
    if os.path.exists(processed_file):
        os.remove(processed_file) 