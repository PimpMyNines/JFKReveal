"""
Interfaces for the JFKReveal application.

This module defines the core interfaces that components of the application must implement.
These interfaces help establish clear boundaries between components and enable
dependency injection for better testing and modularity.
"""
from typing import Protocol, List, Dict, Any, Optional, TypeVar, Generic, Callable, Union, Type

# Type definitions
T = TypeVar('T')
DocumentId = str
ChunkId = str


class ITextCleaner(Protocol):
    """Interface for text cleaning operations."""
    
    def clean_text(self, text: str) -> str:
        """
        Apply all cleaning operations to the text.
        
        Args:
            text: OCR text to clean
            
        Returns:
            Cleaned text
        """
        ...


class IVectorStore(Protocol):
    """Interface for vector storage and retrieval operations."""
    
    def add_documents_from_file(self, file_path: str) -> int:
        """
        Add document chunks from a processed JSON file.
        
        Args:
            file_path: Path to JSON file with processed chunks
            
        Returns:
            Number of chunks added to the vector store
        """
        ...
    
    def add_all_documents(self, processed_dir: str) -> int:
        """
        Add all processed document chunks to the vector store.
        
        Args:
            processed_dir: Directory containing processed chunk files
            
        Returns:
            Total number of chunks added
        """
        ...
    
    def similarity_search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Perform a similarity search against the vector store.
        
        Args:
            query: The query text
            k: Number of results to return
            
        Returns:
            List of document chunks with scores
        """
        ...


class IDocumentProcessor(Protocol):
    """Interface for document processing operations."""
    
    def process_document(self, pdf_path: str) -> Optional[str]:
        """
        Process a single PDF document and save chunks.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Path to the saved chunks file, or None if processing failed
        """
        ...
    
    def process_all_documents(self) -> List[str]:
        """
        Process all PDF documents in the input directory.
        
        Returns:
            List of paths to processed files
        """
        ...
    
    def get_processed_documents(self) -> List[str]:
        """
        Get a list of all processed document files without processing any new ones.
        
        Returns:
            List of paths to all processed chunk files
        """
        ...


class IDocumentScraper(Protocol):
    """Interface for document scraping operations."""
    
    def scrape_all(self) -> List[str]:
        """
        Scrape all documents from the source.
        
        Returns:
            List of paths to downloaded documents
        """
        ...


class IDocumentAnalyzer(Protocol):
    """Interface for document analysis operations."""
    
    def analyze_document_chunk(self, chunk: Dict[str, Any]) -> Any:
        """
        Analyze a single document chunk with LLM.
        
        Args:
            chunk: Document chunk (text and metadata)
            
        Returns:
            Analysis results
        """
        ...
    
    def search_and_analyze_topic(self, topic: str, num_results: int = 20) -> Any:
        """
        Search for a topic and analyze relevant documents.
        
        Args:
            topic: Topic to search for
            num_results: Number of relevant documents to analyze
            
        Returns:
            Topic analysis results
        """
        ...
    
    def analyze_key_topics(self) -> List[Any]:
        """
        Analyze a set of predefined key topics.
        
        Returns:
            List of topic analysis results
        """
        ...
    
    def search_and_analyze_query(self, query: str, num_results: int = 20) -> Any:
        """
        Search for a custom query and analyze relevant documents.
        
        Args:
            query: Custom search query
            num_results: Number of relevant documents to analyze
            
        Returns:
            Query analysis results
        """
        ...


class IFindingsReport(Protocol):
    """Interface for findings report generation operations."""
    
    def generate_executive_summary(self, analyses: List[Dict[str, Any]]) -> str:
        """
        Generate an executive summary of findings.
        
        Args:
            analyses: List of analysis data
            
        Returns:
            Executive summary markdown text
        """
        ...
    
    def generate_detailed_findings(self, analyses: List[Dict[str, Any]]) -> str:
        """
        Generate detailed findings report from analyses.
        
        Args:
            analyses: List of analysis data
            
        Returns:
            Detailed findings markdown text
        """
        ...
    
    def generate_full_report(self) -> Dict[str, str]:
        """
        Generate full findings report.
        
        Returns:
            Dictionary of report sections
        """
        ...


class ILLMProvider(Protocol):
    """Interface for LLM providers."""
    
    def generate_text(self, prompt: str) -> str:
        """
        Generate text using the LLM.
        
        Args:
            prompt: The prompt to send to the LLM
            
        Returns:
            Generated text
        """
        ...
    
    def generate_structured_output(self, prompt: str, output_class: Type[T]) -> T:
        """
        Generate structured output using the LLM.
        
        Args:
            prompt: The prompt to send to the LLM
            output_class: The class to use for structured output
            
        Returns:
            Generated structured output
        """
        ...


class ICredentialProvider(Protocol):
    """Interface for credential providers."""
    
    def get_credential(self, name: str) -> Optional[str]:
        """
        Get a credential by name.
        
        Args:
            name: The name of the credential
            
        Returns:
            The credential value, or None if not found
        """
        ...
    
    def set_credential(self, name: str, value: str) -> None:
        """
        Set a credential.
        
        Args:
            name: The name of the credential
            value: The credential value
        """
        ...
    
    def get_with_fallback(
        self, 
        names: List[str], 
        validate: bool = True,
        raise_on_missing: bool = False
    ) -> Optional[str]:
        """
        Get a credential with fallbacks.
        
        Args:
            names: List of credential names to try in order
            validate: Whether to validate credentials
            raise_on_missing: Whether to raise an exception if no credential is found
            
        Returns:
            The first valid credential found, or None if none are found/valid
        """
        # Default implementation just tries each credential in order
        for name in names:
            value = self.get_credential(name)
            if value:
                return value
        return None