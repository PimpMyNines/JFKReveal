"""
JFKReveal - Main entry point for the JFK documents analysis pipeline.
"""
import os
import sys
import logging
import argparse
import dotenv
from typing import Optional

# Load environment variables from .env file if it exists
dotenv.load_dotenv()

from .scrapers.archives_gov import ArchivesGovScraper
from .database.document_processor import DocumentProcessor
from .database.vector_store import VectorStore
from .analysis.document_analyzer import DocumentAnalyzer
from .summarization.findings_report import FindingsReport

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("jfkreveal.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

class JFKReveal:
    """Main class for JFK document analysis pipeline."""
    
    def __init__(
        self,
        base_dir: str = ".",
        openai_api_key: Optional[str] = None,
        clean_text: bool = True,
    ):
        """
        Initialize the JFK document analysis pipeline.
        
        Args:
            base_dir: Base directory for data
            openai_api_key: OpenAI API key (uses environment variable if not provided)
            clean_text: Whether to clean OCR text before chunking and embedding
        """
        self.base_dir = base_dir
        self.openai_api_key = openai_api_key
        self.clean_text = clean_text
        
        # Create data directories
        os.makedirs(os.path.join(base_dir, "data/raw"), exist_ok=True)
        os.makedirs(os.path.join(base_dir, "data/processed"), exist_ok=True)
        os.makedirs(os.path.join(base_dir, "data/vectordb"), exist_ok=True)
        os.makedirs(os.path.join(base_dir, "data/analysis"), exist_ok=True)
        os.makedirs(os.path.join(base_dir, "data/reports"), exist_ok=True)
    
    def scrape_documents(self):
        """Scrape documents from the National Archives website."""
        logger.info("Starting document scraping")
        
        scraper = ArchivesGovScraper(
            output_dir=os.path.join(self.base_dir, "data/raw")
        )
        
        downloaded_files = scraper.scrape_all()
        
        logger.info(f"Completed document scraping, downloaded {len(downloaded_files)} files")
        return downloaded_files
    
    def process_documents(self, max_workers: int = 20, skip_existing: bool = True, vector_store = None):
        """
        Process PDF documents and extract text.
        
        Args:
            max_workers: Number of documents to process in parallel (default 20)
            skip_existing: Whether to skip already processed documents (default True)
            vector_store: Optional vector store for immediate embedding
        """
        logger.info("Starting document processing")
        
        processor = DocumentProcessor(
            input_dir=os.path.join(self.base_dir, "data/raw"),
            output_dir=os.path.join(self.base_dir, "data/processed"),
            max_workers=max_workers,
            skip_existing=skip_existing,
            vector_store=vector_store,
            clean_text=self.clean_text
        )
        
        processed_files = processor.process_all_documents()
        
        logger.info(f"Completed document processing, processed {len(processed_files)} files")
        return processed_files
    
    def get_processed_documents(self):
        """
        Get a list of already processed documents without processing any new ones.
        
        Returns:
            List of paths to processed documents
        """
        logger.info("Getting already processed documents")
        
        processor = DocumentProcessor(
            input_dir=os.path.join(self.base_dir, "data/raw"),
            output_dir=os.path.join(self.base_dir, "data/processed")
        )
        
        processed_files = processor.get_processed_documents()
        
        logger.info(f"Found {len(processed_files)} already processed documents")
        return processed_files
    
    def build_vector_database(self) -> Optional[VectorStore]:
        """Build the vector database from processed documents."""
        logger.info("Starting vector database build")
        
        try:
            vector_store = VectorStore(
                persist_directory=os.path.join(self.base_dir, "data/vectordb"),
                openai_api_key=self.openai_api_key
            )
            
            return vector_store
            
        except Exception as e:
            logger.error(f"Failed to build vector database: {e}")
            logger.error("Skipping vector database build and analysis steps")
            return None
    
    def add_all_documents_to_vector_store(self, vector_store):
        """Add all processed documents to the vector store."""
        logger.info("Adding all documents to vector store")
        
        total_chunks = vector_store.add_all_documents(
            processed_dir=os.path.join(self.base_dir, "data/processed")
        )
        
        logger.info(f"Completed vector database build, added {total_chunks} chunks")
        return total_chunks
    
    def analyze_documents(self, vector_store):
        """Analyze documents and generate topic analyses."""
        logger.info("Starting document analysis")
        
        # Get model name from environment variables if set
        model_name = os.environ.get("OPENAI_ANALYSIS_MODEL", "gpt-4o")
        
        analyzer = DocumentAnalyzer(
            vector_store=vector_store,
            output_dir=os.path.join(self.base_dir, "data/analysis"),
            model_name=model_name,
            openai_api_key=self.openai_api_key,
            temperature=0.0,
            max_retries=5
        )
        
        topic_analyses = analyzer.analyze_key_topics()
        
        logger.info(f"Completed document analysis, generated {len(topic_analyses)} topic analyses")
        return topic_analyses
    
    def generate_report(self):
        """Generate final report from analysis results."""
        logger.info("Starting report generation")
        
        report_generator = FindingsReport(
            analysis_dir=os.path.join(self.base_dir, "data/analysis"),
            output_dir=os.path.join(self.base_dir, "data/reports"),
            openai_api_key=self.openai_api_key
        )
        
        report = report_generator.generate_full_report()
        
        logger.info("Completed report generation")
        return report
    
    def run_pipeline(self, skip_scraping=False, skip_processing=False, skip_analysis=False, use_existing_processed=False, max_workers=20):
        """
        Run the complete document analysis pipeline.
        
        Args:
            skip_scraping: Skip document scraping
            skip_processing: Skip document processing
            skip_analysis: Skip document analysis
            use_existing_processed: Use existing processed documents without processing new ones
            max_workers: Number of documents to process in parallel
        """
        logger.info("Starting JFK documents analysis pipeline")
        
        # Step 1: Scrape documents
        if not skip_scraping:
            self.scrape_documents()
        else:
            logger.info("Skipping document scraping")
        
        # Initialize vector store early for immediate embedding
        vector_store = None
        if not skip_analysis:
            vector_store = self.build_vector_database()
            if vector_store is None:
                logger.warning("Failed to initialize vector store, reverting to default processing without embedding")
                skip_analysis = True
        
        # Step 2: Process documents or use existing processed documents
        if not skip_processing:
            if use_existing_processed:
                logger.info("Using existing processed documents")
                processed_files = self.get_processed_documents()
                
                # If we have a vector store, ensure all existing documents are added
                if vector_store is not None:
                    logger.info("Adding existing processed documents to vector store")
                    for file_path in processed_files:
                        vector_store.add_documents_from_file(file_path)
            else:
                # Process documents with immediate embedding if vector store is available
                self.process_documents(max_workers=max_workers, vector_store=vector_store)
        else:
            logger.info("Skipping document processing")
            
            # If we're skipping processing but not analysis, we need to ensure the vector store has all documents
            if not skip_analysis and vector_store is not None:
                self.add_all_documents_to_vector_store(vector_store)
        
        # Step 3: Analyze documents if vector store was created successfully
        if not skip_analysis and vector_store is not None:
            self.analyze_documents(vector_store)
        else:
            logger.info("Skipping vector database build and analysis")
        
        # Step 4: Generate report
        if not skip_analysis:
            self.generate_report()
            # Print final report location
            report_path = os.path.join(self.base_dir, "data/reports/full_report.html")
            logger.info(f"Final report available at: {report_path}")
        else:
            # Create a dummy report or use an existing one
            report_path = os.path.join(self.base_dir, "data/reports/dummy_report.html")
            with open(report_path, "w") as f:
                f.write("<html><body><h1>JFK Document Analysis Report</h1><p>Analysis phase was skipped.</p></body></html>")
            logger.info(f"Created dummy report at: {report_path}")
        
        logger.info("Completed JFK documents analysis pipeline")
        
        return report_path


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="JFKReveal - Analyze declassified JFK assassination documents")
    
    parser.add_argument("--base-dir", type=str, default=".",
                        help="Base directory for data storage")
    parser.add_argument("--openai-api-key", type=str,
                        help="OpenAI API key (uses env var OPENAI_API_KEY if not provided)")
    parser.add_argument("--skip-scraping", action="store_true",
                        help="Skip document scraping")
    parser.add_argument("--skip-processing", action="store_true",
                        help="Skip document processing")
    parser.add_argument("--skip-analysis", action="store_true",
                        help="Skip document analysis and report generation")
    parser.add_argument("--use-existing-processed", action="store_true",
                        help="Use existing processed documents without processing new ones")
    parser.add_argument("--max-workers", type=int, default=20,
                        help="Number of documents to process in parallel (default: 20)")
    parser.add_argument("--no-clean-text", action="store_true",
                        help="Disable text cleaning for OCR documents")
    
    args = parser.parse_args()
    
    # Run pipeline
    jfk_reveal = JFKReveal(
        base_dir=args.base_dir,
        openai_api_key=args.openai_api_key,
        clean_text=not args.no_clean_text
    )
    
    report_path = jfk_reveal.run_pipeline(
        skip_scraping=args.skip_scraping,
        skip_processing=args.skip_processing,
        skip_analysis=args.skip_analysis,
        use_existing_processed=args.use_existing_processed,
        max_workers=args.max_workers
    )
    
    print(f"\nAnalysis complete! Final report available at: {report_path}")


if __name__ == "__main__":
    main()