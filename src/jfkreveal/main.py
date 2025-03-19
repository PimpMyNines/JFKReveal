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
        openai_api_key: Optional[str] = None
    ):
        """
        Initialize the JFK document analysis pipeline.
        
        Args:
            base_dir: Base directory for data
            openai_api_key: OpenAI API key (uses environment variable if not provided)
        """
        self.base_dir = base_dir
        self.openai_api_key = openai_api_key
        
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
    
    def process_documents(self):
        """Process PDF documents and extract text."""
        logger.info("Starting document processing")
        
        processor = DocumentProcessor(
            input_dir=os.path.join(self.base_dir, "data/raw"),
            output_dir=os.path.join(self.base_dir, "data/processed")
        )
        
        processed_files = processor.process_all_documents()
        
        logger.info(f"Completed document processing, processed {len(processed_files)} files")
        return processed_files
    
    def build_vector_database(self) -> Optional[VectorStore]:
        """Build the vector database from processed documents."""
        logger.info("Starting vector database build")
        
        try:
            vector_store = VectorStore(
                persist_directory=os.path.join(self.base_dir, "data/vectordb"),
                openai_api_key=self.openai_api_key
            )
            
            total_chunks = vector_store.add_all_documents(
                processed_dir=os.path.join(self.base_dir, "data/processed")
            )
            
            logger.info(f"Completed vector database build, added {total_chunks} chunks")
            return vector_store
            
        except Exception as e:
            logger.error(f"Failed to build vector database: {e}")
            logger.error("Skipping vector database build and analysis steps")
            return None
    
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
    
    def run_pipeline(self, skip_scraping=False, skip_processing=False, skip_analysis=False):
        """
        Run the complete document analysis pipeline.
        
        Args:
            skip_scraping: Skip document scraping
            skip_processing: Skip document processing
            skip_analysis: Skip document analysis
        """
        logger.info("Starting JFK documents analysis pipeline")
        
        # Step 1: Scrape documents
        if not skip_scraping:
            self.scrape_documents()
        else:
            logger.info("Skipping document scraping, will use existing files")
        
        # Step 2: Process documents
        if not skip_processing:
            self.process_documents()
        else:
            logger.info("Skipping document processing")
        
        # Step 3: Build vector database
        if not skip_analysis:
            vector_store = self.build_vector_database()
            
            # Step 4: Analyze documents if vector store was created successfully
            if vector_store is not None:
                self.analyze_documents(vector_store)
            else:
                logger.warning("Skipping analysis phase due to vector store initialization failure")
                skip_analysis = True
        else:
            logger.info("Skipping vector database build and analysis")
        
        # Step 5: Generate report
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
    
    args = parser.parse_args()
    
    # Run pipeline
    jfk_reveal = JFKReveal(
        base_dir=args.base_dir,
        openai_api_key=args.openai_api_key
    )
    
    report_path = jfk_reveal.run_pipeline(
        skip_scraping=args.skip_scraping,
        skip_processing=args.skip_processing,
        skip_analysis=args.skip_analysis
    )
    
    print(f"\nAnalysis complete! Final report available at: {report_path}")


if __name__ == "__main__":
    main()