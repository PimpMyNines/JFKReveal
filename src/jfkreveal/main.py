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
        anthropic_api_key: Optional[str] = None,
        xai_api_key: Optional[str] = None,
        model_provider: str = "openai"
    ):
        """
        Initialize the JFK document analysis pipeline.
        
        Args:
            base_dir: Base directory for data
            openai_api_key: OpenAI API key (uses environment variable if not provided)
            anthropic_api_key: Anthropic API key (uses environment variable if not provided)
            xai_api_key: X AI (Grok) API key (uses environment variable if not provided)
            model_provider: Model provider to use ('openai', 'anthropic', or 'xai')
        """
        self.base_dir = base_dir
        self.openai_api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        self.anthropic_api_key = anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.xai_api_key = xai_api_key or os.environ.get("XAI_API_KEY")
        self.model_provider = model_provider.lower()
        
        if self.model_provider not in ["openai", "anthropic", "xai"]:
            logger.warning(f"Unknown model provider: {model_provider}. Defaulting to 'openai'.")
            self.model_provider = "openai"
        
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
        
        downloaded_files, documents = scraper.scrape_all()
        
        logger.info(f"Completed document scraping, using {len(downloaded_files)} files ({len([d for d in documents if d.downloaded and not d.error])} newly downloaded)")
        return downloaded_files
    
    def process_documents(self):
        """Process PDF documents and extract text."""
        logger.info("Starting document processing")
        
        # Check if OCR is disabled via command-line argument
        enable_ocr = True
        ocr_dpi = 300
        
        if hasattr(self, 'args'):
            if hasattr(self.args, 'disable_ocr') and self.args.disable_ocr:
                enable_ocr = False
                logger.info("OCR is disabled via command-line argument")
            
            if hasattr(self.args, 'ocr_dpi'):
                ocr_dpi = self.args.ocr_dpi
        
        processor = DocumentProcessor(
            input_dir=os.path.join(self.base_dir, "data/raw"),
            output_dir=os.path.join(self.base_dir, "data/processed"),
            enable_ocr=enable_ocr,
            ocr_dpi=ocr_dpi
        )
        
        processed_files = processor.process_all_documents()
        
        logger.info(f"Completed document processing, processed {len(processed_files)} files")
        return processed_files
    
    def build_vector_database(self) -> Optional[VectorStore]:
        """Build the vector database from processed documents."""
        logger.info("Starting vector database build")
        
        try:
            # For now we still use OpenAI embeddings even with Anthropic or XAI
            embedding_model = os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")
            logger.info(f"Using embedding model: {embedding_model}")
            
            vector_store = VectorStore(
                embedding_model=embedding_model,
                openai_api_key=self.openai_api_key
            )
            
            # Add all processed documents
            total_chunks = vector_store.add_all_documents()
            
            # Add summary logging
            if self.args.summarize_embeddings:
                logger.info("=== Vector Database Summary ===")
                logger.info(f"Total chunks embedded: {total_chunks}")
                logger.info(f"Using embedding model: {embedding_model}")
                logger.info(f"Vector store location: {vector_store.persist_directory}")
                logger.info("============================")
            
            return vector_store
            
        except Exception as e:
            logger.error(f"Error building vector database: {e}")
            return None
    
    def analyze_documents(self, vector_store):
        """Analyze documents and generate topic analyses."""
        logger.info("Starting document analysis")
        
        if self.model_provider == "anthropic":
            # Get model name from environment variables if set
            model_name = os.environ.get("ANTHROPIC_ANALYSIS_MODEL", "claude-3-7-sonnet-20240620")
            logger.info(f"Using Anthropic model: {model_name}")
            
            analyzer = DocumentAnalyzer(
                vector_store=vector_store,
                output_dir=os.path.join(self.base_dir, "data/analysis"),
                model_name=model_name,
                model_provider="anthropic",
                anthropic_api_key=self.anthropic_api_key,
                openai_api_key=self.openai_api_key,  # Still include for embedding fallback
                temperature=0.0,
                max_retries=5
            )
        elif self.model_provider == "xai":
            # Get model name from environment variables if set
            model_name = os.environ.get("XAI_ANALYSIS_MODEL", "grok-2")
            logger.info(f"Using X AI (Grok) model: {model_name}")
            
            analyzer = DocumentAnalyzer(
                vector_store=vector_store,
                output_dir=os.path.join(self.base_dir, "data/analysis"),
                model_name=model_name,
                model_provider="xai",
                xai_api_key=self.xai_api_key,
                openai_api_key=self.openai_api_key,  # Still include for embedding fallback
                temperature=0.0,
                max_retries=5
            )
        else:
            # Get model name from environment variables if set
            model_name = os.environ.get("OPENAI_ANALYSIS_MODEL", "gpt-4o")
            logger.info(f"Using OpenAI model: {model_name}")
            
            analyzer = DocumentAnalyzer(
                vector_store=vector_store,
                output_dir=os.path.join(self.base_dir, "data/analysis"),
                model_name=model_name,
                model_provider="openai",
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
        
        if self.model_provider == "anthropic":
            model_name = os.environ.get("ANTHROPIC_REPORT_MODEL", "claude-3-7-sonnet-20240620")
            logger.info(f"Using Anthropic model for report generation: {model_name}")
            
            report_generator = FindingsReport(
                analysis_dir=os.path.join(self.base_dir, "data/analysis"),
                output_dir=os.path.join(self.base_dir, "data/reports"),
                model_name=model_name,
                model_provider="anthropic",
                anthropic_api_key=self.anthropic_api_key,
                openai_api_key=self.openai_api_key  # Include for backward compatibility
            )
        elif self.model_provider == "xai":
            model_name = os.environ.get("XAI_REPORT_MODEL", "grok-2")
            logger.info(f"Using X AI (Grok) model for report generation: {model_name}")
            
            report_generator = FindingsReport(
                analysis_dir=os.path.join(self.base_dir, "data/analysis"),
                output_dir=os.path.join(self.base_dir, "data/reports"),
                model_name=model_name,
                model_provider="xai",
                xai_api_key=self.xai_api_key,
                openai_api_key=self.openai_api_key  # Include for embedding fallback
            )
        else:
            model_name = os.environ.get("OPENAI_REPORT_MODEL", "gpt-4o")
            logger.info(f"Using OpenAI model for report generation: {model_name}")
            
            report_generator = FindingsReport(
                analysis_dir=os.path.join(self.base_dir, "data/analysis"),
                output_dir=os.path.join(self.base_dir, "data/reports"),
                model_name=model_name,
                model_provider="openai",
                openai_api_key=self.openai_api_key
            )
        
        report = report_generator.generate_full_report()
        
        logger.info("Completed report generation")
        return report
    
    def run_pipeline(self, skip_scraping=False, skip_processing=False, skip_analysis=False):
        """Run the full document analysis pipeline."""
        logger.info("Starting JFKReveal pipeline")
        
        # Step 1: Scrape documents
        if not skip_scraping:
            self.scrape_documents()
        else:
            logger.info("Skipping document scraping")
        
        # Step 2: Process documents
        if not skip_processing:
            # Determine OCR settings
            enable_ocr = True
            ocr_dpi = 300
            
            if hasattr(self, 'args'):
                if hasattr(self.args, 'disable_ocr') and self.args.disable_ocr:
                    enable_ocr = False
                    logger.info("OCR is disabled via command-line argument")
                
                if hasattr(self.args, 'ocr_dpi'):
                    ocr_dpi = self.args.ocr_dpi
                    logger.info(f"OCR resolution set to {ocr_dpi} DPI")
            
            # Use token-based chunking if requested
            if hasattr(self, 'args') and self.args.use_token_chunking:
                logger.info(f"Using token-based chunking with size={self.args.token_chunk_size}, overlap={self.args.token_chunk_overlap}")
                processor = DocumentProcessor(
                    input_dir=os.path.join(self.base_dir, "data/raw"),
                    output_dir=os.path.join(self.base_dir, "data/processed"),
                    chunk_size=self.args.token_chunk_size,
                    chunk_overlap=self.args.token_chunk_overlap,
                    use_token_based=True,
                    enable_ocr=enable_ocr,
                    ocr_dpi=ocr_dpi
                )
            else:
                # Use default character-based chunking
                processor = DocumentProcessor(
                    input_dir=os.path.join(self.base_dir, "data/raw"),
                    output_dir=os.path.join(self.base_dir, "data/processed"),
                    enable_ocr=enable_ocr,
                    ocr_dpi=ocr_dpi
                )
            
            processor.process_all_documents()
        else:
            logger.info("Skipping document processing")
        
        # Step 3: Build vector database
        if not skip_analysis:
            # Build vector store with enhanced options if provided
            if hasattr(self, 'args'):
                # Configure embedding options from arguments
                embedding_model = os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")
                logger.info(f"Using embedding model: {embedding_model}")
                
                # Apply enhanced embedding options
                self.vector_store = VectorStore(
                    persist_directory=os.path.join(self.base_dir, "data/vectordb"),
                    embedding_model=embedding_model,
                    openai_api_key=self.openai_api_key,
                    batch_size=self.args.embedding_batch_size if hasattr(self.args, 'embedding_batch_size') else 16,
                    normalize_text=self.args.normalize_text if hasattr(self.args, 'normalize_text') else False,
                    add_document_context=self.args.add_doc_context if hasattr(self.args, 'add_doc_context') else False
                )
            else:
                # Use default vector store configuration
                self.vector_store = self.build_vector_database()
            
            if not self.vector_store:
                logger.error("Failed to build vector database, exiting")
                return
            
            # Step 4: Analyze documents
            if hasattr(self, 'args'):
                # Configure analyzer with enhanced options
                analyzer = DocumentAnalyzer(
                    vector_store=self.vector_store,
                    output_dir=os.path.join(self.base_dir, "data/analysis"),
                    model_name=self._get_model_name(),
                    model_provider=self.model_provider,
                    openai_api_key=self.openai_api_key,
                    anthropic_api_key=self.anthropic_api_key,
                    xai_api_key=self.xai_api_key,
                    # Add enhanced options
                    use_mmr=self.args.use_mmr if hasattr(self.args, 'use_mmr') else False,
                    validate_outputs=self.args.validate_outputs if hasattr(self.args, 'validate_outputs') else False,
                    include_few_shot=self.args.use_few_shot if hasattr(self.args, 'use_few_shot') else False
                )
                
                # Use document-level aggregation if requested
                if hasattr(self.args, 'aggregate_documents') and self.args.aggregate_documents:
                    logger.info("Using document-level aggregation for analysis")
                    for topic in analyzer.KEY_TOPICS:
                        analysis = analyzer.search_and_analyze_with_aggregation(
                            topic=topic,
                            use_mmr=self.args.use_mmr if hasattr(self.args, 'use_mmr') else False,
                            aggregate_documents=True
                        )
                        logger.info(f"Completed analysis for topic: {topic}")
                else:
                    # Standard topic analysis
                    analyzer.analyze_key_topics()
            else:
                # Use standard analysis
                self.analyze_documents(self.vector_store)
            
            # Step 5: Generate report
            self.generate_report()
        else:
            logger.info("Skipping document analysis")
        
        logger.info("JFKReveal pipeline completed")
        
        # Return a meaningful result
        return {
            "success": True,
            "documents_processed": len(os.listdir(os.path.join(self.base_dir, "data/processed"))),
            "reports_generated": len(os.listdir(os.path.join(self.base_dir, "data/reports")))
        }

    def _get_model_name(self):
        """Get the appropriate model name based on provider."""
        if self.model_provider == "anthropic":
            return os.environ.get("ANTHROPIC_ANALYSIS_MODEL", "claude-3-7-sonnet-20240620")
        elif self.model_provider == "xai":
            return os.environ.get("XAI_ANALYSIS_MODEL", "grok-2")
        else:
            return os.environ.get("OPENAI_ANALYSIS_MODEL", "gpt-4o")

def main():
    """Entry point for the JFK document analysis module."""
    parser = argparse.ArgumentParser(description="JFK Documents Analysis Tool")
    
    # Core functionality flags
    parser.add_argument("--skip-scraping", action="store_true", help="Skip document scraping")
    parser.add_argument("--skip-processing", action="store_true", help="Skip document processing")
    parser.add_argument("--skip-analysis", action="store_true", help="Skip document analysis")
    
    # Model selection options
    parser.add_argument("--model-provider", type=str, default="openai", 
                        choices=["openai", "anthropic", "xai"], 
                        help="Model provider to use (openai, anthropic, xai)")
    
    # API keys
    parser.add_argument("--openai-api-key", type=str, help="OpenAI API key")
    parser.add_argument("--anthropic-api-key", type=str, help="Anthropic API key")
    parser.add_argument("--xai-api-key", type=str, help="X AI (Grok) API key")
    
    # Processing options
    parser.add_argument("--use-token-chunking", action="store_true", 
                        help="Use token-based chunking instead of character-based")
    parser.add_argument("--token-chunk-size", type=int, default=500, 
                        help="Token chunk size (default: 500)")
    parser.add_argument("--token-chunk-overlap", type=int, default=100,
                        help="Token chunk overlap (default: 100)")
    parser.add_argument("--disable-ocr", action="store_true",
                        help="Disable OCR for PDF processing")
    parser.add_argument("--ocr-dpi", type=int, default=300,
                        help="DPI resolution for OCR (default: 300)")
    
    # Output options
    parser.add_argument("--summarize-embeddings", action="store_true",
                        help="Show summary of document embeddings")
    
    args = parser.parse_args()
    
    # Create JFKReveal instance with model provider
    jfk = JFKReveal(
        model_provider=args.model_provider,
        openai_api_key=args.openai_api_key,
        anthropic_api_key=args.anthropic_api_key,
        xai_api_key=args.xai_api_key
    )
    jfk.args = args  # Store args for later use
    
    # Run the pipeline
    jfk.run_pipeline(
        skip_scraping=args.skip_scraping,
        skip_processing=args.skip_processing, 
        skip_analysis=args.skip_analysis
    )
    
    # Run custom query if provided
    if args.query:
        if not jfk.vector_store:
            jfk.vector_store = jfk.build_vector_database()
            
        if jfk.vector_store:
            analyzer = DocumentAnalyzer(
                vector_store=jfk.vector_store,
                output_dir=os.path.join(jfk.base_dir, "data/analysis"),
                model_name=jfk._get_model_name(),
                model_provider=jfk.model_provider,
                openai_api_key=jfk.openai_api_key,
                anthropic_api_key=jfk.anthropic_api_key,
                xai_api_key=jfk.xai_api_key,
                # Add new RAG enhancement options
                use_mmr=args.use_mmr,
                validate_outputs=args.validate_outputs,
                include_few_shot=args.use_few_shot
            )
            
            if args.aggregate_documents:
                logger.info(f"Running query with document aggregation: {args.query}")
                analysis = analyzer.search_and_analyze_with_aggregation(
                    topic=args.query,
                    num_results=20,
                    use_mmr=args.use_mmr,
                    aggregate_documents=True
                )
            else:
                logger.info(f"Running query: {args.query}")
                analysis = analyzer.search_and_analyze_topic(
                    topic=args.query,
                    num_results=20,
                    use_mmr=args.use_mmr,
                    mmr_diversity=args.mmr_diversity,
                    hybrid_search=args.use_hybrid_search
                )
            
            logger.info(f"Completed query analysis for: {args.query}")
            logger.info(f"Analysis saved to: {os.path.join(jfk.base_dir, 'data/analysis', args.query.replace(' ', '_').lower() + '.json')}")

if __name__ == "__main__":
    main()