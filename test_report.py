#!/usr/bin/env python3
"""
Generate a test JFK findings report.
"""
import os
import sys
import logging
from jfkreveal.summarization.findings_report import FindingsReport

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger(__name__)

def main():
    """Generate test report"""
    # Update this with your preferred OpenAI model
    model_name = os.environ.get("OPENAI_ANALYSIS_MODEL", "gpt-4o")
    
    # Create the report generator
    report_generator = FindingsReport(
        analysis_dir="data/analysis",
        output_dir="docs/reports",
        raw_docs_dir="data/raw",
        model_name=model_name,
        temperature=0.2,
        max_retries=3,
        pdf_base_url="https://www.archives.gov/files/research/jfk/releases/2025/0318/"
    )
    
    # Generate the report
    logger.info(f"Generating report using model: {model_name}")
    report = report_generator.generate_full_report()
    
    # Success message
    logger.info("Report generation complete!")
    logger.info("Reports saved to docs/reports/")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())