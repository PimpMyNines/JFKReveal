"""
Test script for findings_report.py
"""
import logging
from src.jfkreveal.summarization.findings_report import FindingsReport

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def test_findings_report():
    """Test the FindingsReport class with LangChain implementation."""
    report_generator = FindingsReport(
        analysis_dir="data/analysis",
        output_dir="data/reports",
        model_name="gpt-4o",
        temperature=0.1
    )
    
    # Test loading analyses
    analyses = report_generator.load_analyses()
    print(f"Loaded {len(analyses)} analysis files")
    
    # Test generating executive summary
    if analyses:
        print("Generating executive summary...")
        executive_summary = report_generator.generate_executive_summary(analyses[:2])
        print("Executive summary generated successfully.")
        
        # Save the summary to a file for testing
        with open("data/reports/test_summary.md", "w") as f:
            f.write(executive_summary)
        print("Saved test summary to data/reports/test_summary.md")
    else:
        print("No analyses found to generate summary.")

if __name__ == "__main__":
    test_findings_report()