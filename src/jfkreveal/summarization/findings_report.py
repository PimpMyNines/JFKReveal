"""
Generate comprehensive findings report from document analyses.
"""
import os
import json
import logging
from typing import List, Dict, Any, Optional
import datetime

from openai import OpenAI
import markdown

logger = logging.getLogger(__name__)

class FindingsReport:
    """Generate comprehensive findings report from document analyses."""
    
    def __init__(
        self,
        analysis_dir: str = "data/analysis",
        output_dir: str = "data/reports",
        model: str = "gpt-4o",
        openai_api_key: Optional[str] = None,
    ):
        """
        Initialize the findings report generator.
        
        Args:
            analysis_dir: Directory containing analysis files
            output_dir: Directory to save reports
            model: OpenAI model to use
            openai_api_key: OpenAI API key (uses environment variable if not provided)
        """
        self.analysis_dir = analysis_dir
        self.output_dir = output_dir
        self.model = model
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize OpenAI client
        self.client = OpenAI(api_key=openai_api_key)
    
    def load_analyses(self) -> List[Dict[str, Any]]:
        """
        Load all analysis files.
        
        Returns:
            List of analysis data
        """
        analyses = []
        
        # Find all JSON files
        for file in os.listdir(self.analysis_dir):
            if file.endswith('.json') and not file.endswith('_partial.json'):
                file_path = os.path.join(self.analysis_dir, file)
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        analysis = json.load(f)
                        analyses.append(analysis)
                except Exception as e:
                    logger.error(f"Error loading analysis file {file_path}: {e}")
        
        logger.info(f"Loaded {len(analyses)} analysis files")
        return analyses
    
    def generate_executive_summary(self, analyses: List[Dict[str, Any]]) -> str:
        """
        Generate an executive summary of findings.
        
        Args:
            analyses: List of analysis data
            
        Returns:
            Executive summary markdown text
        """
        # Create a summary of all the analyses
        analyses_summary = []
        for analysis in analyses:
            topic = analysis.get("topic", "Unknown")
            summary = analysis.get("summary", {})
            
            # Add a brief topic summary
            topic_summary = {
                "topic": topic,
                "key_findings": summary.get("key_findings", []),
                "potential_evidence": summary.get("potential_evidence_of_coverup_or_conspiracy", []),
                "credibility": summary.get("level_of_credibility", "Unknown")
            }
            analyses_summary.append(topic_summary)
            
        prompt = f"""
        You are an expert analyst tasked with creating an executive summary of findings from the JFK assassination documents.
        
        Based on the following analyses of declassified JFK documents, create a comprehensive executive summary.
        
        ANALYSES SUMMARY:
        {json.dumps(analyses_summary, indent=2)}
        
        Your executive summary should:
        
        1. Provide an overview of the key findings across all analyzed topics
        2. Highlight the most significant evidence related to the assassination
        3. Discuss potential government involvement or coverup with specific evidence
        4. Address the most credible theories based on document evidence
        5. Identify the most likely culprit(s) based on the evidence
        6. Present alternative suspects with supporting evidence
        7. Discuss any patterns of redaction or information withholding
        8. Evaluate the overall credibility and completeness of the documents
        
        FORMAT YOUR RESPONSE AS MARKDOWN with appropriate headings, bullet points, and emphasis.
        Include specific document references for key claims where possible.
        BE OBJECTIVE AND FACTUAL, but do not hesitate to draw reasonable conclusions from the evidence.
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert analyst examining declassified JFK assassination documents. Create a comprehensive executive summary in Markdown format."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
            )
            
            executive_summary = response.choices[0].message.content
            return executive_summary
            
        except Exception as e:
            logger.error(f"Error generating executive summary: {e}")
            return f"# Error Generating Executive Summary\n\nAn error occurred: {e}"
    
    def generate_detailed_findings(self, analyses: List[Dict[str, Any]]) -> str:
        """
        Generate detailed findings report.
        
        Args:
            analyses: List of analysis data
            
        Returns:
            Detailed findings markdown text
        """
        prompt = f"""
        You are an expert analyst tasked with creating a detailed findings report from the JFK assassination documents.
        
        Based on the following analyses of declassified JFK documents, create a comprehensive detailed report.
        
        ANALYSES:
        {json.dumps(analyses, indent=2)}
        
        Your detailed report should:
        
        1. Provide an in-depth examination of each key topic
        2. Present a chronological timeline of events based on the documents
        3. Detail the roles and actions of key individuals and agencies
        4. Analyze the evidence for various assassination theories
        5. Examine inconsistencies and contradictions in official accounts
        6. Identify patterns of information withholding or redaction
        7. Evaluate the credibility of different pieces of evidence
        8. Draw reasoned conclusions about the most likely scenarios
        9. Identify the most likely culprit(s) with supporting evidence
        10. Present alternative suspects with detailed analysis of their potential involvement
        
        FORMAT YOUR RESPONSE AS MARKDOWN with appropriate headings, subheadings, bullet points, and emphasis.
        Include specific document references (document_id) for all key claims.
        BE OBJECTIVE AND FACTUAL, but do not hesitate to draw reasonable conclusions from the evidence.
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert analyst examining declassified JFK assassination documents. Create a comprehensive detailed report in Markdown format."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=4000
            )
            
            detailed_findings = response.choices[0].message.content
            return detailed_findings
            
        except Exception as e:
            logger.error(f"Error generating detailed findings: {e}")
            return f"# Error Generating Detailed Findings\n\nAn error occurred: {e}"
    
    def generate_suspects_analysis(self, analyses: List[Dict[str, Any]]) -> str:
        """
        Generate an analysis of potential suspects.
        
        Args:
            analyses: List of analysis data
            
        Returns:
            Suspects analysis markdown text
        """
        prompt = f"""
        You are an expert analyst tasked with analyzing potential suspects in the JFK assassination.
        
        Based on the following analyses of declassified JFK documents, create a comprehensive analysis of potential suspects.
        
        ANALYSES:
        {json.dumps(analyses, indent=2)}
        
        Your suspects analysis should:
        
        1. Identify the most likely primary culprit(s) based on document evidence
        2. Provide detailed supporting evidence for this conclusion
        3. Analyze the strength of evidence against this suspect/group
        4. Identify gaps or weaknesses in the case against them
        5. Present alternative suspects in order of likelihood
        6. For each alternative suspect:
           - Summarize the evidence implicating them
           - Assess the credibility of this evidence
           - Identify contradicting evidence
           - Evaluate their capability and motive
        7. Analyze possible collaborations between suspects
        8. Assess government knowledge or involvement in the assassination
        9. Evaluate the evidence for conspiracy vs. lone gunman theories
        
        FORMAT YOUR RESPONSE AS MARKDOWN with appropriate headings, subheadings, bullet points, and emphasis.
        Include specific document references (document_id) for all key claims.
        BE OBJECTIVE AND FACTUAL, but do not hesitate to draw reasonable conclusions from the evidence.
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert analyst examining declassified JFK assassination documents. Create a comprehensive analysis of potential suspects in Markdown format."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
            )
            
            suspects_analysis = response.choices[0].message.content
            return suspects_analysis
            
        except Exception as e:
            logger.error(f"Error generating suspects analysis: {e}")
            return f"# Error Generating Suspects Analysis\n\nAn error occurred: {e}"
    
    def generate_coverup_analysis(self, analyses: List[Dict[str, Any]]) -> str:
        """
        Generate an analysis of potential coverups.
        
        Args:
            analyses: List of analysis data
            
        Returns:
            Coverup analysis markdown text
        """
        prompt = f"""
        You are an expert analyst tasked with analyzing potential coverups in the JFK assassination.
        
        Based on the following analyses of declassified JFK documents, create a comprehensive analysis of potential coverups.
        
        ANALYSES:
        {json.dumps(analyses, indent=2)}
        
        Your coverup analysis should:
        
        1. Identify evidence of information suppression or tampering
        2. Analyze patterns of redaction across documents
        3. Identify inconsistencies in official narratives
        4. Examine the timeline of information releases and their context
        5. Analyze suspicious behaviors by government agencies
        6. Identify disappearance or destruction of evidence
        7. Examine treatment of witnesses and their testimonies
        8. Analyze unusual classification or handling of documents
        9. Identify potential motives for a coverup
        10. Evaluate which entities would have benefited from a coverup
        
        FORMAT YOUR RESPONSE AS MARKDOWN with appropriate headings, subheadings, bullet points, and emphasis.
        Include specific document references (document_id) for all key claims.
        BE OBJECTIVE AND FACTUAL, but do not hesitate to draw reasonable conclusions from the evidence.
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert analyst examining declassified JFK assassination documents. Create a comprehensive analysis of potential coverups in Markdown format."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
            )
            
            coverup_analysis = response.choices[0].message.content
            return coverup_analysis
            
        except Exception as e:
            logger.error(f"Error generating coverup analysis: {e}")
            return f"# Error Generating Coverup Analysis\n\nAn error occurred: {e}"
    
    def generate_full_report(self) -> Dict[str, str]:
        """
        Generate full findings report.
        
        Returns:
            Dictionary of report sections
        """
        # Load all analyses
        analyses = self.load_analyses()
        
        if not analyses:
            logger.error("No analyses found. Cannot generate report.")
            return {"error": "No analyses found"}
        
        # Generate report sections
        executive_summary = self.generate_executive_summary(analyses)
        detailed_findings = self.generate_detailed_findings(analyses)
        suspects_analysis = self.generate_suspects_analysis(analyses)
        coverup_analysis = self.generate_coverup_analysis(analyses)
        
        # Create report structure
        report = {
            "executive_summary": executive_summary,
            "detailed_findings": detailed_findings,
            "suspects_analysis": suspects_analysis,
            "coverup_analysis": coverup_analysis
        }
        
        # Save each section as markdown
        for section_name, content in report.items():
            output_file = os.path.join(self.output_dir, f"{section_name}.md")
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(content)
            logger.info(f"Saved {section_name} to {output_file}")
            
            # Also save as HTML
            html_output = os.path.join(self.output_dir, f"{section_name}.html")
            html_content = markdown.markdown(content, extensions=['tables', 'fenced_code'])
            with open(html_output, 'w', encoding='utf-8') as f:
                f.write(f"""<!DOCTYPE html>
                <html>
                <head>
                    <meta charset="utf-8">
                    <title>JFK Assassination Analysis - {section_name.replace('_', ' ').title()}</title>
                    <style>
                        body {{ font-family: Arial, sans-serif; line-height: 1.6; max-width: 1000px; margin: 0 auto; padding: 20px; }}
                        h1 {{ color: #333; border-bottom: 2px solid #333; padding-bottom: 10px; }}
                        h2 {{ color: #444; margin-top: 30px; border-bottom: 1px solid #ddd; padding-bottom: 5px; }}
                        h3 {{ color: #555; }}
                        blockquote {{ background-color: #f9f9f9; border-left: 10px solid #ccc; margin: 1.5em 10px; padding: 1em 10px; }}
                        code {{ background-color: #f4f4f4; padding: 2px 4px; border-radius: 4px; }}
                        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                        th {{ background-color: #f2f2f2; }}
                        tr:nth-child(even) {{ background-color: #f9f9f9; }}
                    </style>
                </head>
                <body>
                    {html_content}
                    <footer>
                        <p><em>Generated on {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</em></p>
                    </footer>
                </body>
                </html>""")
        
        # Create combined report
        combined_report = f"""# JFK Assassination Analysis - Full Report

{executive_summary}

---

{detailed_findings}

---

{suspects_analysis}

---

{coverup_analysis}

---

*Generated on {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*
"""
        
        # Save combined report
        combined_output = os.path.join(self.output_dir, "full_report.md")
        with open(combined_output, 'w', encoding='utf-8') as f:
            f.write(combined_report)
        logger.info(f"Saved combined report to {combined_output}")
        
        # Save as HTML
        html_output = os.path.join(self.output_dir, "full_report.html")
        html_content = markdown.markdown(combined_report, extensions=['tables', 'fenced_code'])
        with open(html_output, 'w', encoding='utf-8') as f:
            f.write(f"""<!DOCTYPE html>
            <html>
            <head>
                <meta charset="utf-8">
                <title>JFK Assassination Analysis - Full Report</title>
                <style>
                    body {{ font-family: Arial, sans-serif; line-height: 1.6; max-width: 1000px; margin: 0 auto; padding: 20px; }}
                    h1 {{ color: #333; border-bottom: 2px solid #333; padding-bottom: 10px; }}
                    h2 {{ color: #444; margin-top: 30px; border-bottom: 1px solid #ddd; padding-bottom: 5px; }}
                    h3 {{ color: #555; }}
                    blockquote {{ background-color: #f9f9f9; border-left: 10px solid #ccc; margin: 1.5em 10px; padding: 1em 10px; }}
                    code {{ background-color: #f4f4f4; padding: 2px 4px; border-radius: 4px; }}
                    table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #f2f2f2; }}
                    tr:nth-child(even) {{ background-color: #f9f9f9; }}
                </style>
            </head>
            <body>
                {html_content}
                <footer>
                    <p><em>Generated on {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</em></p>
                </footer>
            </body>
            </html>""")
        
        return report