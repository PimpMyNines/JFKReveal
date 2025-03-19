"""
Generate comprehensive findings report from document analyses.
"""
import os
import json
import logging
import datetime
from typing import List, Dict, Any, Optional

from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from langchain_openai import ChatOpenAI
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.exceptions import LangChainException
import markdown

logger = logging.getLogger(__name__)

class ExecutiveSummaryResponse(BaseModel):
    """Executive summary response from LLM."""
    overview: str = Field(..., description="Overview of key findings across all analyzed topics")
    significant_evidence: List[str] = Field(default_factory=list, description="Most significant evidence related to the assassination")
    potential_government_involvement: List[str] = Field(default_factory=list, description="Evidence of potential government involvement or coverup")
    credible_theories: List[str] = Field(default_factory=list, description="Most credible theories based on document evidence")
    likely_culprits: List[str] = Field(default_factory=list, description="Most likely culprit(s) based on the evidence")
    alternative_suspects: List[str] = Field(default_factory=list, description="Alternative suspects with supporting evidence")
    redaction_patterns: List[str] = Field(default_factory=list, description="Patterns of redaction or information withholding")
    document_credibility: str = Field(..., description="Evaluation of overall credibility and completeness of the documents")

class DetailedFindingsResponse(BaseModel):
    """Detailed findings response from LLM."""
    topic_analyses: Dict[str, str] = Field(..., description="In-depth examination of each key topic")
    timeline: str = Field(..., description="Chronological timeline of events based on the documents")
    key_individuals: Dict[str, str] = Field(..., description="Roles and actions of key individuals and agencies")
    theory_analysis: Dict[str, str] = Field(..., description="Analysis of evidence for various assassination theories")
    inconsistencies: List[str] = Field(default_factory=list, description="Inconsistencies and contradictions in official accounts")
    information_withholding: List[str] = Field(default_factory=list, description="Patterns of information withholding or redaction")
    evidence_credibility: Dict[str, str] = Field(..., description="Credibility of different pieces of evidence")
    likely_scenarios: List[str] = Field(default_factory=list, description="Reasoned conclusions about the most likely scenarios")
    primary_suspects: Dict[str, List[str]] = Field(..., description="Most likely culprit(s) with supporting evidence")
    alternative_suspects_analysis: Dict[str, Dict[str, Any]] = Field(..., description="Alternative suspects with detailed analysis")

class SuspectsAnalysisResponse(BaseModel):
    """Suspects analysis response from LLM."""
    primary_culprits: List[str] = Field(default_factory=list, description="Most likely primary culprit(s) based on document evidence")
    supporting_evidence: Dict[str, List[str]] = Field(..., description="Supporting evidence for primary culprits")
    evidence_strength: str = Field(..., description="Strength analysis of evidence against primary suspects")
    case_weaknesses: List[str] = Field(default_factory=list, description="Gaps or weaknesses in the case against primary suspects")
    alternative_suspects: List[Dict[str, Any]] = Field(..., description="Alternative suspects in order of likelihood")
    collaborations: List[str] = Field(default_factory=list, description="Possible collaborations between suspects")
    government_involvement: str = Field(..., description="Assessment of government knowledge or involvement")
    conspiracy_analysis: str = Field(..., description="Evidence evaluation for conspiracy vs. lone gunman theories")

class CoverupAnalysisResponse(BaseModel):
    """Coverup analysis response from LLM."""
    information_suppression: List[str] = Field(default_factory=list, description="Evidence of information suppression or tampering")
    redaction_patterns: Dict[str, Any] = Field(..., description="Patterns of redaction across documents")
    narrative_inconsistencies: List[str] = Field(default_factory=list, description="Inconsistencies in official narratives")
    information_timeline: str = Field(..., description="Timeline of information releases and context")
    agency_behavior: Dict[str, List[str]] = Field(..., description="Suspicious behaviors by government agencies")
    evidence_destruction: List[str] = Field(default_factory=list, description="Disappearance or destruction of evidence")
    witness_treatment: List[str] = Field(default_factory=list, description="Treatment of witnesses and their testimonies")
    document_handling: List[str] = Field(default_factory=list, description="Unusual classification or handling of documents")
    coverup_motives: List[str] = Field(default_factory=list, description="Potential motives for a coverup")
    beneficiaries: List[str] = Field(default_factory=list, description="Entities that would have benefited from a coverup")

class FindingsReport:
    """Generate comprehensive findings report from document analyses."""
    
    def __init__(
        self,
        analysis_dir: str = "data/analysis",
        output_dir: str = "data/reports",
        model_name: str = "gpt-4o",
        openai_api_key: Optional[str] = None,
        temperature: float = 0.1,
        max_retries: int = 5,
    ):
        """
        Initialize the findings report generator.
        
        Args:
            analysis_dir: Directory containing analysis files
            output_dir: Directory to save reports
            model_name: OpenAI model to use
            openai_api_key: OpenAI API key (uses environment variable if not provided)
            temperature: Temperature for LLM generation
            max_retries: Maximum number of retries for API calls
        """
        self.analysis_dir = analysis_dir
        self.output_dir = output_dir
        self.model_name = model_name
        self.temperature = temperature
        self.max_retries = max_retries
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize LangChain model
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=openai_api_key,
            max_retries=max_retries,
        )
    
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
    
    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=1, max=60),
        retry=retry_if_exception_type(LangChainException)
    )
    def generate_executive_summary(self, analyses: List[Dict[str, Any]]) -> str:
        """
        Generate an executive summary of findings using LangChain.
        
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
                "potential_evidence": summary.get("potential_evidence", []),
                "credibility": summary.get("credibility", "Unknown")
            }
            analyses_summary.append(topic_summary)
        
        # Create prompt template
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", "You are an expert analyst examining declassified JFK assassination documents. Create a comprehensive executive summary in Markdown format."),
            ("human", """
            You are an expert analyst tasked with creating an executive summary of findings from the JFK assassination documents.
            
            Based on the following analyses of declassified JFK documents, create a comprehensive executive summary.
            
            ANALYSES SUMMARY:
            {analyses_summary}
            
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
            """)
        ])
        
        try:
            # First try to get structured output
            chain = prompt_template | self.llm.with_structured_output(
                ExecutiveSummaryResponse,
                method="function_calling"
            )
            
            # Run the chain
            response = chain.invoke({
                "analyses_summary": json.dumps(analyses_summary, indent=2)
            })
            
            # Convert structured output to markdown
            sections = [
                f"# Executive Summary: JFK Assassination Document Analysis\n\n## Overview\n\n{response.overview}\n",
                "## Significant Evidence\n\n" + "\n".join([f"- {item}" for item in response.significant_evidence]),
                "## Potential Government Involvement\n\n" + "\n".join([f"- {item}" for item in response.potential_government_involvement]),
                "## Most Credible Theories\n\n" + "\n".join([f"- {item}" for item in response.credible_theories]),
                "## Likely Culprits\n\n" + "\n".join([f"- {item}" for item in response.likely_culprits]),
                "## Alternative Suspects\n\n" + "\n".join([f"- {item}" for item in response.alternative_suspects]),
                "## Patterns of Redaction\n\n" + "\n".join([f"- {item}" for item in response.redaction_patterns]),
                f"## Document Credibility Assessment\n\n{response.document_credibility}"
            ]
            
            executive_summary = "\n\n".join(sections)
            return executive_summary
        
        except Exception as e:
            logger.warning(f"Error generating structured executive summary: {e}. Falling back to text generation.")
            
            # Fall back to unstructured text generation
            unstructured_chain = prompt_template | self.llm
            
            # Run the chain with unstructured output
            response = unstructured_chain.invoke({
                "analyses_summary": json.dumps(analyses_summary, indent=2)
            })
            
            # Extract content from the AIMessage
            if hasattr(response, "content"):
                return response.content
            return str(response)
    
    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=1, max=60),
        retry=retry_if_exception_type(LangChainException)
    )
    def generate_detailed_findings(self, analyses: List[Dict[str, Any]]) -> str:
        """
        Generate detailed findings report using LangChain.
        
        Args:
            analyses: List of analysis data
            
        Returns:
            Detailed findings markdown text
        """
        # Create prompt template
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", "You are an expert analyst examining declassified JFK assassination documents. Create a comprehensive detailed report in Markdown format."),
            ("human", """
            You are an expert analyst tasked with creating a detailed findings report from the JFK assassination documents.
            
            Based on the following analyses of declassified JFK documents, create a comprehensive detailed report.
            
            ANALYSES:
            {analyses}
            
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
            """)
        ])
        
        try:
            # Run the chain with unstructured output (DetailedFindingsResponse is too complex for reliable function calling)
            chain = prompt_template | self.llm
            
            # Run the chain with unstructured output
            response = chain.invoke({
                "analyses": json.dumps(analyses, indent=2)
            })
            
            # Extract content from the AIMessage
            if hasattr(response, "content"):
                return response.content
            return str(response)
            
        except Exception as e:
            logger.error(f"Error generating detailed findings: {e}")
            return f"# Error Generating Detailed Findings\n\nAn error occurred: {e}"
    
    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=1, max=60),
        retry=retry_if_exception_type(LangChainException)
    )
    def generate_suspects_analysis(self, analyses: List[Dict[str, Any]]) -> str:
        """
        Generate an analysis of potential suspects using LangChain.
        
        Args:
            analyses: List of analysis data
            
        Returns:
            Suspects analysis markdown text
        """
        # Create prompt template
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", "You are an expert analyst examining declassified JFK assassination documents. Create a comprehensive analysis of potential suspects in Markdown format."),
            ("human", """
            You are an expert analyst tasked with analyzing potential suspects in the JFK assassination.
            
            Based on the following analyses of declassified JFK documents, create a comprehensive analysis of potential suspects.
            
            ANALYSES:
            {analyses}
            
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
            """)
        ])
        
        try:
            # First try to get structured output
            chain = prompt_template | self.llm.with_structured_output(
                SuspectsAnalysisResponse,
                method="function_calling"
            )
            
            # Run the chain
            response = chain.invoke({
                "analyses": json.dumps(analyses, indent=2)
            })
            
            # Convert structured output to markdown
            sections = [
                "# Suspects Analysis: JFK Assassination\n\n## Primary Culprits\n\n" + "\n".join([f"- {culprit}" for culprit in response.primary_culprits]),
                "## Supporting Evidence\n\n"
            ]
            
            # Add supporting evidence section
            for suspect, evidence_list in response.supporting_evidence.items():
                sections.append(f"### Evidence for {suspect}\n\n" + "\n".join([f"- {item}" for item in evidence_list]))
            
            # Add remaining sections
            sections.extend([
                f"## Strength of Evidence\n\n{response.evidence_strength}",
                "## Case Weaknesses\n\n" + "\n".join([f"- {item}" for item in response.case_weaknesses]),
                "## Alternative Suspects\n\n"
            ])
            
            # Add alternative suspects section
            for suspect_data in response.alternative_suspects:
                suspect_name = suspect_data.get("name", "Unknown Suspect")
                sections.append(f"### {suspect_name}\n\n")
                
                for key, value in suspect_data.items():
                    if key != "name" and value:
                        if isinstance(value, list):
                            sections.append(f"#### {key.replace('_', ' ').title()}\n\n" + "\n".join([f"- {item}" for item in value]))
                        else:
                            sections.append(f"#### {key.replace('_', ' ').title()}\n\n{value}")
            
            # Add final sections
            sections.extend([
                "## Possible Collaborations\n\n" + "\n".join([f"- {item}" for item in response.collaborations]),
                f"## Government Involvement\n\n{response.government_involvement}",
                f"## Conspiracy vs. Lone Gunman Analysis\n\n{response.conspiracy_analysis}"
            ])
            
            suspects_analysis = "\n\n".join(sections)
            return suspects_analysis
            
        except Exception as e:
            logger.warning(f"Error generating structured suspects analysis: {e}. Falling back to text generation.")
            
            # Fall back to unstructured text generation
            unstructured_chain = prompt_template | self.llm
            
            # Run the chain with unstructured output
            response = unstructured_chain.invoke({
                "analyses": json.dumps(analyses, indent=2)
            })
            
            # Extract content from the AIMessage
            if hasattr(response, "content"):
                return response.content
            return str(response)
    
    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=1, max=60),
        retry=retry_if_exception_type(LangChainException)
    )
    def generate_coverup_analysis(self, analyses: List[Dict[str, Any]]) -> str:
        """
        Generate an analysis of potential coverups using LangChain.
        
        Args:
            analyses: List of analysis data
            
        Returns:
            Coverup analysis markdown text
        """
        # Create prompt template
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", "You are an expert analyst examining declassified JFK assassination documents. Create a comprehensive analysis of potential coverups in Markdown format."),
            ("human", """
            You are an expert analyst tasked with analyzing potential coverups in the JFK assassination.
            
            Based on the following analyses of declassified JFK documents, create a comprehensive analysis of potential coverups.
            
            ANALYSES:
            {analyses}
            
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
            """)
        ])
        
        try:
            # First try to get structured output
            chain = prompt_template | self.llm.with_structured_output(
                CoverupAnalysisResponse,
                method="function_calling"
            )
            
            # Run the chain
            response = chain.invoke({
                "analyses": json.dumps(analyses, indent=2)
            })
            
            # Convert structured output to markdown
            sections = [
                "# Coverup Analysis: JFK Assassination\n\n## Information Suppression Evidence\n\n" + "\n".join([f"- {item}" for item in response.information_suppression]),
                "## Redaction Patterns\n\n"
            ]
            
            # Add redaction patterns
            for pattern_name, pattern_details in response.redaction_patterns.items():
                if isinstance(pattern_details, list):
                    sections.append(f"### {pattern_name.replace('_', ' ').title()}\n\n" + "\n".join([f"- {item}" for item in pattern_details]))
                else:
                    sections.append(f"### {pattern_name.replace('_', ' ').title()}\n\n{pattern_details}")
            
            # Add remaining sections
            sections.extend([
                "## Narrative Inconsistencies\n\n" + "\n".join([f"- {item}" for item in response.narrative_inconsistencies]),
                f"## Information Release Timeline\n\n{response.information_timeline}",
                "## Suspicious Agency Behaviors\n\n"
            ])
            
            # Add agency behaviors
            for agency, behaviors in response.agency_behavior.items():
                sections.append(f"### {agency}\n\n" + "\n".join([f"- {item}" for item in behaviors]))
            
            # Add final sections
            sections.extend([
                "## Evidence Destruction\n\n" + "\n".join([f"- {item}" for item in response.evidence_destruction]),
                "## Witness Treatment\n\n" + "\n".join([f"- {item}" for item in response.witness_treatment]),
                "## Unusual Document Handling\n\n" + "\n".join([f"- {item}" for item in response.document_handling]),
                "## Potential Coverup Motives\n\n" + "\n".join([f"- {item}" for item in response.coverup_motives]),
                "## Entities That Benefited\n\n" + "\n".join([f"- {item}" for item in response.beneficiaries])
            ])
            
            coverup_analysis = "\n\n".join(sections)
            return coverup_analysis
            
        except Exception as e:
            logger.warning(f"Error generating structured coverup analysis: {e}. Falling back to text generation.")
            
            # Fall back to unstructured text generation
            unstructured_chain = prompt_template | self.llm
            
            # Run the chain with unstructured output
            response = unstructured_chain.invoke({
                "analyses": json.dumps(analyses, indent=2)
            })
            
            # Extract content from the AIMessage
            if hasattr(response, "content"):
                return response.content
            return str(response)
    
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