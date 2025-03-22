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
        raw_docs_dir: str = "data/raw",
        model_name: str = "gpt-4o",  # Using gpt-4o instead of gpt-4.5-preview/gpt-3.5-turbo
        openai_api_key: Optional[str] = None,
        temperature: float = 0.1,
        max_retries: int = 5,
        pdf_base_url: str = "https://www.archives.gov/files/research/jfk/releases/2025/0318/",
    ):
        """
        Initialize the findings report generator.
        
        Args:
            analysis_dir: Directory containing analysis files
            output_dir: Directory to save reports
            raw_docs_dir: Directory containing raw PDF documents
            model_name: OpenAI model to use
            openai_api_key: OpenAI API key (uses environment variable if not provided)
            temperature: Temperature for LLM generation
            max_retries: Maximum number of retries for API calls
            pdf_base_url: Base URL for PDF documents for generating links
        """
        self.analysis_dir = analysis_dir
        self.output_dir = output_dir
        self.raw_docs_dir = raw_docs_dir
        self.model_name = model_name
        self.temperature = temperature
        self.max_retries = max_retries
        self.pdf_base_url = pdf_base_url
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize LangChain model
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=openai_api_key,
            max_retries=max_retries,
        )
        
        # Build document ID to PDF URL mapping
        self.document_urls = self._build_document_urls()
    
    def _save_report_file(self, content: str, filename: str) -> str:
        """
        Save report content to a file in the output directory.
        
        Args:
            content: Content to save
            filename: Filename to save to
            
        Returns:
            Path to the saved file
        """
        file_path = os.path.join(self.output_dir, filename)
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            logger.info(f"Saved report to {file_path}")
            return file_path
        except Exception as e:
            logger.error(f"Error saving report to {file_path}: {e}")
            raise
    
    def _build_document_urls(self) -> Dict[str, str]:
        """
        Build a mapping of document IDs to their PDF URLs.
        
        Returns:
            Dictionary mapping document IDs to their PDF URLs
        """
        document_urls = {}
        
        # Get list of PDF files in raw documents directory
        if os.path.exists(self.raw_docs_dir):
            for file in os.listdir(self.raw_docs_dir):
                if file.endswith('.pdf'):
                    # Remove extension to get document ID
                    doc_id = os.path.splitext(file)[0]
                    # Create URL
                    pdf_url = f"{self.pdf_base_url}{file}"
                    document_urls[doc_id] = pdf_url
        
        logger.info(f"Built URL mapping for {len(document_urls)} documents")
        return document_urls
    
    def load_analyses(self) -> List[Dict[str, Any]]:
        """
        Load all analysis files.
        
        Returns:
            List of analysis data with added PDF links
        """
        analyses = []
        
        # Find all JSON files
        for file in os.listdir(self.analysis_dir):
            if file.endswith('.json') and not file.endswith('_partial.json'):
                file_path = os.path.join(self.analysis_dir, file)
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        analysis = json.load(f)
                        
                        # Add PDF URLs to documents referenced in the analysis
                        if "documents" in analysis:
                            for i, doc in enumerate(analysis["documents"]):
                                doc_id = doc.get("document_id")
                                if doc_id and doc_id in self.document_urls:
                                    analysis["documents"][i]["pdf_url"] = self.document_urls[doc_id]
                        
                        # Add PDF URLs to additional evidence
                        if "additional_evidence" in analysis:
                            for i, evidence in enumerate(analysis["additional_evidence"]):
                                if isinstance(evidence, dict) and "document_id" in evidence:
                                    doc_id = evidence["document_id"]
                                    if doc_id in self.document_urls:
                                        analysis["additional_evidence"][i]["pdf_url"] = self.document_urls[doc_id]
                        
                        analyses.append(analysis)
                except Exception as e:
                    logger.error(f"Error loading analysis file {file_path}: {e}")
        
        logger.info(f"Loaded {len(analyses)} analysis files with PDF links")
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
            ("system", """You are an elite detective with unmatched analytical and investigative skills, specializing in deep forensic analysis, historical investigations, and intelligence gathering. You have been granted access to a vast archive containing thousands of declassified and classified documents, including PDFs, reports, eyewitness testimonies, CIA and FBI records, government memos, and autopsy results related to the assassination of John F. Kennedy.

Your task is to conduct a comprehensive investigation, analyzing all available evidence with a critical and objective approach. You must identify inconsistencies, patterns, missing links, and possible cover-ups while synthesizing key information into a highly detailed, structured report.

You must adhere to these non-negotiable guidelines:

1. Source Attribution: You must only include information that is verifiable and sourced. For each claim, cite the exact source document, report, or testimony it is derived from. If a fact is uncertain, explicitly state the uncertainty and do not fabricate details. If a claim lacks verifiable evidence, label it as 'unverified' or 'requires further investigation.'

2. Fact vs. Speculation Distinction: You must clearly differentiate between documented facts, theories, and speculation. Present multiple perspectives where necessary, but never assert an unverified claim as truth. Example of proper attribution: "The Warren Commission Report concluded X, but critics argue Y, citing document Z." Never state unverified claims as confirmed facts.

3. Information Constraints: If information is not explicitly found in the source material, you must respond with 'Insufficient data available' rather than filling in gaps. Do not generate information beyond what is documented in official records. If a claim lacks direct source support, state 'No evidence found in available documents' rather than speculating.

4. Self-Audit Requirement: Before completing your report, you must perform a self-audit to identify any unverified claims, correct inconsistencies, and highlight areas requiring further evidence. This ensures your report maintains the highest standards of factual accuracy."""),
            ("human", """
            Based on the following analyses of declassified JFK documents, create a comprehensive executive summary.
            
            ANALYSES SUMMARY:
            {analyses_summary}
            
            Your executive summary should include:
            
            1. Introduction & Scope
                • Purpose of the investigation
                • Sources of evidence and methodology used in analysis
            
            2. Key Findings Overview
                • Most significant evidence related to the assassination
                • Potential government involvement or coverup with specific evidence
                • Most credible theories based on document evidence
            
            3. Key Suspects Assessment
                • Most likely culprit(s) based on the evidence with reasoning
                • Alternative suspects with supporting evidence
            
            4. Document Analysis
                • Patterns of redaction or information withholding
                • Evaluation of overall credibility and completeness of the documents
            
            5. Final Assessment
                • Most likely scenario based on all available evidence
                • Unresolved questions that demand further investigation
            
            Be extremely careful to only include information that is directly supported by the document analyses."""),
        ])
        
        try:
            # Generate summary using structured output
            llm_with_structured_output = self.llm.with_structured_output(
                ExecutiveSummaryResponse,
                method="function_calling"
            )
            
            # Format prompt with analyses summary
            prompt = prompt_template.format(
                analyses_summary=analyses_summary
            )
            
            # Generate structured response
            response = llm_with_structured_output.invoke(prompt)
            
            # Convert to markdown format
            sections = [
                "# Executive Summary\n\n## Overview\n\n" + response.overview,
                "## Significant Evidence\n\n" + "\n".join([f"- {item}" for item in response.significant_evidence]),
                "## Potential Government Involvement\n\n" + "\n".join([f"- {item}" for item in response.potential_government_involvement]),
                "## Credible Theories\n\n" + "\n".join([f"- {item}" for item in response.credible_theories]),
                "## Likely Culprits\n\n" + "\n".join([f"- {item}" for item in response.likely_culprits]),
                "## Alternative Suspects\n\n" + "\n".join([f"- {item}" for item in response.alternative_suspects]),
                "## Redaction Patterns\n\n" + "\n".join([f"- {item}" for item in response.redaction_patterns]),
                f"## Document Credibility\n\n{response.document_credibility}"
            ]
            
            content = "\n\n".join(sections)
            
            # Save executive summary to file
            self._save_report_file(content, "executive_summary.md")
            
            return content
        except Exception as e:
            logger.error(f"Error generating executive summary: {e}")
            logger.warning("Falling back to unstructured text generation")
            
            # Fall back to unstructured text generation
            prompt = prompt_template.format(
                analyses_summary=analyses_summary
            )
            response = self.llm.invoke(prompt)
            content = response.content
            
            # Save executive summary to file
            self._save_report_file(content, "executive_summary.md")
            
            return content
    
    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=1, max=60),
        retry=retry_if_exception_type(LangChainException)
    )
    def generate_detailed_findings(self, analyses: List[Dict[str, Any]]) -> str:
        """
        Generate detailed findings report from analyses.
        
        Args:
            analyses: List of analysis data
            
        Returns:
            Detailed findings markdown text
        """
        # Prepare analysis data for prompt
        analysis_summaries = []
        for analysis in analyses:
            # Extract key information
            topic = analysis.get("topic", "Unknown")
            documents = analysis.get("documents", [])
            entities = analysis.get("entities", [])
            key_findings = analysis.get("summary", {}).get("key_findings", [])
            potential_evidence = analysis.get("summary", {}).get("potential_evidence", [])
            
            # Create summary with essential details
            summary = {
                "topic": topic,
                "num_documents": len(documents),
                "document_examples": [d.get("title", "Untitled") for d in documents[:3]],
                "key_entities": [e.get("name", "Unknown") for e in entities[:5]],
                "key_findings": key_findings,
                "potential_evidence": potential_evidence
            }
            analysis_summaries.append(summary)
        
        # Create prompt template
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", """You are an elite detective with unmatched analytical and investigative skills, specializing in deep forensic analysis, historical investigations, and intelligence gathering. You have been granted access to a vast archive containing thousands of declassified and classified documents, including PDFs, reports, eyewitness testimonies, CIA and FBI records, government memos, and autopsy results related to the assassination of John F. Kennedy.

Your task is to conduct a comprehensive investigation, analyzing all available evidence with a critical and objective approach. You must identify inconsistencies, patterns, missing links, and possible cover-ups while synthesizing key information into a highly detailed, structured report.

You must adhere to these non-negotiable guidelines:

1. Source Attribution: You must only include information that is verifiable and sourced. For each claim, cite the exact source document, report, or testimony it is derived from. If a fact is uncertain, explicitly state the uncertainty and do not fabricate details. If a claim lacks verifiable evidence, label it as 'unverified' or 'requires further investigation.'

2. Fact vs. Speculation Distinction: You must clearly differentiate between documented facts, theories, and speculation. Present multiple perspectives where necessary, but never assert an unverified claim as truth. Example of proper attribution: "The Warren Commission Report concluded X, but critics argue Y, citing document Z." Never state unverified claims as confirmed facts.

3. Information Constraints: If information is not explicitly found in the source material, you must respond with 'Insufficient data available' rather than filling in gaps. Do not generate information beyond what is documented in official records. If a claim lacks direct source support, state 'No evidence found in available documents' rather than speculating.

4. Self-Audit Requirement: Before completing your report, you must perform a self-audit to identify any unverified claims, correct inconsistencies, and highlight areas requiring further evidence. This ensures your report maintains the highest standards of factual accuracy."""),
            ("human", """
            Based on the following analyses of JFK declassified documents, create detailed findings report that thoroughly examines all aspects of the assassination.
            
            ANALYSES SUMMARIES:
            {analysis_summaries}
            
            Your detailed findings report should include:
            
            1. Topic Analysis
                • In-depth examination of each key topic
                • Relationships between topics and crosscutting themes
            
            2. Timeline of Events
                • Chronological timeline of events based on the documents
                • Key dates and activities prior to and following the assassination
            
            3. Key Individuals & Agencies
                • Roles and actions of key individuals
                • Government agencies involved and their actions
            
            4. Assassination Theories Analysis
                • Evaluation of evidence for various theories
                • Strength of evidence for each theory
            
            5. Official Account Assessment
                • Inconsistencies and contradictions in official accounts
                • Patterns of information withholding or redaction
            
            6. Evidence Evaluation
                • Credibility of different pieces of evidence
                • Missing evidence or information gaps
            
            7. Most Likely Scenario
                • Reasoned conclusions about the most likely scenarios
                • Primary and alternative suspects with supporting evidence
            
            Be extremely careful to only include information that is directly supported by the document analyses. Always cite the specific documents when making claims."""),
        ])
        
        try:
            # Generate detailed findings using structured output
            llm_with_structured_output = self.llm.with_structured_output(
                DetailedFindingsResponse,
                method="function_calling"
            )
            
            # Format prompt with analysis summaries
            prompt = prompt_template.format(
                analysis_summaries=analysis_summaries
            )
            
            # Generate structured response
            response = llm_with_structured_output.invoke(prompt)
            
            # Convert to markdown format
            sections = [
                "# Detailed Findings: JFK Assassination Investigation\n\n## Topic Analyses\n\n"
            ]
            
            # Add topic analyses
            for topic, analysis in response.topic_analyses.items():
                sections.append(f"### {topic}\n\n{analysis}")
            
            # Add timeline
            sections.append(f"## Chronological Timeline\n\n{response.timeline}")
            
            # Add key individuals
            sections.append("## Key Individuals & Agencies\n\n")
            for person, role in response.key_individuals.items():
                sections.append(f"### {person}\n\n{role}")
            
            # Add theory analysis
            sections.append("## Assassination Theories Analysis\n\n")
            for theory, analysis in response.theory_analysis.items():
                sections.append(f"### {theory}\n\n{analysis}")
            
            # Add remaining sections
            sections.extend([
                "## Inconsistencies in Official Accounts\n\n" + "\n".join([f"- {item}" for item in response.inconsistencies]),
                "## Information Withholding Patterns\n\n" + "\n".join([f"- {item}" for item in response.information_withholding]),
                "## Evidence Credibility Assessment\n\n"
            ])
            
            # Add evidence credibility
            for evidence, credibility in response.evidence_credibility.items():
                sections.append(f"### {evidence}\n\n{credibility}")
            
            # Add likely scenarios
            sections.append("## Most Likely Scenarios\n\n" + "\n".join([f"- {scenario}" for scenario in response.likely_scenarios]))
            
            # Add primary suspects
            sections.append("## Primary Suspects\n\n")
            for suspect, evidence_list in response.primary_suspects.items():
                sections.append(f"### {suspect}\n\n" + "\n".join([f"- {evidence}" for evidence in evidence_list]))
            
            # Add alternative suspects
            sections.append("## Alternative Suspects Analysis\n\n")
            for suspect, analysis in response.alternative_suspects_analysis.items():
                sections.append(f"### {suspect}\n\n")
                
                if isinstance(analysis, dict):
                    for key, value in analysis.items():
                        if isinstance(value, list):
                            sections.append(f"#### {key.replace('_', ' ').title()}\n\n" + "\n".join([f"- {item}" for item in value]))
                        else:
                            sections.append(f"#### {key.replace('_', ' ').title()}\n\n{value}")
                else:
                    sections.append(str(analysis))
            
            content = "\n\n".join(sections)
            
            # Save detailed findings to file
            self._save_report_file(content, "detailed_findings.md")
            
            return content
        except Exception as e:
            logger.error(f"Error generating detailed findings with structured output: {e}")
            logger.warning("Falling back to unstructured text generation")
            
            # Fall back to unstructured text generation
            prompt = prompt_template.format(
                analysis_summaries=analysis_summaries
            )
            response = self.llm.invoke(prompt)
            content = response.content
            
            # Save detailed findings to file
            self._save_report_file(content, "detailed_findings.md")
            
            return content
    
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
            ("system", """You are an elite detective with unmatched analytical and investigative skills, specializing in deep forensic analysis, historical investigations, and intelligence gathering. You have been granted access to a vast archive containing thousands of declassified and classified documents, including PDFs, reports, eyewitness testimonies, CIA and FBI records, government memos, and autopsy results related to the assassination of John F. Kennedy.

Your task is to conduct a comprehensive investigation, analyzing all available evidence with a critical and objective approach. You must identify inconsistencies, patterns, missing links, and possible cover-ups while synthesizing key information into a highly detailed, structured report.

You must adhere to these non-negotiable guidelines:

1. Source Attribution: You must only include information that is verifiable and sourced. For each claim, cite the exact source document, report, or testimony it is derived from. If a fact is uncertain, explicitly state the uncertainty and do not fabricate details. If a claim lacks verifiable evidence, label it as 'unverified' or 'requires further investigation.'

2. Fact vs. Speculation Distinction: You must clearly differentiate between documented facts, theories, and speculation. Present multiple perspectives where necessary, but never assert an unverified claim as truth. Example of proper attribution: "The Warren Commission Report concluded X, but critics argue Y, citing document Z." Never state unverified claims as confirmed facts.

3. Information Constraints: If information is not explicitly found in the source material, you must respond with 'Insufficient data available' rather than filling in gaps. Do not generate information beyond what is documented in official records. If a claim lacks direct source support, state 'No evidence found in available documents' rather than speculating.

4. Self-Audit Requirement: Before completing your report, you must perform a self-audit to identify any unverified claims, correct inconsistencies, and highlight areas requiring further evidence. This ensures your report maintains the highest standards of factual accuracy."""),
            ("human", """
            Based on the following analyses of declassified JFK documents, create a comprehensive analysis of potential suspects in the assassination.
            
            ANALYSES:
            {analyses}
            
            Your suspects analysis should include:
            
            1. Primary Suspects Assessment
                • Identify the most likely primary culprit(s) based on document evidence
                • Provide detailed supporting evidence for this conclusion, with specific document references
                • Analyze the strength of evidence against this suspect/group
                • Identify gaps or weaknesses in the case against them
            
            2. Alternative Suspects
                • Present alternative suspects in order of likelihood
                • For each alternative suspect:
                   - Summarize the evidence implicating them
                   - Assess the credibility of this evidence
                   - Identify contradicting evidence
                   - Evaluate their capability and motive
                   - Assess their connections to other key figures
            
            3. Conspiracy Analysis
                • Analyze possible collaborations between suspects
                • Evaluate the evidence for conspiracy vs. lone gunman theories
                • Assess possible operational logistics of coordinated action
                • Identify communication patterns or meetings between potential conspirators
            
            4. Government Connection Assessment
                • Assess government knowledge or involvement in the assassination
                • Examine evidence of foreknowledge
                • Analyze unusual behavior by agencies before and after the assassination
                • Evaluate the handling of the investigation by government entities
            
            5. Psychological Profiles
                • Analyze the behavioral patterns of key suspects
                • Assess psychological motivations and capabilities
                • Evaluate consistency with known psychological profiles
                • Identify suspicious behavior changes before or after the assassination
            
            6. Final Assessment
                • Synthesize all evidence into a cohesive theory
                • Rank the likelihood of different suspect scenarios
                • Identify missing evidence that would strengthen or refute the case
                • Provide a final judgment on the most likely perpetrator(s)
            
            IMPORTANT: When referencing specific documents, ALWAYS include PDF links when available. If a document has a "pdf_url" field, format your reference like: "[Document ID](pdf_url)" to create a clickable link to the original document.
            
            FORMAT YOUR RESPONSE AS MARKDOWN with appropriate headings, subheadings, bullet points, and emphasis.
            Include specific document references (document_id) for all key claims WITH PDF LINKS when available.
            Use a fact-driven, objective, and analytical approach with a forensic, intelligence-driven methodology.
            Critically assess every piece of evidence, cross-referencing sources for validity and exposing any inconsistencies.
            Highlight key findings, provide evidence-backed reasoning, and avoid speculative conclusions unless grounded in substantial proof.
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
            ("system", """You are an elite detective with unmatched analytical and investigative skills, specializing in deep forensic analysis, historical investigations, and intelligence gathering. You have been granted access to a vast archive containing thousands of declassified and classified documents, including PDFs, reports, eyewitness testimonies, CIA and FBI records, government memos, and autopsy results related to the assassination of John F. Kennedy.

Your task is to conduct a comprehensive investigation, analyzing all available evidence with a critical and objective approach. You must identify inconsistencies, patterns, missing links, and possible cover-ups while synthesizing key information into a highly detailed, structured report.

You must adhere to these non-negotiable guidelines:

1. Source Attribution: You must only include information that is verifiable and sourced. For each claim, cite the exact source document, report, or testimony it is derived from. If a fact is uncertain, explicitly state the uncertainty and do not fabricate details. If a claim lacks verifiable evidence, label it as 'unverified' or 'requires further investigation.'

2. Fact vs. Speculation Distinction: You must clearly differentiate between documented facts, theories, and speculation. Present multiple perspectives where necessary, but never assert an unverified claim as truth. Example of proper attribution: "The Warren Commission Report concluded X, but critics argue Y, citing document Z." Never state unverified claims as confirmed facts.

3. Information Constraints: If information is not explicitly found in the source material, you must respond with 'Insufficient data available' rather than filling in gaps. Do not generate information beyond what is documented in official records. If a claim lacks direct source support, state 'No evidence found in available documents' rather than speculating.

4. Self-Audit Requirement: Before completing your report, you must perform a self-audit to identify any unverified claims, correct inconsistencies, and highlight areas requiring further evidence. This ensures your report maintains the highest standards of factual accuracy."""),
            ("human", """
            Based on the following analyses of declassified JFK documents, create a comprehensive analysis of potential coverups related to the assassination.
            
            ANALYSES:
            {analyses}
            
            Your coverup analysis should include:
            
            1. Information Control Assessment
                • Identify evidence of information suppression or tampering
                • Analyze patterns of redaction across documents
                • Examine strategic timing of information releases and their context
                • Analyze unusual classification or handling of documents
            
            2. Narrative Inconsistencies
                • Identify contradictions in official narratives
                • Track changes in official statements over time
                • Compare public statements against internal documents
                • Analyze selective emphasis or omission of critical facts
            
            3. Evidence Handling Analysis
                • Document disappearance or destruction of evidence
                • Identify chain of custody irregularities
                • Analyze forensic evidence procedures and integrity
                • Evaluate evidence collection protocols and deviations
            
            4. Witness Intimidation and Manipulation
                • Examine treatment of witnesses and their testimonies
                • Document witness deaths, threats, or unusual circumstances
                • Analyze changes in witness statements over time
                • Identify patterns of pressure or incentives offered to witnesses
            
            5. Agency Involvement
                • Analyze suspicious behaviors by government agencies
                • Identify unusual operational patterns before and after the assassination
                • Document irregular communication or reporting procedures
                • Examine deviations from standard protocols
            
            6. Strategic Assessment
                • Identify potential motives for a coverup
                • Evaluate which entities would have benefited from a coverup
                • Analyze geopolitical and domestic political contexts
                • Assess impact on national security narratives and policies
            
            7. Obstruction Patterns
                • Identify procedural irregularities in the investigations
                • Document interference with investigation efforts
                • Analyze resource allocation and investigative priorities
                • Examine premature conclusion of investigative avenues
            
            8. Final Analysis
                • Synthesize evidence of coverup activities
                • Assess the scope and coordination of coverup efforts
                • Evaluate the key objectives of information control
                • Provide a definitive assessment of coverup evidence and implications
            
            IMPORTANT: When referencing specific documents, ALWAYS include PDF links when available. If a document has a "pdf_url" field, format your reference like: "[Document ID](pdf_url)" to create a clickable link to the original document.
            
            FORMAT YOUR RESPONSE AS MARKDOWN with appropriate headings, subheadings, bullet points, and emphasis.
            Include specific document references (document_id) for all key claims WITH PDF LINKS when available.
            Use a fact-driven, objective, and analytical approach with a forensic, intelligence-driven methodology.
            Critically assess every piece of evidence, cross-referencing sources for validity and exposing any inconsistencies.
            Highlight key findings, provide evidence-backed reasoning, and avoid speculative conclusions unless grounded in substantial proof.
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
                    <meta name="viewport" content="width=device-width, initial-scale=1.0">
                    <title>JFK Assassination Analysis - {section_name.replace('_', ' ').title()}</title>
                    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
                    <style>
                        :root {{
                            --primary: #c41e3a;
                            --secondary: #0a3161;
                            --text: #333;
                            --light: #f5f5f5;
                            --dark: #222;
                            --accent: #8a8d93;
                        }}
                        
                        * {{
                            margin: 0;
                            padding: 0;
                            box-sizing: border-box;
                        }}
                        
                        body {{ 
                            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                            line-height: 1.6;
                            color: var(--text);
                            background-color: var(--light);
                            padding: 0;
                            margin: 0;
                        }}
                        
                        .container {{
                            max-width: 1000px;
                            margin: 0 auto;
                            padding: 0 20px;
                        }}
                        
                        header {{
                            background: linear-gradient(135deg, var(--primary), var(--secondary));
                            color: white;
                            padding: 40px 0;
                            text-align: center;
                            margin-bottom: 40px;
                            box-shadow: 0 3px 10px rgba(0, 0, 0, 0.2);
                        }}
                        
                        header h1 {{
                            font-size: 2.5rem;
                            margin: 0;
                            padding: 0;
                            color: white;
                            border: none;
                            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
                        }}
                        
                        .content {{
                            background: white;
                            padding: 40px;
                            border-radius: 8px;
                            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
                            margin-bottom: 40px;
                        }}
                        
                        h1 {{ color: var(--secondary); font-size: 2.2rem; margin-bottom: 20px; }}
                        
                        h2 {{ 
                            color: var(--secondary); 
                            margin-top: 30px; 
                            margin-bottom: 20px;
                            padding-bottom: 10px;
                            border-bottom: 2px solid var(--primary);
                            font-size: 1.8rem;
                        }}
                        
                        h3 {{ 
                            color: var(--text); 
                            margin-top: 25px;
                            margin-bottom: 15px;
                            font-size: 1.4rem;
                        }}
                        
                        p {{ margin-bottom: 20px; font-size: 1.1rem; }}
                        
                        ul, ol {{ margin-bottom: 20px; padding-left: 20px; }}
                        
                        li {{ margin-bottom: 8px; }}
                        
                        blockquote {{ 
                            background-color: #f9f9f9; 
                            border-left: 4px solid var(--primary); 
                            margin: 1.5em 0; 
                            padding: 1em 20px;
                            font-style: italic;
                            border-radius: 0 4px 4px 0;
                        }}
                        
                        code {{ 
                            background-color: #f4f4f4; 
                            padding: 2px 4px; 
                            border-radius: 4px; 
                            font-family: monospace;
                        }}
                        
                        table {{ 
                            border-collapse: collapse; 
                            width: 100%; 
                            margin: 20px 0; 
                            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                            border-radius: 4px;
                            overflow: hidden;
                        }}
                        
                        th, td {{ 
                            padding: 12px 15px; 
                            text-align: left; 
                            border-bottom: 1px solid #ddd;
                        }}
                        
                        th {{ 
                            background-color: var(--secondary); 
                            color: white;
                            font-weight: bold;
                            text-transform: uppercase;
                            font-size: 0.9rem;
                            letter-spacing: 1px;
                        }}
                        
                        tr:nth-child(even) {{ background-color: #f9f9f9; }}
                        tr:hover {{ background-color: #f1f1f1; }}
                        
                        a {{
                            color: var(--primary);
                            text-decoration: none;
                            transition: color 0.3s;
                        }}
                        
                        a:hover {{
                            color: var(--secondary);
                            text-decoration: underline;
                        }}
                        
                        .back-link {{
                            display: inline-block;
                            margin: 20px 0;
                            color: var(--primary);
                            text-decoration: none;
                            font-weight: bold;
                        }}
                        
                        .back-link:hover {{
                            text-decoration: underline;
                        }}
                        
                        footer {{
                            background: var(--dark);
                            color: white;
                            padding: 20px 0;
                            text-align: center;
                        }}
                        
                        @media (max-width: 768px) {{
                            .content {{
                                padding: 20px;
                            }}
                            
                            header h1 {{
                                font-size: 2rem;
                            }}
                        }}
                    </style>
                </head>
                <body>
                    <header>
                        <div class="container">
                            <h1>JFK<span style="color: #f3da35;">Reveal</span> - {section_name.replace('_', ' ').title()}</h1>
                        </div>
                    </header>
                    
                    <div class="container">
                        <a href="../index.html" class="back-link"><i class="fas fa-arrow-left"></i> Back to Home</a>
                        
                        <div class="content">
                            {html_content}
                        </div>
                        
                        <footer>
                            <p><em>Generated on {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</em></p>
                            <p>JFKReveal - AI Analysis of Declassified Documents</p>
                        </footer>
                    </div>
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
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>JFK Assassination Analysis - Full Report</title>
                <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
                <style>
                    :root {{
                        --primary: #c41e3a;
                        --secondary: #0a3161;
                        --text: #333;
                        --light: #f5f5f5;
                        --dark: #222;
                        --accent: #8a8d93;
                    }}
                    
                    * {{
                        margin: 0;
                        padding: 0;
                        box-sizing: border-box;
                    }}
                    
                    body {{ 
                        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                        line-height: 1.6;
                        color: var(--text);
                        background-color: var(--light);
                        padding: 0;
                        margin: 0;
                    }}
                    
                    .container {{
                        max-width: 1000px;
                        margin: 0 auto;
                        padding: 0 20px;
                    }}
                    
                    header {{
                        background: linear-gradient(135deg, var(--primary), var(--secondary));
                        color: white;
                        padding: 40px 0;
                        text-align: center;
                        margin-bottom: 40px;
                        box-shadow: 0 3px 10px rgba(0, 0, 0, 0.2);
                    }}
                    
                    header h1 {{
                        font-size: 2.5rem;
                        margin: 0;
                        padding: 0;
                        color: white;
                        border: none;
                        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
                    }}
                    
                    .content {{
                        background: white;
                        padding: 40px;
                        border-radius: 8px;
                        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
                        margin-bottom: 40px;
                    }}
                    
                    h1 {{ color: var(--secondary); font-size: 2.2rem; margin-bottom: 20px; }}
                    
                    h2 {{ 
                        color: var(--secondary); 
                        margin-top: 30px; 
                        margin-bottom: 20px;
                        padding-bottom: 10px;
                        border-bottom: 2px solid var(--primary);
                        font-size: 1.8rem;
                    }}
                    
                    h3 {{ 
                        color: var(--text); 
                        margin-top: 25px;
                        margin-bottom: 15px;
                        font-size: 1.4rem;
                    }}
                    
                    p {{ margin-bottom: 20px; font-size: 1.1rem; }}
                    
                    ul, ol {{ margin-bottom: 20px; padding-left: 20px; }}
                    
                    li {{ margin-bottom: 8px; }}
                    
                    blockquote {{ 
                        background-color: #f9f9f9; 
                        border-left: 4px solid var(--primary); 
                        margin: 1.5em 0; 
                        padding: 1em 20px;
                        font-style: italic;
                        border-radius: 0 4px 4px 0;
                    }}
                    
                    code {{ 
                        background-color: #f4f4f4; 
                        padding: 2px 4px; 
                        border-radius: 4px; 
                        font-family: monospace;
                    }}
                    
                    table {{ 
                        border-collapse: collapse; 
                        width: 100%; 
                        margin: 20px 0; 
                        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                        border-radius: 4px;
                        overflow: hidden;
                    }}
                    
                    th, td {{ 
                        padding: 12px 15px; 
                        text-align: left; 
                        border-bottom: 1px solid #ddd;
                    }}
                    
                    th {{ 
                        background-color: var(--secondary); 
                        color: white;
                        font-weight: bold;
                        text-transform: uppercase;
                        font-size: 0.9rem;
                        letter-spacing: 1px;
                    }}
                    
                    tr:nth-child(even) {{ background-color: #f9f9f9; }}
                    tr:hover {{ background-color: #f1f1f1; }}
                    
                    a {{
                        color: var(--primary);
                        text-decoration: none;
                        transition: color 0.3s;
                    }}
                    
                    a:hover {{
                        color: var(--secondary);
                        text-decoration: underline;
                    }}
                    
                    .back-link {{
                        display: inline-block;
                        margin: 20px 0;
                        color: var(--primary);
                        text-decoration: none;
                        font-weight: bold;
                    }}
                    
                    .back-link:hover {{
                        text-decoration: underline;
                    }}
                    
                    .section-divider {{
                        height: 3px;
                        background: linear-gradient(to right, var(--primary), var(--secondary));
                        margin: 40px 0;
                        border-radius: 2px;
                    }}
                    
                    footer {{
                        background: var(--dark);
                        color: white;
                        padding: 20px 0;
                        text-align: center;
                    }}
                    
                    .toc {{
                        background: #f5f5f5;
                        padding: 20px;
                        border-radius: 8px;
                        margin-bottom: 30px;
                    }}
                    
                    .toc h3 {{
                        margin-top: 0;
                        color: var(--secondary);
                    }}
                    
                    .toc ul {{
                        list-style-type: none;
                        padding-left: 0;
                    }}
                    
                    .toc li {{
                        margin-bottom: 10px;
                    }}
                    
                    .toc a {{
                        display: block;
                        padding: 5px 10px;
                        border-left: 3px solid transparent;
                        transition: all 0.3s;
                    }}
                    
                    .toc a:hover {{
                        border-left: 3px solid var(--primary);
                        background: rgba(0,0,0,0.03);
                        text-decoration: none;
                    }}
                    
                    @media (max-width: 768px) {{
                        .content {{
                            padding: 20px;
                        }}
                        
                        header h1 {{
                            font-size: 2rem;
                        }}
                    }}
                </style>
            </head>
            <body>
                <header>
                    <div class="container">
                        <h1>JFK<span style="color: #f3da35;">Reveal</span> - Full Analysis Report</h1>
                    </div>
                </header>
                
                <div class="container">
                    <a href="../index.html" class="back-link"><i class="fas fa-arrow-left"></i> Back to Home</a>
                    
                    <div class="content">
                        <div class="toc">
                            <h3>Table of Contents</h3>
                            <ul>
                                <li><a href="#executive-summary">Executive Summary</a></li>
                                <li><a href="#detailed-findings">Detailed Findings</a></li>
                                <li><a href="#suspects-analysis">Suspects Analysis</a></li>
                                <li><a href="#coverup-analysis">Coverup Analysis</a></li>
                            </ul>
                        </div>
                        
                        <div id="executive-summary">
                            <h2>Executive Summary</h2>
                            {html_content.replace('<h1>JFK Assassination Analysis - Full Report</h1>', '').split('<hr />', 1)[0]}
                        </div>
                        
                        <div class="section-divider"></div>
                        
                        <div id="detailed-findings">
                            <h2>Detailed Findings</h2>
                            {html_content.split('<hr />', 2)[1].split('<hr />', 1)[0]}
                        </div>
                        
                        <div class="section-divider"></div>
                        
                        <div id="suspects-analysis">
                            <h2>Suspects Analysis</h2>
                            {html_content.split('<hr />', 3)[2].split('<hr />', 1)[0]}
                        </div>
                        
                        <div class="section-divider"></div>
                        
                        <div id="coverup-analysis">
                            <h2>Coverup Analysis</h2>
                            {html_content.split('<hr />', 4)[3].split('<hr />', 1)[0]}
                        </div>
                    </div>
                    
                    <footer>
                        <p><em>Generated on {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</em></p>
                        <p>JFKReveal - AI Analysis of Declassified Documents</p>
                    </footer>
                </div>
            </body>
            </html>""")
        
        return report