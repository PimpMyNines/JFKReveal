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
from langchain_core.callbacks import BaseCallbackHandler
import markdown

logger = logging.getLogger(__name__)

class ReportAuditLogCallback(BaseCallbackHandler):
    """Callback handler for logging model thought process during report generation."""
    
    def __init__(self, output_dir: str = "data/audit_logs/reports", report_type: Optional[str] = None):
        """
        Initialize the report audit log callback.
        
        Args:
            output_dir: Directory to save audit logs
            report_type: Type of report (executive_summary, detailed_findings, etc.)
        """
        self.output_dir = output_dir
        self.report_type = report_type
        self.messages = []
        self.current_analysis = {}
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
    
    def on_llm_start(self, serialized, prompts, **kwargs):
        """Log when LLM starts processing."""
        self.messages.append({
            "event": "llm_start",
            "timestamp": self._get_timestamp(),
            "report_type": self.report_type,
            "prompts": prompts
        })
        
    def on_llm_new_token(self, token, **kwargs):
        """Log streaming tokens if available."""
        self.messages.append({
            "event": "llm_token",
            "timestamp": self._get_timestamp(),
            "report_type": self.report_type,
            "token": token
        })
        
    def on_llm_end(self, response, **kwargs):
        """Log when LLM finishes processing."""
        self.messages.append({
            "event": "llm_end",
            "timestamp": self._get_timestamp(),
            "report_type": self.report_type,
            "response": response.dict() if hasattr(response, "dict") else str(response)
        })
        
        # Save the audit log
        self._save_audit_log()
    
    def on_llm_error(self, error, **kwargs):
        """Log any errors during LLM processing."""
        self.messages.append({
            "event": "llm_error",
            "timestamp": self._get_timestamp(),
            "report_type": self.report_type,
            "error": str(error)
        })
        
        # Save the audit log on error as well
        self._save_audit_log()
    
    def on_chain_start(self, serialized, inputs, **kwargs):
        """Log when chain starts processing."""
        # Store the current analysis inputs
        self.current_analysis = inputs
        
        chain_type = serialized.get("name", "unknown")
        self.messages.append({
            "event": "chain_start",
            "timestamp": self._get_timestamp(),
            "report_type": self.report_type,
            "chain_type": chain_type,
            "inputs": {k: v for k, v in inputs.items() if not k.startswith("analyses")} if any(k.startswith("analyses") for k in inputs) else inputs
        })
        
        # Include analyses_summary in a separate field to keep logs cleaner
        if "analyses_summary" in inputs:
            self.messages.append({
                "event": "analyses_summary",
                "timestamp": self._get_timestamp(),
                "report_type": self.report_type,
                "analyses_summary": inputs["analyses_summary"]
            })
    
    def on_chain_end(self, outputs, **kwargs):
        """Log when chain ends processing."""
        self.messages.append({
            "event": "chain_end",
            "timestamp": self._get_timestamp(),
            "report_type": self.report_type,
            "outputs": outputs
        })
    
    def on_chain_error(self, error, **kwargs):
        """Log any errors during chain processing."""
        self.messages.append({
            "event": "chain_error",
            "timestamp": self._get_timestamp(),
            "report_type": self.report_type,
            "error": str(error)
        })
    
    def on_tool_start(self, serialized, input_str, **kwargs):
        """Log when a tool starts being used."""
        self.messages.append({
            "event": "tool_start",
            "timestamp": self._get_timestamp(),
            "report_type": self.report_type,
            "tool": serialized.get("name", "unknown"),
            "input": input_str
        })
    
    def on_tool_end(self, output, **kwargs):
        """Log when a tool finishes being used."""
        self.messages.append({
            "event": "tool_end",
            "timestamp": self._get_timestamp(),
            "report_type": self.report_type,
            "output": output
        })
    
    def on_tool_error(self, error, **kwargs):
        """Log any errors during tool usage."""
        self.messages.append({
            "event": "tool_error",
            "timestamp": self._get_timestamp(),
            "report_type": self.report_type,
            "error": str(error)
        })
    
    def on_text(self, text, **kwargs):
        """Log any text output from intermediate steps."""
        self.messages.append({
            "event": "text",
            "timestamp": self._get_timestamp(),
            "report_type": self.report_type,
            "text": text
        })
    
    def _get_timestamp(self):
        """Get the current timestamp."""
        return datetime.datetime.now().isoformat()
    
    def _save_audit_log(self):
        """Save the current audit log to file."""
        if not self.messages:
            return
            
        # Generate filename with timestamp and report_type if available
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        report_type_str = f"_{self.report_type}" if self.report_type else ""
        filename = f"report_audit_log_{timestamp}{report_type_str}.json"
        filepath = os.path.join(self.output_dir, filename)
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump({
                    "audit_log": self.messages,
                    "metadata": {
                        "report_type": self.report_type,
                        "timestamp": timestamp,
                        "analysis_input": self.current_analysis
                    }
                }, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved report audit log to {filepath}")
        except Exception as e:
            logger.error(f"Error saving report audit log: {e}")
            
    def reset(self):
        """Reset the callback's state."""
        self.messages = []
        self.current_analysis = {}

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
        model_name: str = "gpt-4o",
        model_provider: str = "openai",
        openai_api_key: Optional[str] = None,
        anthropic_api_key: Optional[str] = None,
        xai_api_key: Optional[str] = None,  # API key for xAI (Grok)
        temperature: float = 0.1,
        max_retries: int = 5,
        # Use local PDF files instead of direct links to archives.gov
        # Original source: "https://www.archives.gov/files/research/jfk/releases/2025/0318/"
        pdf_base_url: str = "/data/documents/",
        archive_citation: str = "National Archives, JFK Assassination Records",
        audit_logs_dir: str = "data/audit_logs/reports",
        enable_audit_logging: bool = True,
        stream_tokens: bool = True,
    ):
        """
        Initialize the findings report generator.
        
        Args:
            analysis_dir: Directory containing analysis files
            output_dir: Directory to save reports
            raw_docs_dir: Directory containing raw PDF documents
            model_name: Model name to use for report generation
            model_provider: Model provider to use ('openai' or 'anthropic')
            openai_api_key: OpenAI API key (uses environment variable if not provided)
            anthropic_api_key: Anthropic API key (uses environment variable if not provided)
            temperature: Temperature for LLM generation
            max_retries: Maximum number of retries for API calls
            pdf_base_url: Base URL for PDF documents for generating links
            audit_logs_dir: Directory to save audit logs
            enable_audit_logging: Whether to enable detailed audit logging
            stream_tokens: Whether to stream tokens in audit logs (only works with streaming=True)
        """
        self.analysis_dir = analysis_dir
        self.output_dir = output_dir
        self.raw_docs_dir = raw_docs_dir
        self.model_name = model_name
        self.model_provider = model_provider.lower()
        self.temperature = temperature
        self.max_retries = max_retries
        self.pdf_base_url = pdf_base_url
        self.archive_citation = archive_citation
        self.audit_logs_dir = audit_logs_dir
        self.enable_audit_logging = enable_audit_logging
        self.stream_tokens = stream_tokens
        self.xai_api_key = xai_api_key
        
        if self.model_provider not in ["openai", "anthropic", "xai"]:
            logger.warning(f"Unknown model provider: {model_provider}. Defaulting to 'openai'.")
            self.model_provider = "openai"
        
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        if enable_audit_logging:
            os.makedirs(audit_logs_dir, exist_ok=True)
            logger.info(f"Audit logging enabled for reports. Logs will be saved to {audit_logs_dir}")
        
        # Initialize audit logger callbacks
        self.audit_callbacks = {}
        if enable_audit_logging:
            for report_type in ["executive_summary", "detailed_findings", "suspects_analysis", "coverup_analysis"]:
                self.audit_callbacks[report_type] = ReportAuditLogCallback(
                    output_dir=audit_logs_dir,
                    report_type=report_type
                )
        
        # Initialize LangChain model based on provider (without callbacks - will be added per report generation)
        if self.model_provider == "anthropic":
            # Import Anthropic if needed
            from langchain_anthropic import ChatAnthropic
            
            self.llm = ChatAnthropic(
                model=model_name,
                temperature=temperature,
                anthropic_api_key=anthropic_api_key,
                max_retries=max_retries,
                streaming=stream_tokens,  # Enable streaming for token-by-token audit logs
            )
            logger.info(f"Initialized Anthropic model for report generation: {model_name}")
        elif self.model_provider == "xai":
            # Import xAI (Grok) integration
            try:
                # Note: This is a hypothetical import - may need to adjust based on actual LangChain integration
                from langchain_xai import ChatXAI
                
                self.llm = ChatXAI(
                    model=model_name,
                    temperature=temperature,
                    api_key=xai_api_key,
                    max_retries=max_retries,
                    streaming=stream_tokens,
                )
                logger.info(f"Initialized xAI Grok model for report generation: {model_name}")
            except ImportError:
                logger.warning("LangChain xAI integration not available. Falling back to OpenAI.")
                from langchain_openai import ChatOpenAI
                
                self.llm = ChatOpenAI(
                    model=model_name,
                    temperature=temperature,
                    api_key=openai_api_key,
                    max_retries=max_retries,
                    streaming=stream_tokens,
                )
                logger.info(f"Fallback: Initialized OpenAI model: {model_name}")
        else:
            # Import OpenAI
            from langchain_openai import ChatOpenAI
            
            self.llm = ChatOpenAI(
                model=model_name,
                temperature=temperature,
                api_key=openai_api_key,
                max_retries=max_retries,
                streaming=stream_tokens,  # Enable streaming for token-by-token audit logs
            )
            logger.info(f"Initialized OpenAI model for report generation: {model_name}")
        
        # Build document ID to PDF URL mapping
        self.document_urls = self._build_document_urls()
    
    def _build_document_urls(self) -> Dict[str, Dict[str, str]]:
        """
        Build a mapping of document IDs to their PDF URLs and citation info.
        
        Returns:
            Dictionary mapping document IDs to their metadata (PDF URL and citation)
        """
        document_metadata = {}
        
        # Get list of PDF files in raw documents directory
        if os.path.exists(self.raw_docs_dir):
            for file in os.listdir(self.raw_docs_dir):
                if file.endswith('.pdf'):
                    # Remove extension to get document ID
                    doc_id = os.path.splitext(file)[0]
                    # Create local URL path
                    pdf_url = f"{self.pdf_base_url}{file}"
                    # Add citation info
                    document_metadata[doc_id] = {
                        "url": pdf_url,
                        "citation": f"{self.archive_citation}, Document {doc_id}",
                        "original_source": "National Archives, JFK Assassination Records"
                    }
        
        logger.info(f"Built URL and citation mapping for {len(document_metadata)} documents")
        return document_metadata
    
    def load_analyses(self) -> List[Dict[str, Any]]:
        """
        Load all analysis files.
        
        Returns:
            List of analysis data with added PDF links and citation info
        """
        analyses = []
        
        # Find all JSON files
        for file in os.listdir(self.analysis_dir):
            if file.endswith('.json') and not file.endswith('_partial.json'):
                file_path = os.path.join(self.analysis_dir, file)
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        analysis = json.load(f)
                        
                        # Add PDF URLs and citation info to documents referenced in the analysis
                        if "documents" in analysis:
                            for i, doc in enumerate(analysis["documents"]):
                                doc_id = doc.get("document_id")
                                if doc_id and doc_id in self.document_urls:
                                    doc_metadata = self.document_urls[doc_id]
                                    analysis["documents"][i]["pdf_url"] = doc_metadata["url"]
                                    analysis["documents"][i]["citation"] = doc_metadata["citation"]
                                    analysis["documents"][i]["original_source"] = doc_metadata["original_source"]
                        
                        # Add PDF URLs and citation info to additional evidence
                        if "additional_evidence" in analysis:
                            for i, evidence in enumerate(analysis["additional_evidence"]):
                                if isinstance(evidence, dict) and "document_id" in evidence:
                                    doc_id = evidence["document_id"]
                                    if doc_id in self.document_urls:
                                        doc_metadata = self.document_urls[doc_id]
                                        analysis["additional_evidence"][i]["pdf_url"] = doc_metadata["url"]
                                        analysis["additional_evidence"][i]["citation"] = doc_metadata["citation"]
                                        analysis["additional_evidence"][i]["original_source"] = doc_metadata["original_source"]
                        
                        # Add model information to the analysis for comparison
                        analysis["model_info"] = {
                            "provider": self.model_provider,
                            "model": self.model_name
                        }
                        
                        analyses.append(analysis)
                except Exception as e:
                    logger.error(f"Error loading analysis file {file_path}: {e}")
        
        logger.info(f"Loaded {len(analyses)} analysis files with PDF links and citations")
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
        # Configure LLM with audit logger if enabled
        if self.enable_audit_logging:
            # Reset the callback for this new report
            self.audit_callbacks["executive_summary"].reset()
            # Create a new LLM instance with the callback based on provider
            if self.model_provider == "anthropic":
                from langchain_anthropic import ChatAnthropic
                llm_with_callback = ChatAnthropic(
                    model=self.model_name,
                    temperature=self.temperature,
                    max_retries=self.max_retries,
                    callbacks=[self.audit_callbacks["executive_summary"]],
                    streaming=self.stream_tokens,
                )
            else:
                from langchain_openai import ChatOpenAI
                llm_with_callback = ChatOpenAI(
                    model=self.model_name,
                    temperature=self.temperature,
                    max_retries=self.max_retries,
                    callbacks=[self.audit_callbacks["executive_summary"]],
                    streaming=self.stream_tokens,
                )
            logger.info("Generating executive summary with audit logging")
        else:
            # Use the default LLM without audit logging
            llm_with_callback = self.llm
        
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
        
        # Create prompt template with thought process instructions
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", """You are an elite detective with unmatched analytical and investigative skills, specializing in deep forensic analysis, historical investigations, and intelligence gathering. You have been granted access to a vast archive containing thousands of declassified and classified documents, including PDFs, reports, eyewitness testimonies, CIA and FBI records, government memos, and autopsy results related to the assassination of John F. Kennedy.

Your task is to conduct a comprehensive investigation, analyzing all available evidence with a critical and objective approach. You must identify inconsistencies, patterns, missing links, and possible cover-ups while synthesizing key information into a highly detailed, structured report.

You must adhere to these non-negotiable guidelines:

1. Source Attribution: You must only include information that is verifiable and sourced. For each claim, cite the exact source document, report, or testimony it is derived from. If a fact is uncertain, explicitly state the uncertainty and do not fabricate details. If a claim lacks verifiable evidence, label it as 'unverified' or 'requires further investigation.'

2. Fact vs. Speculation Distinction: You must clearly differentiate between documented facts, theories, and speculation. Present multiple perspectives where necessary, but never assert an unverified claim as truth. Example of proper attribution: "The Warren Commission Report concluded X, but critics argue Y, citing document Z." Never state unverified claims as confirmed facts.

3. Information Constraints: If information is not explicitly found in the source material, you must respond with 'Insufficient data available' rather than filling in gaps. Do not generate information beyond what is documented in official records. If a claim lacks direct source support, state 'No evidence found in available documents' rather than speculating.

4. Self-Audit Requirement: Before completing your report, you must perform a self-audit to identify any unverified claims, correct inconsistencies, and highlight areas requiring further evidence. This ensures your report maintains the highest standards of factual accuracy.

5. Thought Process Documentation: Document your thought process as you analyze the information across all document analyses. Include your reasoning for identifying key patterns, assessment of evidence credibility, how you determined the most significant findings, and your methodology for evaluating competing theories. This will create an audit trail of your executive-level analytical approach."""),
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
            
            IMPORTANT: When referencing specific documents, include PDF links if available. For example, if a document has a "pdf_url" field, format your reference like: "[Document ID](pdf_url)" or with a footnote linking to the PDF.
            
            FORMAT YOUR RESPONSE AS MARKDOWN with appropriate headings, bullet points, and emphasis.
            Include specific document references for key claims where possible with PDF links when available.
            Use a fact-driven, objective, and analytical approach with a forensic, intelligence-driven methodology.
            Critically assess every piece of evidence, cross-referencing sources for validity and exposing any inconsistencies.
            Ensure the language is professional, highly detailed, and structured for clarity.
            
            As you work through this analysis, document your thought process. For each section:
            1. Begin by stating what you're looking to establish in this section
            2. Explain how you're evaluating and synthesizing information across topics
            3. Detail your reasoning for emphasizing certain findings over others
            4. Document your methodology for determining the credibility of competing theories
            5. Explain your process for connecting separate pieces of evidence into coherent patterns
            
            Before finalizing your report, you MUST perform a self-audit:
            1. Identify any unverified claims and mark them as such
            2. Correct any inconsistencies and contradictions 
            3. Highlight areas requiring further evidence
            4. Verify that every claim has proper source attribution
            5. Confirm you've distinguished clearly between facts and speculation
            
            This detailed thought process will be captured in the audit log and provide insight into your executive summary methodology.
            """)
        ])
        
        try:
            # First try to get structured output
            chain = prompt_template | llm_with_callback.with_structured_output(
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
            unstructured_chain = prompt_template | llm_with_callback
            
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
            ("system", """You are an elite detective with unmatched analytical and investigative skills, specializing in deep forensic analysis, historical investigations, and intelligence gathering. You have been granted access to a vast archive containing thousands of declassified and classified documents, including PDFs, reports, eyewitness testimonies, CIA and FBI records, government memos, and autopsy results related to the assassination of John F. Kennedy.

Your task is to conduct a comprehensive investigation, analyzing all available evidence with a critical and objective approach. You must identify inconsistencies, patterns, missing links, and possible cover-ups while synthesizing key information into a highly detailed, structured report.

You must adhere to these non-negotiable guidelines:

1. Source Attribution: You must only include information that is verifiable and sourced. For each claim, cite the exact source document, report, or testimony it is derived from. If a fact is uncertain, explicitly state the uncertainty and do not fabricate details. If a claim lacks verifiable evidence, label it as 'unverified' or 'requires further investigation.'

2. Fact vs. Speculation Distinction: You must clearly differentiate between documented facts, theories, and speculation. Present multiple perspectives where necessary, but never assert an unverified claim as truth. Example of proper attribution: "The Warren Commission Report concluded X, but critics argue Y, citing document Z." Never state unverified claims as confirmed facts.

3. Information Constraints: If information is not explicitly found in the source material, you must respond with 'Insufficient data available' rather than filling in gaps. Do not generate information beyond what is documented in official records. If a claim lacks direct source support, state 'No evidence found in available documents' rather than speculating.

4. Self-Audit Requirement: Before completing your report, you must perform a self-audit to identify any unverified claims, correct inconsistencies, and highlight areas requiring further evidence. This ensures your report maintains the highest standards of factual accuracy."""),
            ("human", """
            Based on the following analyses of declassified JFK documents, create a comprehensive detailed findings report.
            
            ANALYSES:
            {analyses}
            
            Your detailed report should include:
            
            1. Historical Context & Key Figures
                • Overview of JFK's presidency and political climate
                • Key individuals involved (Lee Harvey Oswald, Jack Ruby, government officials, intelligence agencies, etc.)
            
            2. Chronological Timeline
                • Present a detailed chronological timeline of events based on the documents
                • Highlight critical moments before, during, and after the assassination
            
            3. Analysis of Declassified Documents
                • In-depth examination of each key topic
                • Summary of crucial documents and their significance
                • Identification of redactions, inconsistencies, or contradictions in official records
            
            4. Key Individuals and Agencies
                • Detail the roles and actions of key individuals and agencies
                • Behavioral analysis of key individuals before, during, and after the assassination
            
            5. Forensic Analysis
                • Breakdown of bullet trajectories, wounds, and impact reports if available
                • Analysis of the 'single bullet theory' vs. alternative explanations
                • Possible inconsistencies in forensic evidence
            
            6. Evidentiary Assessment
                • Examine inconsistencies and contradictions in official accounts
                • Identify patterns of information withholding or redaction
                • Evaluate the credibility of different pieces of evidence
            
            7. Theories and Scenarios
                • Analyze the evidence for various assassination theories
                • Draw reasoned conclusions about the most likely scenarios
                • Evaluate official Warren Commission conclusions and their flaws
            
            8. Suspects Evaluation
                • Identify the most likely culprit(s) with supporting evidence
                • Present alternative suspects with detailed analysis of their potential involvement
                • Assess possible collaborations between suspects
            
            IMPORTANT: When referencing specific documents, ALWAYS include PDF links when available. For example, if a document has a "pdf_url" field, format your reference like: "[Document ID](pdf_url)" to create a clickable link to the original document.
            
            FORMAT YOUR RESPONSE AS MARKDOWN with appropriate headings, subheadings, bullet points, and emphasis.
            Include specific document references (document_id) for all key claims WITH PDF LINKS when available.
            Use a fact-driven, objective, and analytical approach with a forensic, intelligence-driven methodology.
            Critically assess every piece of evidence, cross-referencing sources for validity and exposing any inconsistencies.
            Highlight key findings, provide evidence-backed reasoning, and avoid speculative conclusions unless grounded in substantial proof.
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
        
        # Create model-specific output directory
        model_dir = os.path.join(self.output_dir, f"{self.model_provider}_{self.model_name}")
        os.makedirs(model_dir, exist_ok=True)
        logger.info(f"Creating model-specific output directory: {model_dir}")
        
        # Generate report sections with audit logging
        logger.info(f"Starting report generation with {self.model_provider} {self.model_name}")
        
        # Generate each report section with appropriate audit logging
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
        
        # Save each section as markdown and copy audit logs
        for section_name, content in report.items():
            # Save markdown in model-specific directory
            output_file = os.path.join(model_dir, f"{section_name}.md")
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(content)
            logger.info(f"Saved {section_name} to {output_file}")
            
            # Also save to main output directory for backward compatibility
            main_output_file = os.path.join(self.output_dir, f"{section_name}.md")
            with open(main_output_file, 'w', encoding='utf-8') as f:
                f.write(content)
            
            # Copy audit logs to reports directory if enabled
            if self.enable_audit_logging:
                # Save audit logs in model-specific directory
                audit_file = os.path.join(model_dir, f"{section_name}_audit.json")
                try:
                    with open(audit_file, 'w', encoding='utf-8') as f:
                        json.dump({
                            "report_type": section_name,
                            "model_info": {
                                "provider": self.model_provider,
                                "model": self.model_name
                            },
                            "audit_log": self.audit_callbacks[section_name].messages,
                            "timestamp": datetime.datetime.now().isoformat()
                        }, f, indent=2, ensure_ascii=False)
                    logger.info(f"Saved {section_name} audit log to {audit_file}")
                except Exception as e:
                    logger.error(f"Error saving audit log for {section_name}: {e}")
            
            # Save HTML in both model-specific directory and main directory
            model_html_output = os.path.join(model_dir, f"{section_name}.html")
            main_html_output = os.path.join(self.output_dir, f"{section_name}.html")
            
            html_content = markdown.markdown(content, extensions=['tables', 'fenced_code'])
            
            # Prepare the HTML content with modern UI
            html_template = f"""<!DOCTYPE html>
                <html lang="en">
                <head>
                    <meta charset="utf-8">
                    <meta name="viewport" content="width=device-width, initial-scale=1.0">
                    <title>JFK Assassination Analysis - {section_name.replace('_', ' ').title()}</title>
                    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
                    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=Special+Elite:wght@400&display=swap" rel="stylesheet">
                    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
                    <script src="https://cdn.jsdelivr.net/npm/d3@7"></script>
                    <style>
                        :root {{
                            --primary: #c41e3a;
                            --primary-dark: #a01830;
                            --primary-light: #f0c0c0;
                            --secondary: #0a3161;
                            --secondary-dark: #062142;
                            --secondary-light: #b3c5e1;
                            --text: #333;
                            --light: #f8f9fa;
                            --dark: #212529;
                            --accent: #8a8d93;
                            --success: #28a745;
                            --info: #17a2b8;
                            --warning: #ffc107;
                            --danger: #dc3545;
                            --font-main: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif;
                            --font-special: 'Special Elite', monospace;
                            --shadow-sm: 0 .125rem .25rem rgba(0,0,0,.075);
                            --shadow: 0 .5rem 1rem rgba(0,0,0,.15);
                            --shadow-lg: 0 1rem 3rem rgba(0,0,0,.175);
                            --radius: 0.5rem;
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
        
        # Save combined audit logs if enabled
        if self.enable_audit_logging:
            combined_audit_file = os.path.join(self.output_dir, "full_report_audit.json")
            try:
                combined_audit_logs = {
                    "timestamp": datetime.datetime.now().isoformat(),
                    "sections": {}
                }
                
                # Collect audit logs from each section
                for section_name in ["executive_summary", "detailed_findings", "suspects_analysis", "coverup_analysis"]:
                    combined_audit_logs["sections"][section_name] = {
                        "thought_process": [msg for msg in self.audit_callbacks[section_name].messages if msg.get("event") in ["text", "llm_token", "chain_start", "chain_end"]],
                        "event_count": len(self.audit_callbacks[section_name].messages)
                    }
                
                with open(combined_audit_file, 'w', encoding='utf-8') as f:
                    json.dump(combined_audit_logs, f, indent=2, ensure_ascii=False)
                logger.info(f"Saved combined audit logs to {combined_audit_file}")
            except Exception as e:
                logger.error(f"Error saving combined audit logs: {e}")
        
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
                        {f'<p><a href="full_report_audit.json" style="color: white; text-decoration: underline;">View Analysis Audit Log</a></p>' if self.enable_audit_logging else ''}
                    </footer>
                </div>
            </body>
            </html>""")
        
        if self.enable_audit_logging:
            logger.info("Completed full report generation with detailed audit logging")
        else:
            logger.info("Completed full report generation")
            
        return report