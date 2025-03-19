"""
Document analyzer using LangChain and OpenAI to extract and analyze information.
"""
import os
import json
import logging
import datetime
from typing import List, Dict, Any, Optional, Union

from pydantic import BaseModel, Field
from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from langchain_openai import ChatOpenAI
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.output_parsers.json import JsonOutputParser
from langchain_core.exceptions import LangChainException
from langchain_core.callbacks import BaseCallbackHandler

from ..database.vector_store import VectorStore

logger = logging.getLogger(__name__)

class AuditLogCallback(BaseCallbackHandler):
    """Callback handler for logging model thought process during analysis."""
    
    def __init__(self, output_dir: str = "data/audit_logs", document_id: Optional[str] = None):
        """
        Initialize the audit log callback.
        
        Args:
            output_dir: Directory to save audit logs
            document_id: Optional document ID to include in log filenames
        """
        self.output_dir = output_dir
        self.document_id = document_id
        self.messages = []
        self.current_analysis = {}
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
    
    def on_llm_start(self, serialized, prompts, **kwargs):
        """Log when LLM starts processing."""
        self.messages.append({
            "event": "llm_start",
            "timestamp": self._get_timestamp(),
            "document_id": self.document_id,
            "prompts": prompts
        })
        
    def on_llm_new_token(self, token, **kwargs):
        """Log streaming tokens if available."""
        self.messages.append({
            "event": "llm_token",
            "timestamp": self._get_timestamp(),
            "document_id": self.document_id,
            "token": token
        })
        
    def on_llm_end(self, response, **kwargs):
        """Log when LLM finishes processing."""
        self.messages.append({
            "event": "llm_end",
            "timestamp": self._get_timestamp(),
            "document_id": self.document_id,
            "response": response.dict() if hasattr(response, "dict") else str(response)
        })
        
        # Save the audit log
        self._save_audit_log()
    
    def on_llm_error(self, error, **kwargs):
        """Log any errors during LLM processing."""
        self.messages.append({
            "event": "llm_error",
            "timestamp": self._get_timestamp(),
            "document_id": self.document_id,
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
            "document_id": self.document_id,
            "chain_type": chain_type,
            "inputs": {k: v for k, v in inputs.items() if k != "text"} if "text" in inputs else inputs
        })
        
        # Include document text in a separate field to keep logs cleaner
        if "text" in inputs:
            self.messages.append({
                "event": "document_text",
                "timestamp": self._get_timestamp(),
                "document_id": self.document_id,
                "text": inputs["text"]
            })
    
    def on_chain_end(self, outputs, **kwargs):
        """Log when chain ends processing."""
        self.messages.append({
            "event": "chain_end",
            "timestamp": self._get_timestamp(),
            "document_id": self.document_id,
            "outputs": outputs
        })
    
    def on_chain_error(self, error, **kwargs):
        """Log any errors during chain processing."""
        self.messages.append({
            "event": "chain_error",
            "timestamp": self._get_timestamp(),
            "document_id": self.document_id,
            "error": str(error)
        })
    
    def on_tool_start(self, serialized, input_str, **kwargs):
        """Log when a tool starts being used."""
        self.messages.append({
            "event": "tool_start",
            "timestamp": self._get_timestamp(),
            "document_id": self.document_id,
            "tool": serialized.get("name", "unknown"),
            "input": input_str
        })
    
    def on_tool_end(self, output, **kwargs):
        """Log when a tool finishes being used."""
        self.messages.append({
            "event": "tool_end",
            "timestamp": self._get_timestamp(),
            "document_id": self.document_id,
            "output": output
        })
    
    def on_tool_error(self, error, **kwargs):
        """Log any errors during tool usage."""
        self.messages.append({
            "event": "tool_error",
            "timestamp": self._get_timestamp(),
            "document_id": self.document_id,
            "error": str(error)
        })
    
    def on_text(self, text, **kwargs):
        """Log any text output from intermediate steps."""
        self.messages.append({
            "event": "text",
            "timestamp": self._get_timestamp(),
            "document_id": self.document_id,
            "text": text
        })
    
    def _get_timestamp(self):
        """Get the current timestamp."""
        return datetime.datetime.now().isoformat()
    
    def _save_audit_log(self):
        """Save the current audit log to file."""
        if not self.messages:
            return
            
        # Generate filename with timestamp and document_id if available
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        doc_id = f"_{self.document_id}" if self.document_id else ""
        filename = f"audit_log_{timestamp}{doc_id}.json"
        filepath = os.path.join(self.output_dir, filename)
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump({
                    "audit_log": self.messages,
                    "metadata": {
                        "document_id": self.document_id,
                        "timestamp": timestamp,
                        "analysis_input": self.current_analysis
                    }
                }, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved audit log to {filepath}")
        except Exception as e:
            logger.error(f"Error saving audit log: {e}")
            
    def reset(self):
        """Reset the callback's state."""
        self.messages = []
        self.current_analysis = {}

class DocumentAnalysisItem(BaseModel):
    """A single item of analysis with the supporting text."""
    information: str = Field(..., description="The extracted information")
    quote: str = Field(..., description="The quote from the document supporting this information")
    page: Optional[str] = Field(None, description="The page number where this information was found")

class DocumentAnalysisResult(BaseModel):
    """Results of analyzing a document chunk."""
    key_individuals: List[DocumentAnalysisItem] = Field(default_factory=list, description="Key individuals mentioned")
    government_agencies: List[DocumentAnalysisItem] = Field(default_factory=list, description="Government agencies mentioned")
    locations: List[DocumentAnalysisItem] = Field(default_factory=list, description="Locations mentioned")
    dates_and_times: List[DocumentAnalysisItem] = Field(default_factory=list, description="Dates and times mentioned")
    potential_coverup: List[DocumentAnalysisItem] = Field(default_factory=list, description="Potential evidence of coverup")
    suspicious_activities: List[DocumentAnalysisItem] = Field(default_factory=list, description="Suspicious activities mentioned")
    assassination_theories: List[DocumentAnalysisItem] = Field(default_factory=list, description="Connections to assassination theories")
    inconsistencies: List[DocumentAnalysisItem] = Field(default_factory=list, description="Inconsistencies in the document")
    weapons_references: List[DocumentAnalysisItem] = Field(default_factory=list, description="References to weapons")
    redacted_information: List[DocumentAnalysisItem] = Field(default_factory=list, description="Missing or redacted information")

class AnalyzedDocument(BaseModel):
    """A document that has been analyzed."""
    text: str = Field(..., description="The text of the document chunk")
    metadata: Dict[str, Any] = Field(..., description="Metadata for the document")
    analysis: DocumentAnalysisResult = Field(..., description="Analysis results")
    error: Optional[str] = Field(None, description="Error message if analysis failed")

class TopicSummary(BaseModel):
    """Summary of analysis for a specific topic."""
    key_findings: List[str] = Field(default_factory=list, description="Key findings from document analysis")
    consistent_information: List[str] = Field(default_factory=list, description="Information that appears in multiple sources")
    contradictions: List[str] = Field(default_factory=list, description="Contradictions between documents")
    potential_evidence: List[str] = Field(default_factory=list, description="Potential evidence of coverup")
    missing_information: List[str] = Field(default_factory=list, description="Notable gaps in information")
    assassination_theories: List[str] = Field(default_factory=list, description="Connections to assassination theories")
    credibility: str = Field(description="Level of credibility of the information (high, medium, or low)")
    document_references: Dict[str, List[str]] = Field(default_factory=dict, description="Document references for key claims")

class TopicAnalysis(BaseModel):
    """Complete analysis of a topic including all document analyses and summary."""
    topic: str = Field(..., description="The topic analyzed")
    summary: TopicSummary = Field(..., description="Summary of findings")
    document_analyses: List[AnalyzedDocument] = Field(..., description="Individual document analyses")
    num_documents: int = Field(..., description="Number of documents analyzed")
    error: Optional[str] = Field(None, description="Error message if analysis failed")

class DocumentAnalyzer:
    """Analyze JFK documents using LangChain and OpenAI models."""
    
    # Categories and topics of interest
    ANALYSIS_CATEGORIES = [
        "Government Agencies Involved",
        "Key Individuals Mentioned",
        "Locations Mentioned",
        "Timeline of Events",
        "Suspicious Activities",
        "Inconsistencies or Contradictions",
        "Missing or Redacted Information",
        "Potential Coverup Evidence",
        "Connections to Other Historical Events",
        "Weapons or Methods Discussed"
    ]
    
    def __init__(
        self,
        vector_store: VectorStore,
        output_dir: str = "data/analysis",
        model_name: str = "gpt-4o",
        model_provider: str = "openai",
        openai_api_key: Optional[str] = None,
        anthropic_api_key: Optional[str] = None,
        xai_api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_retries: int = 5,
        audit_logs_dir: str = "data/audit_logs",
        enable_audit_logging: bool = True,
        stream_tokens: bool = True,
    ):
        """
        Initialize the document analyzer.
        
        Args:
            vector_store: Vector store instance for searching documents
            output_dir: Directory to save analysis results
            model_name: Model name to use for analysis
            model_provider: Model provider to use ('openai', 'anthropic', or 'xai')
            openai_api_key: OpenAI API key (uses environment variable if not provided)
            anthropic_api_key: Anthropic API key (uses environment variable if not provided)
            xai_api_key: X AI (Grok) API key (uses environment variable if not provided)
            temperature: Temperature for LLM generation
            max_retries: Maximum number of retries for API calls
            audit_logs_dir: Directory to save audit logs
            enable_audit_logging: Whether to enable detailed audit logging
            stream_tokens: Whether to stream tokens in audit logs (only works with streaming=True)
        """
        self.vector_store = vector_store
        self.output_dir = output_dir
        self.model_name = model_name
        self.model_provider = model_provider.lower()
        self.temperature = temperature
        self.max_retries = max_retries
        self.audit_logs_dir = audit_logs_dir
        self.enable_audit_logging = enable_audit_logging
        self.stream_tokens = stream_tokens
        
        if self.model_provider not in ["openai", "anthropic", "xai"]:
            logger.warning(f"Unknown model provider: {model_provider}. Defaulting to 'openai'.")
            self.model_provider = "openai"
        
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        if enable_audit_logging:
            os.makedirs(audit_logs_dir, exist_ok=True)
        
        # Initialize audit logger callback
        callbacks = []
        if enable_audit_logging:
            logger.info(f"Audit logging enabled. Logs will be saved to {audit_logs_dir}")
            self.audit_callback = AuditLogCallback(output_dir=audit_logs_dir)
            callbacks.append(self.audit_callback)
        
        # Initialize LangChain model based on provider
        if self.model_provider == "anthropic":
            # Import Anthropic if needed
            from langchain_anthropic import ChatAnthropic

            self.llm = ChatAnthropic(
                model=model_name,
                temperature=temperature,
                anthropic_api_key=anthropic_api_key,
                max_retries=max_retries,
                callbacks=callbacks,
                streaming=stream_tokens,  # Enable streaming for token-by-token audit logs
            )
            logger.info(f"Initialized Anthropic model: {model_name}")
        elif self.model_provider == "xai":
            # Import langchain_groq as a temporary solution for X AI integration
            # In the future, there may be a dedicated langchain_xai package
            try:
                # Try to import LangChain X/Grok integration if available
                from langchain_xai import ChatXAI
                
                self.llm = ChatXAI(
                    model=model_name,
                    temperature=temperature,
                    api_key=xai_api_key,
                    max_retries=max_retries,
                    callbacks=callbacks,
                    streaming=stream_tokens,  # Enable streaming for token-by-token audit logs
                )
                logger.info(f"Initialized X AI (Grok) model: {model_name}")
            except ImportError:
                # If langchain_xai is not available, use a generic model
                from langchain_core.language_models.chat_models import BaseChatModel, LanguageModelInput
                from langchain_core.messages import BaseMessage
                from langchain_core.callbacks.manager import CallbackManagerForLLMRun
                import requests
                
                logger.warning("langchain_xai not found, using generic implementation")
                
                # Create a custom LangChain integration for X AI if official one doesn't exist
                class ChatXAIGeneric(BaseChatModel):
                    model: str = model_name
                    api_key: str = xai_api_key or os.environ.get("XAI_API_KEY", "")
                    temperature: float = temperature
                    streaming: bool = stream_tokens
                    
                    def _generate(
                        self, 
                        messages: List[BaseMessage], 
                        stop: Optional[List[str]] = None,
                        run_manager: Optional[CallbackManagerForLLMRun] = None,
                        **kwargs
                    ):
                        # Implement API call to X AI endpoint
                        headers = {
                            "Content-Type": "application/json",
                            "Authorization": f"Bearer {self.api_key}"
                        }
                        
                        # Convert LangChain messages to API format
                        formatted_messages = []
                        for message in messages:
                            role = message.type
                            # Map roles if needed
                            if role == "human":
                                role = "user"
                            elif role == "ai":
                                role = "assistant"
                            
                            formatted_messages.append({
                                "role": role,
                                "content": message.content
                            })
                        
                        data = {
                            "model": self.model,
                            "messages": formatted_messages,
                            "temperature": self.temperature,
                            "stream": self.streaming
                        }
                        
                        if stop:
                            data["stop"] = stop
                        
                        # Call X AI API
                        # Note: URL might need to be updated based on X AI documentation
                        response = requests.post(
                            "https://api.x.ai/v1/chat/completions",
                            headers=headers,
                            json=data
                        )
                        
                        if response.status_code != 200:
                            raise ValueError(f"Error from X AI API: {response.text}")
                        
                        response_data = response.json()
                        
                        # Format the response according to LangChain expectations
                        # This assumes X API has a similar response format to OpenAI
                        return {
                            "generations": [{
                                "text": response_data["choices"][0]["message"]["content"],
                                "message": {
                                    "role": "assistant",
                                    "content": response_data["choices"][0]["message"]["content"]
                                }
                            }],
                            "llm_output": response_data
                        }
                    
                    @property
                    def _llm_type(self) -> str:
                        return "chat_xai"
                
                self.llm = ChatXAIGeneric(
                    model=model_name,
                    api_key=xai_api_key,
                    temperature=temperature,
                    streaming=stream_tokens,
                    callbacks=callbacks
                )
                logger.info(f"Initialized generic X AI (Grok) implementation with model: {model_name}")
        else:
            # Import OpenAI
            from langchain_openai import ChatOpenAI

            self.llm = ChatOpenAI(
                model=model_name,
                temperature=temperature,
                api_key=openai_api_key,
                max_retries=max_retries,
                callbacks=callbacks,
                streaming=stream_tokens,  # Enable streaming for token-by-token audit logs
            )
            logger.info(f"Initialized OpenAI model: {model_name}")
        
        # Initialize output parsers
        self.json_parser = JsonOutputParser()
    
    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=1, max=60),
        retry=retry_if_exception_type(LangChainException)
    )
    def analyze_document_chunk(self, chunk: Dict[str, Any]) -> AnalyzedDocument:
        """
        Analyze a single document chunk with LangChain and OpenAI.
        
        Args:
            chunk: Document chunk (text and metadata)
            
        Returns:
            AnalyzedDocument: Analysis results
        """
        text = chunk["text"]
        metadata = chunk["metadata"]
        document_id = metadata.get("document_id", "unknown")
        chunk_id = metadata.get("chunk_id", "unknown")
        
        # Configure audit logger with document ID if enabled
        if self.enable_audit_logging:
            # Reset the callback for this new document
            self.audit_callback.reset()
            # Set document ID for this analysis
            self.audit_callback.document_id = f"{document_id}_chunk{chunk_id}"
            logger.info(f"Starting analysis of document {document_id}, chunk {chunk_id} with audit logging")
        
        # Create prompt template with additional thought process instructions
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", """You are an expert analyst examining declassified JFK assassination documents. Extract information in the requested JSON format.

You must adhere to these non-negotiable guidelines:

1. Source Attribution: You must only include information that is verifiable and directly found in the document excerpt. For each item, include the exact quote from the document where the information is derived from. If a fact is uncertain, explicitly state the uncertainty and do not fabricate details.

2. Fact vs. Speculation Distinction: You must clearly differentiate between documented facts and speculation. Never assert an interpretation as a confirmed fact unless it is explicitly stated in the document.

3. Information Constraints: If information related to a category is not explicitly found in the document excerpt, return an empty array for that category. Do not generate information beyond what is documented in the excerpt. Never fill in gaps with assumptions.

4. Self-Audit Requirement: Before completing your extraction, verify that each item is directly supported by text in the document excerpt. Remove any items that contain speculation or inference beyond what the document states.

5. Thought Process Documentation: Document your thought process as you analyze this text. Include your reasoning for identifying specific pieces of information, any uncertainties you have, and how you determined which category information belongs to. This will create an audit trail of your analytical approach."""),
            ("human", """
            Analyze the following document excerpt carefully for any significant information
            related to the JFK assassination, potential coverups, or government involvement.
            
            DOCUMENT EXCERPT:
            {text}
            
            For each of the following categories, extract any relevant information from the text:
            
            1. key_individuals: Key individuals mentioned (include full names and roles if available)
            2. government_agencies: Government agencies or organizations mentioned
            3. locations: Locations mentioned
            4. dates_and_times: Dates and times mentioned
            5. potential_coverup: Potential evidence of coverup or conspiracy (be objective and factual)
            6. suspicious_activities: Suspicious activities or events described
            7. assassination_theories: Connections to known assassination theories
            8. inconsistencies: Inconsistencies or contradictions in the account
            9. weapons_references: References to weapons, bullet trajectory, or cause of death
            10. redacted_information: Missing or redacted information (what seems to be deliberately omitted)
            
            For each item found, include the information, the exact quote from the document, and the page number if available.
            
            Format as JSON with these exact field names matching the categories above.
            For categories with no relevant information, include an empty array.
            
            As you work through this analysis, document your thought process. For each category, first describe what you're looking for, then note what evidence you find or don't find in the text, and finally explain your reasoning for including or excluding specific information. This thought process will be captured in the audit log.
            """)
        ])
        
        try:
            # Create chain and run with function calling method
            chain = prompt_template | self.llm.with_structured_output(
                DocumentAnalysisResult,
                method="function_calling"  # Use function calling method to fix schema issues
            )
            
            # Run the chain
            analysis_result = chain.invoke({
                "text": text,
                "document_id": document_id,
                "chunk_id": chunk_id
            })
            
            # Create analysis document
            analyzed_doc = AnalyzedDocument(
                text=text,
                metadata=metadata,
                analysis=analysis_result
            )
            
            if self.enable_audit_logging:
                logger.info(f"Completed analysis of document {document_id}, chunk {chunk_id} with audit logging")
            
            return analyzed_doc
            
        except Exception as e:
            logger.error(f"Error analyzing chunk {metadata.get('chunk_id', 'unknown')}: {e}")
            # Return error document
            return AnalyzedDocument(
                text=text,
                metadata=metadata,
                analysis=DocumentAnalysisResult(),
                error=str(e)
            )
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=60),
        retry=retry_if_exception_type(LangChainException)
    )
    def search_and_analyze_topic(
        self, 
        topic: str, 
        num_results: int = 20
    ) -> TopicAnalysis:
        """
        Search for a topic and analyze relevant documents.
        
        Args:
            topic: Topic to search for
            num_results: Number of relevant documents to analyze
            
        Returns:
            TopicAnalysis: Topic analysis results
        """
        logger.info(f"Analyzing topic: {topic}")
        
        # Search for relevant documents
        results = self.vector_store.similarity_search(topic, k=num_results)
        
        # Analyze each document chunk
        analyzed_docs = []
        for chunk in tqdm(results, desc=f"Analyzing {topic}"):
            analyzed_doc = self.analyze_document_chunk(chunk)
            analyzed_docs.append(analyzed_doc)
        
        # Configure audit logger for topic analysis if enabled
        if self.enable_audit_logging:
            # Reset the callback for this new topic analysis
            self.audit_callback.reset()
            # Set document ID for this topic analysis
            self.audit_callback.document_id = f"topic_{topic.replace(' ', '_').lower()}"
            logger.info(f"Starting topic analysis for '{topic}' with audit logging")
        
        # Create overall topic analysis prompt with thought process instructions
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", """You are an expert analyst examining declassified JFK assassination documents. Summarize findings across multiple document analyses.

You must adhere to these non-negotiable guidelines:

1. Source Attribution: You must only include information that is verifiable and sourced from the analyzed documents. For key claims, cite the specific document IDs where the information is derived. If a fact is uncertain, explicitly state the uncertainty and do not fabricate details. If a claim lacks verifiable evidence, label it as 'unverified' or 'requires further investigation.'

2. Fact vs. Speculation Distinction: You must clearly differentiate between documented facts, theories, and speculation. Present multiple perspectives where found in the documents, but never assert an unverified claim as truth. Example of proper attribution: "Document X claims Y, while Document Z suggests an alternative view."

3. Information Constraints: If information is not explicitly found in the analyzed documents, you must indicate 'Insufficient data available' rather than filling in gaps. Do not generate information beyond what is documented in the analyzed materials. If a claim lacks direct source support, state 'No evidence found in available documents' rather than speculating.

4. Self-Audit Requirement: Before completing your summary, examine each key finding to ensure it is directly supported by at least one document in the analysis. Remove or flag any findings that lack direct document support.

5. Thought Process Documentation: Document your thought process as you analyze across documents. Include your reasoning for identifying patterns, connections between documents, assessment of document credibility, and how you resolved conflicting information. Explain your methodology for determining what constitutes significant findings versus incidental information. This will create an audit trail of your cross-document analytical approach."""),
            ("human", """
            You've analyzed multiple document excerpts related to: {topic}
            
            Based on the analyzed documents, provide a comprehensive summary of the evidence and information.
            Be objective and factual.
            
            Your response should include:
            1. key_findings: Key findings and patterns across documents
            2. consistent_information: Information that appears in multiple sources
            3. contradictions: Contradictions or inconsistencies between documents
            4. potential_evidence: Potential evidence of coverup or conspiracy (be objective)
            5. missing_information: Notable gaps or missing information 
            6. assassination_theories: Connections to known assassination theories
            7. credibility: Level of credibility of the information (high, medium, low)
            8. document_references: Document references for key claims (use document_id from metadata)
            
            Include specific document references to support claims where possible.
            
            As you work through this analysis, document your thought process. For each section of your analysis:
            1. Begin by describing what you're looking for in the documents
            2. Explain how you're cross-referencing information between documents
            3. Detail your process for evaluating the credibility and significance of findings
            4. Document any challenges in synthesizing contradictory information
            5. Explain your reasoning for including specific findings in your summary
            
            This detailed thought process will be captured in the audit log and provide insight into your analytical methodology.
            """)
        ])
        
        try:
            # Create chain and run with function calling method
            chain = prompt_template | self.llm.with_structured_output(
                TopicSummary,
                method="function_calling"  # Use function calling method to fix schema issues
            )
            
            # Prepare document info for summary
            doc_summaries = []
            for doc in analyzed_docs:
                doc_summary = {
                    "document_id": doc.metadata.get("document_id", "unknown"),
                    "title": doc.metadata.get("title", ""),
                    "key_findings": [item.information for sublist in [
                        doc.analysis.key_individuals,
                        doc.analysis.government_agencies,
                        doc.analysis.suspicious_activities,
                        doc.analysis.potential_coverup
                    ] for item in sublist]
                }
                doc_summaries.append(doc_summary)
            
            # Run the chain
            summary = chain.invoke({
                "topic": topic,
                "documents": doc_summaries
            })
            
            # Complete topic analysis
            topic_analysis = TopicAnalysis(
                topic=topic,
                summary=summary,
                document_analyses=analyzed_docs,
                num_documents=len(analyzed_docs)
            )
            
            # Save to file
            output_file = os.path.join(self.output_dir, f"{topic.replace(' ', '_').lower()}.json")
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(topic_analysis.model_dump_json(indent=2))
            
            # Also save audit logs alongside the analysis results if enabled
            if self.enable_audit_logging:
                audit_file = os.path.join(self.output_dir, f"{topic.replace(' ', '_').lower()}_audit.json")
                with open(audit_file, 'w', encoding='utf-8') as f:
                    # Save a copy of the audit messages in the analysis directory for easier reference
                    json.dump({
                        "topic": topic,
                        "audit_log": self.audit_callback.messages,
                        "timestamp": datetime.datetime.now().isoformat()
                    }, f, indent=2, ensure_ascii=False)
                logger.info(f"Saved topic audit log to {audit_file}")
                
            logger.info(f"Saved topic analysis to {output_file}")
            return topic_analysis
            
        except Exception as e:
            logger.error(f"Error creating topic summary for {topic}: {e}")
            
            # Create error topic analysis
            topic_analysis = TopicAnalysis(
                topic=topic,
                summary=TopicSummary(),
                document_analyses=analyzed_docs,
                num_documents=len(analyzed_docs),
                error=str(e)
            )
            
            # Save partial results
            output_file = os.path.join(self.output_dir, f"{topic.replace(' ', '_').lower()}_partial.json")
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(topic_analysis.model_dump_json(indent=2))
                
            return topic_analysis
    
    def analyze_key_topics(self) -> List[TopicAnalysis]:
        """
        Analyze a set of predefined key topics.
        
        Returns:
            List of topic analysis results
        """
        key_topics = [
            "Lee Harvey Oswald",
            "Jack Ruby",
            "CIA involvement",
            "FBI investigation",
            "Secret Service failures",
            "Cuban connection",
            "Soviet connection",
            "Mafia involvement",
            "Multiple shooters theory",
            "Bullet trajectory evidence",
            "Autopsy inconsistencies",
            "Witness testimonies",
            "Zapruder film analysis",
            "Warren Commission criticism",
            "House Select Committee on Assassinations"
        ]
        
        results = []
        for topic in key_topics:
            analysis = self.search_and_analyze_topic(topic)
            results.append(analysis)
            
        return results
    
    def search_and_analyze_query(
        self, 
        query: str, 
        num_results: int = 20
    ) -> TopicAnalysis:
        """
        Search for a custom query and analyze relevant documents.
        
        Args:
            query: Custom search query
            num_results: Number of relevant documents to analyze
            
        Returns:
            TopicAnalysis: Query analysis results
        """
        return self.search_and_analyze_topic(query, num_results)