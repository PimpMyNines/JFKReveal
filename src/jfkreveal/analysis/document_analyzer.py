"""
Document analyzer using LangChain and OpenAI to extract and analyze information.
"""
import os
import json
import logging
from typing import List, Dict, Any, Optional, Union

from pydantic import BaseModel, Field
from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from langchain_openai import ChatOpenAI
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.output_parsers.json import JsonOutputParser
from langchain_core.exceptions import LangChainException

from ..database.vector_store import VectorStore

logger = logging.getLogger(__name__)

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
        openai_api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_retries: int = 5,
    ):
        """
        Initialize the document analyzer.
        
        Args:
            vector_store: Vector store instance for searching documents
            output_dir: Directory to save analysis results
            model_name: OpenAI model to use for analysis
            openai_api_key: OpenAI API key (uses environment variable if not provided)
            temperature: Temperature for LLM generation
            max_retries: Maximum number of retries for API calls
        """
        self.vector_store = vector_store
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
        
        # Create prompt template
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", "You are an expert analyst examining declassified JFK assassination documents. Extract information in the requested JSON format."),
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
            """)
        ])
        
        try:
            # Create chain and run with function calling method
            chain = prompt_template | self.llm.with_structured_output(
                DocumentAnalysisResult,
                method="function_calling"  # Use function calling method to fix schema issues
            )
            
            # Run the chain
            analysis_result = chain.invoke({"text": text})
            
            # Create analysis document
            analyzed_doc = AnalyzedDocument(
                text=text,
                metadata=metadata,
                analysis=analysis_result
            )
            
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
        
        # Create overall topic analysis prompt
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", "You are an expert analyst examining declassified JFK assassination documents. Summarize findings across multiple document analyses."),
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