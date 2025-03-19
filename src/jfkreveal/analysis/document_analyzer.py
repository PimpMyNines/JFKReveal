"""
Document analyzer using OpenAI to extract and analyze information.
"""
import os
import json
import logging
from typing import List, Dict, Any, Optional, Union

from openai import OpenAI
from tqdm import tqdm

from ..database.vector_store import VectorStore

logger = logging.getLogger(__name__)

class DocumentAnalyzer:
    """Analyze JFK documents using OpenAI models."""
    
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
        model: str = "gpt-4o",
        openai_api_key: Optional[str] = None,
    ):
        """
        Initialize the document analyzer.
        
        Args:
            vector_store: Vector store instance for searching documents
            output_dir: Directory to save analysis results
            model: OpenAI model to use for analysis
            openai_api_key: OpenAI API key (uses environment variable if not provided)
        """
        self.vector_store = vector_store
        self.output_dir = output_dir
        self.model = model
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize OpenAI client
        self.client = OpenAI(api_key=openai_api_key)
    
    def analyze_document_chunk(self, chunk: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a single document chunk with OpenAI.
        
        Args:
            chunk: Document chunk (text and metadata)
            
        Returns:
            Analysis results
        """
        text = chunk["text"]
        metadata = chunk["metadata"]
        
        prompt = f"""
        You are an expert analyst examining declassified JFK assassination documents.
        Analyze the following document excerpt carefully for any significant information
        related to the JFK assassination, potential coverups, or government involvement.
        
        DOCUMENT EXCERPT:
        {text}
        
        For each of the following categories, extract any relevant information from the text:
        
        1. Key individuals mentioned (include full names and roles if available)
        2. Government agencies or organizations mentioned
        3. Locations mentioned
        4. Dates and times mentioned
        5. Potential evidence of coverup or conspiracy (be objective and factual)
        6. Suspicious activities or events described
        7. Connections to known assassination theories
        8. Inconsistencies or contradictions in the account
        9. References to weapons, bullet trajectory, or cause of death
        10. Missing or redacted information (what seems to be deliberately omitted)
        
        FORMAT YOUR RESPONSE AS JSON with these categories as keys.
        If there is no relevant information for a category, use an empty list.
        For each piece of information, include the exact quote from the document.
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert analyst examining declassified JFK assassination documents. Respond only with the requested JSON format."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            
            analysis = json.loads(response.choices[0].message.content)
            
            # Add document metadata
            result = {
                "analysis": analysis,
                "metadata": metadata,
                "text": text
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing chunk {metadata.get('chunk_id', 'unknown')}: {e}")
            return {
                "analysis": {},
                "metadata": metadata,
                "text": text,
                "error": str(e)
            }
    
    def search_and_analyze_topic(
        self, 
        topic: str, 
        num_results: int = 20
    ) -> Dict[str, Any]:
        """
        Search for a topic and analyze relevant documents.
        
        Args:
            topic: Topic to search for
            num_results: Number of relevant documents to analyze
            
        Returns:
            Topic analysis results
        """
        logger.info(f"Analyzing topic: {topic}")
        
        # Search for relevant documents
        results = self.vector_store.similarity_search(topic, k=num_results)
        
        # Analyze each document chunk
        analyses = []
        for chunk in tqdm(results, desc=f"Analyzing {topic}"):
            analysis = self.analyze_document_chunk(chunk)
            analyses.append(analysis)
        
        # Create overall topic analysis using OpenAI
        prompt = f"""
        You are an expert analyst examining declassified JFK assassination documents.
        
        You've analyzed multiple document excerpts related to: {topic}
        
        Based on the following document analyses, provide a comprehensive summary
        of the evidence and information related to this topic. Be objective and factual.
        
        DOCUMENT ANALYSES:
        {json.dumps(analyses, indent=2)}
        
        Your response should include:
        1. Key findings and patterns across documents
        2. Consistent information that appears in multiple sources
        3. Contradictions or inconsistencies between documents
        4. Potential evidence of coverup or conspiracy (be objective)
        5. Notable gaps or missing information
        6. Connections to known assassination theories
        7. Level of credibility of the information (high, medium, low)
        8. Specific document references for key claims (use document_id from metadata)
        
        FORMAT YOUR RESPONSE AS JSON with these categories as keys.
        For key points, include document references to support claims.
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert analyst examining declassified JFK assassination documents. Respond only with the requested JSON format."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            
            summary = json.loads(response.choices[0].message.content)
            
            # Complete topic analysis
            topic_analysis = {
                "topic": topic,
                "summary": summary,
                "document_analyses": analyses,
                "num_documents": len(analyses)
            }
            
            # Save to file
            output_file = os.path.join(self.output_dir, f"{topic.replace(' ', '_').lower()}.json")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(topic_analysis, f, ensure_ascii=False, indent=2)
                
            logger.info(f"Saved topic analysis to {output_file}")
            return topic_analysis
            
        except Exception as e:
            logger.error(f"Error creating topic summary for {topic}: {e}")
            
            # Save partial results
            topic_analysis = {
                "topic": topic,
                "document_analyses": analyses,
                "num_documents": len(analyses),
                "error": str(e)
            }
            
            output_file = os.path.join(self.output_dir, f"{topic.replace(' ', '_').lower()}_partial.json")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(topic_analysis, f, ensure_ascii=False, indent=2)
                
            return topic_analysis
    
    def analyze_key_topics(self) -> List[Dict[str, Any]]:
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
    ) -> Dict[str, Any]:
        """
        Search for a custom query and analyze relevant documents.
        
        Args:
            query: Custom search query
            num_results: Number of relevant documents to analyze
            
        Returns:
            Query analysis results
        """
        return self.search_and_analyze_topic(query, num_results)