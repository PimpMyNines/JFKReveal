"""
Enhanced document analysis with entity recognition and relationship mapping.
"""
import networkx as nx
import spacy
import logging
from typing import Dict, List, Set, Tuple, Any, Optional
import json
from pathlib import Path

logger = logging.getLogger(__name__)

class EnhancedDocumentAnalyzer:
    """Advanced document analysis with entity recognition and relationship mapping."""
    
    def __init__(self, nlp_model: str = "en_core_web_sm", confidence_threshold: float = 0.5):
        """
        Initialize the enhanced document analyzer.
        
        Args:
            nlp_model: The spaCy model to use for NLP tasks
            confidence_threshold: Minimum confidence threshold for relationships
        """
        try:
            self.nlp = spacy.load(nlp_model)
        except OSError:
            # If model isn't installed, download it
            logger.info(f"Downloading spaCy model: {nlp_model}")
            spacy.cli.download(nlp_model)
            self.nlp = spacy.load(nlp_model)
        
        # Configure named entity recognition
        self.ner_labels = {
            "PERSON": "people",
            "ORG": "organizations",
            "GPE": "locations",
            "DATE": "dates",
            "NORP": "groups",  # Nationalities, religious or political groups
            "FAC": "facilities",
            "LOC": "locations",
            "EVENT": "events"
        }
        
        # JFK-specific entity lists for better recognition
        self.known_entities = self._load_known_entities()
        self.confidence_threshold = confidence_threshold
        
        # Graph for entity relationships
        self.entity_graph = nx.Graph()
    
    def _load_known_entities(self) -> Dict[str, Set[str]]:
        """Load known entities related to JFK assassination."""
        # Path to the known entities file
        file_path = Path(__file__).parent / "data" / "known_entities.json"
        
        # Default entities if file doesn't exist
        default_entities = {
            "people": {
                "Lee Harvey Oswald", "Jack Ruby", "John F. Kennedy", 
                "Jacqueline Kennedy", "J. Edgar Hoover", "Earl Warren"
            },
            "organizations": {
                "CIA", "FBI", "Warren Commission", "KGB", "Dallas Police Department"
            },
            "locations": {
                "Dallas", "Texas School Book Depository", "Dealey Plaza", "Grassy Knoll"
            },
            "events": {
                "Assassination", "Warren Commission Hearings"
            }
        }
        
        # Try to load from file, fall back to defaults
        try:
            if file_path.exists():
                with open(file_path, "r") as f:
                    return json.load(f)
            else:
                # Create directory if it doesn't exist
                file_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Write default entities to file for future use
                with open(file_path, "w") as f:
                    json.dump(default_entities, f, indent=2)
                
                return default_entities
        except Exception as e:
            logger.warning(f"Error loading known entities: {e}")
            return default_entities
    
    def _extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract entities from text using spaCy and custom rules.
        
        Args:
            text: Document text
            
        Returns:
            Dictionary of entity types and found entities
        """
        doc = self.nlp(text)
        entities = {category: set() for category in set(self.ner_labels.values())}
        
        # Extract entities from spaCy NER
        for ent in doc.ents:
            if ent.label_ in self.ner_labels:
                category = self.ner_labels[ent.label_]
                entities[category].add(ent.text)
        
        # Add known entities if they appear in the text
        for category, entity_set in self.known_entities.items():
            for entity in entity_set:
                if entity.lower() in text.lower():
                    if category not in entities:
                        entities[category] = set()
                    entities[category].add(entity)
        
        # Convert sets to lists for JSON serialization
        return {k: list(v) for k, v in entities.items() if v}
    
    def analyze_entity_relationships(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Identify relationships between entities across documents.
        
        Args:
            documents: List of documents with text and metadata
            
        Returns:
            Dictionary with entity relationship analysis
        """
        # Reset the entity graph
        self.entity_graph = nx.Graph()
        
        # Track documents where each entity appears
        entity_documents = {}
        
        # Process each document
        for doc in documents:
            doc_id = doc["metadata"]["document_id"]
            entities = self._extract_entities(doc["text"])
            
            # Add entities to the graph
            for category, entity_list in entities.items():
                for entity in entity_list:
                    # Add entity to the graph if not already present
                    if entity not in self.entity_graph:
                        self.entity_graph.add_node(entity, type=category)
                    
                    # Track which documents this entity appears in
                    if entity not in entity_documents:
                        entity_documents[entity] = set()
                    entity_documents[entity].add(doc_id)
        
        # Find co-occurrences across documents
        for entity1, docs1 in entity_documents.items():
            for entity2, docs2 in entity_documents.items():
                if entity1 != entity2:
                    # Find documents where both entities appear
                    common_docs = docs1.intersection(docs2)
                    
                    if common_docs:
                        # Calculate co-occurrence strength
                        strength = len(common_docs) / min(len(docs1), len(docs2))
                        
                        # Only add relationships above the confidence threshold
                        if strength >= self.confidence_threshold:
                            if not self.entity_graph.has_edge(entity1, entity2):
                                self.entity_graph.add_edge(
                                    entity1, entity2, 
                                    weight=strength,
                                    documents=list(common_docs)
                                )
        
        # Generate analysis results
        return self._generate_relationship_report()
    
    def _generate_relationship_report(self) -> Dict[str, Any]:
        """
        Generate a report on entity relationships.
        
        Returns:
            Dictionary with relationship analysis
        """
        # Get central entities by eigenvector centrality
        centrality = nx.eigenvector_centrality(self.entity_graph, weight='weight', max_iter=1000)
        central_entities = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:20]
        
        # Get communities using Louvain method
        try:
            from community import best_partition
            communities = best_partition(self.entity_graph, weight='weight')
            community_groups = {}
            for entity, community_id in communities.items():
                if community_id not in community_groups:
                    community_groups[community_id] = []
                community_groups[community_id].append(entity)
        except ImportError:
            # Fall back to connected components if python-louvain isn't installed
            communities = list(nx.connected_components(self.entity_graph))
            community_groups = {i: list(community) for i, community in enumerate(communities)}
        
        # Strong relationships (high weight edges)
        relationships = []
        for u, v, data in self.entity_graph.edges(data=True):
            relationships.append({
                "entity1": u,
                "entity2": v,
                "strength": data["weight"],
                "documents": data["documents"]
            })
        
        # Sort by strength
        relationships.sort(key=lambda x: x["strength"], reverse=True)
        
        # Create the report
        report = {
            "central_entities": [
                {"entity": entity, "centrality": score} for entity, score in central_entities
            ],
            "entity_communities": [
                {"community_id": comm_id, "entities": entities}
                for comm_id, entities in community_groups.items()
            ],
            "strong_relationships": relationships[:50],  # Top 50 relationships
            "total_entities": self.entity_graph.number_of_nodes(),
            "total_relationships": self.entity_graph.number_of_edges()
        }
        
        return report
    
    def export_network_data(self, output_path: str) -> None:
        """
        Export network data for visualization.
        
        Args:
            output_path: Path to save the network data JSON
        """
        # Convert network to a format suitable for visualization libraries
        nodes = []
        for node, attr in self.entity_graph.nodes(data=True):
            nodes.append({
                "id": node,
                "label": node,
                "type": attr.get("type", "unknown"),
                "size": 1 + (self.entity_graph.degree(node) * 2)
            })
        
        edges = []
        for source, target, attr in self.entity_graph.edges(data=True):
            edges.append({
                "source": source,
                "target": target,
                "weight": attr.get("weight", 1.0),
                "documents": attr.get("documents", [])
            })
        
        network_data = {
            "nodes": nodes,
            "edges": edges
        }
        
        # Save to file
        with open(output_path, "w") as f:
            json.dump(network_data, f, indent=2) 