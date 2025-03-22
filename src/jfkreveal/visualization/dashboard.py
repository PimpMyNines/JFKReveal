"""
Interactive dashboard for JFK documents analysis visualization.
"""
import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go
import plotly.express as px
import networkx as nx
import pandas as pd
import json
from pathlib import Path
import logging
from typing import Dict, List, Any, Optional
import os

logger = logging.getLogger(__name__)

class JFKDashboard:
    """Interactive dashboard for JFK documents analysis visualization."""
    
    def __init__(
        self, 
        data_dir: str,
        host: str = "127.0.0.1",
        port: int = 8050,
        debug: bool = False
    ):
        """
        Initialize the dashboard.
        
        Args:
            data_dir: Directory containing analysis data
            host: Host to run the dashboard on
            port: Port to run the dashboard on
            debug: Whether to run in debug mode
        """
        self.data_dir = Path(data_dir)
        self.host = host
        self.port = port
        self.debug = debug
        
        # Initialize Dash app
        self.app = dash.Dash(__name__, title="JFK Documents Analysis")
        
        # Load data
        self.data = self._load_data()
        
        # Set up layout
        self._setup_layout()
        
        # Set up callbacks
        self._setup_callbacks()
    
    def _load_data(self) -> Dict[str, Any]:
        """
        Load analysis data from files.
        
        Returns:
            Dictionary containing analysis data
        """
        data = {}
        
        # Define expected data files
        data_files = {
            "entity_network": "entity_network.json",
            "document_topics": "document_topics.json",
            "timeline_events": "timeline_events.json",
            "findings": "findings.json"
        }
        
        # Load each file if it exists
        for key, filename in data_files.items():
            file_path = self.data_dir / filename
            if file_path.exists():
                try:
                    with open(file_path, "r") as f:
                        data[key] = json.load(f)
                    logger.info(f"Loaded {key} data from {file_path}")
                except Exception as e:
                    logger.error(f"Error loading {key} data: {e}")
                    data[key] = None
            else:
                logger.warning(f"Data file not found: {file_path}")
                data[key] = None
        
        return data
    
    def _setup_layout(self) -> None:
        """Set up the dashboard layout."""
        self.app.layout = html.Div([
            # Header
            html.Div([
                html.H1("JFK Documents Analysis Dashboard"),
                html.P("Interactive visualization of declassified JFK assassination documents")
            ], className="header"),
            
            # Main content
            html.Div([
                # Tabs for different visualizations
                dcc.Tabs([
                    # Entity Network Tab
                    dcc.Tab(label="Entity Network", children=[
                        html.Div([
                            html.H3("Entity Relationship Network"),
                            html.P("Network of relationships between entities mentioned in the documents"),
                            
                            # Controls
                            html.Div([
                                html.Label("Filter by Entity Type:"),
                                dcc.Dropdown(
                                    id="entity-type-filter",
                                    options=[
                                        {"label": "All", "value": "all"},
                                        {"label": "People", "value": "people"},
                                        {"label": "Organizations", "value": "organizations"},
                                        {"label": "Locations", "value": "locations"}
                                    ],
                                    value="all"
                                ),
                                
                                html.Label("Minimum Relationship Strength:"),
                                dcc.Slider(
                                    id="relationship-strength-slider",
                                    min=0,
                                    max=1,
                                    step=0.05,
                                    value=0.3,
                                    marks={i/10: str(i/10) for i in range(0, 11, 2)}
                                )
                            ], className="controls"),
                            
                            # Network graph
                            dcc.Graph(id="entity-network-graph", style={"height": "700px"}),
                            
                            # Selected entity details
                            html.Div(id="entity-details", className="details-panel")
                        ])
                    ]),
                    
                    # Timeline Tab
                    dcc.Tab(label="Events Timeline", children=[
                        html.Div([
                            html.H3("Events Timeline"),
                            html.P("Timeline of key events related to the JFK assassination"),
                            
                            # Timeline visualization
                            dcc.Graph(id="timeline-graph", style={"height": "600px"}),
                            
                            # Selected event details
                            html.Div(id="event-details", className="details-panel")
                        ])
                    ]),
                    
                    # Key Findings Tab
                    dcc.Tab(label="Key Findings", children=[
                        html.Div([
                            html.H3("Key Findings"),
                            html.P("Summary of key findings from the document analysis"),
                            
                            # Findings list
                            html.Div(id="findings-list", className="findings-panel")
                        ])
                    ]),
                    
                    # Document Search Tab
                    dcc.Tab(label="Document Search", children=[
                        html.Div([
                            html.H3("Document Search"),
                            html.P("Search through the analyzed documents"),
                            
                            # Search controls
                            html.Div([
                                dcc.Input(
                                    id="search-input",
                                    type="text",
                                    placeholder="Enter search query...",
                                    style={"width": "80%"}
                                ),
                                html.Button("Search", id="search-button")
                            ], className="search-controls"),
                            
                            # Search results
                            html.Div(id="search-results", className="search-results")
                        ])
                    ])
                ])
            ], className="main-content")
        ], className="dashboard-container")
    
    def _setup_callbacks(self) -> None:
        """Set up the dashboard callbacks."""
        # Entity Network Graph Callback
        @self.app.callback(
            Output("entity-network-graph", "figure"),
            [
                Input("entity-type-filter", "value"),
                Input("relationship-strength-slider", "value")
            ]
        )
        def update_network_graph(entity_type, min_strength):
            if not self.data.get("entity_network"):
                return self._empty_graph("No entity network data available")
            
            # Create network graph
            return self._create_network_graph(entity_type, min_strength)
        
        # Entity Details Callback
        @self.app.callback(
            Output("entity-details", "children"),
            [Input("entity-network-graph", "clickData")]
        )
        def update_entity_details(click_data):
            if not click_data or not self.data.get("entity_network"):
                return html.Div("Select an entity to see details")
            
            # Get entity name from click data
            try:
                entity_name = click_data["points"][0]["text"]
                
                # Find entity data
                entity_data = None
                for node in self.data["entity_network"]["nodes"]:
                    if node["label"] == entity_name:
                        entity_data = node
                        break
                
                if not entity_data:
                    return html.Div(f"No details found for {entity_name}")
                
                # Find connections
                connections = []
                for edge in self.data["entity_network"]["edges"]:
                    if edge["source"] == entity_name or edge["target"] == entity_name:
                        other_entity = edge["target"] if edge["source"] == entity_name else edge["source"]
                        connections.append({
                            "entity": other_entity,
                            "strength": edge["weight"],
                            "documents": edge.get("documents", [])
                        })
                
                # Sort connections by strength
                connections.sort(key=lambda x: x["strength"], reverse=True)
                
                # Create details panel
                return html.Div([
                    html.H4(entity_name),
                    html.P(f"Type: {entity_data.get('type', 'Unknown')}"),
                    html.P(f"Connections: {len(connections)}"),
                    
                    html.H5("Top Connections:"),
                    html.Ul([
                        html.Li([
                            html.Strong(f"{conn['entity']}"),
                            html.Span(f" (Strength: {conn['strength']:.2f})"),
                            html.Div(f"Mentioned together in {len(conn['documents'])} documents")
                        ]) for conn in connections[:10]  # Show top 10
                    ])
                ])
            except (KeyError, IndexError) as e:
                logger.error(f"Error extracting entity details: {e}")
                return html.Div("Error displaying entity details")
        
        # Timeline Graph Callback
        @self.app.callback(
            Output("timeline-graph", "figure"),
            [Input("timeline-graph", "relayoutData")]
        )
        def update_timeline(relayout_data):
            if not self.data.get("timeline_events"):
                return self._empty_graph("No timeline data available")
            
            # Create timeline visualization
            return self._create_timeline()
        
        # Event Details Callback
        @self.app.callback(
            Output("event-details", "children"),
            [Input("timeline-graph", "clickData")]
        )
        def update_event_details(click_data):
            if not click_data or not self.data.get("timeline_events"):
                return html.Div("Select an event to see details")
            
            # Extract event details
            try:
                point_index = click_data["points"][0]["pointIndex"]
                event = self.data["timeline_events"][point_index]
                
                return html.Div([
                    html.H4(event["event"]),
                    html.P(f"Date: {event['date']}"),
                    html.P(f"Location: {event.get('location', 'Unknown')}"),
                    html.H5("Description:"),
                    html.P(event["description"]),
                    html.H5("Related Documents:"),
                    html.Ul([
                        html.Li(doc_id) for doc_id in event.get("documents", [])
                    ])
                ])
            except (KeyError, IndexError) as e:
                logger.error(f"Error extracting event details: {e}")
                return html.Div("Error displaying event details")
        
        # Key Findings Callback
        @self.app.callback(
            Output("findings-list", "children"),
            [Input("findings-list", "id")]  # Dummy input to trigger callback on load
        )
        def update_findings(dummy):
            if not self.data.get("findings"):
                return html.Div("No findings data available")
            
            # Display key findings
            findings = self.data["findings"]
            categories = [
                "key_findings", 
                "potential_evidence",
                "inconsistencies",
                "important_individuals"
            ]
            
            result = []
            for category in categories:
                if category in findings and findings[category]:
                    result.append(html.H4(category.replace("_", " ").title()))
                    result.append(html.Ul([
                        html.Li(item) for item in findings[category]
                    ]))
            
            return html.Div(result)
        
        # Document Search Callback
        @self.app.callback(
            Output("search-results", "children"),
            [Input("search-button", "n_clicks")],
            [State("search-input", "value")]
        )
        def perform_search(n_clicks, query):
            if not n_clicks or not query:
                return html.Div("Enter a search query and click Search")
            
            # In a real implementation, this would call the SemanticSearchEngine
            # For demo purposes, just return placeholder results
            return html.Div([
                html.P(f"Search results for: {query}"),
                html.Div([
                    html.Div([
                        html.H4(f"Result {i+1}"),
                        html.P(f"This is a placeholder result for '{query}'"),
                        html.P(f"Document ID: doc_{i}", className="document-id")
                    ], className="search-result") for i in range(5)
                ])
            ])
    
    def _create_network_graph(self, entity_type: str, min_strength: float) -> go.Figure:
        """
        Create network graph visualization.
        
        Args:
            entity_type: Type of entity to filter for ('all' for no filter)
            min_strength: Minimum relationship strength to include
            
        Returns:
            Plotly figure with network graph
        """
        # Extract nodes and edges from data
        try:
            nodes = self.data["entity_network"]["nodes"]
            edges = self.data["entity_network"]["edges"]
            
            # Filter nodes by type if specified
            if entity_type != "all":
                filtered_nodes = [n for n in nodes if n.get("type") == entity_type]
                node_ids = {n["id"] for n in filtered_nodes}
                
                # Filter edges to only include connections between filtered nodes
                filtered_edges = [
                    e for e in edges 
                    if e["source"] in node_ids and e["target"] in node_ids
                ]
            else:
                filtered_nodes = nodes
                filtered_edges = edges
            
            # Filter edges by minimum strength
            filtered_edges = [e for e in filtered_edges if e.get("weight", 0) >= min_strength]
            
            # Get the IDs of nodes that have edges after filtering
            connected_nodes = set()
            for edge in filtered_edges:
                connected_nodes.add(edge["source"])
                connected_nodes.add(edge["target"])
            
            # Only keep nodes that have connections
            filtered_nodes = [n for n in filtered_nodes if n["id"] in connected_nodes]
            
            # Create network layout
            G = nx.Graph()
            for node in filtered_nodes:
                G.add_node(node["id"], **node)
            
            for edge in filtered_edges:
                G.add_edge(edge["source"], edge["target"], weight=edge.get("weight", 1.0))
            
            # Use spring layout for node positions
            pos = nx.spring_layout(G, seed=42)
            
            # Create node trace
            node_x = []
            node_y = []
            node_text = []
            node_size = []
            node_color = []
            
            color_map = {
                "people": "rgba(31, 119, 180, 0.8)",  # blue
                "organizations": "rgba(255, 127, 14, 0.8)",  # orange
                "locations": "rgba(44, 160, 44, 0.8)",  # green
                "events": "rgba(214, 39, 40, 0.8)",  # red
            }
            
            for node in filtered_nodes:
                x, y = pos[node["id"]]
                node_x.append(x)
                node_y.append(y)
                node_text.append(node["label"])
                node_size.append(node.get("size", 10))
                
                node_type = node.get("type", "unknown")
                node_color.append(color_map.get(node_type, "rgba(128, 128, 128, 0.8)"))
            
            node_trace = go.Scatter(
                x=node_x, y=node_y,
                mode='markers',
                hoverinfo='text',
                text=node_text,
                marker=dict(
                    color=node_color,
                    size=node_size,
                    line=dict(width=1, color='rgba(50, 50, 50, 0.8)')
                )
            )
            
            # Create edge traces
            edge_traces = []
            for edge in filtered_edges:
                source = edge["source"]
                target = edge["target"]
                
                if source in pos and target in pos:
                    x0, y0 = pos[source]
                    x1, y1 = pos[target]
                    
                    # Adjust line width based on weight
                    weight = edge.get("weight", 1.0)
                    line_width = 1 + (weight * 5)
                    
                    edge_trace = go.Scatter(
                        x=[x0, x1, None],
                        y=[y0, y1, None],
                        mode='lines',
                        line=dict(
                            width=line_width,
                            color='rgba(50, 50, 50, 0.5)'
                        ),
                        hoverinfo='none'
                    )
                    edge_traces.append(edge_trace)
            
            # Create figure
            fig = go.Figure(data=edge_traces + [node_trace],
                            layout=go.Layout(
                                showlegend=False,
                                hovermode='closest',
                                margin=dict(b=20, l=5, r=5, t=40),
                                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                title=f"Entity Network - {len(filtered_nodes)} entities, {len(filtered_edges)} connections"
                            ))
            
            return fig
            
        except (KeyError, IndexError) as e:
            logger.error(f"Error creating network graph: {e}")
            return self._empty_graph(f"Error creating graph: {str(e)}")
    
    def _create_timeline(self) -> go.Figure:
        """
        Create timeline visualization.
        
        Returns:
            Plotly figure with timeline
        """
        try:
            events = self.data["timeline_events"]
            
            # Create dataframe for timeline
            df = pd.DataFrame(events)
            
            # Ensure date column exists
            if "date" not in df.columns:
                return self._empty_graph("Timeline data missing date information")
            
            # Convert to datetime if not already
            df["date"] = pd.to_datetime(df["date"])
            
            # Sort by date
            df = df.sort_values("date")
            
            # Create figure
            fig = px.scatter(
                df,
                x="date",
                y=[1] * len(df),  # All events on same line
                text="event",
                hover_name="event",
                hover_data=["location", "description"],
                title="JFK Assassination Timeline"
            )
            
            # Update layout
            fig.update_layout(
                yaxis=dict(
                    visible=False
                ),
                xaxis=dict(
                    title="Date"
                ),
                showlegend=False
            )
            
            # Add vertical lines for key events
            for i, row in df.iterrows():
                fig.add_vline(
                    x=row["date"],
                    line_width=1,
                    line_dash="dash",
                    line_color="gray"
                )
            
            return fig
            
        except (KeyError, ValueError) as e:
            logger.error(f"Error creating timeline: {e}")
            return self._empty_graph(f"Error creating timeline: {str(e)}")
    
    def _empty_graph(self, message: str) -> go.Figure:
        """
        Create an empty graph with a message.
        
        Args:
            message: Message to display
            
        Returns:
            Empty Plotly figure with message
        """
        return go.Figure().update_layout(
            title=message,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
    
    def run(self) -> None:
        """Run the dashboard."""
        logger.info(f"Starting dashboard at http://{self.host}:{self.port}")
        self.app.run_server(
            host=self.host,
            port=self.port,
            debug=self.debug
        ) 