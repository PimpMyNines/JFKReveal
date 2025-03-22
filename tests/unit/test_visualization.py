"""
Unit tests for the JFKDashboard visualization component
"""
import pytest
from unittest.mock import patch, MagicMock, mock_open
import json
import os
import pandas as pd
import networkx as nx
import plotly.graph_objects as go

from jfkreveal.visualization.dashboard import JFKDashboard


class TestJFKDashboard:
    """Test cases for the JFKDashboard class"""

    @patch('jfkreveal.visualization.dashboard.dash.Dash')
    def test_dashboard_initialization(self, mock_dash, temp_data_dir):
        """Test dashboard initialization"""
        # Setup mock
        mock_app = MagicMock()
        mock_dash.return_value = mock_app
        
        # Create dashboard
        dashboard = JFKDashboard(
            data_dir=temp_data_dir["root"],
            host="127.0.0.1",
            port=8050,
            debug=False
        )
        
        # Verify initialization
        assert dashboard.data_dir.as_posix() == temp_data_dir["root"]
        assert dashboard.host == "127.0.0.1"
        assert dashboard.port == 8050
        assert dashboard.debug is False
        assert dashboard.app == mock_app
        
        # Verify methods were called
        mock_app.layout.__eq__.assert_called_once()

    @patch('jfkreveal.visualization.dashboard.dash.Dash')
    @patch('builtins.open', new_callable=mock_open)
    @patch('pathlib.Path.exists')
    def test_load_data(self, mock_exists, mock_file, mock_dash, temp_data_dir):
        """Test loading dashboard data"""
        # Setup mocks
        mock_app = MagicMock()
        mock_dash.return_value = mock_app
        mock_exists.return_value = True
        
        # Mock data files
        mock_entity_network = {
            "nodes": [
                {"id": "JFK", "type": "person", "documents": ["doc1", "doc2"]},
                {"id": "CIA", "type": "organization", "documents": ["doc1"]}
            ],
            "links": [
                {"source": "JFK", "target": "CIA", "strength": 0.8}
            ]
        }
        
        mock_document_topics = {
            "doc1": ["Assassination", "Government"],
            "doc2": ["Conspiracy", "Evidence"]
        }
        
        mock_timeline_events = [
            {"date": "1963-11-22", "event": "JFK Assassination", "location": "Dallas, TX"},
            {"date": "1964-09-24", "event": "Warren Commission Report Released"}
        ]
        
        mock_findings = {
            "key_findings": ["Finding 1", "Finding 2"],
            "entities_of_interest": ["Oswald", "Ruby"]
        }
        
        # Set up different file content based on filename
        def mock_read_data(file_path, *args, **kwargs):
            if "entity_network.json" in file_path:
                return json.dumps(mock_entity_network)
            elif "document_topics.json" in file_path:
                return json.dumps(mock_document_topics)
            elif "timeline_events.json" in file_path:
                return json.dumps(mock_timeline_events)
            elif "findings.json" in file_path:
                return json.dumps(mock_findings)
            return "{}"
        
        mock_file.return_value.__enter__.return_value.read.side_effect = mock_read_data
        
        # Create dashboard
        dashboard = JFKDashboard(data_dir=temp_data_dir["root"])
        
        # Verify data was loaded correctly
        assert dashboard.data["entity_network"] is not None
        assert dashboard.data["document_topics"] is not None
        assert dashboard.data["timeline_events"] is not None
        assert dashboard.data["findings"] is not None
        
        assert dashboard.data["entity_network"]["nodes"][0]["id"] == "JFK"
        assert dashboard.data["document_topics"]["doc1"] == ["Assassination", "Government"]
        assert dashboard.data["timeline_events"][0]["event"] == "JFK Assassination"
        assert "Finding 1" in dashboard.data["findings"]["key_findings"]

    @patch('jfkreveal.visualization.dashboard.dash.Dash')
    @patch('jfkreveal.visualization.dashboard.nx.Graph')
    @patch('jfkreveal.visualization.dashboard.go.Figure')
    def test_create_network_graph(self, mock_figure, mock_graph, mock_dash, temp_data_dir):
        """Test creating network graph visualization"""
        # Setup mocks
        mock_app = MagicMock()
        mock_dash.return_value = mock_app
        
        # Mock nx.Graph methods
        mock_nx_graph = MagicMock()
        mock_graph.return_value = mock_nx_graph
        mock_nx_graph.nodes.return_value = {"JFK": {}, "CIA": {}}
        
        # Mock for spring layout positioning
        mock_pos = {
            "JFK": (0.1, 0.2),
            "CIA": (0.5, 0.6)
        }
        
        # Create dashboard with mock data
        with patch.object(JFKDashboard, '_load_data') as mock_load_data:
            mock_load_data.return_value = {
                "entity_network": {
                    "nodes": [
                        {"id": "JFK", "type": "person", "documents": ["doc1", "doc2"]},
                        {"id": "CIA", "type": "organization", "documents": ["doc1"]}
                    ],
                    "links": [
                        {"source": "JFK", "target": "CIA", "strength": 0.8}
                    ]
                }
            }
            
            with patch('networkx.spring_layout', return_value=mock_pos):
                dashboard = JFKDashboard(data_dir=temp_data_dir["root"])
                
                # Test _create_network_graph method
                fig = dashboard._create_network_graph(entity_type="all", min_strength=0.3)
                
                # Verify nx.Graph was called correctly
                mock_graph.assert_called_once()
                
                # Verify Figure was created
                mock_figure.assert_called()

    @patch('jfkreveal.visualization.dashboard.dash.Dash')
    @patch('jfkreveal.visualization.dashboard.go.Figure')
    def test_create_timeline(self, mock_figure, mock_dash, temp_data_dir):
        """Test creating timeline visualization"""
        # Setup mocks
        mock_app = MagicMock()
        mock_dash.return_value = mock_app
        
        # Create dashboard with mock data
        with patch.object(JFKDashboard, '_load_data') as mock_load_data:
            mock_load_data.return_value = {
                "timeline_events": [
                    {"date": "1963-11-22", "event": "JFK Assassination", "location": "Dallas, TX"},
                    {"date": "1964-09-24", "event": "Warren Commission Report Released"}
                ]
            }
            
            dashboard = JFKDashboard(data_dir=temp_data_dir["root"])
            
            # Test _create_timeline method
            fig = dashboard._create_timeline()
            
            # Verify Figure was created
            mock_figure.assert_called()

    @patch('jfkreveal.visualization.dashboard.dash.Dash')
    def test_empty_graph(self, mock_dash, temp_data_dir):
        """Test creating empty graph with message"""
        # Setup mocks
        mock_app = MagicMock()
        mock_dash.return_value = mock_app
        
        # Create dashboard
        dashboard = JFKDashboard(data_dir=temp_data_dir["root"])
        
        # Test _empty_graph method
        fig = dashboard._empty_graph("No data available")
        
        # Verify figure has text annotation
        assert isinstance(fig, go.Figure)
        assert len(fig.layout.annotations) == 1
        assert fig.layout.annotations[0].text == "No data available"

    @patch('jfkreveal.visualization.dashboard.dash.Dash')
    def test_run(self, mock_dash, temp_data_dir):
        """Test run method starts the dashboard server"""
        # Setup mocks
        mock_app = MagicMock()
        mock_dash.return_value = mock_app
        
        # Create dashboard
        dashboard = JFKDashboard(
            data_dir=temp_data_dir["root"],
            host="127.0.0.1",
            port=9999,
            debug=True
        )
        
        # Test run method
        dashboard.run()
        
        # Verify app.run_server was called with correct parameters
        mock_app.run_server.assert_called_once_with(
            host="127.0.0.1",
            port=9999,
            debug=True
        ) 