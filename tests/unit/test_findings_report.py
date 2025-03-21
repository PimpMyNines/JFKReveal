"""
Unit tests for the FindingsReport class with multi-model support
"""
import os
import pytest
from unittest.mock import patch, MagicMock, mock_open

from jfkreveal.summarization.findings_report import FindingsReport
from jfkreveal.utils.model_config import ReportType, ModelConfiguration
from jfkreveal.utils.model_registry import ModelProvider


class TestFindingsReport:
    """Test the FindingsReport class with focus on multi-model functionality"""

    @patch('jfkreveal.utils.model_config.ModelConfiguration')
    def test_init(self, mock_model_config_class, temp_data_dir):
        """Test initialization of FindingsReport"""
        # Setup mock config
        mock_config = MagicMock()
        mock_model_config_class.return_value = mock_config
        mock_config.get_report_type.return_value = ReportType.STANDARD
        
        # Create report instance
        report = FindingsReport(
            data_dir=temp_data_dir["root"] + "/data",
            output_dir=temp_data_dir["root"] + "/reports"
        )
        
        # Verify attributes
        assert report.data_dir == temp_data_dir["root"] + "/data"
        assert report.output_dir == temp_data_dir["root"] + "/reports"
        assert report.model_config == mock_config
        assert report.report_type == ReportType.STANDARD
        
        # Verify output directory was created
        assert os.path.exists(temp_data_dir["root"] + "/reports")

    @patch('jfkreveal.utils.model_config.ModelConfiguration')
    def test_get_model_output_dir(self, mock_model_config_class, temp_data_dir):
        """Test getting model-specific output directory"""
        # Setup mock config
        mock_config = MagicMock()
        mock_model_config_class.return_value = mock_config
        mock_config.get_report_type.return_value = ReportType.MULTI_MODEL_COMPARISON
        
        # Create report instance
        report = FindingsReport(
            data_dir=temp_data_dir["root"] + "/data",
            output_dir=temp_data_dir["root"] + "/reports"
        )
        
        # Call method with model name
        model_dir = report._get_model_output_dir("gpt-4o")
        
        # Verify model-specific directory was created
        expected_dir = os.path.join(temp_data_dir["root"] + "/reports", "gpt-4o")
        assert model_dir == expected_dir
        assert os.path.exists(expected_dir)
        
        # Call method without model name (default directory)
        default_dir = report._get_model_output_dir(None)
        
        # Verify default directory
        assert default_dir == temp_data_dir["root"] + "/reports"

    @patch('jfkreveal.utils.model_config.ModelConfiguration')
    @patch('jfkreveal.summarization.findings_report.FindingsReport._save_report_file')
    @patch('jfkreveal.summarization.findings_report.ChatOpenAI')
    def test_create_full_report_for_model(self, mock_chat_openai, mock_save_report, mock_model_config_class, temp_data_dir):
        """Test creating a full report for a specific model"""
        # Setup mock config
        mock_config = MagicMock()
        mock_model_config_class.return_value = mock_config
        mock_config.get_report_type.return_value = ReportType.MULTI_MODEL_COMPARISON
        
        # Setup mock LLM
        mock_llm = MagicMock()
        mock_chat_openai.return_value = mock_llm
        
        # Setup mock for saving reports
        mock_save_report.return_value = None
        
        # Create mock report content
        mock_exec_summary = "Executive Summary Content"
        mock_detailed_findings = "Detailed Findings Content"
        mock_suspects_analysis = "Suspects Analysis Content"
        mock_coverup_analysis = "Coverup Analysis Content"
        
        # Create report instance with mock generation methods
        with patch.multiple(
            'jfkreveal.summarization.findings_report.FindingsReport',
            generate_executive_summary=MagicMock(return_value=mock_exec_summary),
            generate_detailed_findings=MagicMock(return_value=mock_detailed_findings),
            generate_suspects_analysis=MagicMock(return_value=mock_suspects_analysis),
            generate_coverup_analysis=MagicMock(return_value=mock_coverup_analysis)
        ):
            report = FindingsReport(
                data_dir=temp_data_dir["root"] + "/data",
                output_dir=temp_data_dir["root"] + "/reports"
            )
            
            # Call method
            model_info = {"name": "gpt-4o", "provider": ModelProvider.OPENAI}
            report._create_full_report_for_model(model_info)
        
        # Verify _save_report_file was called 4 times (one for each report section)
        assert mock_save_report.call_count == 4
        
        # Verify each report section was generated
        report.generate_executive_summary.assert_called_once()
        report.generate_detailed_findings.assert_called_once()
        report.generate_suspects_analysis.assert_called_once()
        report.generate_coverup_analysis.assert_called_once()

    @patch('jfkreveal.utils.model_config.ModelConfiguration')
    @patch('jfkreveal.summarization.findings_report.FindingsReport._save_report_file')
    @patch('jfkreveal.summarization.findings_report.ChatOpenAI')
    @patch('os.path.exists')
    @patch('builtins.open', new_callable=mock_open)
    def test_create_consolidated_report(self, mock_file, mock_exists, mock_chat_openai, 
                                       mock_save_report, mock_model_config_class, temp_data_dir):
        """Test creating a consolidated report from multiple model reports"""
        # Setup mock config
        mock_config = MagicMock()
        mock_model_config_class.return_value = mock_config
        mock_config.get_report_type.return_value = ReportType.CONSOLIDATED
        
        # Setup mock for checking file existence
        mock_exists.return_value = True
        
        # Setup mock LLM
        mock_llm = MagicMock()
        mock_llm.invoke = MagicMock(return_value=MagicMock(content="Consolidated analysis content"))
        mock_chat_openai.return_value = mock_llm
        
        # Mock for reading individual model reports
        model_report_contents = {
            "gpt-4o": {
                "executive_summary.md": "GPT-4o Executive Summary",
                "detailed_findings.md": "GPT-4o Detailed Findings",
                "suspects_analysis.md": "GPT-4o Suspects Analysis",
                "coverup_analysis.md": "GPT-4o Coverup Analysis"
            },
            "llama3": {
                "executive_summary.md": "Llama3 Executive Summary",
                "detailed_findings.md": "Llama3 Detailed Findings",
                "suspects_analysis.md": "Llama3 Suspects Analysis",
                "coverup_analysis.md": "Llama3 Coverup Analysis"
            }
        }
        
        def mock_file_reader(file_path, *args, **kwargs):
            # Extract model and file name from the path
            # Assuming path format: some/path/model_name/file_name.md
            path_parts = file_path.split(os.sep)
            if len(path_parts) < 2:
                return mock_open().return_value
                
            model_name = path_parts[-2]  # Second to last part is model name
            file_name = path_parts[-1]   # Last part is file name
            
            if model_name in model_report_contents and file_name in model_report_contents[model_name]:
                m = mock_open(read_data=model_report_contents[model_name][file_name])
                return m()
            return mock_open().return_value
        
        # Set the side_effect to use our custom function
        mock_file.side_effect = mock_file_reader
        
        # Create report instance
        report = FindingsReport(
            data_dir=temp_data_dir["root"] + "/data",
            output_dir=temp_data_dir["root"] + "/reports"
        )
        
        # Setup model info
        models = [
            {"name": "gpt-4o", "provider": ModelProvider.OPENAI},
            {"name": "llama3", "provider": ModelProvider.OLLAMA}
        ]
        
        # Call method
        report._create_consolidated_report(models)
        
        # Verify LLM was called for each section
        assert mock_llm.invoke.call_count == 4
        
        # Verify save_report_file was called for each consolidated section
        assert mock_save_report.call_count == 4

    @patch('jfkreveal.utils.model_config.ModelConfiguration')
    @patch('jfkreveal.summarization.findings_report.FindingsReport._create_full_report_for_model')
    @patch('jfkreveal.summarization.findings_report.FindingsReport._create_consolidated_report')
    def test_generate_full_report_standard(self, mock_consolidated, mock_full_report, mock_model_config_class, temp_data_dir):
        """Test generate_full_report with standard report type"""
        # Setup mock config
        mock_config = MagicMock()
        mock_model_config_class.return_value = mock_config
        mock_config.get_report_type.return_value = ReportType.STANDARD
        mock_config.get_analysis_models.return_value = [
            {"name": "gpt-4o", "provider": ModelProvider.OPENAI}
        ]
        
        # Create report instance
        report = FindingsReport(
            data_dir=temp_data_dir["root"] + "/data",
            output_dir=temp_data_dir["root"] + "/reports"
        )
        
        # Call method
        result = report.generate_full_report()
        
        # Verify _create_full_report_for_model was called only once with primary model
        mock_full_report.assert_called_once()
        args = mock_full_report.call_args[0][0]
        assert args["name"] == "gpt-4o"
        
        # Verify _create_consolidated_report was not called
        mock_consolidated.assert_not_called()
        
        # Verify result
        assert result is True

    @patch('jfkreveal.utils.model_config.ModelConfiguration')
    @patch('jfkreveal.summarization.findings_report.FindingsReport._create_full_report_for_model')
    @patch('jfkreveal.summarization.findings_report.FindingsReport._create_consolidated_report')
    def test_generate_full_report_multi_model(self, mock_consolidated, mock_full_report, mock_model_config_class, temp_data_dir):
        """Test generate_full_report with multi-model comparison"""
        # Setup mock config
        mock_config = MagicMock()
        mock_model_config_class.return_value = mock_config
        mock_config.get_report_type.return_value = ReportType.MULTI_MODEL_COMPARISON
        mock_config.get_analysis_models.return_value = [
            {"name": "gpt-4o", "provider": ModelProvider.OPENAI},
            {"name": "claude-3-opus", "provider": ModelProvider.ANTHROPIC},
            {"name": "llama3", "provider": ModelProvider.OLLAMA}
        ]
        
        # Create report instance
        report = FindingsReport(
            data_dir=temp_data_dir["root"] + "/data",
            output_dir=temp_data_dir["root"] + "/reports"
        )
        
        # Call method
        result = report.generate_full_report()
        
        # Verify _create_full_report_for_model was called for each model
        assert mock_full_report.call_count == 3
        model_names = [call[0][0]["name"] for call in mock_full_report.call_args_list]
        assert "gpt-4o" in model_names
        assert "claude-3-opus" in model_names
        assert "llama3" in model_names
        
        # Verify _create_consolidated_report was not called (not relevant for multi-model)
        mock_consolidated.assert_not_called()
        
        # Verify result
        assert result is True

    @patch('jfkreveal.utils.model_config.ModelConfiguration')
    @patch('jfkreveal.summarization.findings_report.FindingsReport._create_full_report_for_model')
    @patch('jfkreveal.summarization.findings_report.FindingsReport._create_consolidated_report')
    def test_generate_full_report_consolidated(self, mock_consolidated, mock_full_report, mock_model_config_class, temp_data_dir):
        """Test generate_full_report with consolidated model reports"""
        # Setup mock config
        mock_config = MagicMock()
        mock_model_config_class.return_value = mock_config
        mock_config.get_report_type.return_value = ReportType.CONSOLIDATED
        mock_config.get_analysis_models.return_value = [
            {"name": "gpt-4o", "provider": ModelProvider.OPENAI},
            {"name": "claude-3-opus", "provider": ModelProvider.ANTHROPIC}
        ]
        
        # Create report instance
        report = FindingsReport(
            data_dir=temp_data_dir["root"] + "/data",
            output_dir=temp_data_dir["root"] + "/reports"
        )
        
        # Call method
        result = report.generate_full_report()
        
        # Verify _create_full_report_for_model was called for each model
        assert mock_full_report.call_count == 2
        
        # Verify _create_consolidated_report was called with all models
        mock_consolidated.assert_called_once()
        models_arg = mock_consolidated.call_args[0][0]
        assert len(models_arg) == 2
        model_names = [model["name"] for model in models_arg]
        assert "gpt-4o" in model_names
        assert "claude-3-opus" in model_names
        
        # Verify result
        assert result is True