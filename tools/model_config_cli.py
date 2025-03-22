#!/usr/bin/env python3
"""
Command-line utility to manage model configuration for JFKReveal.
Allows users to view and change model settings without editing JSON files.
"""

import os
import sys
import argparse
import json
from enum import Enum
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

# Add parent directory to path to allow importing the model configuration
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.jfkreveal.utils.model_registry import ModelProvider, ModelType
from src.jfkreveal.utils.model_config import ModelConfiguration, ReportType, AnalysisTask

# ANSI color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_header(text):
    """Print formatted header text."""
    print(f"\n{Colors.BOLD}{Colors.HEADER}{text}{Colors.ENDC}\n")

def print_section(text):
    """Print formatted section text."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{text}{Colors.ENDC}")

def print_success(text):
    """Print formatted success text."""
    print(f"{Colors.GREEN}{text}{Colors.ENDC}")

def print_warning(text):
    """Print formatted warning text."""
    print(f"{Colors.YELLOW}{text}{Colors.ENDC}")

def print_error(text):
    """Print formatted error text."""
    print(f"{Colors.RED}{text}{Colors.ENDC}")

def list_available_models(config: ModelConfiguration, provider: Optional[str] = None, 
                           model_type: Optional[str] = None, local_only: bool = False):
    """List available models with details."""
    print_header("Available Models")
    
    # Convert string inputs to enums if provided
    provider_enum = None
    if provider:
        try:
            provider_enum = ModelProvider(provider.lower())
        except ValueError:
            print_error(f"Invalid provider: {provider}")
            return
    
    model_type_enum = None
    if model_type:
        try:
            model_type_enum = ModelType(model_type.lower())
        except ValueError:
            print_error(f"Invalid model type: {model_type}")
            return
    
    # Get models
    models = config.list_available_models(
        provider=provider_enum,
        model_type=model_type_enum
    )
    
    if local_only:
        models = [m for m in models if m.local]
    
    if not models:
        print_warning("No matching models found.")
        if provider:
            print(f"Try installing models for {provider} or check another provider.")
        return
    
    # Group by provider
    models_by_provider = {}
    for model in models:
        if model.provider not in models_by_provider:
            models_by_provider[model.provider] = []
        models_by_provider[model.provider].append(model)
    
    # Print models grouped by provider
    for provider, provider_models in models_by_provider.items():
        print_section(f"{provider.value.title()} Models:")
        
        # Group by model type
        models_by_type = {}
        for model in provider_models:
            if model.model_type not in models_by_type:
                models_by_type[model.model_type] = []
            models_by_type[model.model_type].append(model)
        
        # Print by type
        for model_type, type_models in models_by_type.items():
            print(f"\n  {Colors.CYAN}{model_type.value.title()} Models:{Colors.ENDC}")
            
            for model in type_models:
                # Format costs if available
                cost_str = ""
                if model.cost_per_1k_tokens:
                    cost_str = f", ${model.cost_per_1k_tokens}/1K tokens"
                
                # Format size if available
                size_str = ""
                if model.size_mb:
                    size_str = f", {model.size_mb}MB"
                
                # Format performance rating if available
                perf_str = ""
                if model.performance_rating:
                    perf_str = f", Rating: {model.performance_rating}/10"
                
                # Add local indicator
                local_str = f" {Colors.GREEN}[LOCAL]{Colors.ENDC}" if model.local else ""
                
                print(f"    - {Colors.BOLD}{model.name}{Colors.ENDC}{local_str}")
                if model.description:
                    print(f"      {model.description}{size_str}{cost_str}{perf_str}")
                print()

def show_current_config(config: ModelConfiguration):
    """Display the current model configuration."""
    print_header("Current Model Configuration")
    
    # General settings
    print_section("General Settings:")
    print(f"  Prefer Local Models: {Colors.BOLD}{config.prefer_local}{Colors.ENDC}")
    print(f"  Config File: {config.config_path}")
    
    # Embedding model
    print_section("Embedding Model:")
    embed_model, embed_provider = config.get_embedding_model()
    print(f"  Model: {Colors.BOLD}{embed_model}{Colors.ENDC}")
    print(f"  Provider: {embed_provider.value}")
    
    # Task-specific models
    print_section("Task-Specific Models:")
    for task in AnalysisTask:
        model_name, provider = config.get_model_for_task(task)
        print(f"  {task.value}: {Colors.BOLD}{model_name}{Colors.ENDC} ({provider.value})")
    
    # Report configuration
    print_section("Report Configuration:")
    report_config = config.config.get("report_configuration", {})
    report_type = report_config.get("type", "standard")
    print(f"  Report Type: {Colors.BOLD}{report_type}{Colors.ENDC}")
    
    multi_model = report_config.get("multi_model_enabled", False)
    print(f"  Multi-Model Enabled: {Colors.BOLD}{multi_model}{Colors.ENDC}")
    
    if multi_model:
        models = report_config.get("models_to_compare", [])
        print(f"  Models to Compare: {Colors.BOLD}{', '.join(models)}{Colors.ENDC}")
        
        consolidated = report_config.get("consolidated_model", "")
        if consolidated:
            print(f"  Consolidated Model: {Colors.BOLD}{consolidated}{Colors.ENDC}")

def set_embedding_model(config: ModelConfiguration, model_name: str, provider: str):
    """Set the embedding model to use."""
    try:
        provider_enum = ModelProvider(provider.lower())
        config.set_embedding_model(model_name, provider_enum)
        print_success(f"Embedding model set to {model_name} ({provider})")
    except ValueError:
        print_error(f"Invalid provider: {provider}")
        print(f"Available providers: {', '.join([p.value for p in ModelProvider])}")

def set_task_model(config: ModelConfiguration, task: str, model_name: str, provider: str):
    """Set the model to use for a specific task."""
    try:
        provider_enum = ModelProvider(provider.lower())
        # Check if task is a valid enum value
        try:
            task_enum = AnalysisTask(task)
            config.set_model_for_task(task_enum, model_name, provider_enum)
            print_success(f"Model for {task} set to {model_name} ({provider})")
        except ValueError:
            print_error(f"Invalid task: {task}")
            print(f"Available tasks: {', '.join([t.value for t in AnalysisTask])}")
    except ValueError:
        print_error(f"Invalid provider: {provider}")
        print(f"Available providers: {', '.join([p.value for p in ModelProvider])}")

def set_report_type(config: ModelConfiguration, report_type: str):
    """Set the report generation type."""
    try:
        report_type_enum = ReportType(report_type)
        config.set_report_type(report_type_enum)
        print_success(f"Report type set to {report_type}")
    except ValueError:
        print_error(f"Invalid report type: {report_type}")
        print(f"Available report types: {', '.join([t.value for t in ReportType])}")

def set_multi_model(config: ModelConfiguration, enabled: bool, models: Optional[List[str]] = None):
    """Enable or disable multi-model reports."""
    config.enable_multi_model_reports(enabled, models)
    if enabled:
        models_str = ", ".join(models) if models else "default models"
        print_success(f"Multi-model reports enabled with models: {models_str}")
    else:
        print_success("Multi-model reports disabled")

def set_prefer_local(config: ModelConfiguration, prefer_local: bool):
    """Set whether to prefer local models."""
    config.set_prefer_local(prefer_local)
    print_success(f"Prefer local models set to {prefer_local}")

def main():
    parser = argparse.ArgumentParser(description="Configure models for JFKReveal")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Show command
    show_parser = subparsers.add_parser("show", help="Show current configuration")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List available models")
    list_parser.add_argument("--provider", choices=["openai", "ollama", "huggingface"],
                             help="Filter by provider")
    list_parser.add_argument("--type", choices=["embedding", "chat", "completion"],
                             help="Filter by model type")
    list_parser.add_argument("--local-only", action="store_true",
                             help="Only show models that can run locally")
    
    # Set embedding model command
    embed_parser = subparsers.add_parser("set-embedding", help="Set embedding model")
    embed_parser.add_argument("model", help="Model name")
    embed_parser.add_argument("provider", choices=["openai", "ollama", "huggingface"],
                             help="Model provider")
    
    # Set task model command
    task_parser = subparsers.add_parser("set-task", help="Set model for specific task")
    task_parser.add_argument("task", choices=[t.value for t in AnalysisTask],
                             help="Task to configure")
    task_parser.add_argument("model", help="Model name")
    task_parser.add_argument("provider", choices=["openai", "ollama", "huggingface"],
                             help="Model provider")
    
    # Set report type command
    report_parser = subparsers.add_parser("set-report-type", help="Set report type")
    report_parser.add_argument("type", choices=[t.value for t in ReportType],
                               help="Report type")
    
    # Set multi-model command
    multi_parser = subparsers.add_parser("set-multi-model", help="Configure multi-model reports")
    multi_parser.add_argument("enabled", choices=["true", "false"],
                              help="Enable or disable multi-model reports")
    multi_parser.add_argument("--models", nargs="+",
                              help="Models to compare in multi-model reports")
    
    # Set prefer local command
    local_parser = subparsers.add_parser("set-prefer-local", help="Set whether to prefer local models")
    local_parser.add_argument("enabled", choices=["true", "false"],
                              help="Enable or disable preferring local models")
    
    # Reset command
    reset_parser = subparsers.add_parser("reset", help="Reset configuration to defaults")
    
    args = parser.parse_args()
    
    # Initialize configuration
    config = ModelConfiguration()
    
    if args.command == "show":
        show_current_config(config)
    
    elif args.command == "list":
        list_available_models(config, args.provider, args.type, args.local_only)
    
    elif args.command == "set-embedding":
        set_embedding_model(config, args.model, args.provider)
    
    elif args.command == "set-task":
        set_task_model(config, args.task, args.model, args.provider)
    
    elif args.command == "set-report-type":
        set_report_type(config, args.type)
    
    elif args.command == "set-multi-model":
        enabled = args.enabled.lower() == "true"
        set_multi_model(config, enabled, args.models)
    
    elif args.command == "set-prefer-local":
        enabled = args.enabled.lower() == "true"
        set_prefer_local(config, enabled)
    
    elif args.command == "reset":
        # Delete existing config file
        if os.path.exists(config.config_path):
            os.remove(config.config_path)
            # Create new config with defaults
            new_config = ModelConfiguration()
            new_config.save_config()
            print_success("Configuration reset to defaults")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()