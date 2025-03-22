# JFKReveal Model Configuration

This directory contains configuration files for JFKReveal's multi-model system.

## Models Configuration

The `models.json` file controls which AI models are used for various tasks in the JFKReveal pipeline. You can edit this file to customize which models are used.

### Configuration Options

- `prefer_local`: Set to `true` to prefer local Ollama models over cloud-based models (recommended for development)
- `models`: Configuration for different tasks:
  - `embedding`: Models used for vector embeddings
  - `document_analysis`: Models used for analyzing document content  
  - `topic_analysis`: Models used for topic-based analysis
  - `report_generation`: Models used for generating reports
  - `executive_summary`: Models used specifically for the executive summary
  - `suspects_analysis`: Models used for analyzing potential suspects
  - `coverup_analysis`: Models used for analyzing potential coverups
- `report_configuration`: Controls report generation behavior
  - `type`: Report type (`standard`, `multi_model_comparison`, or `consolidated`)
  - `multi_model_enabled`: Whether to enable multi-model reporting
  - `models_to_compare`: List of models to use in multi-model comparisons
  - `consolidated_model`: Model to use for consolidating multi-model reports

### Using Different Models

For each task, you can configure:

```json
"task_name": {
  "default": "model_name",
  "provider": "provider_name",
  "fallback": "fallback_model"
}
```

Where:
- `default`: The primary model to use
- `provider`: The provider (`openai`, `ollama`, or `huggingface`)
- `fallback`: Model to use if the primary model is unavailable

## Model Providers

JFKReveal supports multiple model providers:

1. **Ollama** (Local Models):
   - Fast and private, runs completely on your machine
   - No API costs, but requires more resources
   - Supported models: `llama3`, `mistral`, `gemma`, etc.
   - Embedding models: `nomic-embed-text`, `mxbai-embed-large`, `all-minilm`

2. **OpenAI** (Cloud Models):
   - Higher quality but requires API key and has usage costs
   - Requires internet connection
   - Models: `gpt-4o`, `gpt-4o-mini`, `gpt-3.5-turbo`
   - Embedding models: `text-embedding-3-large`, `text-embedding-3-small`

3. **HuggingFace** (Local or Cloud):
   - Open-source models hosted locally or in the cloud
   - Varying quality depending on the model
   - Requires appropriate dependencies installed

## Command Line Options

You can override the configuration using command line options:

```bash
# Use local models (Ollama)
python -m jfkreveal --prefer-local

# Specify embedding model
python -m jfkreveal --embedding-model nomic-embed-text --embedding-provider ollama

# Specify analysis model  
python -m jfkreveal --analysis-model llama3 --analysis-provider ollama

# Generate multi-model report
python -m jfkreveal --report-type multi_model_comparison --multi-model

# Specify models to compare
python -m jfkreveal --multi-model --models-to-compare gpt-4o llama3 mistral

# Download required Ollama models before running
python -m jfkreveal --download-models
```

## Performance vs. Cost Considerations

- **Development/Testing**: Prefer Ollama models to minimize costs
- **Final Reports**: Consider using OpenAI models for higher quality
- **Multi-Model Reports**: Compare results from multiple models to increase reliability
- **Embeddings**: Local embeddings work well for most use cases

## Example Configurations

### Fast Local Development

```json
{
  "prefer_local": true,
  "models": {
    "embedding": {
      "default": "all-minilm",
      "provider": "ollama"
    }
  }
}
```

### Balanced Approach

```json
{
  "prefer_local": true,
  "models": {
    "embedding": {
      "default": "nomic-embed-text",
      "provider": "ollama"
    },
    "report_generation": {
      "default": "gpt-3.5-turbo",
      "provider": "openai"
    }
  }
}
```

### High-Quality Final Report

```json
{
  "prefer_local": false,
  "models": {
    "embedding": {
      "default": "text-embedding-3-small",
      "provider": "openai"
    },
    "report_generation": {
      "default": "gpt-4o",
      "provider": "openai"
    }
  },
  "report_configuration": {
    "type": "standard"
  }
}
```