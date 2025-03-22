# JFKReveal Configuration Guide

This document provides detailed information about how to configure JFKReveal for different use cases and environments.

## Table of Contents
- [Environment Setup](#environment-setup)
- [Configuration Options](#configuration-options)
- [API Key Management](#api-key-management)
- [Performance Tuning](#performance-tuning)
- [OCR Configuration](#ocr-configuration)
- [Analysis Options](#analysis-options)
- [Report Generation](#report-generation)
- [Visualization Settings](#visualization-settings)
- [Advanced Configuration](#advanced-configuration)

## Environment Setup

### Prerequisites

Before running JFKReveal, ensure you have:

1. **Python 3.8+** installed
2. **Tesseract OCR** installed for text extraction from image-based documents:
   - On macOS: `brew install tesseract`
   - On Ubuntu: `apt-get install tesseract-ocr`
   - On Windows: Download installer from https://github.com/UB-Mannheim/tesseract/wiki
3. **OpenAI API key** for analysis functionality
4. Optional: **Ollama** for local embeddings (https://ollama.ai/)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/PimpMyNines/JFKReveal.git
   cd JFKReveal
   ```

2. Create and activate a virtual environment:
   ```bash
   make setup
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the package in development mode:
   ```bash
   make install-dev
   ```

4. Create an environment variables file:
   ```bash
   cp .env.example .env
   ```

5. Edit the `.env` file to add your API keys and configuration options.

## Configuration Options

JFKReveal can be configured via:
1. Environment variables (in `.env` file)
2. Command-line arguments
3. Configuration within code

The precedence is: Command Line > Environment Variables > Code Defaults

### Core Environment Variables

| Variable | Description | Default | Example |
|----------|-------------|---------|---------|
| `OPENAI_API_KEY` | OpenAI API key | None | `sk-abcd1234...` |
| `OPENAI_EMBEDDING_MODEL` | Model for embeddings | `text-embedding-3-large` | `text-embedding-3-small` |
| `OPENAI_ANALYSIS_MODEL` | Model for analysis | `gpt-4.5-preview` | `gpt-4o` |
| `EMBEDDING_PROVIDER` | Provider for embeddings | `openai` | `ollama` |
| `OLLAMA_EMBEDDING_MODEL` | Ollama model for embeddings | `nomic-embed-text` | `llama3` |
| `OLLAMA_BASE_URL` | Ollama base URL | `http://localhost:11434` | `http://192.168.1.100:11434` |
| `GITHUB_API_KEY` | GitHub API key for repo operations | None | `ghp_abcd1234...` |
| `LOG_LEVEL` | Logging level | `INFO` | `DEBUG` |
| `DATA_DIR` | Data directory | `./data` | `/path/to/data` |

### Example .env File

```
# API Keys
OPENAI_API_KEY=sk-your-api-key-here
GITHUB_API_KEY=ghp-your-github-token-here

# Model Configuration
OPENAI_EMBEDDING_MODEL=text-embedding-3-large
OPENAI_ANALYSIS_MODEL=gpt-4o
EMBEDDING_PROVIDER=openai

# Ollama Configuration (if using)
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
OLLAMA_BASE_URL=http://localhost:11434

# Path Configuration
DATA_DIR=./data

# Logging
LOG_LEVEL=INFO
```

## API Key Management

JFKReveal implements a robust credential management system with multiple sources and fallbacks.

### Credential Sources

Credentials are loaded from the following sources in order:
1. Memory cache (for runtime provided credentials)
2. Environment variables
3. Files in the credentials directory

### API Key Rotation

For handling rate limits, JFKReveal can automatically rotate between multiple API keys:

1. Create a directory for API keys:
   ```bash
   mkdir -p ~/.jfkreveal/credentials
   ```

2. Add multiple OpenAI API keys, one per file:
   ```bash
   echo "sk-key1..." > ~/.jfkreveal/credentials/openai_key1.txt
   echo "sk-key2..." > ~/.jfkreveal/credentials/openai_key2.txt
   ```

3. Enable key rotation in your `.env` file:
   ```
   USE_KEY_ROTATION=true
   CREDENTIALS_DIR=~/.jfkreveal/credentials
   ```

### Multiple API Providers

JFKReveal supports different API providers. Configure in `.env`:

```
# Use Azure OpenAI
API_PROVIDER=azure
AZURE_OPENAI_KEY=your-azure-key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com

# Use Anthropic
API_PROVIDER=anthropic
ANTHROPIC_API_KEY=your-anthropic-key

# Use OpenAI (default)
API_PROVIDER=openai
OPENAI_API_KEY=your-openai-key
```

## Performance Tuning

### Parallel Processing

Control parallel processing behavior for document processing:

```bash
# In .env file
MAX_WORKERS=4          # Number of parallel workers
USE_THREADS=false      # Use processes (false) or threads (true)
CHUNK_SIZE=5           # Batch size for parallel processing
```

Or set at runtime:
```bash
jfkreveal run-analysis --max-workers 8
```

### Memory Management

For systems with limited memory:

```bash
# In .env file
LOW_MEMORY_MODE=true   # Enables memory-efficient processing at cost of speed
MAX_DOCUMENT_CACHE=50  # Maximum documents to keep in memory
```

### Chunking Parameters

Document chunking affects search quality and processing efficiency:

```bash
# In .env file
CHUNK_SIZE=1000        # Target size of each document chunk
CHUNK_OVERLAP=200      # Overlap between chunks to preserve context
```

## OCR Configuration

JFKReveal includes comprehensive OCR configuration options to handle different document qualities.

### Basic OCR Settings

```bash
# In .env file
USE_OCR=true           # Enable OCR for image-based PDFs
OCR_RESOLUTION=2.0     # Resolution multiplier (higher = better quality but slower)
OCR_LANGUAGE=eng       # OCR language (use + for multiple, e.g., 'eng+spa')
```

Or set at runtime:
```bash
jfkreveal run-analysis --ocr-resolution 3.0 --ocr-language eng
```

### OCR Performance Tradeoffs

OCR resolution directly impacts quality and processing time:

| Resolution | Quality | Speed | Memory Usage | Recommended Use |
|------------|---------|-------|--------------|-----------------|
| 1.0 | Low | Fast | Low | High-quality scans, modern documents |
| 2.0 | Medium | Medium | Medium | Good general setting (default) |
| 3.0 | High | Slow | High | Poor quality scans, historical documents |
| 4.0+ | Very High | Very Slow | Very High | Critical documents requiring maximum accuracy |

For more details, see [OCR Performance Documentation](OCR_PERFORMANCE.md).

### Advanced OCR Options

For specialized use cases, you can configure additional parameters in code:

```python
from jfkreveal.database.document_processor import DocumentProcessor

processor = DocumentProcessor(
    use_ocr=True,
    ocr_resolution=2.0,
    ocr_language="eng",
    ocr_psm=6,             # Page segmentation mode
    ocr_oem=3,             # OCR Engine mode
    ocr_timeout=30,        # Timeout in seconds
    ocr_config="--dpi 300" # Additional Tesseract configuration
)
```

## Analysis Options

### LLM Configuration

Configure the language models used for analysis:

```bash
# In .env file
OPENAI_ANALYSIS_MODEL=gpt-4o        # Primary model for analysis
OPENAI_FALLBACK_MODEL=gpt-3.5-turbo # Fallback model if primary unavailable
MODEL_TEMPERATURE=0.0               # Temperature for generation (0.0 = deterministic)
MAX_TOKENS=4000                     # Maximum tokens for LLM responses
```

### Analysis Depth

Control the scope and depth of analysis:

```bash
# In .env file
ANALYSIS_TOPICS=oswald,conspiracy,timeline,bullet,zapruder  # Topics to analyze
ANALYSIS_DEPTH=comprehensive  # basic, standard, or comprehensive
CONFIDENCE_THRESHOLD=0.7      # Minimum confidence for included findings
ENABLE_CROSS_REFERENCES=true  # Enable cross-document reference analysis
```

Or set at runtime:
```bash
jfkreveal run-analysis --analysis-depth comprehensive
```

### Entity Analysis

Configure entity extraction and relationship mapping:

```bash
# In .env file
EXTRACT_ENTITIES=true             # Extract named entities
ENTITY_TYPES=person,org,location  # Entity types to extract
RELATIONSHIP_ANALYSIS=true        # Analyze entity relationships
MIN_ENTITY_MENTIONS=2             # Minimum mentions to include entity
```

## Report Generation

### Report Types

Control which reports are generated:

```bash
# In .env file
GENERATE_EXECUTIVE_SUMMARY=true    # Generate executive summary
GENERATE_DETAILED_FINDINGS=true    # Generate detailed findings
GENERATE_SUSPECTS_ANALYSIS=true    # Generate suspects analysis
GENERATE_COVERUP_ANALYSIS=true     # Generate coverup analysis
GENERATE_FULL_REPORT=true          # Generate comprehensive report
```

Or set at runtime:
```bash
jfkreveal generate-report --executive-only
```

### Report Format

Configure report format and output:

```bash
# In .env file
REPORT_FORMAT=html                # 'html', 'md', or 'both'
INCLUDE_VISUALIZATIONS=true       # Include visualizations in HTML reports
OUTPUT_DIRECTORY=./data/reports   # Output directory for reports
REPORT_STYLE=detective            # 'academic', 'detective', or 'journalistic'
```

## Visualization Settings

### Dashboard Configuration

Configure the visualization dashboard:

```bash
# In .env file
DASHBOARD_HOST=localhost      # Dashboard host
DASHBOARD_PORT=8050           # Dashboard port
DASHBOARD_DEBUG=false         # Run dashboard in debug mode
DASHBOARD_THEME=light         # 'light' or 'dark'
```

Or set at runtime:
```bash
jfkreveal view-dashboard --host 0.0.0.0 --port 8080
```

### Visualization Features

Enable/disable specific visualizations:

```bash
# In .env file
ENABLE_ENTITY_NETWORK=true    # Enable entity network visualization
ENABLE_TIMELINE=true          # Enable timeline visualization
ENABLE_EVIDENCE_MAP=true      # Enable evidence map visualization
ENABLE_DOCUMENT_VIEWER=true   # Enable document viewer
MAX_GRAPH_ENTITIES=100        # Maximum entities in network graph
```

## Advanced Configuration

### Logging Configuration

Configure logging behavior:

```bash
# In .env file
LOG_LEVEL=INFO                # DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_FILE=./logs/jfkreveal.log # Log file path
CONSOLE_LOGGING=true          # Enable console logging
LOG_FORMAT=detailed           # 'simple' or 'detailed'
```

Or set at runtime:
```bash
jfkreveal run-analysis --log-level DEBUG --log-file ./custom_log.log
```

### Caching Configuration

Configure caching behavior:

```bash
# In .env file
ENABLE_CACHE=true             # Enable caching
CACHE_DIR=./data/cache        # Cache directory
CACHE_TTL=86400               # Cache time-to-live in seconds (1 day)
CACHE_EMBEDDINGS=true         # Cache embeddings
CACHE_ANALYSIS=true           # Cache analysis results
```

### Search Configuration

Configure search behavior:

```bash
# In .env file
SEARCH_TYPE=hybrid            # 'vector', 'bm25', or 'hybrid'
VECTOR_WEIGHT=0.7             # Weight for vector search in hybrid search
BM25_WEIGHT=0.3               # Weight for BM25 in hybrid search
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2  # Reranker model
TOP_K=30                      # Number of initial results to rerank
RERANK_TOP_K=10               # Number of results after reranking
```

Or set at runtime:
```bash
jfkreveal search "oswald connections" --search-type hybrid --vector-weight 0.8 --bm25-weight 0.2
```

### Dependency Injection Configuration

JFKReveal uses a lightweight dependency injection container. Configure custom implementations:

```python
from jfkreveal.main import JFKReveal
from jfkreveal.database.custom_vector_store import CustomVectorStore
from jfkreveal.utils.container import DIContainer

# Create container
container = DIContainer()

# Register custom implementation
container.register("vector_store", CustomVectorStore)

# Initialize with custom container
jfk = JFKReveal(container=container)
```

### Webhook Integration

Configure webhooks for pipeline events:

```bash
# In .env file
ENABLE_WEBHOOKS=true           # Enable webhook notifications
WEBHOOK_URL=https://example.com/webhook  # Webhook URL
NOTIFY_ON_COMPLETION=true      # Notify on pipeline completion
NOTIFY_ON_ERROR=true           # Notify on errors
WEBHOOK_SECRET=your-secret-key # Secret for webhook signature
```