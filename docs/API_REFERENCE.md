# JFKReveal API Reference

This document provides detailed information about the JFKReveal API, including module descriptions, class interfaces, and usage examples.

## Table of Contents
- [Core Modules](#core-modules)
- [Scrapers](#scrapers)
- [Database](#database)
- [Analysis](#analysis)
- [Search](#search)
- [Summarization](#summarization)
- [Visualization](#visualization)
- [Utilities](#utilities)
- [Command Line Interface](#command-line-interface)

## Core Modules

### JFKReveal Class

The main class that orchestrates the entire analysis pipeline.

```python
from jfkreveal.main import JFKReveal

# Initialize the JFKReveal pipeline
jfk = JFKReveal(
    data_dir="./data",
    skip_scraping=False,
    skip_processing=False,
    skip_analysis=False
)

# Run the full pipeline
jfk.run()

# Access results
analysis_results = jfk.get_analysis_results()
```

#### Constructor Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `data_dir` | str | Directory for storing data | `"./data"` |
| `skip_scraping` | bool | Skip document scraping step | `False` |
| `skip_processing` | bool | Skip document processing step | `False` |
| `skip_analysis` | bool | Skip document analysis step | `False` |
| `no_clean_text` | bool | Disable text cleaning | `False` |
| `no_ocr` | bool | Disable OCR for image-based PDFs | `False` |
| `ocr_resolution` | float | Resolution multiplier for OCR | `2.0` |
| `ocr_language` | str | Language for OCR | `"eng"` |

#### Methods

| Method | Parameters | Return Type | Description |
|--------|------------|-------------|-------------|
| `run()` | None | None | Run the complete pipeline |
| `scrape_documents()` | None | List[str] | Scrape documents from the archives |
| `process_documents()` | List[str] | List[Dict] | Process documents and extract text |
| `analyze_documents()` | List[Dict] | Dict | Analyze document content |
| `generate_reports()` | Dict | None | Generate analysis reports |
| `get_analysis_results()` | None | Dict | Get the results of the analysis |

## Scrapers

### ArchivesGovScraper

Downloads PDF documents from the National Archives website.

```python
from jfkreveal.scrapers.archives_gov import ArchivesGovScraper

# Initialize the scraper
scraper = ArchivesGovScraper(
    output_dir="./data/pdfs",
    max_documents=100
)

# Download documents
pdf_paths = scraper.download_documents()
```

#### Constructor Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `output_dir` | str | Directory for storing PDFs | `"./data/pdfs"` |
| `max_documents` | int | Maximum number of documents to download | `None` |
| `timeout` | int | Request timeout in seconds | `30` |
| `max_retries` | int | Maximum number of retries | `5` |

#### Methods

| Method | Parameters | Return Type | Description |
|--------|------------|-------------|-------------|
| `download_documents()` | None | List[str] | Download documents and return paths |
| `get_document_urls()` | None | List[str] | Get URLs of available documents |
| `download_document()` | str, str | str | Download a specific document |
| `cleanup_temp_files()` | None | None | Clean up temporary files |

## Database

### DocumentProcessor

Processes PDF documents to extract text.

```python
from jfkreveal.database.document_processor import DocumentProcessor

# Initialize the processor
processor = DocumentProcessor(
    output_dir="./data/processed",
    use_ocr=True,
    ocr_resolution=2.0
)

# Process a document
result = processor.process_document("path/to/document.pdf")

# Process multiple documents in parallel
results = processor.process_documents(["doc1.pdf", "doc2.pdf"])
```

#### Constructor Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `output_dir` | str | Directory for processed outputs | `"./data/processed"` |
| `use_ocr` | bool | Enable OCR for image-based PDFs | `True` |
| `ocr_resolution` | float | Resolution multiplier for OCR | `2.0` |
| `ocr_language` | str | Language for OCR | `"eng"` |
| `max_workers` | int | Maximum worker processes | `None` |

#### Methods

| Method | Parameters | Return Type | Description |
|--------|------------|-------------|-------------|
| `process_document()` | str | Dict | Process a single document |
| `process_documents()` | List[str] | List[Dict] | Process multiple documents |
| `extract_text_from_pdf()` | str | Dict | Extract text from a PDF file |
| `apply_ocr_to_page()` | Image | str | Apply OCR to an image |
| `chunk_document()` | Dict | List[Dict] | Split document into chunks |

### TextCleaner

Cleans and normalizes text from OCR and PDF extraction.

```python
from jfkreveal.database.text_cleaner import TextCleaner

# Initialize the cleaner
cleaner = TextCleaner()

# Clean text
cleaned_text = cleaner.clean_text("Text with OCR  artifacts")
```

#### Methods

| Method | Parameters | Return Type | Description |
|--------|------------|-------------|-------------|
| `clean_text()` | str | str | Clean and normalize text |
| `fix_line_breaks()` | str | str | Fix improper line breaks |
| `fix_spacing()` | str | str | Normalize spacing |
| `fix_typewriter_artifacts()` | str | str | Fix typewriter artifacts |
| `normalize_terms()` | str | str | Normalize JFK-specific terms |
| `fix_character_errors()` | str | str | Fix common OCR character errors |

### VectorStore

Manages embeddings and provides semantic search capabilities.

```python
from jfkreveal.database.vector_store import VectorStore

# Initialize the vector store
vector_store = VectorStore(
    collection_name="jfk_documents",
    embedding_provider="openai"
)

# Add documents
vector_store.add_documents(document_chunks)

# Perform semantic search
results = vector_store.semantic_search("Lee Harvey Oswald CIA connections", k=5)
```

#### Constructor Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `collection_name` | str | Name of the collection | `"jfk_documents"` |
| `embedding_provider` | str | Provider for embeddings | `"openai"` |
| `persist_directory` | str | Directory for persistence | `"./data/vectordb"` |
| `embedding_model` | str | Model for embeddings | `None` |

#### Methods

| Method | Parameters | Return Type | Description |
|--------|------------|-------------|-------------|
| `add_documents()` | List[Dict] | None | Add documents to the vector store |
| `semantic_search()` | str, int | List[Dict] | Perform semantic search |
| `hybrid_search()` | str, int | List[Dict] | Perform hybrid search (vector + BM25) |
| `get_collection()` | None | Collection | Get the underlying collection |
| `count_documents()` | None | int | Count documents in the store |

## Analysis

### DocumentAnalyzer

Analyzes document content using LLMs.

```python
from jfkreveal.analysis.document_analyzer import DocumentAnalyzer

# Initialize the analyzer
analyzer = DocumentAnalyzer(
    model_name="gpt-4o"
)

# Analyze documents
results = analyzer.analyze_documents(document_chunks)

# Analyze specific topic
topic_analysis = analyzer.analyze_topic("Lee Harvey Oswald", relevant_chunks)
```

#### Constructor Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `model_name` | str | Name of the LLM to use | `"gpt-4o"` |
| `api_key` | str | OpenAI API key | `None` |
| `temperature` | float | Temperature for LLM | `0.0` |

#### Methods

| Method | Parameters | Return Type | Description |
|--------|------------|-------------|-------------|
| `analyze_documents()` | List[Dict] | Dict | Analyze document collection |
| `analyze_topic()` | str, List[Dict] | Dict | Analyze a specific topic |
| `generate_topic_summary()` | str, List[Dict] | Dict | Generate summary of a topic |
| `extract_entities()` | List[Dict] | Dict | Extract entities from documents |
| `analyze_entity_relationships()` | Dict | Dict | Analyze relationships between entities |

### EnhancedAnalyzer

Extended analyzer with additional capabilities.

```python
from jfkreveal.analysis.enhanced_analyzer import EnhancedAnalyzer

# Initialize the enhanced analyzer
analyzer = EnhancedAnalyzer()

# Analyze conspiracy theories
conspiracy_analysis = analyzer.analyze_conspiracy_theories(documents)

# Analyze inconsistencies
inconsistencies = analyzer.analyze_inconsistencies(documents)
```

#### Methods

| Method | Parameters | Return Type | Description |
|--------|------------|-------------|-------------|
| `analyze_conspiracy_theories()` | List[Dict] | Dict | Analyze conspiracy theories |
| `analyze_inconsistencies()` | List[Dict] | Dict | Analyze inconsistencies in evidence |
| `analyze_timeline()` | List[Dict] | Dict | Analyze event timeline |
| `analyze_evidence_credibility()` | List[Dict] | Dict | Analyze evidence credibility |

## Search

### SemanticSearch

Provides advanced search capabilities across document collection.

```python
from jfkreveal.search.semantic_search import SemanticSearch

# Initialize the search engine
search = SemanticSearch(
    vector_store_path="./data/vectordb",
    use_hybrid_search=True
)

# Search for documents
results = search.search("Oswald's connections to intelligence agencies", k=10)

# Get document by ID
document = search.get_document_by_id("doc123")
```

#### Constructor Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `vector_store_path` | str | Path to vector store | `"./data/vectordb"` |
| `use_hybrid_search` | bool | Use hybrid search (vector + BM25) | `True` |
| `reranker_model` | str | Model for reranking | `None` |

#### Methods

| Method | Parameters | Return Type | Description |
|--------|------------|-------------|-------------|
| `search()` | str, int | List[Dict] | Search for documents |
| `hybrid_search()` | str, int | List[Dict] | Perform hybrid search |
| `vector_search()` | str, int | List[Dict] | Perform vector-only search |
| `bm25_search()` | str, int | List[Dict] | Perform BM25 search |
| `get_document_by_id()` | str | Dict | Get document by ID |
| `get_similar_documents()` | str, int | List[Dict] | Get similar documents |

## Summarization

### FindingsReport

Generates comprehensive reports from analysis results.

```python
from jfkreveal.summarization.findings_report import FindingsReport

# Initialize the report generator
report_generator = FindingsReport(
    output_dir="./data/reports",
    model_name="gpt-4o"
)

# Generate executive summary
report_generator.generate_executive_summary(analysis_results)

# Generate detailed findings
report_generator.generate_detailed_findings(analysis_results)

# Generate full report
report_generator.generate_full_report(analysis_results)
```

#### Constructor Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `output_dir` | str | Directory for reports | `"./data/reports"` |
| `model_name` | str | Name of the LLM to use | `"gpt-4o"` |
| `api_key` | str | OpenAI API key | `None` |

#### Methods

| Method | Parameters | Return Type | Description |
|--------|------------|-------------|-------------|
| `generate_executive_summary()` | Dict | str | Generate executive summary |
| `generate_detailed_findings()` | Dict | str | Generate detailed findings |
| `generate_full_report()` | Dict | str | Generate full comprehensive report |
| `generate_suspects_analysis()` | Dict | str | Generate suspects analysis |
| `generate_coverup_analysis()` | Dict | str | Generate coverup analysis |
| `save_report()` | str, str, str | None | Save report to file |

## Visualization

### Dashboard

Interactive dashboard for exploring analysis results.

```python
from jfkreveal.visualization.dashboard import Dashboard

# Initialize the dashboard
dashboard = Dashboard(
    data_dir="./data/analysis",
    host="localhost",
    port=8050
)

# Start the dashboard
dashboard.run()
```

#### Constructor Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `data_dir` | str | Directory with analysis data | `"./data/analysis"` |
| `host` | str | Host address | `"localhost"` |
| `port` | int | Port number | `8050` |

#### Methods

| Method | Parameters | Return Type | Description |
|--------|------------|-------------|-------------|
| `run()` | None | None | Start the dashboard |
| `load_data()` | None | Dict | Load analysis data |
| `create_entity_network()` | Dict | Figure | Create entity network visualization |
| `create_timeline()` | Dict | Figure | Create event timeline visualization |
| `create_evidence_map()` | Dict | Figure | Create evidence map visualization |
| `shutdown()` | None | None | Shutdown the dashboard |

## Utilities

### ParallelProcessor

Provides parallel processing capabilities.

```python
from jfkreveal.utils.parallel_processor import ParallelProcessor

# Initialize the processor
processor = ParallelProcessor(max_workers=4)

# Process items in parallel
results = processor.process(items, process_function)
```

#### Constructor Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `max_workers` | int | Maximum number of workers | `None` |
| `use_threads` | bool | Use threads instead of processes | `False` |

#### Methods

| Method | Parameters | Return Type | Description |
|--------|------------|-------------|-------------|
| `process()` | List, callable | List | Process items in parallel |
| `map()` | callable, List | List | Map function over items in parallel |
| `shutdown()` | None | None | Shutdown the processor |

### FileUtils

Utility functions for file operations.

```python
from jfkreveal.utils.file_utils import FileUtils

# Create directory
FileUtils.ensure_dir_exists("./data/output")

# Save JSON data
FileUtils.save_json("./data/results.json", data)

# Load JSON data
data = FileUtils.load_json("./data/results.json")
```

#### Static Methods

| Method | Parameters | Return Type | Description |
|--------|------------|-------------|-------------|
| `ensure_dir_exists()` | str | bool | Ensure directory exists |
| `save_json()` | str, Dict | None | Save data to JSON file |
| `load_json()` | str | Dict | Load data from JSON file |
| `save_pickle()` | str, object | None | Save object to pickle file |
| `load_pickle()` | str | object | Load object from pickle file |
| `get_file_size()` | str | int | Get file size in bytes |
| `get_file_modified_time()` | str | float | Get file modification time |
| `is_file_older_than()` | str, int | bool | Check if file is older than specified seconds |

### Logger

Centralized logging utilities.

```python
from jfkreveal.utils.logger import Logger

# Initialize logger
logger = Logger.get_logger("my_module")

# Log messages
logger.debug("Debug message")
logger.info("Info message")
logger.warning("Warning message")
logger.error("Error message")

# Performance logging decorator
from jfkreveal.utils.logger import log_execution_time

@log_execution_time
def my_function():
    # Function will be performance logged
    pass
```

#### Static Methods

| Method | Parameters | Return Type | Description |
|--------|------------|-------------|-------------|
| `get_logger()` | str | Logger | Get logger for module |
| `set_log_level()` | str, str | None | Set log level for module |
| `set_console_logging()` | bool | None | Enable/disable console logging |
| `set_file_logging()` | str | None | Set file logging path |

#### Decorators

| Decorator | Description |
|-----------|-------------|
| `@log_execution_time` | Log execution time of function |
| `@log_function_calls` | Log function calls with arguments |

## Command Line Interface

JFKReveal provides a comprehensive command-line interface with subcommands.

### Basic Usage

```bash
# Run the full pipeline
jfkreveal run-analysis

# Search for specific terms
jfkreveal search "Oswald CIA connections"

# Generate only reports
jfkreveal generate-report --data-dir ./data/analysis

# Launch visualization dashboard
jfkreveal view-dashboard --port 8080
```

### Available Commands

| Command | Description | Examples |
|---------|-------------|----------|
| `run-analysis` | Run full analysis pipeline | `jfkreveal run-analysis --skip-scraping` |
| `search` | Search document collection | `jfkreveal search "bullet trajectory" --format json` |
| `generate-report` | Generate reports from analysis | `jfkreveal generate-report --executive-only` |
| `view-dashboard` | Launch visualization dashboard | `jfkreveal view-dashboard --port 8080` |
| `version` | Display version information | `jfkreveal version` |
| `examples` | Show example commands | `jfkreveal examples` |

### Global Options

| Option | Description | Default |
|--------|-------------|---------|
| `--data-dir` | Data directory | `./data` |
| `--log-level` | Logging level | `INFO` |
| `--log-file` | Log file path | `None` |
| `--no-console-log` | Disable console logging | `False` |

### Command-Specific Options

#### run-analysis

| Option | Description | Default |
|--------|-------------|---------|
| `--skip-scraping` | Skip document scraping | `False` |
| `--skip-processing` | Skip document processing | `False` |
| `--skip-analysis` | Skip analysis | `False` |
| `--no-clean-text` | Disable text cleaning | `False` |
| `--no-ocr` | Disable OCR | `False` |
| `--ocr-resolution` | OCR resolution multiplier | `2.0` |
| `--ocr-language` | OCR language | `eng` |
| `--max-workers` | Maximum worker processes | `None` |

#### search

| Option | Description | Default |
|--------|-------------|---------|
| `--count` | Number of results | `10` |
| `--format` | Output format (text, json, csv, html) | `text` |
| `--hybrid` | Use hybrid search | `True` |
| `--output` | Output file | `None` |

#### generate-report

| Option | Description | Default |
|--------|-------------|---------|
| `--executive-only` | Only generate executive summary | `False` |
| `--format` | Output format (html, md) | `html` |
| `--include-visualizations` | Include visualizations | `True` |

#### view-dashboard

| Option | Description | Default |
|--------|-------------|---------|
| `--host` | Dashboard host | `localhost` |
| `--port` | Dashboard port | `8050` |
| `--debug` | Run in debug mode | `False` |