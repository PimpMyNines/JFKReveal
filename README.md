# JFKReveal 🔍

**Uncover the Truth: Declassified JFK Assassination Documents Analysis**

JFKReveal is a powerful tool that analyzes over 1,100 declassified documents from the National Archives to uncover evidence about the JFK assassination. It uses advanced AI techniques and RAG (Retrieval Augmented Generation) to provide comprehensive analysis and insights with detailed audit logging of the reasoning process.

## 📋 Features

- **Automated Document Collection**: Scrapes 1,123 PDF documents from the National Archives JFK Release 2025 collection.
- **Advanced Text Extraction**: Processes PDFs to extract text with page references and metadata.
- **Semantic Search**: Creates vector embeddings for efficient document searching and retrieval.
- **AI-Powered Analysis**: Leverages OpenAI models to analyze document content for key information.
- **Comprehensive Reports**: Generates detailed reports on findings, suspects, and potential coverups.
- **Evidence-Based Conclusions**: Presents the most likely explanations based on document evidence.
- **Detailed Thought Process Audit**: Captures the model's reasoning in JSON logs for transparency and verification.

## 🚀 Quick Start

### Prerequisites

- Python 3.8+ installed
- API key for at least one of these providers:
  - OpenAI API key
  - Anthropic API key
  - X AI (Grok) API key

### Installation

Clone the repository and set up the environment:

```bash
# Clone the repository
git clone https://github.com/PimpMyNines/JFKReveal.git
cd JFKReveal

# Set up virtual environment and install dependencies
make setup

# Configure your API key(s)
cp .env.example .env
# Edit the .env file and add your API key(s)
# OPENAI_API_KEY=your_openai_key
# ANTHROPIC_API_KEY=your_anthropic_key
# XAI_API_KEY=your_xai_key

# Install the package
make install-dev
```

### Running the Analysis

To run the complete analysis pipeline:

```bash
make run
```

This will:
1. Download all PDF documents from the National Archives
2. Process the documents and extract text
3. Create a vector database for efficient searching
4. Analyze the documents using AI
5. Generate comprehensive reports

The final report will be available at `data/reports/full_report.html`.

### Options

Skip certain pipeline steps if needed:

```bash
# Skip the document scraping (if you already have the PDFs)
make run SKIP_SCRAPING=1

# Skip document processing (if PDFs are already processed)
make run SKIP_PROCESSING=1

# Use a specific model provider
make run MODEL_PROVIDER=anthropic  # Use Anthropic Claude models
make run MODEL_PROVIDER=xai        # Use X AI (Grok) models
make run MODEL_PROVIDER=openai     # Use OpenAI models (default)
```

## 📊 Understanding the Results

The analysis produces several detailed reports:

- **Executive Summary**: High-level overview of key findings
- **Detailed Analysis**: In-depth examination of all evidence
- **Suspects Analysis**: Evaluation of potential culprits with supporting evidence
- **Coverup Analysis**: Assessment of potential government involvement or information suppression
- **Audit Logs**: JSON files containing the model's detailed thought process for each analysis step

All reports include document references, supporting evidence, confidence levels, and traceable reasoning paths.

## 🧠 How It Works

JFKReveal follows a sophisticated pipeline:

1. **Document Collection**: Scrapes PDF documents from the National Archives website
2. **Text Extraction**: Processes PDFs to extract text with page numbers and metadata
3. **Chunking & Vectorization**: Splits documents into manageable chunks and creates vector embeddings
4. **Topic Analysis**: Analyzes documents for specific topics, individuals, and events
5. **Thought Process Auditing**: Records the model's reasoning step-by-step in detailed logs
6. **Report Generation**: Synthesizes findings into comprehensive reports with traceable conclusions

The system uses OpenAI embeddings for vectorization and GPT-4o for analysis, ensuring high-quality results with transparent reasoning.

## 🛠️ Configuration

Key configuration options:

- **API Keys**: Set in the `.env` file (copy from `.env.example`)
  - `OPENAI_API_KEY`: Your OpenAI API key
  - `ANTHROPIC_API_KEY`: Your Anthropic API key 
  - `XAI_API_KEY`: Your X AI (Grok) API key
- **Model Providers**: Choose between OpenAI, Anthropic, or X AI (Grok)
- **Embedding Model**: Configure in `.env` file (defaults to `text-embedding-3-large` for local development)
- **Analysis Models**:
  - OpenAI: `OPENAI_ANALYSIS_MODEL` (defaults to `gpt-4o`)
  - Anthropic: `ANTHROPIC_ANALYSIS_MODEL` (defaults to `claude-3-7-sonnet-20240620`)
  - X AI: `XAI_ANALYSIS_MODEL` (defaults to `grok-2`)
- **Report Models**:
  - OpenAI: `OPENAI_REPORT_MODEL` (defaults to `gpt-4o`)
  - Anthropic: `ANTHROPIC_REPORT_MODEL` (defaults to `claude-3-7-sonnet-20240620`)
  - X AI: `XAI_REPORT_MODEL` (defaults to `grok-2`)
- **Chunking Parameters**: Adjust chunk size and overlap in `document_processor.py`
- **Analysis Topics**: Modify topics list in `document_analyzer.py`
- **Audit Logging**: Enable/disable or customize in both `document_analyzer.py` and `findings_report.py`
- **Streaming Tokens**: Control token-level logging for detailed reasoning capture

## 📚 Documentation

For more detailed documentation:

- [API Reference](docs/API_REFERENCE.md)
- [Configuration Guide](docs/CONFIGURATION.md)
- [Technical Architecture](docs/ARCHITECTURE.md)
- [Live Reports](https://pimpmynines.github.io/JFKReveal/)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

This tool is designed for educational and research purposes only. The analysis presents evidence-based conclusions but should not be considered definitive. All findings should be critically evaluated alongside other historical research.

## 🙏 Acknowledgments

- National Archives for making these documents available
- OpenAI, Anthropic, and X for providing the AI capabilities
- LangChain and ChromaDB for vector search functionality