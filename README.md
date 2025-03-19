# JFKReveal 🔍

**Uncover the Truth: Declassified JFK Assassination Documents Analysis**

JFKReveal is a powerful tool that analyzes over 1,100 declassified documents from the National Archives to uncover evidence about the JFK assassination. It uses advanced AI techniques and RAG (Retrieval Augmented Generation) to provide comprehensive analysis and insights.

## 📋 Features

- **Automated Document Collection**: Scrapes 1,123 PDF documents from the National Archives JFK Release 2025 collection.
- **Advanced Text Extraction**: Processes PDFs to extract text with page references and metadata.
- **Semantic Search**: Creates vector embeddings for efficient document searching and retrieval.
- **AI-Powered Analysis**: Leverages OpenAI models to analyze document content for key information.
- **Comprehensive Reports**: Generates detailed reports on findings, suspects, and potential coverups.
- **Evidence-Based Conclusions**: Presents the most likely explanations based on document evidence.

## 🚀 Quick Start

### Prerequisites

- Python 3.8+ installed
- OpenAI API key

### Installation

Clone the repository and set up the environment:

```bash
# Clone the repository
git clone https://github.com/yourusername/JFKReveal.git
cd JFKReveal

# Set up virtual environment and install dependencies
make setup

# Configure your OpenAI API key
cp .env.example .env
# Edit the .env file and add your OpenAI API key

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
```

## 📊 Understanding the Results

The analysis produces several detailed reports:

- **Executive Summary**: High-level overview of key findings
- **Detailed Analysis**: In-depth examination of all evidence
- **Suspects Analysis**: Evaluation of potential culprits with supporting evidence
- **Coverup Analysis**: Assessment of potential government involvement or information suppression

All reports include document references, supporting evidence, and confidence levels for each conclusion.

## 🧠 How It Works

JFKReveal follows a sophisticated pipeline:

1. **Document Collection**: Scrapes PDF documents from the National Archives website
2. **Text Extraction**: Processes PDFs to extract text with page numbers and metadata
3. **Chunking & Vectorization**: Splits documents into manageable chunks and creates vector embeddings
4. **Topic Analysis**: Analyzes documents for specific topics, individuals, and events
5. **Report Generation**: Synthesizes findings into comprehensive reports

The system uses OpenAI embeddings for vectorization and GPT-4o for analysis, ensuring high-quality results.

## 🛠️ Configuration

Key configuration options:

- **OpenAI API Key**: Set in the `.env` file (copy from `.env.example`)
- **Embedding Model**: Configure in `.env` file (defaults to `text-embedding-ada-002` for local development)
- **Analysis Model**: Configure in `.env` file (defaults to `gpt-4o`)
- **Chunking Parameters**: Adjust chunk size and overlap in `document_processor.py`
- **Analysis Topics**: Modify topics list in `document_analyzer.py`

## 📚 Documentation

For more detailed documentation:

- [API Reference](docs/API_REFERENCE.md)
- [Configuration Guide](docs/CONFIGURATION.md)
- [Technical Architecture](docs/ARCHITECTURE.md)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

This tool is designed for educational and research purposes only. The analysis presents evidence-based conclusions but should not be considered definitive. All findings should be critically evaluated alongside other historical research.

## 🙏 Acknowledgments

- National Archives for making these documents available
- OpenAI for providing the AI capabilities
- LangChain and ChromaDB for vector search functionality