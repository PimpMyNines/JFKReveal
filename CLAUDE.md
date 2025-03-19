# Project Setup and Commands

## Environment
- You need to set up an OpenAI API key in the `.env` file
- Environment variables can be set in `.env` file:
  - `OPENAI_API_KEY`: Your OpenAI API key
  - `OPENAI_EMBEDDING_MODEL`: Model for embeddings (default: text-embedding-3-large)
  - `OPENAI_ANALYSIS_MODEL`: Model for analysis (default: gpt-4.5-preview)
  - `GITHUB_API_KEY`: Your GitHub API key for repository operations

## Common Commands
- `make setup`: Create virtual environment and install dependencies
- `make install-dev`: Install package in development mode
- `make run`: Run the full pipeline
- `make run SKIP_SCRAPING=1 SKIP_PROCESSING=1`: Run only the analysis part of the pipeline

## Code Refactoring Summary

### What Works
- ✅ PDF scraper with robust retry, backoff, and jitter using `backoff` library
- ✅ Request handling with proper error management and graceful degradation
- ✅ Structured data validation using Pydantic models
- ✅ LangChain integration for LLM calls with built-in typed outputs
- ✅ Environment variable configuration for models

### What's Broken
- ❌ The findings_report.py module still needs to be updated to use LangChain and Pydantic
- ❌ The test suite is not comprehensive enough to verify all components

### What to Do Next
1. Update findings_report.py to use LangChain and Pydantic
2. Create proper unit tests for each module
3. Add meaningful logging with different levels (DEBUG, INFO, WARNING, ERROR)
4. Implement proper dependency injection for easier testing
5. Create a better CLI interface with detailed help and options

## Lessons Learned

1. **Library Choices Matter**:
   - Using established libraries like `backoff`, `tenacity`, and `Pydantic` saves significant time over custom implementations
   - LangChain provides built-in error handling, retries, and structured outputs that simplify LLM interactions

2. **Error Handling Strategy**:
   - Implement proper retry with backoff early in the development process
   - Use jitter to avoid thundering herd problems with API rate limits
   - Define clear exception hierarchies and handle specific exceptions differently

3. **Configuration Management**:
   - Use environment variables and `.env` files for secrets and configuration
   - Create typed configuration objects (like our `ScraperConfig`) for better validation and documentation

4. **Type Safety**:
   - Pydantic models provide runtime validation and better documentation
   - Structured outputs from LLMs are more reliable when validated against a schema

5. **Testing**:
   - Test network-dependent code with small, isolated examples before full integration
   - Create mock responses for testing to avoid hitting real APIs
   - Test failure modes (timeouts, rate limits) as well as success paths

## Architecture Notes

The project follows a pipeline architecture:
1. **Scraping**: Downloads PDF documents with retry capability
2. **Processing**: Extracts text from PDFs and splits into chunks
3. **Vectorization**: Creates embeddings and stores in ChromaDB
4. **Analysis**: Analyzes documents using OpenAI/LangChain
5. **Reporting**: Generates comprehensive reports from analysis

Each component can be run independently using the appropriate flags.

## GitHub Pages Setup

The project is configured with GitHub Pages to showcase the analysis reports:

1. **Page Structure**:
   - GitHub Pages site is built from the `/docs` directory
   - Analysis reports are available at `/docs/reports/`
   - Main site is available at https://pimpmynines.github.io/JFKReveal/

2. **Content Organization**:
   - `docs/index.html`: Main landing page
   - `docs/reports/`: Contains all analysis HTML reports
   - `docs/data/`: Contains analysis data in JSON format
   - `docs/_config.yml`: Jekyll configuration file

3. **Theme Configuration**:
   - Using the `minimal` theme for GitHub Pages
   - Custom styling can be added through the theme's layout options

4. **Updating the GitHub Pages Site**:
   - Add new reports to the `docs/reports/` directory
   - Commit and push changes to the `main` branch
   - GitHub Pages will automatically rebuild and deploy the site

## Repository Configuration

1. **Branch Protection**:
   - Main branch is protected with required reviews
   - Direct pushes to main are restricted to repository owners
   - All changes must go through pull requests

2. **GitHub Actions**:
   - Configured to run tests on pull requests
   - Uses Python 3.10 environment
   - Tests run with pytest
   - Linting with flake8

3. **Templates**:
   - Issue template for bug reports and feature requests
   - Pull request template with checklist
   - CODEOWNERS file designating repository owners
