# Project Setup and Commands

## Environment
- You need to set up an OpenAI API key in the `.env` file
- For local development, use the `text-embedding-ada-002` model instead of newer models
- Environment variables can be set in `.env` file:
  - `OPENAI_API_KEY`: Your OpenAI API key
  - `OPENAI_EMBEDDING_MODEL`: Model for embeddings (default: text-embedding-ada-002)
  - `OPENAI_ANALYSIS_MODEL`: Model for analysis (default: gpt-4o)

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

## Fixes for OpenAI Embedding Issues

### Plan to complete reverting to OpenAI embeddings

1. **Fix the document loading in vector_store.py**:
   - There seems to be an issue with the JSON document format in `test_document.json` and `sample_document.json`
   - Check the format and update the code to properly parse them

2. **Use function_calling method for LLM output parsing**:
   - In `document_analyzer.py`, ensure all calls to `with_structured_output()` use `method="function_calling"`
   - This fixes the schema validation issue with TopicSummary and DocumentAnalysisResult models

3. **Remove the metadata default value in the TopicSummary model**:
   - Update the Pydantic model to fix the error: "In context=('properties', 'credibility'), 'default' is not permitted"
   - The credibility field should not have a default value with OpenAI function calling

4. **Ensure proper JSON processing for ChromaDB**:
   - Update `filter_complex_metadata` to properly handle the list values in the metadata
   - Make sure chunk_id is properly extracted from metadata

5. **Clean up the database for fresh testing**:
   - Remove the `data/vectordb` directory to start fresh
   - This ensures dimensions and other DB settings are consistent

6. **Verify OpenAI API key permission**:
   - Ensure the API key in `.env` has access to the specified embedding model
   - If needed, upgrade the account or request access to the embedding models
