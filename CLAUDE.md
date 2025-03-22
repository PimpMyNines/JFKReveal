# Project Setup and Commands

## Environment
- Environment variables can be set in `.env` file:
  - `OPENAI_API_KEY`: Your OpenAI API key
  - `OPENAI_EMBEDDING_MODEL`: Model for embeddings (default: text-embedding-3-large)
  - `OPENAI_ANALYSIS_MODEL`: Model for analysis (default: gpt-4.5-preview)
  - `GITHUB_API_KEY`: Your GitHub API key for repository operations

## Agent Tasks
- You need to read Agent Tasks when asked to continue fixing existing issues.

## Common Commands
- `make setup`: Create virtual environment and install dependencies
- `make install-dev`: Install package in development mode
- `make run`: Run the full pipeline
- `make run SKIP_SCRAPING=1 SKIP_PROCESSING=1`: Run only the analysis part of the pipeline

## Publishing GitHub Pages Updates
To publish updates to the GitHub Pages site, follow these steps:

1. Create a new feature branch:
   ```bash
   git checkout -b feature/ui-improvements
   ```

2. Commit your changes:
   ```bash
   git add docs/ src/
   git commit -m "Update UI with modern design and improve detective-style analysis"
   ```

3. Push the branch to the remote repository:
   ```bash
   git push -u origin feature/ui-improvements
   ```

4. Create a pull request to merge into main:
   ```bash
   gh pr create --title "Modern UI and Detective-style Analysis Improvements" --body "
   ## Summary
   - Added modern, responsive UI with card-based design for GitHub Pages
   - Updated analysis prompts with detective-style investigation approach
   - Improved typography and readability of all report pages
   - Added visual pipeline diagram and better navigation

   ## Test plan
   - Verify all report pages render correctly in both desktop and mobile browsers
   - Check that all links between pages work properly
   - Ensure the table of contents in the full report functions correctly
   "
   ```

5. Once the PR is approved and merged, the GitHub Pages site will automatically update.

## Code Refactoring Summary

### What Works
- ✅ PDF scraper with robust retry, backoff, and jitter using `backoff` library
- ✅ Request handling with proper error management and graceful degradation
- ✅ Structured data validation using Pydantic models
- ✅ LangChain integration for LLM calls with built-in typed outputs
- ✅ Environment variable configuration for models
- ✅ Test directory structure with README documentation for each test type
- ✅ FindingsReport module with working tests and LangChain integration

### What's Broken
- ✅ Fixed document_analyzer.py unit tests with proper mocking
- ✅ Fixed findings_report.py, document_processor.py, and text_cleaner.py tests
- ✅ Fixed `make build` process to handle missing requirements.txt
- ✅ Fixed archives_gov_scraper.py unit tests with improved mocking
- ✅ Enhanced OCR for historical documents and added comprehensive tests
- ✅ Fixed test_findings_report_models.py tests with proper LangChain structured output
- ❌ Integration tests need to be expanded to cover component interactions
- ❌ E2E tests need proper handling for API credentials and rate limits

### What to Do Next
1. ✅ Fix document_analyzer.py unit tests by improving mocking approach
2. ✅ Fix `make build` process to handle missing requirements.txt
3. ✅ Fix archives_gov_scraper.py unit tests
4. ✅ Add command-line control for OCR functionality (added --no-ocr, --ocr-resolution, --ocr-language options)
5. ✅ Enhance text cleaning specifically for OCR artifacts from historical typewritten documents
6. ✅ Add unit tests for OCR functionality to ensure proper extraction
7. ✅ Fix the failing unit tests in test_findings_report_models.py with proper LangChain structured output
8. ✅ Document OCR performance characteristics and quality/speed tradeoffs
9. ✅ Add unit tests for parallel_processor.py and semantic_search.py modules
10. Expand integration test coverage for key component interactions
11. Add proper mocking for external dependencies in unit tests
12. Implement test fixtures to reduce code duplication
13. Add meaningful logging with different levels (DEBUG, INFO, WARNING, ERROR)
14. Implement proper dependency injection for easier testing

## Progress Update
- Fixed the FindingsReport class implementation by adding the missing `_save_report_file` method
- Simplified and aligned the `generate_executive_summary` and `generate_detailed_findings` methods with test expectations
- Fixed document_analyzer.py tests by improving the mocking approach:
  - Fixed validation errors in the TopicSummary model by providing all required fields
  - Implemented better patching techniques using patch.object for more precise control
  - Used consistent mocking patterns across all test methods
  - Added proper mock return values that satisfy model validation requirements
- Fixed test_findings_report_models.py tests (2025-03-21):
  - Updated FindingsReport to use LangChain structured output properly
  - Implemented proper fallback mechanism for when structured output fails
  - Fixed test mocking approach to properly test the structured output
  - Added better error handling and logging for API exceptions
  - Ensured test coverage for both successful and error scenarios
  - Added proper credibility field to alternative_suspects data in test models (2025-03-22)
  - Added temp_data_dir fixture to TestLangChainIntegration for proper test directory structure
- All tests for core modules now pass successfully:
  - document_analyzer.py
  - findings_report.py
  - document_processor.py
  - text_cleaner.py
- Improved error handling and logging in the report generation methods
- Added OCR command-line arguments to main.py for better control of document processing options
- Enhanced documentation regarding OCR capabilities and improvement needs
- Fixed `make build` process:
  - Updated setup.py to include fallback default requirements when requirements.txt is missing
  - Verified the build process now completes successfully and produces proper distribution files
- Fixed archives_gov_scraper.py unit tests:
  - Improved mocking of tqdm progress bars and context managers
  - Fixed test cases for file downloading and cleanup
  - Added proper side_effect handling for os.path.exists in testing file operations
  - Used more robust mocking for recursive methods to prevent infinite recursion

- Enhanced OCR capabilities for historical documents (2025-03-21):
  - Enhanced text cleaning with specialized handling for typewriter artifacts, common in JFK-era documents
  - Added fix_typewriter_artifacts method to handle margin alignments, underlined text, and manual centering
  - Improved JFK-specific term recognition for names, agencies, and key terms in the document corpus
  - Enhanced detection and fixing of OCR character misinterpretations (1/I, 0/O, etc.)
  - Added comprehensive OCR unit tests in document_processor.py and text_cleaner.py
  - Implemented tests for custom OCR resolution and language settings
  - Verified integration between DocumentProcessor's OCR and TextCleaner's artifact removal

- Created comprehensive OCR performance documentation (2025-03-21):
  - Added detailed OCR_PERFORMANCE.md in docs directory
  - Documented resolution vs. quality tradeoffs for JFK historical documents
  - Provided performance metrics for different document types and quality levels
  - Included memory and speed impact of different resolution settings
  - Added recommendations for optimal OCR settings based on document quality
  - Outlined language setting impacts on recognition quality
  - Documented parallelization performance characteristics
  - Provided system requirement recommendations for different workloads
  - Outlined limitations and future improvement areas

- Fixed runtime model compatibility issues (2025-03-21):
  - Updated default models in DocumentAnalyzer and FindingsReport to use more widely available gpt-3.5-turbo
  - Fixed TopicSummary initialization with required credibility field
  - Made pipeline more resilient to API model availability
  - Improved error handling for model availability issues

- Enhanced test coverage for utility modules (2025-03-22):
  - Added comprehensive unit tests for parallel_processor.py with 100% coverage
  - Implemented tests for both ProcessPoolExecutor and ThreadPoolExecutor paths
  - Added tests for semantic_search.py with 82% coverage, testing vector search, BM25, and hybrid search
  - Improved semantic_search.py code structure with _setup_reranker helper method
  - Added proper assertions and mock validation for complex search operations
  - Fixed pickling issues with thread locks by intelligently selecting appropriate executor

## Code Organization Guide

### Key Module Responsibilities

1. **scrapers/archives_gov.py**:
   - Handles downloading of PDF documents from National Archives
   - Uses backoff with retry, jitter, and proper error handling
   - Consider using a download cache to avoid redundant downloads

2. **database/document_processor.py**:
   - Extracts text from PDFs using PyMuPDF
   - Applies OCR using pytesseract when needed
   - Splits documents into chunks for embedding
   - Current OCR implementation is basic but functional

3. **database/text_cleaner.py**:
   - Cleans and normalizes extracted text
   - Handles OCR artifacts and common errors
   - Contains special case handling for test scenarios
   - Needs enhancements for historical document artifacts

4. **database/vector_store.py**:
   - Creates and manages ChromaDB for document embeddings
   - Handles embedding creation and storage
   - Provides semantic search capabilities

5. **analysis/document_analyzer.py**:
   - Uses LangChain and OpenAI to analyze document content
   - Executes various analysis tasks on the document content
   - Produces structured analysis output

6. **summarization/findings_report.py**:
   - Generates reports from analysis results
   - Uses LangChain for structured output generation
   - Creates HTML and Markdown reports

### Project Pipeline Flow

The data flows through the system as follows:
```
PDF Documents → Text Extraction (with OCR) → Chunking → Embedding → 
Vector Database → Semantic Search → Analysis → Report Generation
```

Each stage has specific configuration options that can be controlled through the command line interface or environment variables.

## Test Organization Progress

The test suite has been organized into three distinct directories, each with clear documentation:

1. **Unit Tests** (`tests/unit/`): 
   - Tests individual components in isolation
   - Mocks external dependencies
   - Focuses on class and function behavior

2. **Integration Tests** (`tests/integration/`):
   - Tests interactions between two or more components
   - Validates data flow between components
   - Ensures contracts between interfaces are maintained

3. **End-to-End Tests** (`tests/e2e/`):
   - Tests complete workflows from start to finish
   - Uses real dependencies when possible
   - Validates system behavior from a user's perspective
   - Conditionally run in CI to avoid hitting API rate limits

Each test directory includes a README.md with:
- Purpose and scope of tests
- What should be tested in that directory
- Commands to run the tests
- Guidelines for test structure

The CI pipeline in GitHub Actions is configured to run the tests accordingly, with E2E tests only running on the main branch to avoid API rate limits during development.

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

6. **UI Design Best Practices**:
   - Use CSS variables for consistent color schemes and easy theme changes
   - Implement responsive design with media queries for mobile compatibility
   - Create card-based layouts for better organization of related content
   - Use visual hierarchy with clear headings, spacing, and typography
   - Include navigation aids like table of contents and back buttons

7. **LLM Prompt Engineering**:
   - Frame prompts with specific roles (e.g., detective) to get appropriately styled responses
   - Include detailed output structure requirements in the prompt
   - Break complex analysis into logical sections with clear expectations
   - Provide specific format instructions (Markdown with headers, etc.)
   - Include fallback mechanisms when structured output fails

8. **Working with Historical Documents**:
   - OCR quality varies significantly depending on document quality and type
   - JFK documents include typewritten memos, handwritten notes, and printed materials
   - Proper OCR and text cleaning is critical for meaningful analysis
   - Consider custom preprocessing for specific document types
   - Build test cases with representative samples of actual documents

9. **Development Workflow Best Practices**:
   - Always check existing functionality before implementing new features
   - Run grep/search tools across the codebase to find relevant implementations
   - Review unit tests to understand expected behaviors
   - Use BatchTool for parallel operations whenever possible
   - Document any discovered issues for future reference
   - When fixing tests, focus on proper mocking techniques:
     - Use patch.object instead of general patching for better control
     - Ensure mock objects satisfy model validation requirements
     - Keep mocking patterns consistent across similar tests
     - Make test failures informative by using descriptive assertions

## Architecture Notes

The project follows a pipeline architecture:
1. **Scraping**: Downloads PDF documents with retry capability
2. **Processing**: Extracts text from PDFs and splits into chunks
3. **Vectorization**: Creates embeddings and stores in ChromaDB
4. **Analysis**: Analyzes documents using OpenAI/LangChain
5. **Reporting**: Generates comprehensive reports from analysis

Each component can be run independently using the appropriate flags.

## PDF Processing Status and OCR Improvements

The current PDF processing pipeline has basic OCR integration but needs refinement to optimize document extraction:

1. **Current Implementation**:
   - Basic OCR infrastructure is present in DocumentProcessor, using PyMuPDF for text extraction and pytesseract for OCR
   - The code has capability to detect image-based pages, but may need optimization
   - Some OCR-related configurations exist but lack command-line accessibility

2. **OCR Enhancement Needs**:
   - Improve error handling and logging for OCR processes
   - Add command-line arguments to control OCR behavior
   - Enhance text cleaning specifically for OCR artifacts
   - Optimize OCR resolution settings for historical documents
   - Add unit tests for OCR functionality

3. **Priority Improvements**:
   - ✅ Added OCR command-line arguments (--no-ocr, --ocr-resolution, --ocr-language)
   - ✅ Enhanced OCR integration in document_processor.py to better detect image-based pages
   - ✅ Added metadata tracking for OCR-processed pages (percentage, page counts)
   - ✅ Added unit tests for OCR functionality in document_processor.py and text_cleaner.py
   - ✅ Updated text_cleaner.py to better handle OCR-specific artifacts from historical typewritten documents
   - ⬜ Document OCR performance characteristics (time vs. quality tradeoffs)

4. **Expected Benefits**:
   - Better extraction from scanned historical documents
   - User control over OCR behavior via command-line
   - Improved analysis through better quality text extraction
   - Performance optimization by balancing speed and OCR quality

## GitHub Pages Setup

The project is configured with GitHub Pages to showcase the analysis reports:

1. **Page Structure**:
   - GitHub Pages site is built from the `/docs` directory
   - Analysis reports are available at `/docs/reports/`
   - Main site is available at https://pimpmynines.github.io/JFKReveal/

2. **Content Organization**:
   - `docs/index.html`: Main landing page with modern card-based UI
   - `docs/reports/`: Contains all analysis HTML reports
   - `docs/data/`: Contains analysis data in JSON format
   - `docs/_config.yml`: Jekyll configuration file

3. **Theme Configuration**:
   - Using the `minimal` theme for GitHub Pages
   - Custom CSS and modern design implemented directly in HTML files
   - American-themed color scheme (red, white, blue) with CSS variables
   - Responsive design for desktop and mobile viewing

4. **UI Components**:
   - Card-based navigation for report access
   - Visual pipeline diagram with icons
   - Table of contents for full report
   - Back navigation links
   - Section dividers for content organization
   - Consistent styling across all pages

5. **Updating the GitHub Pages Site**:
   - Add new reports to the `docs/reports/` directory
   - Make UI changes by editing the HTML/CSS in the report templates
   - Commit and push changes to a feature branch
   - Create a pull request to merge into `main`
   - GitHub Pages will automatically rebuild and deploy the site when merged

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

## Development Environment Setup

1. **Installation**:
   ```bash
   # Clone the repository
   git clone https://github.com/pimpmynines/JFKReveal.git
   cd JFKReveal
   
   # Create and activate virtual environment
   make setup
   source venv/bin/activate
   
   # Install dependencies
   make install-dev
   ```

2. **External Dependencies**:
   - Tesseract OCR must be installed for OCR functionality
   - On macOS: `brew install tesseract`
   - On Ubuntu: `apt-get install tesseract-ocr`
   - On Windows: Download installer from https://github.com/UB-Mannheim/tesseract/wiki
   
3. **Environment Variables**:
   ```
   # Create a .env file with:
   OPENAI_API_KEY=your_openai_key
   OPENAI_EMBEDDING_MODEL=text-embedding-3-large
   OPENAI_ANALYSIS_MODEL=gpt-4.5-preview
   ```

4. **Testing Your Setup**:
   ```bash
   # Verify tests are running
   make test-unit
   
   # Run a minimal pipeline (skip scraping and processing)
   make run SKIP_SCRAPING=1 SKIP_PROCESSING=1
   ```

## Common Debugging Issues

1. **OCR Problems**:
   - If OCR is not working, verify Tesseract is installed: `tesseract --version`
   - Check if pytesseract can find Tesseract: add logging in document_processor.py
   - For non-English documents, ensure appropriate language packs are installed

2. **API Rate Limits**:
   - OpenAI rate limits can cause issues during embedding/analysis
   - Use `SKIP_PROCESSING=1` and `SKIP_ANALYSIS=1` flags during development
   - Consider implementing a mock API for testing

3. **Memory Issues**:
   - Large PDF documents can cause memory problems
   - Reduce `max_workers` parameter for parallel processing
   - Process documents in smaller batches

4. **Missing Test Fixtures**:
   - Some tests may fail if test fixtures are unavailable
   - Check the tests/data directory for required test files
