# JFKReveal Project Agent Guide

You are assisting with the JFKReveal project, a sophisticated analysis tool for declassified JFK assassination documents. The project uses AI (OpenAI, LangChain) to extract insights from over 1,100 PDFs from the National Archives. Please follow these guidelines to efficiently improve the codebase.

## Current Status

The project has a functional pipeline architecture:
1. **Scraping**: Downloads PDF documents from National Archives
2. **Processing**: Extracts and cleans text from PDFs 
3. **Vectorization**: Creates embeddings in ChromaDB
4. **Analysis**: Analyzes documents using LLMs (OpenAI)
5. **Reporting**: Generates comprehensive analysis reports

However, several issues need immediate attention, particularly with the build process, failing tests, and PDF text extraction challenges.

## Priority Issues

### Immediate Fixes (Critical)
1. ✅ **Fix `make build` failures**: 
   - Problem: Build fails due to missing requirements.txt during build
   - Solution: Updated setup.py to handle missing requirements.txt file with fallback default requirements

2. **Fix failing unit tests**:
   - ✅ Fixed document_analyzer.py tests with improved mocking techniques
   - ✅ Fixed findings_report.py, document_processor.py, and text_cleaner.py tests
   - ✅ Fixed archives_gov_scraper.py tests with proper mocking of async operations
   - ✅ Fixed test_findings_report_models.py by adding extra field handling in Pydantic models
   - Run `make test-unit` to identify any remaining failing tests

3. **Ensure Makefile targets work correctly**:
   - Add dedicated test targets for different test categories
   - Test each target to ensure proper functionality

4. **Improve OCR for scanned PDFs**:
   - Status: Basic OCR infrastructure exists but needs refinement
   - Actions:
     - ✅ Added command-line arguments for OCR control (--no-ocr, --ocr-resolution, --ocr-language)
     - ✅ Enhanced text cleaning for better OCR results with specific handling for historical typewritten documents
     - ✅ Added unit tests for OCR functionality in document_processor.py and text_cleaner.py
     - ✅ Improved JFK-specific term handling for better recognition of names, agencies, and key terms
     - ✅ Document OCR performance characteristics

5. ✅ **Fix E2E tests**:
   - ✅ Added proper handling for API credentials and rate limits
   - ✅ Implemented smart credential management with fallbacks
   - ✅ Added retry logic with exponential backoff for rate limits
   - ✅ Created CI/CD integration for safe testing
   - ✅ Added comprehensive mocking for API calls in CI environment

### High Priority
1. Improve API credential handling with better fallbacks
2. Enhance error handling in main pipeline components
3. Implement better test fixtures
4. ✅ Enhance text cleaning for OCR artifacts (especially for historical typewritten documents)

### Medium Priority
1. Implement proper dependency injection for easier testing
2. Add robust logging throughout
3. Expand integration test coverage
4. Reduce code duplication in tests
5. Enhance text preprocessing pipeline for better extraction quality

### Low Priority
1. Improve CLI interface
2. Clean up code duplication
3. Update documentation
4. Add more pipeline control options

## Working Components

- PDF scraper with robust retry/backoff
- Request handling with proper error management
- Structured data validation (Pydantic)
- LangChain integration for LLM calls
- Environment variable configuration
- Test directory structure
- FindingsReport module
- Text cleaning for digital PDFs (but lacks OCR capability)

## Development Guidelines

1. **Fix one issue at a time**: Complete one task before moving to the next
2. **Document all changes**: Update CLAUDE.md with your progress
3. **Follow existing patterns**: Maintain consistent code style
4. **Add or update tests**: Ensure test coverage doesn't decrease
5. **Mark completed items**: ❌ → ✅ in CLAUDE.md with dated entries immediatly after validating the updates were succuesful

## Testing Workflow

1. Run specific test categories:
   ```bash
   make test-unit       # Unit tests only
   make test-integration # Integration tests only
   make test-e2e        # End-to-end tests only
   ```

2. After fixing issues, run full test suite:
   ```bash
   make test            # All tests
   ```

## Key Files

1. `src/jfkreveal/main.py`: Main pipeline implementation (JFKReveal class)
2. `src/jfkreveal/database/document_processor.py`: PDF processing and text extraction
3. `src/jfkreveal/database/text_cleaner.py`: OCR text cleaning functionality
4. `src/jfkreveal/analysis/document_analyzer.py`: Core analysis functionality  
5. `Makefile`: Build and run targets
6. `run_tests.sh`: Test execution script
7. `tests/`: Test directory with unit, integration, and e2e subdirectories
8. `CLAUDE.md`: Project documentation and progress tracking

## Immediate Action Plan

1. First, fix the `make build` process:
   ```makefile
   build:
     $(PYTHON) -m pip install --upgrade pip
     $(PYTHON) -m pip install --upgrade build
     # Create temporary requirements file if it doesn't exist
     test -f requirements.txt || touch requirements.txt
     $(PYTHON) -m build
   ```

2. Add proper test targets to Makefile:
   ```makefile
   test-unit:
     $(VENV)/bin/python -m pytest tests/unit/ -v

   test-integration:
     $(VENV)/bin/python -m pytest tests/integration/ -v

   test-e2e:
     $(VENV)/bin/python -m pytest tests/e2e/ -v --skip-slow
   ```

3. Add OCR capability to PDF processing:
   - Add pytesseract and Pillow to requirements.txt
   - Modify document_processor.py to detect image-based PDFs
   - Implement OCR fallback in extract_text_from_pdf method
   - Enhance text cleaning to handle OCR artifacts

4. Run tests to identify failing unit tests:
   ```bash
   make test-unit
   ```

5. Fix each failing test one by one, updating CLAUDE.md after each fix.

## PDF Text Extraction Improvement Plan

### Current Status
- Basic OCR infrastructure exists using PyMuPDF (fitz) with pytesseract fallback
- Some image detection and OCR capability is implemented
- Command-line control has been added for OCR options
- Further refinement needed for optimization and quality

### OCR Enhancement Plan
1. **Dependency Verification**:
   - Verify pytesseract and Pillow are properly installed
   - Check Tesseract binaries are accessible in the environment
   - Add error handling for missing dependencies

2. **Text Cleaning for OCR**:
   - Enhance OCR-specific text cleaning rules:
     - Fix erroneous word breaks and hyphens
     - Handle typewriter artifacts (mixed spaces, alignment issues)
     - Fix common OCR recognition errors for numbers/letters (0/O, 1/I, etc.)
     - Preserve document structure while removing headers/footers

3. **OCR Performance Optimization**:
   - Document optimal resolution settings for JFK documents
   - Balance quality vs. speed for different document types
   - Consider parallelizing OCR for multi-page documents

4. **Testing and Quality Assurance**:
   - Create unit tests with sample scanned documents
   - Measure extraction accuracy with known content
   - Add logging for OCR quality metrics

## Progress Tracking

Always update CLAUDE.md when making changes:
1. Update the "Last Updated" timestamp
2. Mark completed items (❌ → ✅)
3. Add dated entries to "Recent Fixes" section (Check current date from local system)
4. Document any new issues discovered

## Recent Progress (2025-03-22)

- ✅ Fixed document_analyzer.py tests with proper mocking:
  - Fixed validation errors in TopicSummary model by providing all required fields
  - Implemented better patching using patch.object for more precise control
  - Used consistent mocking patterns across all test methods 
  - Added proper mock return values that satisfy Pydantic validation
- ✅ Fixed test_findings_report_models.py tests:
  - Added extra field handling in Pydantic models using `extra="ignore"` config
  - Updated to modern Pydantic v2 ConfigDict approach to remove deprecation warnings
  - Improved model compatibility with LangChain structured outputs
  - Fixed test cases for all response models (Executive Summary, Detailed Findings, etc.)
  - Ensured test coverage for both successful and error scenarios
- ✅ Created comprehensive OCR performance documentation:
  - Added detailed OCR_PERFORMANCE.md in docs directory
  - Documented resolution vs. quality tradeoffs
  - Provided performance metrics for different document types
  - Added recommendations for optimal OCR settings
  - Outlined limitations and future improvements
- ✅ Fixed runtime model compatibility issues:
  - Updated default models to use more widely available gpt-3.5-turbo
  - Fixed TopicSummary initialization with required credibility field
  - Made pipeline more resilient to API model availability
- ✅ Verified core module tests are now passing:
  - document_analyzer.py
  - findings_report.py
  - document_processor.py
  - text_cleaner.py
  - findings_report_models.py
- ✅ Fixed `make build` process:
  - Updated setup.py to handle missing requirements.txt file with fallback default requirements
  - Verified build process now successfully creates distribution packages
- ✅ Enhanced test coverage for critical utility modules:
  - Added comprehensive unit tests for parallel_processor.py with 100% coverage
  - Implemented tests for both ProcessPoolExecutor and ThreadPoolExecutor paths
  - Created thorough tests for semantic_search.py with 82% coverage
  - Tested vector search, BM25, and hybrid search functionality
  - Improved semantic_search.py with _setup_reranker helper method for testing
  - Fixed pickling issues with thread locks using intelligent executor selection
- Updated CLAUDE.md with progress details
- Updated AGENT_TASKS.md with completed items

Your goal is to ensure the entire pipeline functions correctly, with passing tests and proper documentation.