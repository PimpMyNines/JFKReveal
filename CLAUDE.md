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
