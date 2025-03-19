# Project Setup and Commands

## Environment
- You need to set up at least one API key in the `.env` file (OpenAI, Anthropic, or X AI)
- Environment variables can be set in `.env` file:
  - `OPENAI_API_KEY`: Your OpenAI API key
  - `ANTHROPIC_API_KEY`: Your Anthropic API key
  - `XAI_API_KEY`: Your X AI (Grok) API key
  - `OPENAI_EMBEDDING_MODEL`: Model for embeddings (default: text-embedding-3-large)
  - `OPENAI_ANALYSIS_MODEL`: OpenAI model for analysis (default: gpt-4o)
  - `OPENAI_REPORT_MODEL`: OpenAI model for report generation (default: gpt-4o)
  - `ANTHROPIC_ANALYSIS_MODEL`: Anthropic model for analysis (default: claude-3-7-sonnet-20240620)
  - `ANTHROPIC_REPORT_MODEL`: Anthropic model for report generation (default: claude-3-7-sonnet-20240620)
  - `XAI_ANALYSIS_MODEL`: X AI model for analysis (default: grok-2)
  - `XAI_REPORT_MODEL`: X AI model for report generation (default: grok-2)
  - `GITHUB_API_KEY`: Your GitHub API key for repository operations
  - `ENABLE_AUDIT_LOGGING`: Set to "true" or "false" to enable/disable detailed thought process logging (default: true)

## Common Commands
- `make setup`: Create virtual environment and install dependencies
- `make install-dev`: Install package in development mode
- `make run`: Run the full pipeline with OpenAI models
- `make run MODEL_PROVIDER=anthropic`: Run the pipeline with Anthropic Claude models
- `make run MODEL_PROVIDER=xai`: Run the pipeline with X AI Grok models
- `make run SKIP_SCRAPING=1 SKIP_PROCESSING=1`: Run only the analysis part of the pipeline
- `make run ENABLE_AUDIT_LOGGING=false`: Run without detailed thought process logging
- `make run MODEL_PROVIDER=anthropic ANTHROPIC_ANALYSIS_MODEL=claude-3-7-opus-20240620`: Use a specific Anthropic model
- `make run MODEL_PROVIDER=xai XAI_ANALYSIS_MODEL=grok-2`: Use a specific X AI model
- `make run-all-models`: Run analysis with all supported models for comparison
- `make audit-viewer`: Launch a simple web interface to explore audit logs

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
- ✅ Detailed thought process audit logging with LangChain callbacks
- ✅ Findings report generation with modern HTML templates

### What's Broken
- ❌ The audit log viewer is not yet implemented
- ❌ The test suite is not comprehensive enough to verify all components

### What to Do Next
1. Create a simple web interface for exploring and visualizing audit logs
2. Develop unit tests for document analysis with mocked LLM responses
3. Implement end-to-end tests for the full pipeline
4. Add audit log comparison tools to identify reasoning differences between model versions
5. Create a CLI interface with detailed help and options
6. Develop visualization tools for audit log analysis

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
   - Add explicit instructions for documenting thought processes to create audit trails
   
8. **Claude 3.7-Specific Prompt Engineering**:
   - Leverage Claude 3.7's extended thinking capabilities with detailed reasoning instructions
   - Use explicit "think step by step" instructions for complex analytical tasks
   - Structure prompts with clear hierarchical sections using numbered lists
   - Request explicit reasoning traces with format like "Thought process: [reasoning]"
   - Include self-verification steps asking Claude to review its own work
   - For complex analyses, use a two-stage approach: first analysis, then synthesis
   - Provide clear evaluation criteria for Claude to judge evidence quality
   - Specify exactly how to handle contradictions or inconsistent evidence
   - Ask for confidence levels on specific conclusions (high/medium/low)
   - Use detailed schemas with `function_calling` for reliable structured outputs

9. **Audit Logging Best Practices**:
   - Use LangChain callbacks to capture every step of the model's reasoning
   - Store thought processes in structured JSON for easy analysis
   - Include timestamps and document/topic identifiers in all logs
   - Separate large document content from log entries for better readability
   - Create different log granularity levels (token-by-token vs summary)
   - Implement streaming tokens for the most detailed reasoning capture

## Architecture Notes

The project follows a pipeline architecture:
1. **Scraping**: Downloads PDF documents with retry capability
2. **Processing**: Extracts text from PDFs and splits into chunks
3. **Vectorization**: Creates embeddings and stores in ChromaDB
4. **Analysis**: Analyzes documents using LLMs (OpenAI, Anthropic, or xAI) via LangChain with detailed thought process logging
5. **Audit Logging**: Captures and stores the model's reasoning in structured JSON files
6. **Reporting**: Generates comprehensive reports from analysis with traceable reasoning paths
7. **Model Comparison**: Generates reports with multiple models and provides UI for comparison

Each component can be run independently using the appropriate flags.

### Multi-Model Support
The system supports OpenAI, Anthropic Claude, and xAI (Grok) models:

1. **Model Selection**:
   - Use the `--model-provider` flag to choose between "openai", "anthropic", and "xai"
   - Each provider has dedicated environment variables for model selection
   - Recommended providers by task:
     * Complex analysis tasks: Anthropic's Claude 3.7 models
     * Visualization and statistical reasoning: OpenAI's GPT-4o
     * Creative analysis and insights: xAI's Grok-2

2. **Environment Variable Mapping**:
   - For OpenAI: `OPENAI_EMBEDDING_MODEL`, `OPENAI_ANALYSIS_MODEL`, `OPENAI_REPORT_MODEL`
   - For Anthropic: `ANTHROPIC_ANALYSIS_MODEL`, `ANTHROPIC_REPORT_MODEL`
   - For xAI: `XAI_ANALYSIS_MODEL`, `XAI_REPORT_MODEL`
   
3. **API Keys**:
   - OpenAI API key should be set with `OPENAI_API_KEY`
   - Anthropic API key should be set with `ANTHROPIC_API_KEY`
   - xAI API key should be set with `XAI_API_KEY`
   
4. **Vector Database**:
   - Currently uses OpenAI embeddings regardless of the model provider
   - Future versions may add support for Anthropic embeddings when available

5. **Claude 3.7 Extended Thinking**:
   - Default models are now set to Claude 3.7 Sonnet, which is specifically designed for extended thinking
   - Claude 3.7 models excel at maintaining context over long documents and complex analyses
   - For most analytical tasks, Claude 3.7 Sonnet provides the best balance of performance and cost
   - For particularly complex analyses, consider using Claude 3.7 Opus (`ANTHROPIC_ANALYSIS_MODEL=claude-3-7-opus-20240620`)
   - These models are particularly strong at:
     * Multi-document analysis with cross-referencing
     * Tracking complex timelines and relationships between entities
     * Detailed evidence evaluation and credibility assessment
     * Detecting patterns and inconsistencies across large document sets
     * Reasoning through complex hypotheses with structured thinking

6. **Agent Usage Notes**:
   - When using Claude models for analysis, consider increasing the chunk size for document processing
   - Claude 3.7 models can handle longer contexts more effectively than previous versions
   - The audit logs from Claude 3.7 provide more detailed reasoning traces, useful for debugging
   - Consider setting a lower temperature (0.0-0.1) for analysis tasks and slightly higher (0.1-0.3) for report generation

## Agentic Workflow Architecture

The system now implements an agentic workflow approach for document analysis:

1. **Agent-Based Processing**:
   - Each major task (analysis, report generation) is handled by specialized agents
   - Agents maintain their own state and reasoning chain
   - Audit logs capture the agent's thought process for transparency

2. **Pipeline Orchestration**:
   - The main pipeline controller coordinates agents across different stages
   - Each agent focuses on its specialized task while sharing results through structured outputs
   - This enables parallel processing where possible

3. **Context Management**:
   - Agents are designed to manage their own context windows efficiently
   - Claude 3.7 models have enhanced context management capabilities
   - Document chunking strategies are optimized for Claude's extended thinking mode

4. **Task Specialization**:
   - Document Analysis Agent: Focuses on extracting and categorizing information
   - Cross-Document Analysis Agent: Identifies patterns and connections across documents
   - Report Generation Agent: Synthesizes findings into structured reports
   - Each agent uses optimized prompts specific to its task

5. **Audit Trail**:
   - All agent reasoning is captured in detailed audit logs
   - This provides a complete trail of how conclusions were reached
   - Useful for validating findings and improving agent performance

6. **Best Practices**:
   - Use chain-of-thought prompting for complex analytical tasks
   - Structure agent outputs with JSON schemas for reliable parsing
   - Keep prompt instructions clear and hierarchical
   - For complex analyses, break down into subtasks with separate agents

## Audit Logging Structure

The audit logging system captures the model's thought process at multiple levels:

1. **Document Analysis Logs**: 
   - Stored in: `data/audit_logs/`
   - Format: JSON files with timestamp and document ID
   - Contents: Model thought process during individual document analysis

2. **Topic Analysis Logs**:
   - Stored in: `data/analysis/[topic]_audit.json`
   - Format: JSON with hierarchical structure
   - Contents: Cross-document reasoning and pattern identification

3. **Report Generation Logs**:
   - Stored in: `data/audit_logs/reports/`
   - Format: Per-report JSON files
   - Contents: Reasoning for report synthesis and conclusion formation

4. **Combined Audit Logs**:
   - Stored in: `data/reports/full_report_audit.json`
   - Format: Consolidated JSON with section mapping
   - Contents: Summarized thought process for the entire report

The logs can be viewed directly as JSON files or through the audit log viewer (when implemented).

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

## Current System Improvements (In Progress)

The following improvements are currently being implemented:

1. **Local Document References**: 
   - Updated to use local PDF references instead of direct links to archives.gov
   - Added proper citation info for original sources
   - Status: ✅ Implemented in FindingsReport

2. **Grok Model Support**:
   - Added xAI as a model provider option
   - Added environment variables for Grok models
   - Implemented fallback mechanism for xAI integration
   - Status: ✅ Fully implemented

3. **Multi-Model Comparison**:
   - Added Makefile target to run analysis with multiple models
   - Created model-specific output directories
   - Status: ✅ Backend implementation complete
   - TODO: Complete model comparison UI  

4. **Modern UI with Visualizations**:
   - Added new CSS variables and improved styling
   - Added data visualization capabilities with Chart.js
   - Status: 🔄 Partially implemented
   - TODO: Complete the UI and model selector component

5. **Next Steps for Next Agent**:
   - Complete the `generate_comparison.py` tool in the `tools` directory to create comparison pages
   - Finish implementing the model selector in HTML templates
   - Add data visualization charts for logical/statistical examples
   - Update the docs/index.html to include the modern styling
   - Ensure HTML templates are properly written to both model-specific and main directories
