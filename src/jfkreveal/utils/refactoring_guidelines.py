"""
Refactoring Guidelines to Clean Up Code Duplication

This module provides guidelines for addressing code duplication identified in the JFKReveal project.
The goal is to improve code maintainability, reduce potential for bugs, and make future
enhancements easier to implement.
"""

# 1. Template Management
# Problem: HTML templates are duplicated across main.py and findings_report.py
# Solution: Extract all templates to a dedicated templates directory

TEMPLATE_REFACTORING_PLAN = """
1. Create a templates directory structure:
   - src/jfkreveal/templates/
     - base.html       (common layout, CSS, header/footer)
     - reports/
       - executive_summary.html
       - detailed_findings.html
       - suspects_analysis.html 
       - coverup_analysis.html
       - full_report.html
     - components/
       - navigation.html
       - table_of_contents.html
       - card.html

2. Implement a simple template engine or use an existing one:
   - Create a TemplateEngine class in src/jfkreveal/utils/template_engine.py
   - Support template inheritance (extends, include)
   - Support variable interpolation
   
3. Update findings_report.py and main.py to use the template engine:
   - Replace inline HTML with template loading
   - Move CSS to separate files
"""

# 2. Argument Parsing
# Problem: Multiple similar add_*_args functions in main.py
# Solution: Create a more composable argument parser system

ARG_PARSER_REFACTORING_PLAN = """
1. Create a dedicated module for argument parsing:
   - src/jfkreveal/utils/arg_parser.py

2. Implement an ArgumentGroup class that encapsulates related arguments:
   - OCR arguments group
   - Processing arguments group
   - Analysis arguments group
   - Report arguments group
   - Logging arguments group

3. Create a function to compose argument groups:
   - register_argument_groups(parser, groups)
   
4. Update main.py to use the new argument parser:
   - Replace all add_*_args functions with argument group registrations
"""

# 3. File Operations
# Problem: Similar file scanning logic in document_processor.py and text_cleaner.py
# Solution: Create a FileUtils module with shared functionality

FILE_UTILS_REFACTORING_PLAN = """
1. Create a dedicated module for file operations:
   - src/jfkreveal/utils/file_utils.py

2. Extract common functions:
   - list_files(directory, pattern, recursive=False)
   - ensure_directory_exists(directory)
   - get_file_extension(file_path)
   - is_pdf_file(file_path)
   - get_output_path(input_path, output_dir, extension)
   - clean_filename(filename)

3. Update document_processor.py and text_cleaner.py to use the file utils:
   - Replace inline file operations with calls to file_utils functions
"""

# 4. LLM Prompts
# Problem: Duplicated LLM prompts across findings_report.py
# Solution: Extract prompts to a dedicated module and implement a prompt template system

PROMPT_REFACTORING_PLAN = """
1. Create a prompts directory:
   - src/jfkreveal/prompts/
     - system/
       - detective_role.txt
       - historian_role.txt
     - user/
       - executive_summary.txt
       - detailed_findings.txt
       - suspects_analysis.txt
       - coverup_analysis.txt

2. Create a PromptTemplate class in src/jfkreveal/utils/prompt_template.py:
   - Support loading prompts from files
   - Support variable interpolation
   - Support combining system and user prompts

3. Update findings_report.py to use the PromptTemplate class:
   - Replace inline prompts with template loading and formatting
"""

# 5. Error Handling
# Problem: Similar error handling blocks throughout the code
# Solution: Implement a more structured error handling system

ERROR_HANDLING_REFACTORING_PLAN = """
1. Create a dedicated module for error handling:
   - src/jfkreveal/utils/error_handler.py

2. Define a hierarchy of custom exceptions:
   - JFKRevealError (base class)
     - ScraperError
     - ProcessingError
     - AnalysisError
     - ReportingError

3. Implement error handling decorators:
   - @handle_errors(fallback=None, retry=False)
   - @with_retry(max_retries=3, backoff_factor=2.0)

4. Update main.py and other modules to use the error handling system:
   - Replace try/except blocks with decorated functions
   - Use custom exceptions for specific error conditions
"""

# 6. Report Generation
# Problem: Duplicated code in report generation functions
# Solution: Create a base report generator that specific reports extend

REPORT_GENERATOR_REFACTORING_PLAN = """
1. Create a BaseReportGenerator class in findings_report.py:
   - Implement common functionality for all report types
   - Define abstract methods for report-specific behavior
   - Handle common error cases and LLM interactions

2. Refactor specific report generators to extend BaseReportGenerator:
   - ExecutiveSummaryGenerator
   - DetailedFindingsGenerator
   - SuspectsAnalysisGenerator
   - CoverupAnalysisGenerator

3. Implement a ReportFactory to create appropriate generator instances:
   - create_report_generator(report_type, config)
"""

# 7. Progress Tracking
# Problem: Duplicated progress tracking code in main.py
# Solution: Create a unified progress tracking module

PROGRESS_TRACKING_REFACTORING_PLAN = """
1. Create a dedicated module for progress tracking:
   - src/jfkreveal/utils/progress_tracker.py

2. Implement a ProgressTracker class:
   - Support both CLI and silent modes
   - Support nested progress tracking
   - Support colorized output
   - Support timing information

3. Update main.py to use the ProgressTracker class:
   - Replace print_colored and show_progress functions with ProgressTracker
"""

# 8. Dependency Management
# Problem: Duplicated dependency registration logic
# Solution: Enhance the existing container implementation

DEPENDENCY_MANAGEMENT_REFACTORING_PLAN = """
1. Enhance the container implementation:
   - Add support for dependency profiles (e.g., production, testing)
   - Add support for configuration-based registration
   - Add support for automatic detection of circular dependencies

2. Create a dedicated module for dependency registration:
   - src/jfkreveal/config/dependencies.py

3. Update main.py to use the enhanced container:
   - Replace inline registration with a call to register_dependencies
"""

# Implementation Strategy
IMPLEMENTATION_STRATEGY = """
Recommended implementation order:

1. File Utils (easiest, high impact)
2. Error Handling (foundation for other improvements)
3. Progress Tracking (improves user experience)
4. Template Management (improves maintainability)
5. Prompt Refactoring (simplifies LLM interactions)
6. Report Generator (builds on templates and prompts)
7. Argument Parsing (improves CLI experience)
8. Dependency Management (most complex, final step)

For each refactoring:
1. Create new module/files
2. Implement the shared functionality
3. Update one module to use the new approach
4. Run tests to verify behavior
5. Update remaining modules
6. Run full test suite
7. Update documentation
"""

"""
This module serves as a guide for refactoring work to address code duplication.
Each section identifies a specific area of duplication and provides a plan for
consolidating the code into more maintainable components.

The implementation strategy suggests an order based on complexity and impact.
"""