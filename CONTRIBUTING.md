# Contributing to JFKReveal

Thank you for your interest in contributing to JFKReveal! This document provides guidelines and instructions for contributing to this project.

## Code of Conduct

By participating in this project, you agree to abide by our Code of Conduct. Please read it carefully.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR-USERNAME/JFKReveal.git`
3. Create a new branch for your changes: `git checkout -b feature/your-feature-name`

## Development Setup

1. Create a virtual environment: `python -m venv venv`
2. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - macOS/Linux: `source venv/bin/activate`
3. Install dependencies: `make setup`
4. Install development dependencies: `make install-dev`

## Environment Variables

Create a `.env` file in the root directory and add your OpenAI API key:

```
OPENAI_API_KEY=your_key_here
OPENAI_EMBEDDING_MODEL=text-embedding-3-large
OPENAI_ANALYSIS_MODEL=gpt-4.5-preview
```

## Pull Request Process

1. Update the README.md with details of changes if applicable
2. Update the requirements.txt file if you've added new dependencies
3. Make sure your code follows the existing code style
4. Make sure all tests pass
5. Create a pull request with a clear title and description

## Commit Messages

Please use clear, descriptive commit messages that explain the changes you've made.

## Testing

All new features should include appropriate tests. Run tests using:

```
pytest tests/
```

## Code Review

All submissions require review. We use GitHub pull requests for this purpose.

## License

By contributing to this project, you agree that your contributions will be licensed under the project's MIT License.