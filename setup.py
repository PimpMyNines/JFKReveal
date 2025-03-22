"""
Setup script for JFKReveal.
"""
import os
from setuptools import setup, find_packages

# Read the contents of the README file
with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

# Read requirements.txt if it exists, otherwise use minimal requirements
try:
    with open("requirements.txt", "r", encoding="utf-8") as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]
except FileNotFoundError:
    # Minimal requirements for the package to function
    requirements = [
        "langchain>=0.1.0",
        "langchain-openai>=0.0.2",
        "langchain-chroma>=0.0.1",
        "openai>=1.3.0",
        "pymupdf>=1.22.5",
        "pydantic>=2.5.0",
        "chromadb>=0.4.18",
        "tqdm>=4.66.1",
        "spacy>=3.7.2",
        "pytesseract>=0.3.10",
        "pillow>=10.0.0",
        "python-dotenv>=1.0.0",
        "requests>=2.31.0",
        "beautifulsoup4>=4.12.2",
        "backoff>=2.2.1",
        "nltk>=3.8.1",
        "markdown>=3.4.4",
        "colorama>=0.4.6",
        "tabulate>=0.9.0",
    ]

setup(
    name="jfkreveal",
    version="0.2.0",
    author="PimpMyNines",
    author_email="info@pimpmynines.com",
    description="Analysis tool for declassified JFK assassination documents",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/PimpMyNines/JFKReveal",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "jfkreveal=jfkreveal.__main__:main",
        ],
    },
    extras_require={
        "dev": [
            "pytest>=7.4.3",
            "pytest-cov>=4.1.0",
            "pytest-benchmark>=4.0.0",
            "responses>=0.25.0",
            "black>=23.0.0",
            "isort>=5.0.0",
            "mypy>=1.0.0",
            "flake8>=6.0.0",
        ],
        "visualization": [
            "plotly>=5.18.0",
            "dash>=2.14.1",
            "pandas>=2.1.1",
            "networkx>=3.1",
        ],
        "search": [
            "rank-bm25>=0.2.2",
            "scikit-learn>=1.3.0",
            "sentence-transformers>=2.2.2",
        ],
        "local": [
            "langchain-ollama>=0.0.1",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)

# Display a message about installing the spaCy model and other optional components
print("\n==========================================================")
print("IMPORTANT: After installation, you'll need to:")
print("")
print("1. Download the spaCy language model:")
print("   python -m spacy download en_core_web_sm")
print("")
print("2. For local model support, install Ollama:")
print("   make setup-ollama")
print("")
print("3. For visualization features:")
print("   pip install jfkreveal[visualization]")
print("")
print("4. For enhanced search features:")
print("   pip install jfkreveal[search]")
print("")
print("5. For full development environment:")
print("   pip install jfkreveal[dev]")
print("==========================================================\n")