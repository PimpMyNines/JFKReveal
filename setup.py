"""
Setup script for JFKReveal.
"""
from setuptools import setup, find_packages
import os

# Read requirements from the same directory as setup.py
requirements_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
try:
    with open(requirements_path) as f:
        requirements = f.read().splitlines()
except FileNotFoundError:
    # Fallback for build environment
    requirements = [
        # Web scraping
        "requests>=2.31.0",
        "beautifulsoup4>=4.12.2",
        "tqdm>=4.66.1",
        "backoff>=2.2.1",
        
        # PDF processing
        "PyMuPDF>=1.22.5",
        "nltk>=3.8.1",
        
        # Vector database and LLMs
        "langchain>=0.1.0",
        "langchain-openai>=0.0.2",
        "langchain-chroma>=0.0.1",
        "chromadb>=0.4.18",
        "pydantic>=2.5.0",
        "tenacity>=8.2.3",
        "openai>=1.3.0",
        "python-dotenv>=1.0.0",
        
        # Report generation
        "markdown>=3.4.4",
        "plotly>=5.18.0",
        "pandas>=2.1.1"
    ]

setup(
    name="jfkreveal",
    version="0.1.0",
    description="JFK Declassified Documents Analysis Tool",
    author="JFKReveal Team",
    author_email="info@jfkreveal.org",
    url="https://github.com/jfkreveal/jfkreveal",
    packages=find_packages("src"),
    package_dir={"": "src"},
    include_package_data=True,
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "jfkreveal=jfkreveal.main:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
)