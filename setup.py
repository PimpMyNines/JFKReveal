"""
Setup script for JFKReveal.
"""
import os
from setuptools import setup, find_packages

# Read the contents of the README file
with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

# Read requirements.txt
with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="jfkreveal",
    version="0.1.0",
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
            "pytest>=7.0.0",
            "black>=23.0.0",
            "isort>=5.0.0",
            "mypy>=1.0.0",
            "flake8>=6.0.0",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)

# Display a message about installing the spaCy model
print("\n==========================================================")
print("IMPORTANT: After installation, you'll need to download the")
print("spaCy language model with this command:")
print("python -m spacy download en_core_web_sm")
print("==========================================================\n")