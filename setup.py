"""
Setup script for JFKReveal.
"""
from setuptools import setup, find_packages

# Read requirements
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

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