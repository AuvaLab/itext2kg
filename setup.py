"""Setup script for iText2KG package."""

from setuptools import setup, find_packages
from pathlib import Path

# Read the contents of README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

setup(
    name="itext2kg",
    version="0.0.8",
    author="Auvalab - Yassir LAIRGI",
    author_email="yassir.lairgi@auvalie.com",
    description="Incremental Knowledge Graphs Construction Using Large Language Models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/auvalab/itext2kg",
    project_urls={
        "Bug Tracker": "https://github.com/auvalab/itext2kg/issues",
        "Documentation": "https://github.com/auvalab/itext2kg",
        "Source Code": "https://github.com/auvalab/itext2kg",
        "Paper": "https://arxiv.org/abs/2409.03284",
    },
    packages=find_packages(exclude=["tests", "tests.*", "examples", "docs", "Data", "datasets"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    install_requires=[
        "langchain>=0.3.0",
        "langchain-core>=0.3.0",
        "langchain-openai>=0.2.0",
        "neo4j>=5.24.0",
        "numpy>=1.24.0",
        "openai>=1.45.0",
        "openpyxl>=3.1.5",
        "pydantic>=2.0.0",
        "pydantic-settings>=2.5.2",
        "scikit-learn>=1.5.2",
        "pypdf>=4.3.1",
    ],
    extras_require={
        "dev": [
            "pytest>=8.2.2",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
            "isort>=5.12.0",
        ],
        "docs": [
            "sphinx>=7.0.0",
            "sphinx-rtd-theme>=1.3.0",
        ],
        "ollama": [
            "langchain-ollama>=0.1.0",
        ],
        "mistral": [
            "langchain-mistralai>=0.1.0",
        ],
    },
    keywords=[
        "knowledge graph",
        "knowledge graph construction",
        "large language models",
        "llm",
        "neo4j",
        "graph database",
        "entity extraction",
        "relation extraction",
        "incremental learning",
        "nlp",
        "natural language processing",
    ],
    include_package_data=True,
    zip_safe=False,
)
