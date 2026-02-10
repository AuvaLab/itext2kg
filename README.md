# iText2KG for Biomedical Literature

A customized version of [iText2KG](https://github.com/auvalab/itext2kg) optimized for biomedical literature knowledge graph construction from PubTator-formatted documents.

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Examples](#usage-examples)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Development](#development)
- [FAQ](#faq)

## Overview

This project extends the original iText2KG framework with specialized support for biomedical literature processing. It seamlessly integrates with **PubTator** format, enabling automatic extraction of biomedical entities (genes, diseases, chemicals, mutations, etc.) and their relationships from scientific publications.

**Key Enhancements:**

- 🔬 **PubTator Integration**: Native support for PubTator-annotated biomedical entities
- 🌐 **Domain Agnostic**: Works with any biomedical domain (oncology, neurology, cardiology, etc.)
- 🚀 **Scalable Processing**: Multi-process batch processing for large literature corpora
- 🔗 **Entity Resolution**: Precise entity matching using unique identifiers (MESH, Gene IDs, etc.)
- 📊 **Neo4j Visualization**: Interactive knowledge graph exploration
- 🔄 **Incremental Updates**: Continuously expand existing knowledge graphs

## Key Features

### ✨ Core Capabilities

- **PubTator Parser**: Automatic extraction of pre-annotated biomedical entities from PubTator files
- **Incremental Graph Construction**: Add new publications to existing knowledge graphs without reprocessing
- **Entity Deduplication**: Similarity-based entity merging (default threshold: 0.9)
- **LLM-Powered Relation Extraction**: Extract semantic relationships between entities using large language models
- **Source Provenance**: Track every relationship back to its source publication (PMID)
- **Multi-Model Support**: Compatible with Ollama, OpenAI, and other LangChain-supported LLMs

### 🛠️ Technology Stack

- **LLM**: DeepSeek-R1, GPT-4, Llama, or any LangChain-compatible model
- **Embeddings**: nomic-embed-text, OpenAI embeddings, or custom models
- **Database**: Neo4j for graph storage and visualization
- **Input Format**: PubTator (standard biomedical entity annotation format)

## Installation

### Prerequisites

- Python 3.9 or higher
- Neo4j (optional, for visualization)
- Ollama (optional, for local LLM inference)

### Basic Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd itext2kg

# Install dependencies
pip install -r requirements.txt

# Or install in development mode
pip install -e .
```

### Install Ollama (Recommended for Local Inference)

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull models
ollama pull deepseek-r1:32b
ollama pull nomic-embed-text
```

## Quick Start

### 1. Single Document Processing

```python
from langchain_ollama import ChatOllama, OllamaEmbeddings
from itext2kg.utils import PubtatorProcessor
from itext2kg import iText2KG
import pickle

# Initialize models
llm = ChatOllama(
    model="deepseek-r1:32b",
    temperature=0,
)

embeddings = OllamaEmbeddings(
    model="nomic-embed-text:latest",
)

# Process PubTator file
pubtator_file = "Data/pubmed/12345678.txt"
pubtator_processor = PubtatorProcessor(pubtator_file, llm)

# Extract semantic blocks and entity information
semantic_blocks = pubtator_processor.block
properties_info = pubtator_processor.properties_info
pubtator_info = pubtator_processor.pubtator_info

# Add abstract context
pubtator_info['abstract'] = {
    'context': semantic_blocks[-1],
    'source': properties_info['source']
}

# Build knowledge graph
itext2kg = iText2KG(llm_model=llm, embeddings_model=embeddings)
kg = itext2kg.build_graph(
    sections=[semantic_blocks],
    source=properties_info,
    entities_info=pubtator_info,
    ent_threshold=0.9,  # Entity similarity threshold
    rel_threshold=0.4   # Relationship similarity threshold
)

# Save results
with open('output_kg/12345678.pkl', 'wb') as f:
    pickle.dump(kg, f)

print(f"Extracted {len(kg.entities)} entities and {len(kg.relationships)} relationships")
```

### 2. Batch Processing with Multiprocessing

```python
import os
import logging
from multiprocessing import Pool
from functools import partial
from tqdm import tqdm

# Configuration
DATA_PATH = "Data/pubmed_articles"
OUTPUT_PATH = "output_kg/pubmed"
NUM_WORKERS = 20

# Collect PMIDs to process
pmid_list = []
for file_name in os.listdir(DATA_PATH):
    if file_name.endswith('.txt'):
        pmid = file_name.split('.')[0]
        output_file = f"{OUTPUT_PATH}/{pmid}.pkl"

        # Skip already processed files
        if not os.path.exists(output_file):
            pmid_list.append(pmid)

print(f"Processing {len(pmid_list)} documents...")

# Define processing function
def process_pmid(pmid, data_path, output_path):
    # Your processing logic here
    # See abstract2KG.py or AD2KG.py for complete implementation
    pass

# Process in parallel
with Pool(NUM_WORKERS) as pool:
    list(tqdm(pool.imap(
        partial(process_pmid, data_path=DATA_PATH, output_path=OUTPUT_PATH),
        pmid_list
    ), total=len(pmid_list)))
```

### 3. Visualize in Neo4j

```python
from itext2kg.graph_integration import GraphIntegrator
import pickle

# Neo4j connection
URI = "bolt://localhost:7687"
USERNAME = "neo4j"
PASSWORD = "your-password"

# Load knowledge graph
with open('output_kg/12345678.pkl', 'rb') as f:
    kg = pickle.load(f)

# Visualize
integrator = GraphIntegrator(
    uri=URI,
    username=USERNAME,
    password=PASSWORD
)

integrator.visualize_graph(knowledge_graph=kg)
```

## Usage Examples

### PubTator File Format

PubTator is a standard format for biomedical entity annotations:

```
12345678|t|Title of the paper
12345678|a|Abstract text here...
12345678	0	10	Alzheimer	Disease	MESH:D000544
12345678	20	25	tau	Gene	4137
12345678	30	40	amyloid	Chemical	MESH:D000682
```

**Format Explanation:**
- Line 1: `PMID|t|Title`
- Line 2: `PMID|a|Abstract`
- Subsequent lines: `PMID [TAB] start [TAB] end [TAB] mention [TAB] type [TAB] identifier`

### Example Scripts

#### abstract2KG.py - Single Document Processing

Process one PubTator file at a time. Useful for testing or small-scale processing.

```bash
python abstract2KG.py
```

#### AD2KG.py - Batch Processing Example (Alzheimer's Disease)

Demonstrates batch processing with filtering and multiprocessing. Can be adapted for any disease domain.

**Features:**
- Multi-process parallel processing (configurable workers)
- Automatic skip of processed files
- Custom filtering logic (easily adaptable)
- Comprehensive error handling and logging

```bash
python AD2KG.py
```

**Adapting for Your Domain:**
1. Change `DATA_PATH` to your PubTator files directory
2. Modify filtering criteria (e.g., keyword matching)
3. Adjust `NUM_WORKERS` based on your system resources

## Project Structure

```
itext2kg/
├── itext2kg/                      # Core package
│   ├── __init__.py
│   ├── itext2kg.py               # Main KG construction logic
│   ├── documents_distiller/      # Document processing
│   ├── ientities_extraction/     # Entity extraction
│   ├── irelations_extraction/    # Relation extraction
│   ├── graph_integration/        # Neo4j integration
│   │   └── graph_integrator.py
│   ├── models/                   # Data models
│   │   ├── entity.py
│   │   ├── relation.py
│   │   └── knowledge_graph.py
│   └── utils/                    # Utility functions
│       ├── pubtator_processor.py # PubTator parser
│       ├── matcher.py            # Entity matching
│       └── schemas.py            # Data schemas
├── Data/                         # Data directory (not tracked)
│   └── pubmed_articles/          # Your PubTator files
├── output_kg/                    # Output directory (not tracked)
│   └── pubmed/                   # Generated knowledge graphs
├── tests/                        # Unit tests
├── examples/                     # Example notebooks and scripts
├── docs/                         # Documentation
│   └── reviewer_response/        # Academic materials
├── abstract2KG.py               # Single document processing
├── AD2KG.py                     # Batch processing example
├── setup.py                     # Installation script
├── requirements.txt             # Dependencies
├── CHANGELOG.md                 # Version history
└── README.md                    # This file
```

## Configuration

### Model Configuration

#### Using Ollama (Recommended for Local Deployment)

```python
from langchain_ollama import ChatOllama, OllamaEmbeddings

llm = ChatOllama(
    model="deepseek-r1:32b",  # or llama3, mistral, etc.
    temperature=0,
)

embeddings = OllamaEmbeddings(
    model="nomic-embed-text:latest",
)
```

#### Using OpenAI

```python
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

llm = ChatOpenAI(
    api_key="your-api-key",
    model="gpt-4o",
    temperature=0,
)

embeddings = OpenAIEmbeddings(
    api_key="your-api-key",
    model="text-embedding-3-large",
)
```

#### Using Local OpenAI-Compatible API

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    api_key="EMPTY",
    base_url="http://localhost:8102/v1",
    model="path/to/your/model",
    temperature=0,
)
```

### Parameter Tuning

#### Entity Matching Threshold (`ent_threshold`)
- **Default**: 0.9
- **Description**: Controls entity deduplication strictness
- **Recommendations**:
  - 0.9-1.0: Strict matching, reduces false merges
  - 0.7-0.9: Moderate matching
  - <0.7: Loose matching, may increase false positives

#### Relationship Matching Threshold (`rel_threshold`)
- **Default**: 0.4
- **Description**: Controls relationship deduplication
- **Recommendations**:
  - 0.4-0.6: Suitable for most scenarios
  - <0.4: More aggressive merging

#### Entity Embedding Weights

```python
kg = itext2kg.build_graph(
    sections=sections,
    entity_name_weight=0.6,   # Weight for entity name
    entity_label_weight=0.4,  # Weight for entity type/label
)
```

This prevents false merges like "Python (programming language)" and "Python (snake)".

## Development

### Running Tests

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_matcher.py

# Run with coverage
pytest --cov=itext2kg --cov-report=html
```

### Code Formatting

```bash
# Format code with black
black itext2kg/

# Sort imports
isort itext2kg/

# Lint code
flake8 itext2kg/
```

### Creating Domain-Specific Processing Scripts

1. Copy `abstract2KG.py` as a template
2. Modify data paths and output paths
3. Customize filtering logic (e.g., disease keywords, journal filters)
4. Add domain-specific entity type handling if needed

Example filtering for cancer research:

```python
# Filter for cancer-related papers
if any(keyword in abstract.lower() for keyword in ['cancer', 'tumor', 'oncology', 'carcinoma']):
    pmid_list.append(pmid)
```

## FAQ

### Q: What biomedical domains are supported?
**A**: All domains! As long as you have PubTator-formatted files, you can process literature from any biomedical field (oncology, cardiology, immunology, etc.).

### Q: How do I get PubTator files?
**A**:
- Use [PubTator Central](https://www.ncbi.nlm.nih.gov/research/pubtator/) API
- Download pre-annotated datasets from PubTator FTP
- Use PubTator3 for custom annotations

### Q: How to handle large literature corpora?
**A**: Use the multiprocessing approach in `AD2KG.py`. Adjust `NUM_WORKERS` based on your CPU cores and memory.

### Q: How to avoid reprocessing?
**A**: The scripts automatically check for existing `.pkl` files in the output directory and skip them.

### Q: How to view the knowledge graph?
**A**:
- Use Neo4j for interactive visualization
- Load `.pkl` files and inspect entities/relationships programmatically
- Export to other graph formats (GraphML, JSON, etc.)

### Q: Memory issues with large batches?
**A**:
- Reduce `NUM_WORKERS`
- Process in smaller batches
- Use a machine with more RAM

### Q: How to merge multiple knowledge graphs?
**A**: Use the `existing_knowledge_graph` parameter:

```python
kg2 = itext2kg.build_graph(
    sections=new_sections,
    existing_knowledge_graph=kg1
)
```

### Q: Can I use custom entity types?
**A**: Yes! The PubTator processor handles any entity types in your PubTator files (Disease, Gene, Chemical, Species, Mutation, CellLine, etc.).

## License

This project is licensed under the MIT License. Original iText2KG copyright belongs to Auvalab.

## Acknowledgments

- **Original Project**: [iText2KG](https://github.com/auvalab/itext2kg) by Auvalab
- **Paper**: Lairgi, Y., et al. (2024). iText2KG: Incremental Knowledge Graphs Construction Using Large Language Models. arXiv:2409.03284
- **Conference**: Accepted at WISE 2024

## Citation

If you use this tool in your research, please cite the original iText2KG paper:

```bibtex
@article{lairgi2024itext2kg,
  title={iText2KG: Incremental Knowledge Graphs Construction Using Large Language Models},
  author={Lairgi, Yassir and Moncla, Ludovic and Cazabet, R{\'e}my and Benabdeslem, Khalid and Cl{\'e}au, Pierre},
  journal={arXiv preprint arXiv:2409.03284},
  year={2024},
  note={Accepted at WISE 2024}
}
```

## Contact

For questions or suggestions, please open an issue on GitHub.

---

⭐ **Star this repository if you find it useful for your biomedical research!**
