# Project Summary

## Overview

This is a customized version of **iText2KG** optimized for biomedical literature knowledge graph construction from PubTator-formatted documents.

## Key Modifications

### 1. PubTator Integration
- Added `PubtatorProcessor` in `itext2kg/utils/` for parsing PubTator format
- Automatic extraction of pre-annotated biomedical entities (genes, diseases, chemicals, etc.)
- Support for entity unique identifiers (MESH IDs, Gene IDs)

### 2. Batch Processing
- Multi-process support for large-scale literature processing
- Example scripts: `AD2KG.py`, `abstract2KG.py`
- Automatic skip of already processed files

### 3. Enhanced Entity Matching
- Modified `itext2kg/utils/matcher.py` for better biomedical entity resolution
- Support for unique ID-based exact matching
- Configurable similarity thresholds (default: 0.9 for entities, 0.4 for relations)

### 4. Graph Integration
- Updated `itext2kg/graph_integration/graph_integrator.py`
- Neo4j visualization support
- Source provenance tracking (PMID-based)

## File Structure

```
itext2kg/
├── itext2kg/                      # Core package
│   ├── itext2kg.py               # Main KG construction
│   ├── utils/
│   │   ├── pubtator_processor.py # PubTator parser (NEW)
│   │   └── matcher.py            # Enhanced matching (MODIFIED)
│   └── graph_integration/
│       └── graph_integrator.py   # Neo4j integration (MODIFIED)
├── abstract2KG.py                # Single document processing
├── AD2KG.py                      # Batch processing example
├── setup.py                      # Package setup
├── requirements.txt              # Dependencies
├── README.md                     # Documentation
└── CHANGELOG.md                  # Version history
```

## Usage

### Single Document
```bash
python abstract2KG.py
```

### Batch Processing
```bash
python AD2KG.py
```

### Custom Domain
Modify `AD2KG.py`:
1. Change `DATA_PATH` to your PubTator directory
2. Update filtering logic (keywords, criteria)
3. Adjust `NUM_WORKERS` for your system

## Dependencies

- Python 3.9+
- LangChain (0.3.0+)
- Neo4j (5.24.0+)
- Ollama (for local LLM)
- See `requirements.txt` for full list

## Output

- `.pkl` files containing `KnowledgeGraph` objects
- Each graph includes:
  - Entities with unique IDs and embeddings
  - Relationships with source tracking
  - Metadata (PMID, publication info)

## Next Steps

1. Process your PubTator files
2. Visualize in Neo4j
3. Analyze the knowledge graph
4. Expand with new literature

## Credits

Based on [iText2KG](https://github.com/auvalab/itext2kg) by Auvalab (WISE 2024).
