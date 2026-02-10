# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added
- PubTator format support for biomedical literature processing
- Multi-process batch processing for large-scale literature analysis
- Disease-specific processing scripts (AD, Delirium)
- Source tracking for all extracted relationships (PMID-based)
- Enhanced entity resolution with unique ID matching (MESH, Gene IDs)
- Support for DeepSeek-R1 and other local LLM models
- Comprehensive documentation in Chinese

### Changed
- Optimized for biomedical literature (neurodegenerative diseases)
- Updated entity matching thresholds (0.9 for entities, 0.4 for relations)
- Improved error handling and logging in batch processing
- Restructured project for better organization

### Fixed
- Memory management in multi-process scenarios
- File existence checks before processing
- Error handling for malformed PubTator files

## [0.0.8] - 2024-XX-XX

Based on original iText2KG v0.0.7 with biomedical customizations.

### Core Features from Original iText2KG
- Incremental knowledge graph construction
- LLM-based entity and relation extraction
- Neo4j integration for visualization
- Support for multiple LLM providers (OpenAI, Mistral, Ollama)
- Entity deduplication using cosine similarity
- Configurable entity name/label weights

## Original iText2KG Credits

This project is based on [iText2KG](https://github.com/auvalab/itext2kg) by Auvalab.

**Original Paper**:
```
Lairgi, Y., Moncla, L., Cazabet, R., Benabdeslem, K., & Cléau, P. (2024).
iText2KG: Incremental Knowledge Graphs Construction Using Large Language Models.
arXiv preprint arXiv:2409.03284.
```

**Accepted at**: WISE 2024 (The International Web Information Systems Engineering conference)
