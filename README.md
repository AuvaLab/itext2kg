# iText2KG: Incremental Knowledge Graphs Construction Using Large Language Models

## Overview

iText2KG is a Python package designed to construct knowledge graphs incrementally by leveraging large language models for entity and relation extraction from text documents. It features zero-shot capability, allowing for knowledge extraction across various domains without specific training. The package includes modules for document distillation, entity extraction, and relation extraction, ensuring resolved and unique entities and relationships. It continuously updates the KG with new documents and integrates them into Neo4j for visual representation

## Installation

To install iText2KG, ensure you have Python installed, then use pip to install
```bash
pip install itext2kg
```
## The Overall Architecture

The ```iText2KG``` package consists of four main modules that work together to construct and visualize knowledge graphs from unstructured text. An overview of the overall architecture:

1. **Document Distiller**: This module processes raw documents and reformulates them into semantic blocks based on a user-defined schema. It improves the signal-to-noise ratio by focusing on relevant information and structuring it in a predefined format.

2. **Incremental Entity Extractor**: This module extracts unique entities from the semantic blocks and resolves ambiguities to ensure each entity is clearly defined. It uses cosine similarity measures to match local entities with global entities.

3. **Incremental Relation Extractor**: This module identifies relationships between the extracted entities. It can operate in two modes: using global entities to enrich the graph with potential information or using local entities for more precise relationships.

4. **Graph Integrator and Visualization**: This module integrates the extracted entities and relationships into a Neo4j database, providing a visual representation of the knowledge graph. It allows for interactive exploration and analysis of the structured data.

Each module performs a distinct role in the pipeline, from initial document processing to final graph visualization, ensuring that the knowledge graph is accurate, comprehensive, and easy to analyze.

![itext2kg](./docs/itext2kg.png)


## Modules and Examples

The Document Distiller module reformulates raw documents into predefined and semantic blocks using LLMs. It utilizes a schema to guide the extraction of specific information from each document.

### The ```DocumentDistiller```
Example

```python
from itext2kg import DocumentDistiller
# You can define a schema or upload some predefined ones.
from itext2kg.utils import Article

# Define your OpenAI API key.
OPENAI_API_KEY = "####"

# Initialize the DocumentDistiller with the OpenAI API key.
document_distiller = DocumentDistiller(openai_api_key=OPENAI_API_KEY)

# List of documents to be distilled.
documents = ["doc1", "doc2", "doc3"]

# Information extraction query.
IE_query = '''
# DIRECTIVES : 
- Act like an experienced information extractor. 
- You have a chunk of a scientific paper.
- If you do not find the right information, keep its place empty.
'''

# Distill the documents using the defined query and output data structure.
distilled_doc = document_distiller.distill(documents=documents, IE_query=IE_query, output_data_structure=Article)

```

You can define a custom schema using  ```pydantic_v1``` of ```langchain_core```. 

```python
from typing import List, Optional
from langchain_core.pydantic_v1 import BaseModel, Field

# Define an Author model with name and affiliation fields.
class Author(BaseModel):
    name: str = Field(description="The name of the author")
    affiliation: str = Field(description="The affiliation of the author")
    
# Define an Article model with various fields describing a scientific article.
class Article(BaseModel):
    title: str = Field(description="The title of the scientific article")
    authors: List[Author] = Field(description="The list of the article's authors and their affiliation")
    abstract: str = Field(description="The article's abstract")
    key_findings: str = Field(description="The key findings of the article")
    limitation_of_sota: str = Field(description="limitation of the existing work")
    proposed_solution: str = Field(description="The proposed solution in details")
    paper_limitations: str = Field(description="The limitations of the proposed solution of the paper")

```

### The ```iText2KG```
The iText2KG module is the core component of the package, responsible for integrating various functionalities to construct the knowledge graph. It uses the distilled semantic sections from documents to extract entities and relationships, and then builds the knowledge graph incrementally. 

```python
# Initialize iText2KG with the OpenAI API key.
itext2kg = iText2KG(openai_api_key=OPENAI_API_KEY)

# Format the distilled document into semantic sections.
semantic_blocks = [f"{key} - {value}".replace("{", "[").replace("}", "]") for key, value in distilled_doc]

# Build the knowledge graph using the semantic sections.
global_ent, global_rel = itext2kg.build_graph(sections=semantic_blocks)

```

The Arguments of ```iText2KG```:
- `openai_api_key` (str): The API key for accessing OpenAI services. This key is used to authenticate and make requests to OpenAI's language models for entity and relation extraction.

- `embeddings_model_name` (str, optional): The name of the embeddings model to be used for generating text embeddings. Default is `"text-embedding-3-large"`.

- `model_name` (str, optional): The name of the language model to be used for entity and relation extraction. Default is `"gpt-4-turbo"`.

- `temperature` (float, optional): The temperature setting for the language model, controlling the randomness of the output. Default is `0`.

- `sleep_time` (int, optional): The sleep time between API requests to avoid rate limiting. Default is `5` seconds.



## The ```GraphIntegrator```
It integrates the extracted entities and relationships into a Neo4j graph database and provides a visualization of the knowledge graph. This module allows users to easily explore and analyze the structured data using Neo4j's graph capabilities.

```python
from itext2kg.graph_integration import GraphIntegrator

URI = "bolt://localhost:####"
USERNAME = "####"
PASSWORD = "####"


new_graph = {}
new_graph["nodes"] = global_ent
new_graph["relationships"] = global_rel

GraphIntegrator(uri=URI, username=USERNAME, password=PASSWORD).visualize_graph(json_graph=new_graph)
```

## Some ```iText2KG``` use-cases
![text2kg](./docs/text_2_kg.png)

## Public Collaboration
We welcome contributions from the community to improve iText2KG.