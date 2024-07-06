# iText2KG: Incremental Knowledge Graphs Construction Using Large Language Models

## Overview

iText2KG is a Python package designed to construct knowledge graphs incrementally by leveraging large language models for entity and relation extraction from text documents. It features zero-shot capability, allowing for knowledge extraction across various domains without specific training. The package includes modules for document distillation, entity extraction, and relation extraction, ensuring resolved and unique entities and relationships. It continuously updates the KG with new documents and integrates them into Neo4j for visual representation

## Installation

To install iText2KG, ensure you have Python installed, then use pip to install
```bash
pip install itext2kg
```

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

This version is when you want to extract

```python
# Initialize iText2KG with the OpenAI API key.
itext2kg = iText2KG(openai_api_key=OPENAI_API_KEY)

# Format the distilled document into semantic sections.
semantic_blocks = [f"{key} - {value}".replace("{", "[").replace("}", "]") for key, value in distilled_doc]

# Build the knowledge graph using the semantic sections.
global_ent, global_rel = itext2kg.build_graph(sections=semantic_blocks)

```



## Citation

```md
@inproceedings{Lairgi2024,
  author    = {Yassir Lairgi and
               Ludovic Moncla and
               R{\'{e}}my Cazabet and
               Khalid Benabdeslem and
               Pierre Cl{\'{e}}au},
  title     = {iText2KG: Incremental Knowledge Graphs Construction Using Large Language Models},
  booktitle = {WISE Conference},
  year      = {2024},
}
```