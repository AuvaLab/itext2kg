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

Example

```python
from itext2kg import DocumentDistiller
#You can define a schema or upload some predefined ones
from itext2kg.utils import Article


# Define your OpenAI api key
OPENAI_API_KEY = "####"

document_distiller = DocumentsDisiller(openai_api_key=OPENAI_API_KEY)


documents = ["doc1", "doc2","doc3"]

IE_query = '''
# DIRECTIVES : 
- Act like an experienced information extractor. 
- You have a chunk of a scientific paper.
- If you do not find the right information, keep its place empty.
'''

output_file = document_distiller.distill(documents= documents, IE_query=IE_query, output_data_structure=Article)
```

You can define a custom schema using  ```pydantic_v1``` of ```langchain_core```. 

```python
from typing import List, Optional
from langchain_core.pydantic_v1 import BaseModel, Field

class Author(BaseModel):
    name : str=Field(description="The name of the author")
    affiliation: str = Field(description="The affiliation of the author")
    
class Article(BaseModel):
    title : str = Field(description="The title of the scientific article")
    authors : List[Author] = Field(description="The list of the article's authors and their affiliation")
    abstract:str = Field(description="The article's abstract")
    key_findings:str = Field(description="The key findings of the article")
    limitation_of_sota : str=Field(description="limitation of the existing work")
    proposed_solution : str = Field(description="the proposed solution in details")
    paper_limitations : str=Field(description="The limitations of the proposed solution of the paper")

```

