# ATOM: AdapTive and OptiMized Dynamic Temporal Knowledge Graph Construction Using LLMs

iText2KG is now ATOM. ATOM is a few-shot and scalable approach for building and continuously updating Temporal Knowledge Graphs (TKGs) from unstructured texts. 
(We kept the legacy iText2KG in the repository, please check ![README](./README_itext2kg.md).)


<p align="center">
  <img src="./docs/banner-atom.png" width="851px" alt="ATOM Banner">
</p>

![GitHub stars](https://img.shields.io/github/stars/auvalab/itext2kg?style=social)
![GitHub forks](https://img.shields.io/github/forks/auvalab/itext2kg?style=social)
![PyPI](https://img.shields.io/pypi/dm/itext2kg)
![Total Downloads](https://img.shields.io/pepy/dt/itext2kg)
[![Paper](https://img.shields.io/badge/Paper-View-green?style=flat&logo=adobeacrobatreader)]()
![PyPI](https://img.shields.io/pypi/v/itext2kg)
[![Demo](https://img.shields.io/badge/Demo-Available-blue)](./examples/)
![Status](https://img.shields.io/badge/Status-Work%20in%20Progress-yellow)



<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="./docs/logo_atom_white.png" width="300">
    <source media="(prefers-color-scheme: light)" srcset="./docs/logo_atom_black.png" width="300">
    <img alt="Logo" src="./docs/logo_atom_white.png" width="300">
  </picture>
</p>



## Overview
Traditional static KG construction often overlooks the dynamic and time-sensitive nature of real-world data, limiting adaptability to continuous changes. Moreover, recent zero- or few-shot approaches that avoid domain-specific fine-tuning or reliance on prebuilt ontologies often suffer from instability across multiple runs, as well as incomplete coverage of key facts.

To address these challenges, we introduce ATOM (AdapTive and OptiMized), a few-shot and scalable approach that builds and continuously updates Temporal Knowledge Graphs (TKGs) from unstructured texts. ATOM splits input documents into minimal, self-contained “atomic” facts, improving extraction exhaustivity and stability. From these atomic facts, atomic KGs are derived and then merged in parallel. 

In a nutshell, ATOM adresses these limitations by:

- ✅ **Improving exhaustivity**: Capturing comprehensive fact coverage from longer texts (~31% gain on factual exhaustivity, ~18% improvement in temporal exhaustivity)
- ✅ **Ensuring stability**: Producing consistent TKGs across multiple runs (~17% improvement)
- ✅ **Enabling scalability**: Supporting large-scale dynamic temporal updates through parallel architecture

## Key Features

### Atomic Fact Decomposition

ATOM decomposes unstructured text into **atomic facts** - short, self-contained snippets that convey exactly one piece of information. This addresses the "forgetting effect" where LLMs prioritize salient information in longer contexts while omitting key relationships.

### Dual-Time Modeling
ATOM incorporates dual-time modeling, differentiating between:
- **Observation time** (`t_obs`): When facts are observed
- **Validity period** (`t_start`, `t_end`): Temporal information conveyed by the facts themselves

### Parallel Architecture
The framework employs three modules running in parallel:
1. **Module-1**: Atomic Fact Decomposition
2. **Module-2**: Atomic TKGs Construction (parallel 5-tuple extraction)
3. **Module-3**: Parallel Atomic Merge of TKGs and DTKG Update

## 🔥 News
* [20/10/2025] ATOM - Major Enhancements :
  - **Complete Architectural Redesign**: ATOM employs a three-module parallel pipeline that constructs and continuously updates Dynamic Temporal Knowledge Graphs (DTKGs) from unstructured text.
  - **Atomic Fact Decomposition**: Implemented atomic fact decomposition that converts complex unstructured text into short, self-contained atomic facts that convey exactly one piece of information, addressing the "forgetting effect" where LLMs prioritize salient information while omitting key relationships in longer contexts.
  - **Enhanced Exhaustivity and Stability**: The architecture achieved ~31% improvement in factual exhaustivity, ~18% improvement in temporal exhaustivity, ~17% improvement in stability across multiple runs, and ~31% reduction in factual omission compared to direct paragraph extraction methods.
  - **Dual-Time Modeling**: Introduced a temporal modeling that differentiates between observation time (`t_obs`) - when facts are observed - and validity period (`t_start`, `t_end`) - temporal information conveyed by the facts themselves, preventing temporal misattribution in dynamic TKGs (DTKGs).
  - **Parallel 5-Tuple Extraction**: Replaced separate entity and relation extraction with direct parallel extraction of 5-tuples `(subject, predicate, object, t_start, t_end)` from atomic facts, reducing processing time, token consumption, and eliminating invented/isolated entity handling while maintaining higher accuracy.
  - **Parallel Atomic Merge Architecture**: Implemented an efficient parallel merge algorithm that processes atomic TKGs through iterative pairwise merging with configurable thread pools, achieving 93.8% latency reduction vs. Graphiti and 95.3% vs. iText2KG (with 8 threads, batch size of 40 atomic facts).
  - **LLM-Independent Resolution**: Enhanced entity and relation resolution using distance metrics (cosine similarity thresholds: θ_E = 0.8 for entities, θ_R = 0.7 for relations) instead of LLM-based resolution, enabling true parallelization and scalability to millions of nodes without computational bottlenecks.


* [29/07/2025] iText2KG - New Features and Enhanced Capabilities:
  - **iText2KG_Star**: Introduced a simpler and more efficient version of iText2KG that eliminates the separate entity extraction step. Instead of extracting entities and relations separately, iText2KG_Star directly extracts relationships from text, automatically deriving entities from those relationships. This approach is more efficient as it reduces processing time and token consumption and does not need to handle invented/isolated entities.
  - **Facts-Based KG Construction**: Enhanced the framework with facts-based knowledge graph construction using the Document Distiller to extract structured facts from documents, which are then used for incremental KG building. This approach provides more exhaustive and precise knowledge graphs by focusing on factual information extraction.
  - **Dynamic Knowledge Graphs**: iText2KG now supports building dynamic knowledge graphs that evolve over time. By leveraging the incremental nature of the framework and document snapshots with observation dates, users can track how knowledge changes and grows. See example: [Dynamic KG Construction](./examples/building_dynamic_kg_openai_posts.ipynb). **NB: The temporal/logical conflicts resolution is not handled in this version. But you can apply a post processing filter to resolve them**

* [19/07/2025] iText2KG - Major Performance and Reliability Updates:
  - **Asynchronous Architecture**: Complete migration to async/await patterns for all core methods (`build_graph`, `extract_entities`, `extract_relations`, etc.) enabling better performance and non-blocking I/O operations with LLM APIs.
  - **Logging System**: Implemented comprehensive logging infrastructure to replace all print statements with structured, configurable logging (DEBUG, INFO, WARNING, ERROR levels) with timestamps and module identification.
  - **Enhanced Batch Processing**: Improved efficiency through async batch processing for multiple document handling and LLM API calls.
  - **Better Error Handling**: Enhanced error handling and retry mechanisms with proper logging for production environments.

* [07/10/2024] iText2KG - Latest features:
  - The entire iText2KG code has been refactored by adding data models that describe an Entity, a Relation, and a KnowledgeGraph.
  - Each entity is embedded using both its name and label to avoid merging concepts with similar names but different labels. For example, Python:Language and Python:Snake.
    - The weights for entity name embedding and entity label are configurable, with defaults set to 0.4 for the entity label and 0.6 for the entity name.
  - A max_tries parameter has been added to the iText2KG.build_graph function for entity and relation extraction to prevent hallucinatory effects in structuring the output. Additionally, a max_tries_isolated_entities parameter has been added to the same method to handle hallucinatory effects when processing isolated entities.

* [17/09/2024] iText2KG - Latest features: 
  - Now, iText2KG is compatible with all the chat/embeddings models supported by LangChain. For available chat models, refer to the options listed at: https://python.langchain.com/v0.2/docs/integrations/chat/. For embedding models, explore the choices at: https://python.langchain.com/v0.2/docs/integrations/text_embedding/.

  - The constructed graph can be expanded by passing the already extracted entities and relationships as arguments to the `build_graph` function in iText2KG.
  - iText2KG is compatible with all Python versions above 3.9.


* [16/07/2024] iText2KG - We have addressed two major LLM hallucination issues related to KG construction with LLMs when passing the entities list and context to the LLM. These issues are:

  - The LLM might invent entities that do not exist in the provided entities list. We handled this problem by replacing the invented entities with the most similar ones from the input entities list.
  - The LLM might fail to assign a relation to some entities from the input entities list, causing a "forgetting effect." We handled this problem by reprompting the LLM to extract relations for those entities.

## Architecture

ATOM employs a three-module parallel pipeline that constructs and continuously updates Dynamic Temporal Knowledge Graphs (DTKGs) from unstructured text. 

**Module-1 (Atomic Fact Decomposition)** splits input documents `D_t` observed at time `t` into temporal atomic facts `{f_{t,1}, ..., f_{t,m_t}}` using LLM-based prompting with an optimal chunk size of <400 tokens, where each temporal atomic fact is a short, self-contained snippet that conveys exactly one piece of information. 

**Module-2 (Atomic TKGs Construction)** extracts 5-tuples (quintuples) in parallel from each atomic fact `f_{t,i}` to construct atomic temporal KGs `G^t_i = ExtractQuintuplesLLM(f_{t,i}) ⊆ P(E^t × R^t × E^t × T^t_start × T^t_end)`, while embedding nodes and relations and addressing temporal resolution during extraction by transforming end validity facts into affirmative counterparts while modifying only the `t_end` time (e.g., "John Doe is no longer CEO of X on 01-01-2026" → `(John_Doe, is_ceo, X, [.], [01-01-2026])`). 

**Module-3 (Parallel Atomic Merge)** employs a binary merge algorithm to merge pairs of atomic TKGs through iterative pairwise merging in parallel until convergence, with three resolution phases: (1) entity resolution using exact match or cosine similarity threshold `θ_E = 0.8`, (2) relation resolution merging relation names regardless of endpoints and timestamps using threshold `θ_R = 0.7`, and (3) temporal resolution that merges observation and validity time sets for relations with similar `(e_s, r_p, e_o)`. 

The resulting TKG snapshot `G^t_s` is then merged with the previous DTKG `G^{t-1}` using the merge operator ⊕ to yield the updated DTKG: `G^t = G^{t-1} ⊕ G^t_s`. 


<p align="center">
  <img src="./docs/atom_architecture.png" width="800px" alt="ATOM Architecture">
</p>


---
## Example of the ATOM Workflow

 On observation date 09-01-2007, ATOM processes the fact "Steve Jobs was the CEO of Apple Inc. on January 9, 2007" to create the 5-tuple `(Steve Jobs, is_ceo, Apple Inc., [09-01-2007], [.])` where `t_start = [09-01-2007]` and `t_end = [.]` (empty/unknown). Later, on observation date 05-10-2011, when processing the incoming update "Steve Jobs is no longer the CEO of Apple Inc. on 05-10-2011", ATOM's Module-2 transforms this **end validity fact** into its affirmative counterpart while modifying only the `t_end` time, producing `(Steve Jobs, is_ceo, Apple Inc., [.], [05-10-2011])` rather than creating a contradictory relation like `is_no_longer_ceo`. During Module-3's temporal resolution phase, ATOM detects that both 5-tuples share the same `(e_s, r_p, e_o)` triple and merges their time lists to produce the final 5-tuple: `(Steve Jobs, is_ceo, Apple Inc., [09-01-2007], [05-10-2011])`, which correctly represents that Steve Jobs was CEO from January 9, 2007 to October 5, 2011, while maintaining dual-time modeling with `t_obs = [09-01-2007, 05-10-2011]` to track when each piece of information was observed. This preprocessing of end-actions during extraction enables ATOM's LLM-independent merging approach, preventing temporal inconsistencies where separate quintuples describing the same temporal fact would coexist in the TKG.


<p align="center">
  <img src="./docs/example_atom.png" width="800px" alt="ATOM Workflow Diagram">
</p>

For more technical details, check out:
- **`atom/atom.py`**: Core logic for building, merging, and updating the knowledge graphs.
- **`evaluation/`**: Scripts to reconduct the experiments. 

---

## Latency & Scalability

ATOM achieves significant latency reduction of 93.8% compared to Graphiti and 95.3% compared to iText2KG by employing a parallel architecture that addresses computational bottlenecks in traditional approaches. While iText2KG and Graphiti separate entity and relation extraction steps (increasing latency and doubling LLM calls), and use incremental entity/relation resolution that restricts parallel requests (with Graphiti's LLM-based resolution limiting parallelization as the graph scales to millions of nodes), ATOM's architecture facilitates (1) parallel LLM calls for 5-tuple extraction using 8 threads with batch size of 40 atomic facts, (2) parallel merge of atomic TKGs through iterative pairwise merging, (3) LLM-independent merging using distance metrics for entity/relation resolution, and (4) temporal resolution during extraction rather than during merging. Notably, Module-3 (the parallel atomic merge) accounts for only 13% of ATOM's total latency, with the remainder attributed to API calls—which can be further minimized by leveraging the parallel architecture through either increasing the batch size (by upgrading the API tier) or scaling hardware for local LLM deployment

<p align="center">
  <img src="./docs/latency_comparison_plot.png" width="800px" alt="Latency Comparison">
</p>

---

## Example

The following figure demonstrates the difference between ATOM's and Graphiti's temporal modeling using COVID-19 news from 09-01-2020 to 23-01-2020. For the fact "The mysterious respiratory virus spread to at least 10 other countries" observed on 23-01-2020, Graphiti treats the observation time as the validity start time (t_start), setting `valid_at = 23-01-2020` and implying the spread occurred on that specific date. In contrast, ATOM's dual-time modeling preserves the observation time (t_obs = 23-01-2020) separately from the validity period, recognizing that the article was published on 23-01-2020 but this does not guarantee the spread occurred at that exact time—the spread could have happened days or weeks earlier. This distinction is essential for temporal reasoning: Graphiti would infer that all events in a news article happened on the publication date, while ATOM correctly models when information was observed versus when events actually occurred. This prevents temporal misattribution in dynamic knowledge graphs.

<p align="center">
  <img src="./docs/comparison_example.png" width="800px" alt="OpenAI posts DTKG">
</p>



## Installation

1. **Clone or Fork** the repository:
   ```bash
   git clone https://github.com/geeekai/atom.git
   cd atom


2. **Install Requirements**

Install all dependencies by running:

```bash
pip install -r requirements.txt
```

3. **(Optional) Set Up a Virtual Environment**
It is recommended to use a virtual environment (e.g., conda, venv) to isolate dependencies.

# Example: Building a Temporal Knowledge Graph (TKG) with ATOM from LLMS History

In this example, we demonstrate how to use ATOM to extractatomic factsfrom a dataset, build a dynamic Temporal Knowledge Graph (TKG) across different observation timestamps, and finally visualize the graph using Neo4j.

The process involves:
1. **Loading Data**: Reading an Excel file containing LLMS history with associated observation dates.
2. **Factoid Extraction**: Using the `LangchainOutputParser` to extractatomic factsfrom the text.
3. **Graph Construction**: Groupingatomic factsby observation date and building a knowledge graph that merges atomic KGs from different timestamps.
4. **Visualization**: Rendering the final graph using the GraphIntegrator module connected to a Neo4j database.

Below is the derived example code:

---

```python
import pandas as pd
import asyncio
import ast

# Import LLM and Embeddings models using LangChain wrappers
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from atom import Atom, Neo4jStorage

# Set up the OpenAI LLM and embeddings models (replace "##" with your API key)
openai_api_key = "##"
openai_llm_model = ChatOpenAI(
    api_key=openai_api_key,
    model="gpt-4.1-2025-04-14",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

openai_embeddings_model = OpenAIEmbeddings(
    api_key=openai_api_key,
    model="text-embedding-3-large",
)

# Load the 2020-COVID-NYT dataset pickle
news_covid = pd.read_pickle("../datasets/nyt_news/2020_nyt_COVID_last_version_ready.pkl")

# Define a helper function to convert the dataframe'satomic factsinto a dictionary,
# where keys are observation dates and values are the combined list ofatomic factsfor that date.
def to_dictionary(df): 

    if isinstance(df['factoids_g_truth'][0], str):
        df["factoids_g_truth"] = df["factoids_g_truth"].apply(lambda x:ast.literal_eval(x))
    grouped_df = df.groupby("date")["factoids_g_truth"].sum().reset_index()
    return {
        str(date): factoids for date, factoids in grouped_df.set_index("date")["factoids_g_truth"].to_dict().items()
        }

# Convert the dataframe into the required dictionary format
news_covid_dict = to_dictionary(news_covid)

# Initialize the ATOM pipeline with the OpenAI models
atom = Atom(llm_model=openai_llm_model, embeddings_model=openai_embeddings_model)

# Build the knowledge graph across different observation timestamps
kg = await atom.build_graph_from_different_obs_times(
    atomic_facts_with_obs_timestamps=news_covid_dict,
    
)

# Visualize the resulting knowledge graph using Neo4j
URI = "bolt://localhost:7687"
USERNAME = "neo4j"
PASSWORD = "##"
Neo4jStorage(uri=URI, username=USERNAME, password=PASSWORD).visualize_graph(knowledge_graph=kg)
```


# Contributing

We welcome contributions! To help improve ATOM:
	1.	Fork this repository to your GitHub account.
	2.	Create a feature branch with your enhancements or bug fixes.
	3.	Submit a pull request detailing the changes.

Please report any issues via the Issues tab. Community feedback is invaluable!
