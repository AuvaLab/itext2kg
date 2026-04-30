"""
Token Cost Estimation for Atomic Fact Decomposition

This script calculates the token costs for two scenarios:
(F) With Factoids: Lead Paragraphs -> Atomic Facts -> 5-Tuples
(L) Without Factoids: Lead Paragraphs -> 5-Tuples

Uses tiktoken to count tokens and estimates costs for different LLM models.
"""

import sys
import json
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime
import pandas as pd
import tiktoken

# Add the project root to Python path
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
sys.path.append(str(project_root))

# Import Pydantic models for schema generation (after sys.path is set)
from itext2kg.atom.models.schemas import AtomicFact, RelationshipsExtractor

# ==========================
# Configuration & Constants
# ==========================

# Model pricing per 1M tokens (input, output, batch_input, batch_output)
MODEL_PRICING = {
    'claude-sonnet-4-2025-01-31': {
        'name': 'Claude Sonnet 4',
        'input': 3.00,
        'output': 15.00,
        'batch_input': 1.50,  # 50% discount for batch
        'batch_output': 7.50
    },
    'gpt-4o-2024-11-20': {
        'name': 'GPT-4o',
        'input': 2.50,
        'output': 10.00,
        'batch_input': 1.25,  # 50% discount for cached/batch
        'batch_output': 5.00
    },
    'mistral-large-2411': {
        'name': 'Mistral Large',
        'input': 2.00,
        'output': 6.00,
        'batch_input': 1.00,  # 50% discount
        'batch_output': 3.00
    },
    'o3-mini-2025-01-31': {
        'name': 'O3 Mini',
        'input': 1.10,
        'output': 4.40,
        'batch_input': 0.55,  # 50% discount
        'batch_output': 2.20
    },
    'gpt-4.1-2025-04-14': {
        'name': 'GPT-4.1',
        'input': 2.00,
        'output': 8.00,
        'batch_input': 1,  # 75% discount for cached inputs
        'batch_output': 4.00
    }
}

# Embeddings pricing (per 1M tokens)
EMBEDDINGS_PRICING = {
    'text-embedding-3-large': {
        'name': 'OpenAI text-embedding-3-large',
        'cost_per_million': 0.07
    }
}

# Prompt templates extracted from the codebase
ATOMIC_FACTS_PROMPT = '''You are an expert factoid extraction engine. Your primary function is to read a news paragraph and its associated observation date, and then decompose the text into a comprehensive list of atomic, self-contained, and temporally-grounded facts.

## Task
Given an input paragraph and an `observation_date`, generate a list of all distinct factoids present in the text.

## Guidelines for Generating Temporal Factoids

### 1. Atomic Factoids
- Convert compound or complex sentences into short, single-fact statements
- Each factoid must contain exactly one piece of information or relationship
- Ensure that each factoid is expressed directly and concisely, without redundancies or duplicating the same information across multiple statements
- **Example:** "Unsupervised learning is dedicated to discovering intrinsic patterns in unlabeled datasets" becomes "Unsupervised learning discovers patterns in unlabeled data"

### 2. Decontextualization
- Replace pronouns (e.g., "it," "he," "they") with the full entity name or a clarifying noun phrase
- Include any necessary modifiers so that each factoid is understandable in isolation

### 3. Temporal Context
- Convert ALL time references to absolute dates/times using the observation_date

#### Conversion Rules:
- "today" → exact observation_date
- "yesterday" → observation_date minus 1 day
- "this week" → Monday of observation_date's week
- "last week" → Monday of the week before observation_date
- "this month" → first day of observation_date's month
- "last month" → first day of the month before observation_date
- "this year" → January 1st of observation_date's year
- "last year" → January 1st of the year before observation_date
- Keep explicit dates as-is (e.g., "June 18, 2024")

#### Additional Temporal Guidelines:
- Position time references naturally within factoids
- Split sentences with multiple time references into separate factoids
- **NEVER include relative terms like "today," "yesterday," "last week" in the final factoids**

### 4. Accuracy & Completeness
- Preserve the original meaning without combining multiple facts into a single statement
- Avoid adding details not present in the source text

### 5. End Actions
- If the text indicates the end of a role or an action (for example, someone leaving a position), be explicit about the role/action and the time it ended

### 6. Redundancies
- Eliminate redundancies by simplifying phrases
- **Example:** Convert "the method is crucial for maintaining X" into "the method maintains X"

## Example

**Input:** "On June 18, 2024, Real Madrid won the Champions League final with a 2-1 victory. Following the triumph, fans of Real Madrid celebrated the Champions League victory across the city."

**Output:**
- Real Madrid won the Champions League final on June 18, 2024
- The Champions League final ended with a 2-1 victory for Real Madrid on June 18, 2024
- Fans of Real Madrid celebrated the Champions League victory across the city on June 18, 2024
'''

QUINTUPLES_BASE_PROMPT = '''You are a top-tier algorithm designed for extracting information in structured 
formats to build a knowledge graph.
Try to capture as much information from the text as possible without 
sacrificing accuracy. Do not add any information that is not explicitly mentioned in the text
Remember, the knowledge graph should be coherent and easily understandable, 
so maintaining consistency in entity references is crucial.'''

QUINTUPLES_EXAMPLES = ''' 
FEW SHOT EXAMPLES 

* Michel served as CFO at Acme Corp from 2019 to 2021. He was hired by Beta Inc in 2021, but left that role in 2023.
-> (Michel, is_CFO_of, Acme Corp, ["01-01-2019"], ["01-01-2021"]), (Michel, works_at, Beta Inc, ["01-01-2021"], ["01-01-2023"])

* Subsequent experiments confirmed the role of microRNAs in modulating cell growth.
-> (Experiments, confirm_role_of, microRNAs, [], []), (microRNAs, modulate, Cell Growth, [], [])

* Researchers used high-resolution imaging in a study on neural plasticity.
-> (Researchers, use, High-Resolution Imaging, [], []), (High-Resolution Imaging, is_used_in, Study on Neural Plasticity, [], [])

* Sarah was a board member of GreenFuture until 2019.
-> (Sarah, is_board_member_of, GreenFuture, [], ["01-01-2019"])

* Dr. Lee was the head of the Oncology Department until 2022.
-> (Dr. Lee, is_head_of, Oncology Department, [], ["01-01-2022"])

* Activity-dependent modulation of receptor trafficking is crucial for maintaining synaptic efficacy.
-> (Activity-Dependent Modulation, involves, Receptor Trafficking, [], []), (Receptor Trafficking, maintains, Synaptic Efficacy, [], [])

* (observation_date = 2024-06-15) John Doe is no longer the CEO of GreenIT a few months ago.
-> (John Doe, is_CEO_of, GreenIT, [], ["2024-03-15"])
# "a few months ago" ≈ 3 months → 2024-06-15 minus 3 months = 2024-03-15

* John Doe's marriage is happening on 26-02-2026.
-> (John Doe, has_status, Married, ["2026-02-26"], [])

* (observation_date = 2024-03-20) The AI Summit conference started yesterday and will end tomorrow.
-> (AI Summit, has_status, Started, ["2024-03-19"], ["2024-03-21"])

* The independence day of Morocco is celebrated on January 1st each year since 1956.
-> (Morocco, celebrates, Independence Day, ["1956-01-01"], [])

* (observation_date = 2024-08-10) The product launch event is scheduled for next month.
-> (Product Launch, has_status, Scheduled, ["2024-09-01"], [])
# "next month" = first day of September 2024
'''

# Dataset paths
DATASET_PATH = project_root / "datasets" / "atom" / "nyt_news" / "2020_nyt_COVID_last_version_ready.pkl"
OUTPUT_JSON_PATH = project_root / "evaluation" / "costs" / "detailed_costs.json"

# Column names
LEAD_COL = "lead_paragraph_observation_date"
DATE_COL = "date"
FACTOIDS_COL = "factoids_claude"
QUINTUPLES_FROM_FACTOIDS_COL = "quintuples_gpt41_from_factoids"
QUINTUPLES_DIRECT_COL = "quintuples_gpt41"


# ==========================
# Configuration & Constants for Token Counting
# ==========================

# LangChain prompt wrapper format (from langchain_output_parser.py line 269)
LANGCHAIN_WRAPPER_FORMAT = "# Context: {context}\n\n# Question: {system_query}\n\nAnswer: "

# Cache for Pydantic schema tokens (to avoid regenerating)
_SCHEMA_TOKENS_CACHE: Dict[str, int] = {}

# ==========================
# Core Functions
# ==========================

def count_tokens(text: str, encoding_name: str = "cl100k_base") -> int:
    """Count tokens in text using tiktoken"""
    if not text or pd.isna(text):
        return 0
    if isinstance(text, (list, dict)):
        text = json.dumps(text, ensure_ascii=False)
    encoding = tiktoken.get_encoding(encoding_name)
    return len(encoding.encode(str(text)))


def get_pydantic_schema_tokens(model_class) -> int:
    """
    Get token count for Pydantic model JSON schema.
    This represents the schema that LangChain sends to the LLM for structured output.
    
    Args:
        model_class: Pydantic model class (e.g., AtomicFact, RelationshipsExtractor)
    
    Returns:
        Number of tokens in the JSON schema
    """
    model_name = model_class.__name__
    
    # Check cache first
    if model_name in _SCHEMA_TOKENS_CACHE:
        return _SCHEMA_TOKENS_CACHE[model_name]
    
    try:
        # Generate JSON schema from Pydantic model
        schema = model_class.model_json_schema()
        # Convert to compact JSON string
        schema_json = json.dumps(schema, ensure_ascii=False, separators=(',', ':'))
        # Count tokens
        tokens = count_tokens(schema_json)
        # Cache result
        _SCHEMA_TOKENS_CACHE[model_name] = tokens
        return tokens
    except Exception as e:
        print(f"Warning: Could not generate schema for {model_name}: {e}")
        # Return conservative estimate if schema generation fails
        return 500  # Conservative estimate for a typical schema


def wrap_prompt_langchain_style(context: str, system_query: str) -> str:
    """
    Wrap prompt in LangChain's format as used in langchain_output_parser.py
    
    Args:
        context: The input context (lead paragraph or atomic facts)
        system_query: The system query/instruction
    
    Returns:
        Formatted prompt string matching LangChain's format
    """
    return LANGCHAIN_WRAPPER_FORMAT.format(context=context, system_query=system_query)


def format_atomic_facts_as_json(factoids_list: List[str]) -> str:
    """
    Reconstruct the full JSON output structure for AtomicFact model.
    This matches what the LLM actually outputs.
    
    Args:
        factoids_list: List of factoid strings
    
    Returns:
        JSON string representing the full Pydantic output
    """
    if not factoids_list:
        return '{"atomic_fact":[]}'
    
    # Create the full JSON structure
    output = {
        "atomic_fact": factoids_list
    }
    # Return compact JSON (matching LLM output format)
    return json.dumps(output, ensure_ascii=False, separators=(',', ':'))


def format_relationships_as_json(quintuples_list: List[tuple]) -> str:
    """
    Reconstruct the full JSON output structure for RelationshipsExtractor model.
    This matches what the LLM actually outputs.
    
    Args:
        quintuples_list: List of quintuples as tuples (head, relation, tail, t_start, t_end)
    
    Returns:
        JSON string representing the full Pydantic output
    """
    if not quintuples_list:
        return '{"relationships":[]}'
    
    relationships = []
    for quintuple in quintuples_list:
        if not quintuple or len(quintuple) < 3:
            continue
        
        head = str(quintuple[0]) if quintuple[0] is not None else ""
        relation = str(quintuple[1]) if quintuple[1] is not None else ""
        tail = str(quintuple[2]) if quintuple[2] is not None else ""
        t_start = quintuple[3] if len(quintuple) > 3 and quintuple[3] is not None else []
        t_end = quintuple[4] if len(quintuple) > 4 and quintuple[4] is not None else []
        
        # Convert t_start and t_end to lists if they aren't already
        if not isinstance(t_start, list):
            t_start = [t_start] if t_start else []
        if not isinstance(t_end, list):
            t_end = [t_end] if t_end else []
        
        # Create relationship object matching Pydantic structure
        relationship = {
            "startNode": {
                "label": "Entity",  # Conservative: use generic label (not stored in dataset)
                "name": head
            },
            "endNode": {
                "label": "Entity",  # Conservative: use generic label
                "name": tail
            },
            "name": relation,
            "t_start": t_start,
            "t_end": t_end
        }
        relationships.append(relationship)
    
    output = {
        "relationships": relationships
    }
    # Return compact JSON (matching LLM output format)
    return json.dumps(output, ensure_ascii=False, separators=(',', ':'))


def load_dataset(path: Path) -> pd.DataFrame:
    """Load the NYT COVID dataset"""
    print(f"Loading dataset from {path}...")
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")
    df = pd.read_pickle(path)
    print(f"Loaded {len(df)} rows")
    return df


def format_factoids(factoids: Any) -> str:
    """Format factoids list to string for token counting (plain text, for input)"""
    if factoids is None:
        return ""
    try:
        if pd.isna(factoids):
            return ""
    except (ValueError, TypeError):
        # If pd.isna fails (e.g., for lists), continue
        pass
    if isinstance(factoids, list):
        return "\n".join(str(f) for f in factoids if f)
    return str(factoids)


def format_quintuples(quintuples: Any) -> str:
    """Format quintuples list to string for token counting (plain text, for input)"""
    if quintuples is None:
        return ""
    try:
        if pd.isna(quintuples):
            return ""
    except (ValueError, TypeError):
        # If pd.isna fails (e.g., for lists), continue
        pass
    if isinstance(quintuples, list):
        # Format as list of tuples
        return str(quintuples)
    return str(quintuples)


def count_embedding_tokens_from_quintuples(quintuples_list: List[tuple]) -> Dict[str, int]:
    """
    Count tokens for embeddings from quintuples.
    Embeddings are needed for: entity labels, entity names, and relation names.
    Counts all instances (not unique) - every occurrence is embedded.
    
    Args:
        quintuples_list: List of quintuples (head, relation, tail, t_start, t_end)
    
    Returns:
        Dictionary with token counts for each embedding type:
        - entity_label_tokens: Total tokens for all entity labels
        - entity_name_tokens: Total tokens for all entity names
        - relation_name_tokens: Total tokens for all relation names
        - total_embedding_tokens: Sum of all embedding tokens
    """
    if not quintuples_list:
        return {
            'entity_label_tokens': 0,
            'entity_name_tokens': 0,
            'relation_name_tokens': 0,
            'total_embedding_tokens': 0
        }
    
    entity_label_tokens = 0
    entity_name_tokens = 0
    relation_name_tokens = 0
    
    for quintuple in quintuples_list:
        if not quintuple or len(quintuple) < 3:
            continue
        
        # Extract components from quintuple
        head = str(quintuple[0]) if quintuple[0] is not None else ""
        relation = str(quintuple[1]) if quintuple[1] is not None else ""
        tail = str(quintuple[2]) if quintuple[2] is not None else ""
        
        # Use generic "Entity" label (as per format_relationships_as_json line 313)
        # This matches the conservative approach used in the JSON reconstruction
        head_label = "Entity"
        tail_label = "Entity"
        
        # Count tokens for entity labels (both head and tail)
        entity_label_tokens += count_tokens(head_label)
        entity_label_tokens += count_tokens(tail_label)
        
        # Count tokens for entity names (both head and tail)
        if head:
            entity_name_tokens += count_tokens(head)
        if tail:
            entity_name_tokens += count_tokens(tail)
        
        # Count tokens for relation name
        if relation:
            relation_name_tokens += count_tokens(relation)
    
    total_embedding_tokens = entity_label_tokens + entity_name_tokens + relation_name_tokens
    
    return {
        'entity_label_tokens': entity_label_tokens,
        'entity_name_tokens': entity_name_tokens,
        'relation_name_tokens': relation_name_tokens,
        'total_embedding_tokens': total_embedding_tokens
    }


def analyze_scenario_F(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze token usage for Scenario F (With Factoids)
    Lead Paragraphs -> Atomic Facts -> 5-Tuples
    """
    print("\n" + "="*80)
    print("SCENARIO F: With Factoids (Lead -> Atomic Facts -> 5-Tuples)")
    print("="*80)
    
    # Filter rows that have all required columns
    required_cols = [LEAD_COL, FACTOIDS_COL, QUINTUPLES_FROM_FACTOIDS_COL]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Warning: Missing columns: {missing_cols}")
        print(f"Available columns: {list(df.columns)}")
        return {}
    
    df_valid = df.dropna(subset=[LEAD_COL, FACTOIDS_COL, QUINTUPLES_FROM_FACTOIDS_COL])
    print(f"Analyzing {len(df_valid)} valid rows...")
    
    if len(df_valid) == 0:
        print("No valid rows found for Scenario F")
        return {}
    
    # Step 1: Lead Paragraphs -> Atomic Facts
    print("\n--- Step 1: Extracting Atomic Facts from Lead Paragraphs ---")
    
    # Get Pydantic schema tokens (cached, calculated once)
    atomic_fact_schema_tokens = get_pydantic_schema_tokens(AtomicFact)
    
    step1_input_tokens = []
    step1_output_tokens = []
    step1_details = []
    
    for idx, row in df_valid.iterrows():
        lead_text = str(row[LEAD_COL]) if not pd.isna(row[LEAD_COL]) else ""
        obs_date = str(row[DATE_COL]) if DATE_COL in df.columns and not pd.isna(row[DATE_COL]) else ""
        factoids = row[FACTOIDS_COL]
        
        # Build system query with observation date
        system_query = f"Observation Date: {obs_date}\n\n{ATOMIC_FACTS_PROMPT}\n\nParagraph: {lead_text}"
        
        # Wrap in LangChain format
        wrapped_prompt = wrap_prompt_langchain_style(lead_text, system_query)
        
        # Input tokens: LangChain wrapper + Pydantic schema
        lead_token_count = count_tokens(lead_text)
        wrapped_prompt_tokens = count_tokens(wrapped_prompt)
        prompt_token_count = wrapped_prompt_tokens + atomic_fact_schema_tokens
        
        # Output: Atomic facts as JSON (full Pydantic structure)
        if isinstance(factoids, list):
            factoids_json = format_atomic_facts_as_json(factoids)
        else:
            factoids_list = [str(factoids)] if factoids else []
            factoids_json = format_atomic_facts_as_json(factoids_list)
        output_token_count = count_tokens(factoids_json)
        
        step1_input_tokens.append(prompt_token_count)
        step1_output_tokens.append(output_token_count)
        
        step1_details.append({
            'row_idx': int(idx),
            'lead_tokens': lead_token_count,
            'input_tokens': prompt_token_count,
            'output_tokens': output_token_count,
            'schema_tokens': atomic_fact_schema_tokens
        })
    
    # Step 2: Atomic Facts -> 5-Tuples
    print("\n--- Step 2: Extracting 5-Tuples from Atomic Facts ---")
    
    # Get Pydantic schema tokens (cached, calculated once)
    relationships_schema_tokens = get_pydantic_schema_tokens(RelationshipsExtractor)
    
    step2_input_tokens = []
    step2_output_tokens = []
    step2_details = []
    step2_embedding_tokens = []
    
    for idx, row in df_valid.iterrows():
        factoids = row[FACTOIDS_COL]
        obs_date = str(row[DATE_COL]) if DATE_COL in df.columns and not pd.isna(row[DATE_COL]) else ""
        quintuples = row[QUINTUPLES_FROM_FACTOIDS_COL]
        
        # Format factoids as text for context
        factoids_text = format_factoids(factoids)
        
        # Build system query
        system_query = f"Observation Time: {obs_date}\n\n{QUINTUPLES_BASE_PROMPT}\n\n{QUINTUPLES_EXAMPLES}\n\nAtomic Facts:\n{factoids_text}"
        
        # Wrap in LangChain format
        wrapped_prompt = wrap_prompt_langchain_style(factoids_text, system_query)
        
        # Input tokens: LangChain wrapper + Pydantic schema
        factoids_token_count = count_tokens(factoids_text)
        wrapped_prompt_tokens = count_tokens(wrapped_prompt)
        prompt_token_count = wrapped_prompt_tokens + relationships_schema_tokens
        
        # Output: 5-tuples as JSON (full Pydantic structure)
        if isinstance(quintuples, list):
            quintuples_json = format_relationships_as_json(quintuples)
            # Count embedding tokens from quintuples
            embedding_tokens_dict = count_embedding_tokens_from_quintuples(quintuples)
        else:
            quintuples_json = format_relationships_as_json([])
            embedding_tokens_dict = count_embedding_tokens_from_quintuples([])
        output_token_count = count_tokens(quintuples_json)
        
        step2_input_tokens.append(prompt_token_count)
        step2_output_tokens.append(output_token_count)
        step2_embedding_tokens.append(embedding_tokens_dict['total_embedding_tokens'])
        
        step2_details.append({
            'row_idx': int(idx),
            'factoids_tokens': factoids_token_count,
            'input_tokens': prompt_token_count,
            'output_tokens': output_token_count,
            'schema_tokens': relationships_schema_tokens,
            'embedding_tokens': embedding_tokens_dict['total_embedding_tokens']
        })
    
    # Calculate statistics
    total_step1_input = sum(step1_input_tokens)
    total_step1_output = sum(step1_output_tokens)
    total_step2_input = sum(step2_input_tokens)
    total_step2_output = sum(step2_output_tokens)
    total_embedding_tokens = sum(step2_embedding_tokens)
    
    print(f"\nStep 1 Statistics (Atomic Facts Extraction):")
    print(f"  Total Input Tokens:  {total_step1_input:,} ({total_step1_input/1e6:.3f}M)")
    print(f"  Total Output Tokens: {total_step1_output:,} ({total_step1_output/1e6:.3f}M)")
    print(f"  Avg Input per Article:  {total_step1_input/len(df_valid):.0f} tokens")
    print(f"  Avg Output per Article: {total_step1_output/len(df_valid):.0f} tokens")
    
    print(f"\nStep 2 Statistics (5-Tuples from Factoids):")
    print(f"  Total Input Tokens:  {total_step2_input:,} ({total_step2_input/1e6:.3f}M)")
    print(f"  Total Output Tokens: {total_step2_output:,} ({total_step2_output/1e6:.3f}M)")
    print(f"  Avg Input per Article:  {total_step2_input/len(df_valid):.0f} tokens")
    print(f"  Avg Output per Article: {total_step2_output/len(df_valid):.0f} tokens")
    print(f"  Total Embedding Tokens: {total_embedding_tokens:,} ({total_embedding_tokens/1e6:.3f}M)")
    print(f"  Avg Embedding Tokens per Article: {total_embedding_tokens/len(df_valid):.0f} tokens")
    
    return {
        'scenario': 'F',
        'num_articles': len(df_valid),
        'step1': {
            'input_tokens': total_step1_input,
            'output_tokens': total_step1_output,
            'details': step1_details
        },
        'step2': {
            'input_tokens': total_step2_input,
            'output_tokens': total_step2_output,
            'details': step2_details,
            'batch_discount': True
        },
        'total_input_tokens': total_step1_input + total_step2_input,
        'total_output_tokens': total_step1_output + total_step2_output,
        'total_embedding_tokens': total_embedding_tokens
    }


def analyze_scenario_F_FT(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze token usage for Scenario F-FT (Fine-Tuned for Atomic Facts)
    Only Step 2 is counted: Atomic Facts -> 5-Tuples (Step 1 tokens are not counted)
    """
    print("\n" + "="*80)
    print("SCENARIO F-FT: Fine-Tuned LLM for Atomic Facts (Only Step 2 counted)")
    print("="*80)
    
    # Filter rows that have all required columns
    required_cols = [LEAD_COL, FACTOIDS_COL, QUINTUPLES_FROM_FACTOIDS_COL]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Warning: Missing columns: {missing_cols}")
        print(f"Available columns: {list(df.columns)}")
        return {}
    
    df_valid = df.dropna(subset=[LEAD_COL, FACTOIDS_COL, QUINTUPLES_FROM_FACTOIDS_COL])
    print(f"Analyzing {len(df_valid)} valid rows...")
    
    if len(df_valid) == 0:
        print("No valid rows found for Scenario F-FT")
        return {}
    
    # Only Step 2: Atomic Facts -> 5-Tuples
    print("\n--- Step 2: Extracting 5-Tuples from Atomic Facts ---")
    print("Note: Step 1 tokens are not counted (fine-tuned model used)")
    
    # Get Pydantic schema tokens (cached, calculated once)
    relationships_schema_tokens = get_pydantic_schema_tokens(RelationshipsExtractor)
    
    step2_input_tokens = []
    step2_output_tokens = []
    step2_details = []
    step2_embedding_tokens = []
    
    for idx, row in df_valid.iterrows():
        factoids = row[FACTOIDS_COL]
        obs_date = str(row[DATE_COL]) if DATE_COL in df.columns and not pd.isna(row[DATE_COL]) else ""
        quintuples = row[QUINTUPLES_FROM_FACTOIDS_COL]
        
        # Format factoids as text for context
        factoids_text = format_factoids(factoids)
        
        # Build system query
        system_query = f"Observation Time: {obs_date}\n\n{QUINTUPLES_BASE_PROMPT}\n\n{QUINTUPLES_EXAMPLES}\n\nAtomic Facts:\n{factoids_text}"
        
        # Wrap in LangChain format
        wrapped_prompt = wrap_prompt_langchain_style(factoids_text, system_query)
        
        # Input tokens: LangChain wrapper + Pydantic schema
        factoids_token_count = count_tokens(factoids_text)
        wrapped_prompt_tokens = count_tokens(wrapped_prompt)
        prompt_token_count = wrapped_prompt_tokens + relationships_schema_tokens
        
        # Output: 5-tuples as JSON (full Pydantic structure)
        if isinstance(quintuples, list):
            quintuples_json = format_relationships_as_json(quintuples)
            # Count embedding tokens from quintuples
            embedding_tokens_dict = count_embedding_tokens_from_quintuples(quintuples)
        else:
            quintuples_json = format_relationships_as_json([])
            embedding_tokens_dict = count_embedding_tokens_from_quintuples([])
        output_token_count = count_tokens(quintuples_json)
        
        step2_input_tokens.append(prompt_token_count)
        step2_output_tokens.append(output_token_count)
        step2_embedding_tokens.append(embedding_tokens_dict['total_embedding_tokens'])
        
        step2_details.append({
            'row_idx': int(idx),
            'factoids_tokens': factoids_token_count,
            'input_tokens': prompt_token_count,
            'output_tokens': output_token_count,
            'schema_tokens': relationships_schema_tokens,
            'embedding_tokens': embedding_tokens_dict['total_embedding_tokens']
        })
    
    # Calculate statistics
    total_step2_input = sum(step2_input_tokens)
    total_step2_output = sum(step2_output_tokens)
    total_embedding_tokens = sum(step2_embedding_tokens)
    
    print(f"\nStep 2 Statistics (5-Tuples from Factoids):")
    print(f"  Total Input Tokens:  {total_step2_input:,} ({total_step2_input/1e6:.3f}M)")
    print(f"  Total Output Tokens: {total_step2_output:,} ({total_step2_output/1e6:.3f}M)")
    print(f"  Avg Input per Article:  {total_step2_input/len(df_valid):.0f} tokens")
    print(f"  Avg Output per Article: {total_step2_output/len(df_valid):.0f} tokens")
    print(f"  Total Embedding Tokens: {total_embedding_tokens:,} ({total_embedding_tokens/1e6:.3f}M)")
    print(f"  Avg Embedding Tokens per Article: {total_embedding_tokens/len(df_valid):.0f} tokens")
    
    return {
        'scenario': 'F-FT',
        'num_articles': len(df_valid),
        'step2': {
            'input_tokens': total_step2_input,
            'output_tokens': total_step2_output,
            'details': step2_details,
            'batch_discount': True
        },
        'total_input_tokens': total_step2_input,
        'total_output_tokens': total_step2_output,
        'total_embedding_tokens': total_embedding_tokens
    }


def analyze_scenario_L(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze token usage for Scenario L (Without Factoids)
    Lead Paragraphs -> 5-Tuples directly
    """
    print("\n" + "="*80)
    print("SCENARIO L: Without Factoids (Lead -> 5-Tuples Direct)")
    print("="*80)
    
    # Filter rows that have required columns
    required_cols = [LEAD_COL, QUINTUPLES_DIRECT_COL]
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    # Determine which quintuples column to use
    quintuples_col = QUINTUPLES_DIRECT_COL
    if missing_cols:
        print(f"Warning: Missing columns: {missing_cols}")
        print(f"Available columns: {list(df.columns)}")
        # Try alternative column names
        alt_cols = [col for col in df.columns if 'quintuples' in col.lower() and 'factoids' not in col.lower()]
        if alt_cols:
            print(f"Alternative quintuples columns found: {alt_cols}")
            quintuples_col = alt_cols[0]
            print(f"Using alternative column: {quintuples_col}")
        else:
            print("No suitable quintuples column found for Scenario L")
            return {}
    
    df_valid = df.dropna(subset=[LEAD_COL, quintuples_col])
    
    print(f"Analyzing {len(df_valid)} valid rows...")
    
    if len(df_valid) == 0:
        print("No valid rows found for Scenario L")
        return {}
    
    # Step 1: Lead Paragraphs -> 5-Tuples directly
    print("\n--- Step 1: Extracting 5-Tuples Directly from Lead Paragraphs ---")
    
    # Get Pydantic schema tokens (cached, calculated once)
    relationships_schema_tokens = get_pydantic_schema_tokens(RelationshipsExtractor)
    
    step1_input_tokens = []
    step1_output_tokens = []
    step1_details = []
    step1_embedding_tokens = []
    
    for idx, row in df_valid.iterrows():
        lead_text = str(row[LEAD_COL]) if not pd.isna(row[LEAD_COL]) else ""
        obs_date = str(row[DATE_COL]) if DATE_COL in df.columns and not pd.isna(row[DATE_COL]) else ""
        quintuples = row[quintuples_col]
        
        # Build system query
        system_query = f"Observation Time: {obs_date}\n\n{QUINTUPLES_BASE_PROMPT}\n\n{QUINTUPLES_EXAMPLES}\n\nParagraph: {lead_text}"
        
        # Wrap in LangChain format
        wrapped_prompt = wrap_prompt_langchain_style(lead_text, system_query)
        
        # Input tokens: LangChain wrapper + Pydantic schema
        lead_token_count = count_tokens(lead_text)
        wrapped_prompt_tokens = count_tokens(wrapped_prompt)
        prompt_token_count = wrapped_prompt_tokens + relationships_schema_tokens
        
        # Output: 5-tuples as JSON (full Pydantic structure)
        if isinstance(quintuples, list):
            quintuples_json = format_relationships_as_json(quintuples)
            # Count embedding tokens from quintuples
            embedding_tokens_dict = count_embedding_tokens_from_quintuples(quintuples)
        else:
            quintuples_json = format_relationships_as_json([])
            embedding_tokens_dict = count_embedding_tokens_from_quintuples([])
        output_token_count = count_tokens(quintuples_json)
        
        step1_input_tokens.append(prompt_token_count)
        step1_output_tokens.append(output_token_count)
        step1_embedding_tokens.append(embedding_tokens_dict['total_embedding_tokens'])
        
        step1_details.append({
            'row_idx': int(idx),
            'lead_tokens': lead_token_count,
            'input_tokens': prompt_token_count,
            'output_tokens': output_token_count,
            'schema_tokens': relationships_schema_tokens,
            'embedding_tokens': embedding_tokens_dict['total_embedding_tokens']
        })
    
    # Calculate statistics
    total_step1_input = sum(step1_input_tokens)
    total_step1_output = sum(step1_output_tokens)
    total_embedding_tokens = sum(step1_embedding_tokens)
    
    print(f"\nStep 1 Statistics (Direct 5-Tuples Extraction):")
    print(f"  Total Input Tokens:  {total_step1_input:,} ({total_step1_input/1e6:.3f}M)")
    print(f"  Total Output Tokens: {total_step1_output:,} ({total_step1_output/1e6:.3f}M)")
    print(f"  Avg Input per Article:  {total_step1_input/len(df_valid):.0f} tokens")
    print(f"  Avg Output per Article: {total_step1_output/len(df_valid):.0f} tokens")
    print(f"  Total Embedding Tokens: {total_embedding_tokens:,} ({total_embedding_tokens/1e6:.3f}M)")
    print(f"  Avg Embedding Tokens per Article: {total_embedding_tokens/len(df_valid):.0f} tokens")
    
    return {
        'scenario': 'L',
        'num_articles': len(df_valid),
        'step1': {
            'input_tokens': total_step1_input,
            'output_tokens': total_step1_output,
            'details': step1_details
        },
        'total_input_tokens': total_step1_input,
        'total_output_tokens': total_step1_output,
        'total_embedding_tokens': total_embedding_tokens
    }


def calculate_costs(token_stats: Dict[str, Any], use_batch: bool = False) -> Dict[str, Dict[str, float]]:
    """
    Calculate costs for all models based on token statistics
    
    Args:
        token_stats: Dictionary with 'total_input_tokens' and 'total_output_tokens'
        use_batch: Whether to apply batch processing discounts (if True, ALL tokens use batch pricing)
    
    Returns:
        Dictionary mapping model IDs to cost breakdowns
    """
    if not token_stats:
        return {}
    
    input_tokens = token_stats.get('total_input_tokens', 0)
    output_tokens = token_stats.get('total_output_tokens', 0)
    
    costs = {}
    
    for model_id, pricing in MODEL_PRICING.items():
        if use_batch:
            # Use batch pricing for all tokens
            input_cost = (input_tokens / 1e6) * pricing['batch_input']
            output_cost = (output_tokens / 1e6) * pricing['batch_output']
        else:
            # Use regular pricing
            input_cost = (input_tokens / 1e6) * pricing['input']
            output_cost = (output_tokens / 1e6) * pricing['output']
        
        total_cost = input_cost + output_cost
        
        costs[model_id] = {
            'model_name': pricing['name'],
            'input_cost': input_cost,
            'output_cost': output_cost,
            'total_cost': total_cost,
            'cost_per_article': total_cost / token_stats.get('num_articles', 1) if token_stats.get('num_articles', 0) > 0 else 0
        }
    
    return costs


def calculate_embeddings_costs(embedding_tokens: int) -> Dict[str, Any]:
    """
    Calculate embeddings costs based on token count.
    
    Args:
        embedding_tokens: Total tokens to embed
    
    Returns:
        Dictionary with cost breakdown for each embedding model
    """
    if embedding_tokens <= 0:
        return {}
    
    costs = {}
    
    for model_id, pricing in EMBEDDINGS_PRICING.items():
        cost = (embedding_tokens / 1e6) * pricing['cost_per_million']
        
        costs[model_id] = {
            'model_name': pricing['name'],
            'total_cost': cost,
            'tokens': embedding_tokens
        }
    
    return costs


def print_summary(scenario_f_stats: Dict[str, Any], scenario_l_stats: Dict[str, Any],
                  scenario_f_costs: Dict[str, Dict[str, float]], 
                  scenario_l_costs: Dict[str, Dict[str, float]],
                  scenario_f_ft_stats: Dict[str, Any] = None,
                  scenario_f_ft_costs: Dict[str, Dict[str, float]] = None,
                  scenario_f_embeddings_costs: Dict[str, Dict[str, Any]] = None,
                  scenario_l_embeddings_costs: Dict[str, Dict[str, Any]] = None,
                  scenario_f_ft_embeddings_costs: Dict[str, Dict[str, Any]] = None):
    """Print formatted summary tables comparing all scenarios"""
    
    print("\n" + "="*80)
    print("COST COMPARISON SUMMARY")
    print("="*80)
    
    # Token comparison
    print("\n--- Token Usage Comparison ---")
    print(f"{'Metric':<40} {'Scenario F':<20} {'Scenario L':<20} {'Difference':<20}")
    print("-" * 100)
    
    if scenario_f_stats and scenario_l_stats:
        f_input = scenario_f_stats.get('total_input_tokens', 0)
        f_output = scenario_f_stats.get('total_output_tokens', 0)
        l_input = scenario_l_stats.get('total_input_tokens', 0)
        l_output = scenario_l_stats.get('total_output_tokens', 0)
        
        print(f"{'Total Input Tokens':<40} {f_input:>15,} {f_input/1e6:>4.3f}M  {l_input:>15,} {l_input/1e6:>4.3f}M  {f_input-l_input:>15,} ({((f_input-l_input)/l_input*100):+.1f}%)")
        print(f"{'Total Output Tokens':<40} {f_output:>15,} {f_output/1e6:>4.3f}M  {l_output:>15,} {l_output/1e6:>4.3f}M  {f_output-l_output:>15,} ({((f_output-l_output)/l_output*100):+.1f}%)")
        
        f_articles = scenario_f_stats.get('num_articles', 0)
        l_articles = scenario_l_stats.get('num_articles', 0)
        if f_articles > 0 and l_articles > 0:
            print(f"{'Avg Input per Article':<40} {f_input/f_articles:>15.0f}        {l_input/l_articles:>15.0f}        {(f_input/f_articles)-(l_input/l_articles):>15.0f}")
            print(f"{'Avg Output per Article':<40} {f_output/f_articles:>15.0f}        {l_output/l_articles:>15.0f}        {(f_output/f_articles)-(l_output/l_articles):>15.0f}")
    
    # Embedding tokens comparison
    if scenario_f_stats and scenario_l_stats:
        f_emb_tokens = scenario_f_stats.get('total_embedding_tokens', 0)
        l_emb_tokens = scenario_l_stats.get('total_embedding_tokens', 0)
        
        if f_emb_tokens > 0 or l_emb_tokens > 0:
            print("\n--- Embedding Token Usage Comparison ---")
            print(f"{'Metric':<40} {'Scenario F':<20} {'Scenario L':<20} {'Difference':<20}")
            print("-" * 100)
            print(f"{'Total Embedding Tokens':<40} {f_emb_tokens:>15,} {f_emb_tokens/1e6:>4.3f}M  {l_emb_tokens:>15,} {l_emb_tokens/1e6:>4.3f}M  {f_emb_tokens-l_emb_tokens:>15,} ({((f_emb_tokens-l_emb_tokens)/l_emb_tokens*100):+.1f}%)" if l_emb_tokens > 0 else f"{'Total Embedding Tokens':<40} {f_emb_tokens:>15,} {f_emb_tokens/1e6:>4.3f}M  {l_emb_tokens:>15,} {l_emb_tokens/1e6:>4.3f}M  {f_emb_tokens-l_emb_tokens:>15,}")
            
            f_articles = scenario_f_stats.get('num_articles', 0)
            l_articles = scenario_l_stats.get('num_articles', 0)
            if f_articles > 0 and l_articles > 0:
                print(f"{'Avg Embedding Tokens per Article':<40} {f_emb_tokens/f_articles:>15.0f}        {l_emb_tokens/l_articles:>15.0f}        {(f_emb_tokens/f_articles)-(l_emb_tokens/l_articles):>15.0f}")
    
    # Embeddings cost comparison
    if scenario_f_embeddings_costs and scenario_l_embeddings_costs:
        print("\n--- Embeddings Cost Comparison (USD) ---")
        print(f"{'Model':<30} {'Scenario F':<20} {'Scenario L':<20} {'Difference':<20}")
        print("-" * 90)
        
        for model_id in EMBEDDINGS_PRICING.keys():
            if model_id in scenario_f_embeddings_costs and model_id in scenario_l_embeddings_costs:
                f_emb_cost = scenario_f_embeddings_costs[model_id]['total_cost']
                l_emb_cost = scenario_l_embeddings_costs[model_id]['total_cost']
                diff = f_emb_cost - l_emb_cost
                
                model_name = scenario_f_embeddings_costs[model_id]['model_name']
                print(f"{model_name:<30} ${f_emb_cost:>15.2f}  ${l_emb_cost:>15.2f}  ${diff:>15.2f}")
    
    # Cost comparison (LLM only)
    print("\n--- LLM Cost Comparison (USD) ---")
    print(f"{'Model':<25} {'Scenario F':<20} {'Scenario L':<20} {'Difference':<20} {'Savings % (L vs F)':<20}")
    print("-" * 105)
    
    for model_id in MODEL_PRICING.keys():
        if model_id in scenario_f_costs and model_id in scenario_l_costs:
            f_cost = scenario_f_costs[model_id]['total_cost']
            l_cost = scenario_l_costs[model_id]['total_cost']
            diff = f_cost - l_cost
            # Calculate savings as percentage saved by using L instead of F
            savings_pct = (diff / f_cost * 100) if f_cost > 0 else 0
            
            model_name = scenario_f_costs[model_id]['model_name']
            print(f"{model_name:<25} ${f_cost:>15.2f}  ${l_cost:>15.2f}  ${diff:>15.2f}  {savings_pct:>18.1f}%")
    
    # Total cost comparison (LLM + Embeddings)
    if scenario_f_embeddings_costs and scenario_l_embeddings_costs:
        print("\n--- Total Cost Comparison (LLM + Embeddings) (USD) ---")
        print(f"{'Model':<25} {'Scenario F':<20} {'Scenario L':<20} {'Difference':<20} {'Savings % (L vs F)':<20}")
        print("-" * 105)
        
        # Get embeddings cost (using first embedding model)
        emb_model_id = list(EMBEDDINGS_PRICING.keys())[0] if EMBEDDINGS_PRICING else None
        
        for model_id in MODEL_PRICING.keys():
            if model_id in scenario_f_costs and model_id in scenario_l_costs:
                f_llm_cost = scenario_f_costs[model_id]['total_cost']
                l_llm_cost = scenario_l_costs[model_id]['total_cost']
                
                f_emb_cost = scenario_f_embeddings_costs.get(emb_model_id, {}).get('total_cost', 0) if emb_model_id else 0
                l_emb_cost = scenario_l_embeddings_costs.get(emb_model_id, {}).get('total_cost', 0) if emb_model_id else 0
                
                f_total = f_llm_cost + f_emb_cost
                l_total = l_llm_cost + l_emb_cost
                diff = f_total - l_total
                savings_pct = (diff / f_total * 100) if f_total > 0 else 0
                
                model_name = scenario_f_costs[model_id]['model_name']
                print(f"{model_name:<25} ${f_total:>15.2f}  ${l_total:>15.2f}  ${diff:>15.2f}  {savings_pct:>18.1f}%")
    
    # Cost per article (LLM only)
    print("\n--- LLM Cost per Article (USD) ---")
    print(f"{'Model':<25} {'Scenario F':<20} {'Scenario L':<20} {'Difference':<20}")
    print("-" * 100)
    
    for model_id in MODEL_PRICING.keys():
        if model_id in scenario_f_costs and model_id in scenario_l_costs:
            f_per_article = scenario_f_costs[model_id]['cost_per_article']
            l_per_article = scenario_l_costs[model_id]['cost_per_article']
            diff = f_per_article - l_per_article
            
            model_name = scenario_f_costs[model_id]['model_name']
            print(f"{model_name:<25} ${f_per_article:>15.4f}  ${l_per_article:>15.4f}  ${diff:>15.4f}")
    
    # Total cost per article (LLM + Embeddings)
    if scenario_f_embeddings_costs and scenario_l_embeddings_costs:
        print("\n--- Total Cost per Article (LLM + Embeddings) (USD) ---")
        print(f"{'Model':<25} {'Scenario F':<20} {'Scenario L':<20} {'Difference':<20}")
        print("-" * 100)
        
        emb_model_id = list(EMBEDDINGS_PRICING.keys())[0] if EMBEDDINGS_PRICING else None
        f_articles = scenario_f_stats.get('num_articles', 0) if scenario_f_stats else 0
        l_articles = scenario_l_stats.get('num_articles', 0) if scenario_l_stats else 0
        
        for model_id in MODEL_PRICING.keys():
            if model_id in scenario_f_costs and model_id in scenario_l_costs:
                f_llm_per_article = scenario_f_costs[model_id]['cost_per_article']
                l_llm_per_article = scenario_l_costs[model_id]['cost_per_article']
                
                f_emb_cost = scenario_f_embeddings_costs.get(emb_model_id, {}).get('total_cost', 0) if emb_model_id else 0
                l_emb_cost = scenario_l_embeddings_costs.get(emb_model_id, {}).get('total_cost', 0) if emb_model_id else 0
                
                f_emb_per_article = f_emb_cost / f_articles if f_articles > 0 else 0
                l_emb_per_article = l_emb_cost / l_articles if l_articles > 0 else 0
                
                f_total_per_article = f_llm_per_article + f_emb_per_article
                l_total_per_article = l_llm_per_article + l_emb_per_article
                diff = f_total_per_article - l_total_per_article
                
                model_name = scenario_f_costs[model_id]['model_name']
                print(f"{model_name:<25} ${f_total_per_article:>15.4f}  ${l_total_per_article:>15.4f}  ${diff:>15.4f}")
    
    # Fine-Tuned scenario comparison with Scenario L
    if scenario_f_ft_stats and scenario_f_ft_costs:
        print("\n" + "="*80)
        print("FINE-TUNED SCENARIO (F-FT) vs DIRECT (L) COMPARISON")
        print("="*80)
        
        print("\n--- Token Usage Comparison (F-FT vs L) ---")
        print(f"{'Metric':<40} {'Scenario F-FT':<20} {'Scenario L':<20} {'Difference':<20}")
        print("-" * 100)
        
        ft_input = scenario_f_ft_stats.get('total_input_tokens', 0)
        ft_output = scenario_f_ft_stats.get('total_output_tokens', 0)
        l_input = scenario_l_stats.get('total_input_tokens', 0)
        l_output = scenario_l_stats.get('total_output_tokens', 0)
        
        print(f"{'Total Input Tokens':<40} {ft_input:>15,} {ft_input/1e6:>4.3f}M  {l_input:>15,} {l_input/1e6:>4.3f}M  {ft_input-l_input:>15,} ({((ft_input-l_input)/l_input*100):+.1f}%)")
        print(f"{'Total Output Tokens':<40} {ft_output:>15,} {ft_output/1e6:>4.3f}M  {l_output:>15,} {l_output/1e6:>4.3f}M  {ft_output-l_output:>15,} ({((ft_output-l_output)/l_output*100):+.1f}%)")
        
        ft_articles = scenario_f_ft_stats.get('num_articles', 0)
        l_articles = scenario_l_stats.get('num_articles', 0)
        if ft_articles > 0 and l_articles > 0:
            print(f"{'Avg Input per Article':<40} {ft_input/ft_articles:>15.0f}        {l_input/l_articles:>15.0f}        {(ft_input/ft_articles)-(l_input/l_articles):>15.0f}")
            print(f"{'Avg Output per Article':<40} {ft_output/ft_articles:>15.0f}        {l_output/l_articles:>15.0f}        {(ft_output/ft_articles)-(l_output/l_articles):>15.0f}")
        
        # Embedding tokens comparison for F-FT vs L
        ft_emb_tokens = scenario_f_ft_stats.get('total_embedding_tokens', 0)
        l_emb_tokens = scenario_l_stats.get('total_embedding_tokens', 0) if scenario_l_stats else 0
        
        if ft_emb_tokens > 0 or l_emb_tokens > 0:
            print("\n--- Embedding Token Usage Comparison (F-FT vs L) ---")
            print(f"{'Metric':<40} {'Scenario F-FT':<20} {'Scenario L':<20} {'Difference':<20}")
            print("-" * 100)
            print(f"{'Total Embedding Tokens':<40} {ft_emb_tokens:>15,} {ft_emb_tokens/1e6:>4.3f}M  {l_emb_tokens:>15,} {l_emb_tokens/1e6:>4.3f}M  {ft_emb_tokens-l_emb_tokens:>15,} ({((ft_emb_tokens-l_emb_tokens)/l_emb_tokens*100):+.1f}%)" if l_emb_tokens > 0 else f"{'Total Embedding Tokens':<40} {ft_emb_tokens:>15,} {ft_emb_tokens/1e6:>4.3f}M  {l_emb_tokens:>15,} {l_emb_tokens/1e6:>4.3f}M  {ft_emb_tokens-l_emb_tokens:>15,}")
            
            if ft_articles > 0 and l_articles > 0:
                print(f"{'Avg Embedding Tokens per Article':<40} {ft_emb_tokens/ft_articles:>15.0f}        {l_emb_tokens/l_articles:>15.0f}        {(ft_emb_tokens/ft_articles)-(l_emb_tokens/l_articles):>15.0f}")
        
        # Embeddings cost comparison for F-FT vs L
        if scenario_f_ft_embeddings_costs and scenario_l_embeddings_costs:
            print("\n--- Embeddings Cost Comparison: Fine-Tuned (F-FT) vs Direct (L) (USD) ---")
            print(f"{'Model':<30} {'Scenario F-FT':<20} {'Scenario L':<20} {'Difference':<20}")
            print("-" * 90)
            
            for model_id in EMBEDDINGS_PRICING.keys():
                if model_id in scenario_f_ft_embeddings_costs and model_id in scenario_l_embeddings_costs:
                    ft_emb_cost = scenario_f_ft_embeddings_costs[model_id]['total_cost']
                    l_emb_cost = scenario_l_embeddings_costs[model_id]['total_cost']
                    diff = ft_emb_cost - l_emb_cost
                    
                    model_name = scenario_f_ft_embeddings_costs[model_id]['model_name']
                    print(f"{model_name:<30} ${ft_emb_cost:>15.2f}  ${l_emb_cost:>15.2f}  ${diff:>15.2f}")
        
        print("\n--- LLM Cost Comparison: Fine-Tuned (F-FT) vs Direct (L) (USD) ---")
        print(f"{'Model':<25} {'Scenario F-FT':<20} {'Scenario L':<20} {'Difference':<20} {'Savings % (F-FT vs L)':<25}")
        print("-" * 110)
        
        for model_id in MODEL_PRICING.keys():
            if model_id in scenario_f_ft_costs and model_id in scenario_l_costs:
                ft_cost = scenario_f_ft_costs[model_id]['total_cost']
                l_cost = scenario_l_costs[model_id]['total_cost']
                diff = ft_cost - l_cost
                # Calculate savings: if F-FT costs less, show positive savings; if more, show negative
                if ft_cost < l_cost:
                    savings_pct = ((l_cost - ft_cost) / l_cost * 100) if l_cost > 0 else 0
                    savings_label = f"{savings_pct:.1f}% (F-FT cheaper)"
                else:
                    extra_pct = ((ft_cost - l_cost) / l_cost * 100) if l_cost > 0 else 0
                    savings_label = f"-{extra_pct:.1f}% (F-FT more expensive)"
                
                model_name = scenario_f_ft_costs[model_id]['model_name']
                print(f"{model_name:<25} ${ft_cost:>15.2f}  ${l_cost:>15.2f}  ${diff:>15.2f}  {savings_label:>25}")
        
        # Total cost comparison (LLM + Embeddings) for F-FT vs L
        if scenario_f_ft_embeddings_costs and scenario_l_embeddings_costs:
            print("\n--- Total Cost Comparison (LLM + Embeddings): Fine-Tuned (F-FT) vs Direct (L) (USD) ---")
            print(f"{'Model':<25} {'Scenario F-FT':<20} {'Scenario L':<20} {'Difference':<20} {'Savings % (F-FT vs L)':<25}")
            print("-" * 110)
            
            emb_model_id = list(EMBEDDINGS_PRICING.keys())[0] if EMBEDDINGS_PRICING else None
            
            for model_id in MODEL_PRICING.keys():
                if model_id in scenario_f_ft_costs and model_id in scenario_l_costs:
                    ft_llm_cost = scenario_f_ft_costs[model_id]['total_cost']
                    l_llm_cost = scenario_l_costs[model_id]['total_cost']
                    
                    ft_emb_cost = scenario_f_ft_embeddings_costs.get(emb_model_id, {}).get('total_cost', 0) if emb_model_id else 0
                    l_emb_cost = scenario_l_embeddings_costs.get(emb_model_id, {}).get('total_cost', 0) if emb_model_id else 0
                    
                    ft_total = ft_llm_cost + ft_emb_cost
                    l_total = l_llm_cost + l_emb_cost
                    diff = ft_total - l_total
                    
                    if ft_total < l_total:
                        savings_pct = ((l_total - ft_total) / l_total * 100) if l_total > 0 else 0
                        savings_label = f"{savings_pct:.1f}% (F-FT cheaper)"
                    else:
                        extra_pct = ((ft_total - l_total) / l_total * 100) if l_total > 0 else 0
                        savings_label = f"-{extra_pct:.1f}% (F-FT more expensive)"
                    
                    model_name = scenario_f_ft_costs[model_id]['model_name']
                    print(f"{model_name:<25} ${ft_total:>15.2f}  ${l_total:>15.2f}  ${diff:>15.2f}  {savings_label:>25}")
        
        print("\n--- LLM Cost per Article: Fine-Tuned (F-FT) vs Direct (L) (USD) ---")
        print(f"{'Model':<25} {'Scenario F-FT':<20} {'Scenario L':<20} {'Difference':<20}")
        print("-" * 100)
        
        for model_id in MODEL_PRICING.keys():
            if model_id in scenario_f_ft_costs and model_id in scenario_l_costs:
                ft_per_article = scenario_f_ft_costs[model_id]['cost_per_article']
                l_per_article = scenario_l_costs[model_id]['cost_per_article']
                diff = ft_per_article - l_per_article
                
                model_name = scenario_f_ft_costs[model_id]['model_name']
                print(f"{model_name:<25} ${ft_per_article:>15.4f}  ${l_per_article:>15.4f}  ${diff:>15.4f}")
        
        # Total cost per article (LLM + Embeddings) for F-FT vs L
        if scenario_f_ft_embeddings_costs and scenario_l_embeddings_costs:
            print("\n--- Total Cost per Article (LLM + Embeddings): Fine-Tuned (F-FT) vs Direct (L) (USD) ---")
            print(f"{'Model':<25} {'Scenario F-FT':<20} {'Scenario L':<20} {'Difference':<20}")
            print("-" * 100)
            
            emb_model_id = list(EMBEDDINGS_PRICING.keys())[0] if EMBEDDINGS_PRICING else None
            ft_articles = scenario_f_ft_stats.get('num_articles', 0)
            l_articles = scenario_l_stats.get('num_articles', 0) if scenario_l_stats else 0
            
            for model_id in MODEL_PRICING.keys():
                if model_id in scenario_f_ft_costs and model_id in scenario_l_costs:
                    ft_llm_per_article = scenario_f_ft_costs[model_id]['cost_per_article']
                    l_llm_per_article = scenario_l_costs[model_id]['cost_per_article']
                    
                    ft_emb_cost = scenario_f_ft_embeddings_costs.get(emb_model_id, {}).get('total_cost', 0) if emb_model_id else 0
                    l_emb_cost = scenario_l_embeddings_costs.get(emb_model_id, {}).get('total_cost', 0) if emb_model_id else 0
                    
                    ft_emb_per_article = ft_emb_cost / ft_articles if ft_articles > 0 else 0
                    l_emb_per_article = l_emb_cost / l_articles if l_articles > 0 else 0
                    
                    ft_total_per_article = ft_llm_per_article + ft_emb_per_article
                    l_total_per_article = l_llm_per_article + l_emb_per_article
                    diff = ft_total_per_article - l_total_per_article
                    
                    model_name = scenario_f_ft_costs[model_id]['model_name']
                    print(f"{model_name:<25} ${ft_total_per_article:>15.4f}  ${l_total_per_article:>15.4f}  ${diff:>15.4f}")


def save_to_json(results: Dict[str, Any], output_path: Path):
    """Save detailed results to JSON file"""
    print(f"\nSaving detailed results to {output_path}...")
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert numpy types to native Python types for JSON serialization
    def convert_to_json_serializable(obj):
        import numpy as np
        if isinstance(obj, dict):
            return {k: convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, (int, float)):
            return float(obj) if isinstance(obj, float) else int(obj)
        elif pd.isna(obj):
            return None
        else:
            return obj
    
    json_results = convert_to_json_serializable(results)
    
    with open(output_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"Results saved successfully!")


def main():
    """Main execution flow"""
    print("="*80)
    print("TOKEN COST ESTIMATION FOR ATOMIC FACT DECOMPOSITION")
    print("="*80)
    
    # Load dataset
    df = load_dataset(DATASET_PATH)
    
    # Analyze Scenario F (with batch pricing for all steps)
    scenario_f_stats = analyze_scenario_F(df)
    scenario_f_costs = calculate_costs(scenario_f_stats, use_batch=True) if scenario_f_stats else {}
    scenario_f_embeddings_costs = calculate_embeddings_costs(scenario_f_stats.get('total_embedding_tokens', 0)) if scenario_f_stats else {}
    
    # Analyze Scenario L (with batch pricing)
    scenario_l_stats = analyze_scenario_L(df)
    scenario_l_costs = calculate_costs(scenario_l_stats, use_batch=True) if scenario_l_stats else {}
    scenario_l_embeddings_costs = calculate_embeddings_costs(scenario_l_stats.get('total_embedding_tokens', 0)) if scenario_l_stats else {}
    
    # Analyze Scenario F-FT (Fine-Tuned - only Step 2 counted, with batch pricing)
    scenario_f_ft_stats = analyze_scenario_F_FT(df)
    scenario_f_ft_costs = calculate_costs(scenario_f_ft_stats, use_batch=True) if scenario_f_ft_stats else {}
    scenario_f_ft_embeddings_costs = calculate_embeddings_costs(scenario_f_ft_stats.get('total_embedding_tokens', 0)) if scenario_f_ft_stats else {}
    
    # Print summary
    if scenario_f_stats and scenario_l_stats:
        print_summary(scenario_f_stats, scenario_l_stats, scenario_f_costs, scenario_l_costs,
                     scenario_f_ft_stats, scenario_f_ft_costs,
                     scenario_f_embeddings_costs, scenario_l_embeddings_costs, scenario_f_ft_embeddings_costs)
    
    # Save detailed results
    results = {
        'metadata': {
            'dataset_path': str(DATASET_PATH),
            'analysis_date': datetime.now().isoformat(),
            'num_articles_scenario_f': scenario_f_stats.get('num_articles', 0) if scenario_f_stats else 0,
            'num_articles_scenario_l': scenario_l_stats.get('num_articles', 0) if scenario_l_stats else 0,
            'num_articles_scenario_f_ft': scenario_f_ft_stats.get('num_articles', 0) if scenario_f_ft_stats else 0,
        },
        'scenario_f': {
            'statistics': scenario_f_stats,
            'costs': scenario_f_costs,
            'embeddings_costs': scenario_f_embeddings_costs
        },
        'scenario_l': {
            'statistics': scenario_l_stats,
            'costs': scenario_l_costs,
            'embeddings_costs': scenario_l_embeddings_costs
        },
        'scenario_f_ft': {
            'statistics': scenario_f_ft_stats,
            'costs': scenario_f_ft_costs,
            'embeddings_costs': scenario_f_ft_embeddings_costs,
            'description': 'Fine-tuned LLM for atomic facts decomposition - Step 1 tokens not counted'
        },
        'model_pricing': MODEL_PRICING,
        'embeddings_pricing': EMBEDDINGS_PRICING
    }
    
    save_to_json(results, OUTPUT_JSON_PATH)
    
    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)


if __name__ == "__main__":
    main()

