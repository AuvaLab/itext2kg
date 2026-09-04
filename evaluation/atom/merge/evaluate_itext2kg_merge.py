"""
Evaluate iText2KG Entity Resolution Precision and Recall

This script evaluates the entity resolution (entity merging) performance of the iText2KG system
by comparing its entity merging decisions against ground truth similar entities. It computes
precision (how many merged entities should have been merged) and recall (how many similar
entities were correctly identified and merged) using embedding-based similarity matching.

Usage:
    python evaluate_itext2kg_merge.py

Output:
    - Precision, recall, and F1 scores for entity resolution
    - Analysis of false positives and false negatives in entity merging
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Any, Tuple
from langchain_openai import OpenAIEmbeddings
import asyncio


# ============================================================================
# GLOBAL CONFIGURATION
# ============================================================================

# Path to the iText2KG knowledge graph pickle file (last batch)
ITEXT2KG_KG_PATH = "/Users/yassirlairgi/Developer/Projects/ATOM_Article/batch_cache_itext2kg/batch_56_kg.pkl"

# Path to the df_nyt pickle file (contains ground truth data)
DF_NYT_PATH = "/Users/yassirlairgi/Developer/Projects/ATOM_Article/datasets/nyt_news/2020_nyt_COVID_last_version_ready_quintuples_gpt41_from_factoids_run3_run3_itext2kg.pkl"

# Similarity threshold for determining duplicates
THRESHOLD = 0.8

# Cache file for ground truth entity embeddings
ENTITY_EMBEDDINGS_CACHE = "./entity_embeddings_ground_truth_itext2kg.pkl"

# OpenAI API Key
OPENAI_API_KEY = "###"

# Initialize embeddings model
embeddings_model = OpenAIEmbeddings(
    api_key=OPENAI_API_KEY,
    model="text-embedding-3-large",
)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def load_itext2kg_kg(path: str) -> Any:
    """Load iText2KG knowledge graph from pickle file."""
    print(f"📂 Loading iText2KG knowledge graph from: {path}")
    with open(path, 'rb') as f:
        kg = pickle.load(f)
    print(f"   ✅ Loaded KG with {len(kg.entities)} entities and {len(kg.relationships)} relationships")
    return kg


def load_df_nyt(path: str) -> pd.DataFrame:
    """Load NYT dataframe from pickle file."""
    print(f"📂 Loading NYT dataframe from: {path}")
    df = pd.read_pickle(path)
    print(f"   ✅ Loaded dataframe with {len(df)} rows")
    return df


def load_cached_embeddings(cache_path: str) -> Dict[str, Any]:
    """
    Load cached embeddings if they exist.
    
    Args:
        cache_path: Path to the cache file
    
    Returns:
        Dictionary with cached embeddings or None if cache doesn't exist
    """
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'rb') as f:
                cache = pickle.load(f)
            print(f"   ✅ Loaded cached embeddings for {len(cache['entity_names'])} entities")
            return cache
        except Exception as e:
            print(f"   ⚠️  Error loading cache: {e}")
            return None
    return None


def save_embeddings_cache(cache_path: str, entity_names: List[str], embeddings: np.ndarray):
    """
    Save embeddings to cache.
    
    Args:
        cache_path: Path to save the cache file
        entity_names: List of entity names
        embeddings: Numpy array of embeddings
    """
    try:
        cache = {
            'entity_names': entity_names,
            'embeddings': embeddings,
            'model': 'text-embedding-3-large'
        }
        with open(cache_path, 'wb') as f:
            pickle.dump(cache, f)
        print(f"   💾 Saved embeddings cache to {cache_path}")
    except Exception as e:
        print(f"   ⚠️  Error saving cache: {e}")


# ============================================================================
# ENTITY RESOLUTION FUNCTIONS
# ============================================================================

def find_similar_nodes_itext2kg(itext2kg_kg: Any, similarity_threshold: float = 0.9) -> List[Dict[str, Any]]:
    """
    Find nodes with cosine similarity greater than the specified threshold for iText2KG graph structure.
    
    Args:
        itext2kg_kg: iText2KG knowledge graph object with .entities attribute
        similarity_threshold: Minimum cosine similarity score
    
    Returns:
        List of dictionaries containing similar entity pairs with their similarity scores
    """
    print(f"\n🔍 Finding similar entities with threshold > {similarity_threshold}")
    
    # Extract entities
    entities = itext2kg_kg.entities
    print(f"   Found {len(entities)} total entities")
    
    # Filter entities that have embeddings
    entities_with_embeddings = []
    embeddings = []
    
    for entity in entities:
        if hasattr(entity.properties, 'embeddings') and entity.properties.embeddings is not None:
            entities_with_embeddings.append(entity)
            embeddings.append(entity.properties.embeddings)
    
    print(f"   Found {len(entities_with_embeddings)} entities with embeddings")
    
    if len(embeddings) < 2:
        print("   ⚠️  Not enough entities with embeddings found.")
        return []
    
    # Convert to numpy array
    embeddings_matrix = np.array(embeddings)
    print(f"   Embeddings matrix shape: {embeddings_matrix.shape}")
    
    # Calculate cosine similarity matrix
    similarity_matrix = cosine_similarity(embeddings_matrix)
    
    # Find similar pairs
    similar_pairs = []
    n_entities = len(entities_with_embeddings)
    
    for i in range(n_entities):
        for j in range(i + 1, n_entities):  # Only upper triangle to avoid duplicates
            similarity_score = similarity_matrix[i][j]
            
            if similarity_score > similarity_threshold:
                similar_pairs.append({
                    'entity_1_name': entities_with_embeddings[i].name,
                    'entity_1_label': entities_with_embeddings[i].label,
                    'entity_2_name': entities_with_embeddings[j].name,
                    'entity_2_label': entities_with_embeddings[j].label,
                    'similarity_score': float(similarity_score)
                })
    
    # Sort by similarity score (descending)
    similar_pairs.sort(key=lambda x: x['similarity_score'], reverse=True)
    
    print(f"   ✅ Found {len(similar_pairs)} pairs with similarity > {similarity_threshold}")
    
    return similar_pairs

def calculate_number_of_entities(df_nyt):
    all_entities = [relation[0] for relation in df_nyt["quintuples_g_truth"].cumsum().iloc[-1]] + [relation[2] for relation in df_nyt["quintuples_g_truth"].cumsum().iloc[-1]]
    #non_duplicated_entities = list(set([entity.lower() for entity in all_entities if entity]))
    return len(all_entities)

def calculate_number_of_entities_(df_nyt: pd.DataFrame) -> int:
    """Calculate the number of unique entities in the ground truth."""
    all_entities = []
    
    # Collect all entities from quintuples_g_truth
    for quintuples in df_nyt["quintuples_g_truth"]:
        if isinstance(quintuples, list):
            for relation in quintuples:
                if len(relation) >= 3:
                    all_entities.append(relation[0])  # head entity
                    all_entities.append(relation[2])  # tail entity
    
    # Remove duplicates (case-insensitive)
    non_duplicated_entities = list(set([entity.lower() for entity in all_entities if entity]))
    
    return len(non_duplicated_entities)


def calculate_number_of_entities_itext2kg(itext2kg_kg: Any) -> int:
    """Calculate the number of entities in iText2KG graph."""
    return len(itext2kg_kg.entities)


def calculate_ER_precision(itext2kg_kg: Any, df_nyt: pd.DataFrame, threshold: float = 0.9) -> Tuple[float, List[Dict[str, Any]]]:
    """
    Calculate Entity Resolution (ER) precision.
    
    Args:
        itext2kg_kg: iText2KG knowledge graph object
        df_nyt: Ground truth dataframe
        threshold: Similarity threshold for finding duplicates
    
    Returns:
        Tuple of (ER precision score, list of similar entity pairs)
    """
    print("\n📊 Calculating Entity Resolution (ER) Precision")
    
    similar_nodes = find_similar_nodes_itext2kg(itext2kg_kg, similarity_threshold=threshold)
    n_entities_itext2kg = calculate_number_of_entities_itext2kg(itext2kg_kg)
    n_entities_ground_truth = calculate_number_of_entities(df_nyt)
    
    print(f"   Ground truth entities: {n_entities_ground_truth}")
    print(f"   iText2KG entities: {n_entities_itext2kg}")
    print(f"   Similar entity pairs (potential duplicates): {len(similar_nodes)}")
    
    # Precision formula: 1 - (duplicates / expected_duplicates)
    # where expected_duplicates = ground_truth_count - itext2kg_count
    expected_duplicates = n_entities_ground_truth - n_entities_itext2kg
    
    if expected_duplicates <= 0:
        print("   ⚠️  Warning: iText2KG has more entities than ground truth!")
        return 0.0, similar_nodes
    
    er_precision = 1.0 - (len(similar_nodes) / expected_duplicates)
    er_precision = max(0.0, min(1.0, er_precision))  # Clamp between 0 and 1
    
    print(f"   ✅ ER Precision: {er_precision:.4f}")
    
    return er_precision, similar_nodes


async def number_ground_truth_merged_entities_itext2kg(
    df_nyt: pd.DataFrame, 
    embeddings_model,
    threshold: float = 0.9,
    cache_path: str = None
) -> int:
    """
    Calculate the number of entities that should be merged in ground truth based on embeddings similarity.
    Uses caching to avoid re-embedding on subsequent runs.
    
    This function:
    1. Gets all entities from ground truth (with duplicates)
    2. Creates unique set of entities (case-insensitive)
    3. Embeds the unique entities (with caching)
    4. Finds pairs with cosine similarity above threshold
    5. Returns: total_entities - unique_entities - similar_pairs
    
    Args:
        df_nyt: Ground truth dataframe
        embeddings_model: OpenAI embeddings model
        threshold: Similarity threshold for determining duplicates
        cache_path: Optional path to cache embeddings
    
    Returns:
        Number of entities that should be merged according to ground truth
    """
    print(f"\n🔍 Calculating ground truth merged entities with threshold > {threshold}")
    
    # Step 1: Get all entities (with duplicates)
    all_entities = []
    for quintuples in df_nyt["quintuples_g_truth"]:
        if isinstance(quintuples, list):
            for relation in quintuples:
                if len(relation) >= 3:
                    all_entities.append(relation[0])  # head entity
                    all_entities.append(relation[2])  # tail entity
    
    total_entities = len(all_entities)
    print(f"   Total entities (with duplicates): {total_entities}")
    
    # Step 2: Get unique entities (case-insensitive, remove None/empty)
    unique_entities_list = list(set([entity.lower() for entity in all_entities if entity]))
    num_unique = len(unique_entities_list)
    print(f"   Unique entities: {num_unique}")
    
    if num_unique < 2:
        print("   ⚠️  Not enough unique entities found.")
        return 0
    
    # Step 3: Try to load from cache
    embeddings_array = None
    if cache_path:
        cached = load_cached_embeddings(cache_path)
        if cached is not None:
            # Check if the cached entity names match
            if set(cached['entity_names']) == set(unique_entities_list):
                print("   🎯 Cache matches current entities, using cached embeddings")
                embeddings_array = cached['embeddings']
                # Reorder if necessary
                cached_names = cached['entity_names']
                if cached_names != unique_entities_list:
                    name_to_embedding = {name: emb for name, emb in zip(cached_names, embeddings_array)}
                    embeddings_array = np.array([name_to_embedding[name] for name in unique_entities_list])
            else:
                print("   ⚠️  Cache doesn't match current entities, re-embedding...")
    
    # Step 4: Embed unique entities if not cached
    if embeddings_array is None:
        print(f"   🔮 Embedding {num_unique} unique entities...")
        embeddings = await embeddings_model.aembed_documents(unique_entities_list)
        embeddings_array = np.array(embeddings)
        print(f"   ✅ Generated embeddings with shape: {embeddings_array.shape}")
        
        # Save to cache
        if cache_path:
            save_embeddings_cache(cache_path, unique_entities_list, embeddings_array)
    
    # Step 5: Calculate cosine similarity and find pairs above threshold
    similarity_matrix = cosine_similarity(embeddings_array)
    
    similar_pairs_count = 0
    n_entities = len(unique_entities_list)
    
    for i in range(n_entities):
        for j in range(i + 1, n_entities):
            if similarity_matrix[i][j] > threshold:
                similar_pairs_count += 1
    
    print(f"   Similar pairs (should be merged): {similar_pairs_count}")
    
    # Step 6: Calculate ground truth merged entities
    # Formula: total - unique - similar_pairs
    ground_truth_merged = total_entities - num_unique - similar_pairs_count
    
    print(f"   ✅ Ground truth merged entities: {ground_truth_merged}")
    
    return ground_truth_merged


async def calculate_ER_recall_itext2kg(
    itext2kg_kg: Any, 
    df_nyt: pd.DataFrame, 
    embeddings_model,
    threshold: float = 0.9,
    cache_path: str = None
) -> Tuple[float, int]:
    """
    Calculate Entity Resolution (ER) recall for iText2KG.
    
    Recall measures how well iText2KG is merging entities compared to ground truth.
    Formula: recall = 1 - (len(similar_nodes) / ground_truth_merged)
    
    Args:
        itext2kg_kg: iText2KG knowledge graph object
        df_nyt: Ground truth dataframe
        embeddings_model: OpenAI embeddings model
        threshold: Similarity threshold for finding duplicates
        cache_path: Optional path to cache embeddings
    
    Returns:
        Tuple of (ER recall score, ground truth merged count)
    """
    print("\n📊 Calculating Entity Resolution (ER) Recall")
    
    similar_nodes = find_similar_nodes_itext2kg(itext2kg_kg, similarity_threshold=threshold)
    ground_truth_merged = await number_ground_truth_merged_entities_itext2kg(
        df_nyt, 
        embeddings_model, 
        threshold, 
        cache_path
    )
    
    print(f"   Ground truth merged entities: {ground_truth_merged}")
    print(f"   iText2KG similar pairs (unresolved): {len(similar_nodes)}")
    
    if ground_truth_merged <= 0:
        print("   ⚠️  Warning: No entities should be merged in ground truth!")
        return 1.0, ground_truth_merged
    
    er_recall = 1.0 - (len(similar_nodes) / ground_truth_merged)
    er_recall = max(0.0, min(1.0, er_recall))  # Clamp between 0 and 1
    
    print(f"   ✅ ER Recall: {er_recall:.4f}")
    
    return er_recall, ground_truth_merged


# ============================================================================
# RELATION RESOLUTION FUNCTIONS
# ============================================================================

def extract_unique_relations_with_embeddings(itext2kg_kg: Any) -> Tuple[List[str], np.ndarray]:
    """
    Extract unique relation names and their corresponding embeddings from iText2KG graph.
    IMPORTANT: Deduplicates based on relation name only, keeping first occurrence's embedding.
    
    Args:
        itext2kg_kg: iText2KG knowledge graph object
    
    Returns:
        Tuple of (unique_relation_names list, embeddings array)
    """
    print("\n📋 Extracting unique relations with embeddings...")
    
    # Dictionary to store first occurrence of each relation name with its embedding
    relation_dict = {}
    
    for relationship in itext2kg_kg.relationships:
        rel_name = relationship.name
        
        # Only add if we haven't seen this relation name before
        if rel_name not in relation_dict:
            # Check if relationship has embeddings
            if hasattr(relationship.properties, 'embeddings') and relationship.properties.embeddings is not None:
                relation_dict[rel_name] = relationship.properties.embeddings
    
    # Extract unique names and their embeddings
    unique_relation_names = list(relation_dict.keys())
    embeddings_list = [relation_dict[name] for name in unique_relation_names]
    
    print(f"   Found {len(itext2kg_kg.relationships)} total relationships")
    print(f"   Deduplicated to {len(unique_relation_names)} unique relation names")
    print(f"   {len(embeddings_list)} relations have embeddings")
    
    if embeddings_list:
        embeddings_array = np.array(embeddings_list)
        print(f"   Embeddings shape: {embeddings_array.shape}")
        return unique_relation_names, embeddings_array
    else:
        print("   ⚠️  No relations with embeddings found")
        return [], np.array([])


def find_similar_relations_itext2kg(
    itext2kg_kg: Any,
    threshold: float = 0.9
) -> List[Dict[str, Any]]:
    """
    Find similar relations based on cosine similarity of embeddings.
    
    Args:
        itext2kg_kg: iText2KG knowledge graph object
        threshold: Minimum cosine similarity score
    
    Returns:
        List of dictionaries containing similar relation pairs with their similarity scores
    """
    print(f"\n🔍 Finding similar relations with threshold > {threshold}")
    
    # Extract unique relation names and embeddings
    relation_names, embeddings = extract_unique_relations_with_embeddings(itext2kg_kg)
    
    if len(relation_names) < 2 or len(embeddings) < 2:
        print("   ⚠️  Not enough relations with embeddings found.")
        return []
    
    # Calculate cosine similarity matrix
    similarity_matrix = cosine_similarity(embeddings)
    
    # Find similar pairs
    similar_pairs = []
    n_relations = len(relation_names)
    
    for i in range(n_relations):
        for j in range(i + 1, n_relations):  # Only upper triangle to avoid duplicates
            similarity_score = similarity_matrix[i][j]
            
            if similarity_score > threshold:
                similar_pairs.append({
                    'relation_1': relation_names[i],
                    'relation_2': relation_names[j],
                    'similarity_score': float(similarity_score)
                })
    
    # Sort by similarity score (descending)
    similar_pairs.sort(key=lambda x: x['similarity_score'], reverse=True)
    
    print(f"   ✅ Found {len(similar_pairs)} pairs with similarity > {threshold}")
    
    return similar_pairs


def calculate_number_of_relations_itext2kg(itext2kg_kg: Any) -> int:
    """Calculate the number of unique relations in iText2KG graph."""
    unique_relation_names = list(set([relationship.name for relationship in itext2kg_kg.relationships]))
    return len(unique_relation_names)

def calculate_number_of_relations(df_nyt):
    all_relations = [relation[1] for relation in df_nyt["quintuples_g_truth"].cumsum().iloc[-1]]
    return len(all_relations)


def calculate_number_of_relations_(df_nyt: pd.DataFrame) -> int:
    """Calculate the number of unique relations in the ground truth."""
    all_relations = []
    
    # Collect all relations from quintuples_g_truth
    for quintuples in df_nyt["quintuples_g_truth"]:
        if isinstance(quintuples, list):
            for relation in quintuples:
                if len(relation) >= 3:
                    all_relations.append(relation[1].lower())  # relation type
    
    # Remove duplicates
    non_duplicated_relations = list(set(all_relations))
    
    return len(non_duplicated_relations)


def calculate_RR_precision(
    itext2kg_kg: Any, 
    df_nyt: pd.DataFrame,
    threshold: float = 0.9
) -> Tuple[float, List[Dict[str, Any]]]:
    """
    Calculate Relation Resolution (RR) precision.
    
    Args:
        itext2kg_kg: iText2KG knowledge graph object
        df_nyt: Ground truth dataframe
        threshold: Similarity threshold for finding duplicates
    
    Returns:
        Tuple of (RR precision score, list of similar relation pairs)
    """
    print("\n📊 Calculating Relation Resolution (RR) Precision")
    
    similar_relations = find_similar_relations_itext2kg(itext2kg_kg, threshold)
    n_relations_itext2kg = calculate_number_of_relations_itext2kg(itext2kg_kg)
    n_relations_ground_truth = calculate_number_of_relations(df_nyt)
    
    print(f"   Ground truth relations: {n_relations_ground_truth}")
    print(f"   iText2KG relations: {n_relations_itext2kg}")
    print(f"   Similar relation pairs (potential duplicates): {len(similar_relations)}")
    
    # Precision formula: 1 - (duplicates / expected_duplicates)
    expected_duplicates = n_relations_ground_truth - n_relations_itext2kg
    
    if expected_duplicates <= 0:
        print("   ⚠️  Warning: iText2KG has more or equal relations than ground truth!")
        precision = 1.0 if n_relations_itext2kg == n_relations_ground_truth else 0.0
        return precision, similar_relations
    
    rr_precision = 1.0 - (len(similar_relations) / expected_duplicates)
    rr_precision = max(0.0, min(1.0, rr_precision))  # Clamp between 0 and 1
    
    print(f"   ✅ RR Precision: {rr_precision:.4f}")
    
    return rr_precision, similar_relations


async def number_ground_truth_merged_relations_itext2kg(
    df_nyt: pd.DataFrame, 
    embeddings_model,
    threshold: float = 0.9,
    cache_path: str = None
) -> int:
    """
    Calculate the number of relations that should be merged in ground truth based on embeddings similarity.
    Uses caching to avoid re-embedding on subsequent runs.
    
    This function:
    1. Gets all relations from ground truth (with duplicates)
    2. Creates unique set of relations (case-insensitive)
    3. Embeds the unique relations (with caching)
    4. Finds pairs with cosine similarity above threshold
    5. Returns: total_relations - unique_relations - similar_pairs
    
    Args:
        df_nyt: Ground truth dataframe
        embeddings_model: OpenAI embeddings model
        threshold: Similarity threshold for determining duplicates
        cache_path: Optional path to cache embeddings
    
    Returns:
        Number of relations that should be merged according to ground truth
    """
    print(f"\n🔍 Calculating ground truth merged relations with threshold > {threshold}")
    
    # Step 1: Get all relations (with duplicates)
    all_relations = []
    for quintuples in df_nyt["quintuples_g_truth"]:
        if isinstance(quintuples, list):
            for relation in quintuples:
                if len(relation) >= 3:
                    all_relations.append(relation[1])  # relation type
    
    total_relations = len(all_relations)
    print(f"   Total relations (with duplicates): {total_relations}")
    
    # Step 2: Get unique relations (case-insensitive, remove None/empty)
    unique_relations_list = list(set([relation.lower() for relation in all_relations if relation]))
    num_unique = len(unique_relations_list)
    print(f"   Unique relations: {num_unique}")
    
    if num_unique < 2:
        print("   ⚠️  Not enough unique relations found.")
        return 0
    
    # Step 3: Try to load from cache
    cache_path_relations = cache_path.replace('entity_', 'relation_') if cache_path else None
    embeddings_array = None
    if cache_path_relations:
        cached = load_cached_embeddings(cache_path_relations)
        if cached is not None and 'entity_names' in cached:
            # Check if the cached relation names match (using entity_names key for consistency)
            if set(cached['entity_names']) == set(unique_relations_list):
                print("   🎯 Cache matches current relations, using cached embeddings")
                embeddings_array = cached['embeddings']
                # Reorder if necessary
                cached_names = cached['entity_names']
                if cached_names != unique_relations_list:
                    name_to_embedding = {name: emb for name, emb in zip(cached_names, embeddings_array)}
                    embeddings_array = np.array([name_to_embedding[name] for name in unique_relations_list])
            else:
                print("   ⚠️  Cache doesn't match current relations, re-embedding...")
    
    # Step 4: Embed unique relations if not cached
    if embeddings_array is None:
        print(f"   🔮 Embedding {num_unique} unique relations...")
        embeddings = await embeddings_model.aembed_documents(unique_relations_list)
        embeddings_array = np.array(embeddings)
        print(f"   ✅ Generated embeddings with shape: {embeddings_array.shape}")
        
        # Save to cache
        if cache_path_relations:
            save_embeddings_cache(cache_path_relations, unique_relations_list, embeddings_array)
    
    # Step 5: Calculate cosine similarity and find pairs above threshold
    similarity_matrix = cosine_similarity(embeddings_array)
    
    similar_pairs_count = 0
    n_relations = len(unique_relations_list)
    
    for i in range(n_relations):
        for j in range(i + 1, n_relations):
            if similarity_matrix[i][j] > threshold:
                similar_pairs_count += 1
    
    print(f"   Similar pairs (should be merged): {similar_pairs_count}")
    
    # Step 6: Calculate ground truth merged relations
    # Formula: total - unique - similar_pairs
    ground_truth_merged = total_relations - num_unique - similar_pairs_count
    
    print(f"   ✅ Ground truth merged relations: {ground_truth_merged}")
    
    return ground_truth_merged


async def calculate_RR_recall_itext2kg(
    itext2kg_kg: Any, 
    df_nyt: pd.DataFrame, 
    embeddings_model,
    threshold: float = 0.9,
    cache_path: str = None
) -> Tuple[float, int]:
    """
    Calculate Relation Resolution (RR) recall for iText2KG.
    
    Recall measures how well iText2KG is merging relations compared to ground truth.
    Formula: recall = 1 - (len(similar_relations) / ground_truth_merged)
    
    Args:
        itext2kg_kg: iText2KG knowledge graph object
        df_nyt: Ground truth dataframe
        embeddings_model: OpenAI embeddings model
        threshold: Similarity threshold for finding duplicates
        cache_path: Optional path to cache embeddings
    
    Returns:
        Tuple of (RR recall score, ground truth merged count)
    """
    print("\n📊 Calculating Relation Resolution (RR) Recall")
    
    similar_relations = find_similar_relations_itext2kg(itext2kg_kg, threshold)
    ground_truth_merged = await number_ground_truth_merged_relations_itext2kg(
        df_nyt, 
        embeddings_model, 
        threshold, 
        cache_path
    )
    
    print(f"   Ground truth merged relations: {ground_truth_merged}")
    print(f"   iText2KG similar pairs (unresolved): {len(similar_relations)}")
    
    if ground_truth_merged <= 0:
        print("   ⚠️  Warning: No relations should be merged in ground truth!")
        return 1.0, ground_truth_merged
    
    rr_recall = 1.0 - (len(similar_relations) / ground_truth_merged)
    rr_recall = max(0.0, min(1.0, rr_recall))  # Clamp between 0 and 1
    
    print(f"   ✅ RR Recall: {rr_recall:.4f}")
    
    return rr_recall, ground_truth_merged


# ============================================================================
# DISPLAY FUNCTIONS
# ============================================================================

def display_similar_entities_examples(similar_entities: List[Dict[str, Any]], n_examples: int = 20):
    """
    Display examples of similar entity pairs.
    
    Args:
        similar_entities: List of similar entity pairs
        n_examples: Number of examples to display
    """
    if not similar_entities:
        print("   No similar entities found.")
        return
    
    print(f"\n{'='*80}")
    print(f"🔍 UNRESOLVED ENTITIES - Top {min(n_examples, len(similar_entities))} Examples")
    print(f"{'='*80}")
    
    for i, pair in enumerate(similar_entities[:n_examples], 1):
        print(f"\n{i}. Similarity: {pair['similarity_score']:.4f}")
        print(f"   Entity 1: {pair['entity_1_name']} ({pair['entity_1_label']})")
        print(f"   Entity 2: {pair['entity_2_name']} ({pair['entity_2_label']})")
        if i < min(n_examples, len(similar_entities)):
            print("   " + "-" * 70)


def display_similar_relations_examples(similar_relations: List[Dict[str, Any]], n_examples: int = 20):
    """
    Display examples of similar relation pairs.
    
    Args:
        similar_relations: List of similar relation pairs
        n_examples: Number of examples to display
    """
    if not similar_relations:
        print("   No similar relations found.")
        return
    
    print(f"\n{'='*80}")
    print(f"🔍 UNRESOLVED RELATIONS - Top {min(n_examples, len(similar_relations))} Examples")
    print(f"{'='*80}")
    
    for i, pair in enumerate(similar_relations[:n_examples], 1):
        print(f"\n{i}. Similarity: {pair['similarity_score']:.4f}")
        print(f"   Relation 1: {pair['relation_1']}")
        print(f"   Relation 2: {pair['relation_2']}")
        if i < min(n_examples, len(similar_relations)):
            print("   " + "-" * 70)


# ============================================================================
# MAIN FUNCTION
# ============================================================================

async def main():
    """Main function to evaluate entity and relation resolution precision and recall."""
    print("=" * 80)
    print("🚀 iText2KG Entity & Relation Resolution Evaluation")
    print("=" * 80)
    
    # Load data
    itext2kg_kg = load_itext2kg_kg(ITEXT2KG_KG_PATH)
    df_nyt = load_df_nyt(DF_NYT_PATH)
    
    print("\n🔧 Using pre-initialized embeddings model: text-embedding-3-large")
    print(f"💾 Entity embeddings cache: {ENTITY_EMBEDDINGS_CACHE}")
    if os.path.exists(ENTITY_EMBEDDINGS_CACHE):
        print("   ✅ Cache file exists - will use cached embeddings if they match")
    else:
        print("   📝 Cache file doesn't exist - will create after first run")
    
    # Calculate Entity Resolution (ER) Precision
    print("\n" + "=" * 80)
    er_precision, similar_entities = calculate_ER_precision(itext2kg_kg, df_nyt, threshold=THRESHOLD)
    
    # Calculate Entity Resolution (ER) Recall
    print("\n" + "=" * 80)
    er_recall, ground_truth_merged = await calculate_ER_recall_itext2kg(
        itext2kg_kg, 
        df_nyt, 
        embeddings_model,
        threshold=THRESHOLD,
        cache_path=ENTITY_EMBEDDINGS_CACHE
    )
    
    # Calculate Relation Resolution (RR) Precision
    print("\n" + "=" * 80)
    rr_precision, similar_relations = calculate_RR_precision(itext2kg_kg, df_nyt, threshold=THRESHOLD)
    
    # Calculate Relation Resolution (RR) Recall
    print("\n" + "=" * 80)
    rr_recall, ground_truth_merged_relations = await calculate_RR_recall_itext2kg(
        itext2kg_kg, 
        df_nyt, 
        embeddings_model,
        threshold=THRESHOLD,
        cache_path=ENTITY_EMBEDDINGS_CACHE
    )
    
    # Display examples of unresolved entities and relations
    display_similar_entities_examples(similar_entities, n_examples=20)
    display_similar_relations_examples(similar_relations, n_examples=20)
    
    # Print final summary
    print("\n" + "=" * 80)
    print("📋 FINAL RESULTS")
    print("=" * 80)
    print(f"Similarity Threshold: {THRESHOLD}")
    print("\n--- Entity Resolution (ER) ---")
    print(f"ER Precision: {er_precision:.4f} ({er_precision*100:.2f}%)")
    print(f"ER Recall: {er_recall:.4f} ({er_recall*100:.2f}%)")
    if er_precision > 0 and er_recall > 0:
        er_f1 = 2 * (er_precision * er_recall) / (er_precision + er_recall)
        print(f"ER F1-Score: {er_f1:.4f} ({er_f1*100:.2f}%)")
    print("\n--- Relation Resolution (RR) ---")
    print(f"RR Precision: {rr_precision:.4f} ({rr_precision*100:.2f}%)")
    print(f"RR Recall: {rr_recall:.4f} ({rr_recall*100:.2f}%)")
    if rr_precision > 0 and rr_recall > 0:
        rr_f1 = 2 * (rr_precision * rr_recall) / (rr_precision + rr_recall)
        print(f"RR F1-Score: {rr_f1:.4f} ({rr_f1*100:.2f}%)")
    print("\n--- Details ---")
    print(f"Total unresolved entities in iText2KG: {len(similar_entities)}")
    print(f"Ground truth merged entities: {ground_truth_merged}")
    print(f"Total unresolved relations in iText2KG: {len(similar_relations)}")
    print(f"Ground truth merged relations: {ground_truth_merged_relations}")
    print("=" * 80)
    
    return {
        'er_precision': er_precision,
        'er_recall': er_recall,
        'rr_precision': rr_precision,
        'rr_recall': rr_recall,
        'threshold': THRESHOLD,
        'similar_entities': similar_entities,
        'similar_relations': similar_relations,
        'ground_truth_merged': ground_truth_merged,
        'ground_truth_merged_relations': ground_truth_merged_relations
    }


if __name__ == "__main__":
    # Run the async main function
    results = asyncio.run(main())


