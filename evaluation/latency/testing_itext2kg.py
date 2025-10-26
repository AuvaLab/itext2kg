"""
iText2KG Latency Testing and Benchmarking

This script measures the latency performance of the iText2KG knowledge graph construction system
(a baseline approach) by processing batches of factoids and tracking timing metrics. It provides
a comparative baseline for evaluating ATOM's performance in terms of processing speed and
scalability when building knowledge graphs from text.

Usage:
    python testing_itext2kg.py

Output:
    - Cached knowledge graph states after each batch
    - JSON file with detailed latency metrics per batch
    - Processing logs and timing information
"""

import os
import pickle
import json
import time
import asyncio
import logging
from datetime import datetime
from typing import Tuple, Optional
from itext2kg import iText2KG
from atom.models import KnowledgeGraph
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Global Parameters - Modify these as needed
CACHE_DIR = "./batch_cache_itext2kg"
BATCH_SIZE = 40
ENT_THRESHOLD = 0.8
REL_THRESHOLD = 0.7
MAX_WORKERS = 8

# OpenAI Configuration
OPENAI_API_KEY = "###"

# Initialize LLM and Embeddings models
openai_llm_model = ChatOpenAI(
    api_key=OPENAI_API_KEY,
    model="gpt-4.1-2025-04-14",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

openai_embeddings_model = OpenAIEmbeddings(
    api_key=OPENAI_API_KEY,
    model="text-embedding-3-large",
)

# Data file path
DATA_FILE_PATH = "/Users/yassirlairgi/Developer/Projects/ATOM_Article/evaluation/2020-covid-news_factoids.pkl"

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def find_last_completed_batch(cache_dir: str) -> Tuple[int, Optional[KnowledgeGraph]]:
    """
    Find the last successfully completed batch and load its knowledge graph.
    
    Args:
        cache_dir: Directory containing batch cache files
    
    Returns:
        Tuple of (last_batch_index, knowledge_graph)
        - last_batch_index: -1 if no batches found, otherwise the highest batch index
        - knowledge_graph: None if no batches found, otherwise the KG from the last batch
    """
    if not os.path.exists(cache_dir):
        logger.info(f"Cache directory {cache_dir} does not exist. Starting fresh.")
        return -1, None
    
    batch_files = []
    for filename in os.listdir(cache_dir):
        if filename.startswith('batch_') and filename.endswith('_kg.pkl'):
            try:
                # Extract batch index from filename like 'batch_5_kg.pkl'
                batch_idx = int(filename.split('_')[1])
                batch_files.append((batch_idx, filename))
            except (ValueError, IndexError):
                continue
    
    if not batch_files:
        logger.info("No existing batch files found. Starting fresh.")
        return -1, None
    
    # Sort by batch index and get the highest one
    batch_files.sort(key=lambda x: x[0])
    last_batch_idx, last_batch_file = batch_files[-1]
    
    # Load the knowledge graph from the last batch
    try:
        kg_path = os.path.join(cache_dir, last_batch_file)
        with open(kg_path, 'rb') as f:
            knowledge_graph = pickle.load(f)
        
        logger.info(f"🔄 RESUMING: Found {len(batch_files)} existing batches. "
                   f"Last completed batch: {last_batch_idx}")
        logger.info(f"📊 Loaded KG with {len(knowledge_graph.entities)} entities "
                   f"and {len(knowledge_graph.relationships)} relationships")
        
        return last_batch_idx, knowledge_graph
    except Exception as e:
        logger.error(f"Error loading batch {last_batch_idx}: {e}")
        logger.info("Starting fresh due to loading error.")
        return -1, None


def load_existing_latency_stats(cache_dir: str) -> list:
    """
    Load existing latency statistics if they exist, or reconstruct them from batch files.
    
    Args:
        cache_dir: Directory containing batch cache files
    
    Returns:
        List of existing latency statistics, or empty list if none found
    """
    latency_path = os.path.join(cache_dir, 'batch_latency_stats.json')
    
    # Try to load existing stats file first
    if os.path.exists(latency_path):
        try:
            with open(latency_path, 'r') as f:
                existing_stats = json.load(f)
            logger.info(f"📈 Loaded {len(existing_stats)} existing latency records")
            return existing_stats
        except Exception as e:
            logger.error(f"Error loading existing latency stats: {e}")
            logger.info("🔧 Attempting to reconstruct from batch files...")
    else:
        logger.info("No existing latency statistics file found.")
        logger.info("🔧 Attempting to reconstruct from batch files...")
    
    # Fallback: try to reconstruct from existing batch files
    reconstructed_stats = reconstruct_missing_batch_stats(cache_dir)
    
    if reconstructed_stats:
        # Save the reconstructed stats for future use
        try:
            with open(latency_path, 'w') as f:
                json.dump(reconstructed_stats, f, indent=2)
            logger.info(f"💾 Saved reconstructed statistics to {latency_path}")
        except Exception as e:
            logger.error(f"Could not save reconstructed stats: {e}")
    
    return reconstructed_stats


def reconstruct_missing_batch_stats(cache_dir: str) -> list:
    """
    Reconstruct basic batch statistics from existing batch files when the stats file is missing.
    This is a fallback recovery mechanism.
    
    Args:
        cache_dir: Directory containing batch cache files
    
    Returns:
        List of reconstructed basic batch statistics
    """
    if not os.path.exists(cache_dir):
        return []
    
    batch_files = []
    for filename in os.listdir(cache_dir):
        if filename.startswith('batch_') and filename.endswith('_kg.pkl'):
            try:
                batch_idx = int(filename.split('_')[1])
                batch_files.append((batch_idx, filename))
            except (ValueError, IndexError):
                continue
    
    if not batch_files:
        return []
    
    batch_files.sort(key=lambda x: x[0])
    reconstructed_stats = []
    
    for batch_idx, filename in batch_files:
        try:
            kg_path = os.path.join(cache_dir, filename)
            with open(kg_path, 'rb') as f:
                kg = pickle.load(f)
            
            # Create basic reconstructed stats (missing timing info)
            reconstructed_stat = {
                'batch_idx': batch_idx,
                'batch_size': 'unknown',
                'total_facts_processed': 'unknown',
                'total_latency_seconds': 0,  # Can't recover this
                'api_latencies': {
                    'extraction_calls': 'unknown',
                    'extraction_total_seconds': 0,
                    'extraction_avg_seconds': 0,
                    'entity_embedding_calls': 'unknown',
                    'entity_embedding_total_seconds': 0,
                    'entity_embedding_avg_seconds': 0,
                    'relationship_embedding_calls': 'unknown',
                    'relationship_embedding_total_seconds': 0,
                    'relationship_embedding_avg_seconds': 0,
                },
                'timestamp': 'reconstructed',
                'entities_count': len(kg.entities),
                'relationships_count': len(kg.relationships),
                'is_reconstructed': True  # Flag to indicate this was reconstructed
            }
            reconstructed_stats.append(reconstructed_stat)
            
        except Exception as e:
            logger.warning(f"Could not reconstruct stats for {filename}: {e}")
            continue
    
    if reconstructed_stats:
        logger.info(f"🔧 RECOVERED: Reconstructed basic statistics for {len(reconstructed_stats)} batches")
        logger.warning("⚠️  Timing information was lost and cannot be recovered")
    
    return reconstructed_stats


def show_batch_stats_summary(existing_stats: list, start_batch_idx: int):
    """
    Show a summary of existing batch statistics during resume.
    
    Args:
        existing_stats: List of existing batch statistics
        start_batch_idx: Index of the batch we're starting from
    """
    if not existing_stats:
        logger.info("📊 No existing batch statistics to summarize")
        return
    
    completed_batches = len(existing_stats)
    total_facts = sum(stat.get('total_facts_processed', 0) for stat in existing_stats if isinstance(stat.get('total_facts_processed', 0), (int, float)))
    total_time = sum(stat.get('total_latency_seconds', 0) for stat in existing_stats if isinstance(stat.get('total_latency_seconds', 0), (int, float)))
    last_entities = existing_stats[-1].get('entities_count', 0) if existing_stats else 0
    last_relationships = existing_stats[-1].get('relationships_count', 0) if existing_stats else 0
    
    # Check if any stats were reconstructed
    has_reconstructed = any(stat.get('is_reconstructed', False) for stat in existing_stats)
    
    logger.info(f"📊 BATCH STATS SUMMARY:")
    logger.info(f"   ✅ Completed batches: {completed_batches}")
    if total_facts > 0:
        logger.info(f"   🔢 Total facts processed: {total_facts}")
    else:
        logger.info(f"   🔢 Total facts processed: [unknown - reconstructed data]")
    if total_time > 0:
        logger.info(f"   ⏱️  Total processing time: {total_time:.2f}s ({total_time/60:.1f} minutes)")
    else:
        logger.info(f"   ⏱️  Total processing time: [unknown - timing data lost]")
    logger.info(f"   📈 Final KG size: {last_entities} entities, {last_relationships} relationships")
    logger.info(f"   🎯 Next batch to process: {start_batch_idx}")
    
    if has_reconstructed:
        logger.info(f"   ⚠️  Some statistics were reconstructed (timing data unavailable)")


def cleanup_incomplete_batch_files(cache_dir: str, last_completed_batch: int):
    """
    Clean up any incomplete batch files that might exist beyond the last completed batch.
    
    Args:
        cache_dir: Directory containing batch cache files
        last_completed_batch: Index of the last successfully completed batch
    """
    if not os.path.exists(cache_dir):
        return
    
    cleaned_files = 0
    for filename in os.listdir(cache_dir):
        if filename.startswith('batch_') and filename.endswith('_kg.pkl'):
            try:
                batch_idx = int(filename.split('_')[1])
                if batch_idx > last_completed_batch:
                    file_path = os.path.join(cache_dir, filename)
                    os.remove(file_path)
                    cleaned_files += 1
                    logger.info(f"🧹 Cleaned up incomplete batch file: {filename}")
            except (ValueError, IndexError):
                continue
    
    if cleaned_files > 0:
        logger.info(f"🧹 Cleaned up {cleaned_files} incomplete batch files")


async def batch_build_graph_itext2kg(itext2kg_instance,
                                      all_facts: list,
                                      existing_knowledge_graph: KnowledgeGraph = None,
                                      cache_dir: str = CACHE_DIR,
                                      ent_threshold: float = ENT_THRESHOLD,
                                      rel_threshold: float = REL_THRESHOLD,
                                      batch_size: int = BATCH_SIZE,
                                      enable_retry: bool = True,
                                      ):
    """
    Batch processing function for building knowledge graphs using iText2KG.
    
    Args:
        itext2kg_instance: An instance of the iText2KG class
        all_facts: List of facts to process
        existing_knowledge_graph: Optional existing KG to merge with
        cache_dir: Directory to save batch results and latency stats
        ent_threshold: Entity matching threshold
        rel_threshold: Relationship matching threshold
        batch_size: Number of facts to process per batch
        enable_retry: Whether to enable retry mechanism (resume from last batch)
    
    Returns:
        Final merged KnowledgeGraph
    """
    
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
    
    # 🔄 RETRY MECHANISM: Check for existing batches and resume if enabled
    start_batch_idx = 0
    current_kg = existing_knowledge_graph
    existing_latency_stats = []
    
    if enable_retry:
        last_batch_idx, resumed_kg = find_last_completed_batch(cache_dir)
        existing_latency_stats = load_existing_latency_stats(cache_dir)
        
        if last_batch_idx >= 0:
            start_batch_idx = last_batch_idx + 1
            current_kg = resumed_kg
            cleanup_incomplete_batch_files(cache_dir, last_batch_idx)
            show_batch_stats_summary(existing_latency_stats, start_batch_idx)
            
            logger.info(f"🚀 RESUMING from batch {start_batch_idx}")
        else:
            logger.info("🆕 STARTING fresh - no previous batches found")
    else:
        logger.info("🔄 RETRY DISABLED - starting fresh regardless of existing batches")
    
    # Create batches
    batches = []
    for i in range(0, len(all_facts), batch_size):
        batch = all_facts[i:i + batch_size]
        batches.append(batch)
    
    latency_stats = existing_latency_stats.copy()  # Start with existing stats
    
    # Determine which batches to process
    batches_to_process = batches[start_batch_idx:]
    total_batches = len(batches)
    
    if start_batch_idx > 0:
        logger.info(f"📊 PROGRESS: Skipping {start_batch_idx} already completed batches")
    
    logger.info(f"Processing {len(batches_to_process)} remaining batches "
               f"(batches {start_batch_idx} to {total_batches-1}) with batch_size={batch_size}")
    
    for relative_idx, batch_facts in enumerate(batches_to_process):
        actual_batch_idx = start_batch_idx + relative_idx
        logger.info(f"Processing batch {actual_batch_idx + 1}/{total_batches} "
                   f"(batch index {actual_batch_idx})")
        
        batch_start_time = time.time()
        
        # Track API latencies by monkey-patching the methods temporarily
        api_latencies = {'extraction': [], 'entity_embedding': [], 'relationship_embedding': []}
        
        # Store original methods - iText2KG uses similar structure to ATOM
        original_extract = itext2kg_instance.langchain_output_parser.extract_information_as_json_for_context
        
        # Wrap extraction method
        async def tracked_extract(*args, **kwargs):
            start_time = time.time()
            result = await original_extract(*args, **kwargs)
            api_latencies['extraction'].append(time.time() - start_time)
            return result
        
        # Temporarily replace method
        itext2kg_instance.langchain_output_parser.extract_information_as_json_for_context = tracked_extract
        
        # Store original embedding methods from KnowledgeGraph class
        original_embed_entities_class = KnowledgeGraph.embed_entities
        original_embed_relationships_class = KnowledgeGraph.embed_relationships
        
        # Create tracked versions that replace the class methods
        async def tracked_embed_entities(self, *args, **kwargs):
            start_time = time.time()
            result = await original_embed_entities_class(self, *args, **kwargs)
            api_latencies['entity_embedding'].append(time.time() - start_time)
            return result
        
        async def tracked_embed_relationships(self, *args, **kwargs):
            start_time = time.time()
            result = await original_embed_relationships_class(self, *args, **kwargs)
            api_latencies['relationship_embedding'].append(time.time() - start_time)
            return result
        
        # Temporarily replace the class methods
        KnowledgeGraph.embed_entities = tracked_embed_entities
        KnowledgeGraph.embed_relationships = tracked_embed_relationships
        
        try:
            # Build graph for current batch using iText2KG API
            batch_kg = await itext2kg_instance.build_graph(
                sections=batch_facts,
                existing_knowledge_graph=current_kg,
                ent_threshold=ent_threshold,
                rel_threshold=rel_threshold,
            )
            
            batch_end_time = time.time()
            batch_latency = batch_end_time - batch_start_time
            
            # Calculate batch statistics
            total_facts_in_batch = len(batch_facts)
            
            batch_stats = {
                'batch_idx': actual_batch_idx,  # Use actual batch index
                'batch_size': batch_size,
                'total_facts_processed': total_facts_in_batch,
                'total_latency_seconds': batch_latency,
                'api_latencies': {
                    'extraction_calls': len(api_latencies['extraction']),
                    'extraction_total_seconds': sum(api_latencies['extraction']),
                    'extraction_avg_seconds': sum(api_latencies['extraction']) / len(api_latencies['extraction']) if api_latencies['extraction'] else 0,
                    'entity_embedding_calls': len(api_latencies['entity_embedding']),
                    'entity_embedding_total_seconds': sum(api_latencies['entity_embedding']),
                    'entity_embedding_avg_seconds': sum(api_latencies['entity_embedding']) / len(api_latencies['entity_embedding']) if api_latencies['entity_embedding'] else 0,
                    'relationship_embedding_calls': len(api_latencies['relationship_embedding']),
                    'relationship_embedding_total_seconds': sum(api_latencies['relationship_embedding']),
                    'relationship_embedding_avg_seconds': sum(api_latencies['relationship_embedding']) / len(api_latencies['relationship_embedding']) if api_latencies['relationship_embedding'] else 0,
                },
                'timestamp': datetime.now().isoformat(),
                'entities_count': len(batch_kg.entities),
                'relationships_count': len(batch_kg.relationships),
                'is_resumed_batch': start_batch_idx > 0  # Flag to indicate if this was a resumed run
            }
            
            latency_stats.append(batch_stats)
            
            # 💾 SAVE INCREMENTAL BATCH STATISTICS after each successful batch
            if cache_dir:
                latency_path = os.path.join(cache_dir, 'batch_latency_stats.json')
                with open(latency_path, 'w') as f:
                    json.dump(latency_stats, f, indent=2)
                logger.info(f"📊 Updated latency statistics with batch {actual_batch_idx}")
            
            # Save batch KG to cache
            if cache_dir:
                kg_path = os.path.join(cache_dir, f'batch_{actual_batch_idx}_kg.pkl')
                with open(kg_path, 'wb') as f:
                    pickle.dump(batch_kg, f)
                logger.info(f"💾 Saved batch {actual_batch_idx} KG to {kg_path}")
            
            # Update current KG for next iteration
            current_kg = batch_kg
            
            logger.info(f"✅ Batch {actual_batch_idx + 1} completed in {batch_latency:.2f}s - "
                       f"Entities: {len(batch_kg.entities)}, Relationships: {len(batch_kg.relationships)}")
            
        except Exception as e:
            logger.error(f"❌ ERROR in batch {actual_batch_idx}: {e}")
            logger.info(f"💾 Progress saved up to batch {actual_batch_idx - 1}")
            logger.info(f"🔄 To resume, run the script again - it will continue from batch {actual_batch_idx}")
            raise  # Re-raise the exception to stop processing
            
        finally:
            # Restore original methods
            itext2kg_instance.langchain_output_parser.extract_information_as_json_for_context = original_extract
            KnowledgeGraph.embed_entities = original_embed_entities_class
            KnowledgeGraph.embed_relationships = original_embed_relationships_class
    
    # Save latency statistics (this will include both existing and new stats)
    if cache_dir:
        latency_path = os.path.join(cache_dir, 'batch_latency_stats.json')
        with open(latency_path, 'w') as f:
            json.dump(latency_stats, f, indent=2)
        logger.info(f"📊 Saved complete latency statistics to {latency_path}")
    
    total_entities = len(current_kg.entities) if current_kg else 0
    total_relationships = len(current_kg.relationships) if current_kg else 0
    
    if start_batch_idx > 0:
        logger.info(f"🎉 RESUMED PROCESSING COMPLETED! Final KG - Entities: {total_entities}, Relationships: {total_relationships}")
    else:
        logger.info(f"🎉 FRESH PROCESSING COMPLETED! Final KG - Entities: {total_entities}, Relationships: {total_relationships}")
    
    return current_kg


def load_covid_data():
    """Load the COVID-19 factoids data from pickle file and flatten into a list."""
    try:
        with open(DATA_FILE_PATH, 'rb') as f:
            data = pickle.load(f)
        
        # Flatten the data structure (dict of dates -> list of facts) into a single list
        all_facts = []
        for date, facts in data.items():
            all_facts.extend(facts)
        
        logger.info(f"Loaded data with {len(data)} dates from {DATA_FILE_PATH}")
        logger.info(f"Total facts: {len(all_facts)}")
        return all_facts
    except Exception as e:
        logger.error(f"Error loading data from {DATA_FILE_PATH}: {e}")
        return []


async def run_itext2kg_batch_processing():
    """
    Main function to run batch processing on COVID-19 factoids data using iText2KG.
    """
    
    # Load the COVID-19 factoids data
    all_facts = load_covid_data()
    
    if not all_facts:
        logger.error("No data loaded. Exiting.")
        return None
    
    # Initialize iText2KG instance
    itext2kg_instance = iText2KG(llm_model=openai_llm_model, embeddings_model=openai_embeddings_model)
    
    logger.info("Starting iText2KG batch processing...")
    logger.info(f"Total facts in dataset: {len(all_facts)}")
    
    # Run batch processing with retry enabled
    final_kg = await batch_build_graph_itext2kg(
        itext2kg_instance=itext2kg_instance,
        all_facts=all_facts,
        existing_knowledge_graph=None,  # Start fresh (retry mechanism will handle resume)
        cache_dir=CACHE_DIR,
        ent_threshold=ENT_THRESHOLD,
        rel_threshold=REL_THRESHOLD,
        batch_size=BATCH_SIZE,
        enable_retry=True  # Enable retry mechanism
    )
    
    logger.info("iText2KG batch processing completed!")
    return final_kg


if __name__ == "__main__":
    print("COVID-19 iText2KG Batch Processing Script")
    print("="*50)
    print(f"Global Parameters:")
    print(f"  CACHE_DIR: {CACHE_DIR}")
    print(f"  BATCH_SIZE: {BATCH_SIZE}")
    print(f"  ENT_THRESHOLD: {ENT_THRESHOLD}")
    print(f"  REL_THRESHOLD: {REL_THRESHOLD}")
    print(f"  DATA_FILE: {DATA_FILE_PATH}")
    print(f"  LLM Model: {openai_llm_model.model_name}")
    print(f"  Embeddings Model: {openai_embeddings_model.model}")
    print("="*50)
    
    # Run the batch processing
    asyncio.run(run_itext2kg_batch_processing())

