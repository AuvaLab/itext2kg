"""
Stability Evaluation of Quintuple Extraction

This script evaluates the stability of quintuple extraction across multiple runs
of the same LLM on the same input data. It compares quintuples extracted from different runs
using semantic similarity (embeddings) to compute stability metrics. High stability indicates
that the extraction process produces similar (stable) results across runs.

Usage:
    python calculate_stability.py

Output:
    - JSON file with stability metrics for different model configurations
    - Comparison of stability across different extraction approaches (direct extraction vs. factoid-based extraction)
"""

import asyncio
import json
import logging
import time
import argparse
import numpy as np
import pandas as pd
import pickle
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
import sys
from pathlib import Path

# Add the project root to Python path (same pattern as other scripts)
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
sys.path.append(str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Console output
    ]
)
logger = logging.getLogger(__name__)

print("🚀 Starting stability evaluation script...")
logger.info("Setting up configuration and API connections...")

# ============================================================================
# GLOBAL CONFIGURATION VARIABLES
# ============================================================================

# Data configuration - assume same data file structure as quality evaluation
DATA_PATH = project_root / "datasets" / "nyt_news" / "2020_nyt_COVID_last_version_ready_quintuples_gpt41_from_factoids_run3_run3.pkl"

# Column pairs to compare for stability
STABILITY_COMPARISONS = [
    {
        'name': 'factoids_stability',
        'col1': 'quintuples_gpt41_from_factoids_run3',
        'col2': 'quintuples_gpt41_from_factoids',
        'description': 'GPT-4.1 from Factoids: Run 3 vs Run 1'
    },
    {
        'name': 'direct_stability',
        'col1': 'quintuples_gpt41_run3',
        'col2': 'quintuples_gpt41',
        'description': 'GPT-4.1 Direct: Run 3 vs Run 1'
    }
]

# Analysis parameters
MAX_SAMPLES = None  # Set to None for all samples, or integer for limit

# Output configuration
OUTPUT_JSON = project_root / "evaluation" / "stability_results.json"
EMBEDDINGS_CACHE = project_root / "evaluation" / "stability_embeddings_cache.pkl"

# ============================================================================
# CORE FUNCTIONS
# ============================================================================

def format_quintuple_text(quintuple):
    """
    Format a quintuple as '{subject} {predicate} {object} {t_start}-{t_end}'.
    
    Args:
        quintuple: Tuple/list with (subject, predicate, object, t_start, t_end)
        
    Returns:
        Formatted string representation
    """
    if not quintuple or len(quintuple) < 3:
        return ""
    
    subject = str(quintuple[0]) if quintuple[0] is not None else ""
    predicate = str(quintuple[1]) if quintuple[1] is not None else ""
    obj = str(quintuple[2]) if quintuple[2] is not None else ""
    
    # Handle temporal information
    t_start = str(quintuple[3]) if len(quintuple) > 3 and quintuple[3] is not None else ""
    t_end = str(quintuple[4]) if len(quintuple) > 4 and quintuple[4] is not None else ""
    
    # Format temporal part
    if t_start or t_end:
        temporal = f"{t_start}-{t_end}"
    else:
        temporal = ""
    
    # Combine all parts
    if temporal:
        return f"{subject} {predicate} {obj} {temporal}"
    else:
        return f"{subject} {predicate} {obj}"

def save_embeddings_cache(cache_data, cache_path):
    """Save embeddings to cache file."""
    cache_data['timestamp'] = datetime.now().isoformat()
    with open(cache_path, 'wb') as f:
        pickle.dump(cache_data, f)
    logger.info(f"Embeddings cached to {cache_path}")

def load_embeddings_cache(cache_path):
    """Load embeddings from cache file."""
    try:
        with open(cache_path, 'rb') as f:
            cache_data = pickle.load(f)
        logger.info(f"Loaded cached embeddings from {cache_path}")
        return cache_data
    except (FileNotFoundError, Exception) as e:
        logger.info(f"No valid cache found: {e}")
        return {}

async def calculate_quintuple_embeddings(quintuples, lg_kg_construction, cache_dict=None):
    """
    Calculate embeddings for a list of quintuples.
    
    Args:
        quintuples: List of quintuples
        lg_kg_construction: Language model construction object
        cache_dict: Dictionary to use for caching
        
    Returns:
        List of embeddings
    """
    if not quintuples:
        return []
    
    # Format quintuples as text
    quintuple_texts = [format_quintuple_text(q) for q in quintuples]
    
    # Filter out empty texts
    valid_texts = [text for text in quintuple_texts if text.strip()]
    if not valid_texts:
        return []
    
    # Check cache if provided
    if cache_dict is not None:
        cached_embeddings = []
        uncached_texts = []
        uncached_indices = []
        
        for i, text in enumerate(valid_texts):
            if text in cache_dict:
                cached_embeddings.append((i, cache_dict[text]))
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)
        
        # Calculate embeddings for uncached texts
        if uncached_texts:
            new_embeddings = await lg_kg_construction.calculate_embeddings(text=uncached_texts)
            
            # Update cache
            for text, embedding in zip(uncached_texts, new_embeddings):
                cache_dict[text] = embedding
            
            # Combine cached and new embeddings in correct order
            final_embeddings = [None] * len(valid_texts)
            
            # Add cached embeddings
            for idx, embedding in cached_embeddings:
                final_embeddings[idx] = embedding
            
            # Add new embeddings
            for i, embedding in enumerate(new_embeddings):
                original_idx = uncached_indices[i]
                final_embeddings[original_idx] = embedding
            
            return final_embeddings
        else:
            # All embeddings were cached
            return [embedding for _, embedding in sorted(cached_embeddings)]
    else:
        # No cache, calculate all embeddings
        return await lg_kg_construction.calculate_embeddings(text=valid_texts)

async def calculate_row_stability(quintuples1, quintuples2, lg_kg_construction, cache_dict=None):
    """
    Calculate stability between two sets of quintuples for a single row.
    
    Args:
        quintuples1: First set of quintuples
        quintuples2: Second set of quintuples  
        lg_kg_construction: Language model construction object
        cache_dict: Embeddings cache dictionary
        
    Returns:
        Dictionary with similarity scores and metadata
    """
    # Handle empty cases
    if not quintuples1 and not quintuples2:
        return {'similarity': 1.0, 'count1': 0, 'count2': 0, 'comparison_type': 'both_empty'}
    
    if not quintuples1 or not quintuples2:
        return {'similarity': 0.0, 'count1': len(quintuples1) if quintuples1 else 0, 
                'count2': len(quintuples2) if quintuples2 else 0, 'comparison_type': 'one_empty'}
    
    # Calculate embeddings for both sets
    embeddings1 = await calculate_quintuple_embeddings(quintuples1, lg_kg_construction, cache_dict)
    embeddings2 = await calculate_quintuple_embeddings(quintuples2, lg_kg_construction, cache_dict)
    
    if not embeddings1 or not embeddings2:
        return {'similarity': 0.0, 'count1': len(quintuples1), 'count2': len(quintuples2), 
                'comparison_type': 'no_embeddings'}
    
    # Convert to numpy arrays
    embeddings1 = np.array(embeddings1)
    embeddings2 = np.array(embeddings2)
    
    # Ensure 2D arrays
    if embeddings1.ndim == 1:
        embeddings1 = embeddings1.reshape(1, -1)
    if embeddings2.ndim == 1:
        embeddings2 = embeddings2.reshape(1, -1)
    
    # Calculate cosine similarity matrix
    similarity_matrix = cosine_similarity(embeddings1, embeddings2)
    
    # Calculate various similarity measures
    max_similarities = []
    
    # For each embedding in set1, find the best match in set2
    for i in range(similarity_matrix.shape[0]):
        max_sim = np.max(similarity_matrix[i, :])
        max_similarities.append(max_sim)
    
    # Average of maximum similarities
    avg_max_similarity = np.mean(max_similarities)
    
    # Overall matrix mean (alternative measure)
    overall_mean_similarity = np.mean(similarity_matrix)
    
    return {
        'similarity': float(avg_max_similarity),
        'overall_mean_similarity': float(overall_mean_similarity),
        'count1': len(quintuples1),
        'count2': len(quintuples2),
        'comparison_type': 'full_comparison',
        'similarity_matrix_shape': similarity_matrix.shape,
        'max_similarities': [float(x) for x in max_similarities]
    }

async def evaluate_stability(df, lg_kg_construction, max_samples=None):
    """
    Evaluate stability across all specified column comparisons.
    
    Args:
        df: DataFrame containing the data
        lg_kg_construction: Language model construction object
        max_samples: Maximum number of samples to process
        
    Returns:
        Dictionary with results for all comparisons
    """
    print("🚀 Evaluating stability across column comparisons")
    logger.info("Starting stability evaluation")
    
    # Check if required columns exist
    all_required_cols = []
    valid_comparisons = []
    
    for comparison in STABILITY_COMPARISONS:
        col1, col2 = comparison['col1'], comparison['col2']
        if col1 in df.columns and col2 in df.columns:
            valid_comparisons.append(comparison)
            all_required_cols.extend([col1, col2])
        else:
            logger.warning(f"Skipping comparison {comparison['name']}: missing columns {col1} or {col2}")
    
    if not valid_comparisons:
        logger.error("No valid column comparisons found")
        return None
    
    # Filter to rows that have data in at least one comparison
    valid_mask = pd.Series([False] * len(df))
    for comparison in valid_comparisons:
        col1, col2 = comparison['col1'], comparison['col2']
        valid_mask |= (df[col1].notna() | df[col2].notna())
    
    valid_df = df[valid_mask].copy()
    
    if max_samples:
        valid_df = valid_df.head(max_samples)
    
    if len(valid_df) == 0:
        logger.error("No valid data found")
        return None
    
    logger.info(f"Processing {len(valid_df)} valid samples across {len(valid_comparisons)} comparisons")
    
    # Load embeddings cache
    cache_data = load_embeddings_cache(EMBEDDINGS_CACHE)
    embeddings_cache = cache_data.get('embeddings_cache', {})
    
    results = {}
    
    # Process each comparison
    for comparison in valid_comparisons:
        comparison_name = comparison['name']
        col1, col2 = comparison['col1'], comparison['col2']
        
        print(f"\n🔍 Processing {comparison['description']}")
        logger.info(f"Processing comparison: {comparison_name}")
        
        comparison_results = []
        
        # Process each row
        for row_idx, idx in enumerate(valid_df.index):
            if row_idx % 10 == 0:
                logger.info(f"Processing row {row_idx + 1}/{len(valid_df)} for {comparison_name}")
            
            quintuples1 = valid_df[col1].loc[idx]
            quintuples2 = valid_df[col2].loc[idx]
            
            # Calculate stability for this row
            row_result = await calculate_row_stability(
                quintuples1=quintuples1,
                quintuples2=quintuples2,
                lg_kg_construction=lg_kg_construction,
                cache_dict=embeddings_cache
            )
            
            row_result['row_idx'] = int(idx)
            comparison_results.append(row_result)
        
        results[comparison_name] = comparison_results
    
    # Save updated embeddings cache
    updated_cache_data = cache_data.copy()
    updated_cache_data['embeddings_cache'] = embeddings_cache
    save_embeddings_cache(updated_cache_data, EMBEDDINGS_CACHE)
    
    logger.info("Stability evaluation completed")
    return results

def calculate_stability_statistics(results):
    """
    Calculate summary statistics for stability results.
    
    Args:
        results: Dictionary with results for each comparison
        
    Returns:
        Dictionary with summary statistics
    """
    logger.info("Calculating stability statistics")
    
    summary = {}
    
    for comparison_name, comparison_results in results.items():
        if not comparison_results:
            continue
        
        # Extract similarities
        similarities = [result['similarity'] for result in comparison_results]
        overall_similarities = [result.get('overall_mean_similarity', result['similarity']) 
                               for result in comparison_results]
        
        # Calculate basic statistics
        comparison_summary = {
            'mean_similarity': float(np.mean(similarities)),
            'std_similarity': float(np.std(similarities)),
            'min_similarity': float(np.min(similarities)),
            'max_similarity': float(np.max(similarities)),
            'median_similarity': float(np.median(similarities)),
            'mean_overall_similarity': float(np.mean(overall_similarities)),
            'std_overall_similarity': float(np.std(overall_similarities)),
            'n_samples': len(comparison_results)
        }
        
        # Calculate distribution of comparison types
        comparison_types = {}
        for result in comparison_results:
            comp_type = result.get('comparison_type', 'unknown')
            comparison_types[comp_type] = comparison_types.get(comp_type, 0) + 1
        
        comparison_summary['comparison_types'] = comparison_types
        
        # Calculate count statistics
        count1_values = [result['count1'] for result in comparison_results]
        count2_values = [result['count2'] for result in comparison_results]
        
        comparison_summary.update({
            'mean_count1': float(np.mean(count1_values)),
            'std_count1': float(np.std(count1_values)),
            'mean_count2': float(np.mean(count2_values)),
            'std_count2': float(np.std(count2_values)),
            'total_count1': int(np.sum(count1_values)),
            'total_count2': int(np.sum(count2_values))
        })
        
        summary[comparison_name] = comparison_summary
    
    return summary

def save_stability_results(results, summary, output_path):
    """
    Save stability results and summary to JSON file.
    
    Args:
        results: Dictionary with detailed results
        summary: Dictionary with summary statistics
        output_path: Path to save JSON file
    """
    logger.info(f"Saving stability results to: {output_path}")
    
    # Add metadata
    output_data = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'data_path': str(DATA_PATH),
            'comparisons': STABILITY_COMPARISONS,
            'total_samples_per_comparison': {
                name: len(results.get(name, [])) for name in [comp['name'] for comp in STABILITY_COMPARISONS]
            }
        },
        'summary_statistics': summary,
        'detailed_results': results
    }
    
    try:
        with open(str(output_path), 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"💾 Results saved to: {output_path}")
        logger.info(f"Successfully saved results to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save results to {output_path}: {e}")
        raise

def print_stability_report(summary):
    """Print a formatted stability report."""
    print("\n📊 STABILITY EVALUATION SUMMARY")
    print("=" * 80)
    
    for comparison_name, comparison_summary in summary.items():
        # Find the comparison details
        comparison_desc = next(
            (comp['description'] for comp in STABILITY_COMPARISONS if comp['name'] == comparison_name),
            comparison_name
        )
        
        print(f"\n🔍 {comparison_desc}")
        print("-" * 60)
        
        mean_sim = comparison_summary.get('mean_similarity', 0.0)
        std_sim = comparison_summary.get('std_similarity', 0.0)
        n_samples = comparison_summary.get('n_samples', 0)
        
        print(f"  📊 SIMILARITY STATISTICS:")
        print(f"    Mean Similarity    : {mean_sim:.4f} ± {std_sim:.4f}")
        print(f"    Min Similarity     : {comparison_summary.get('min_similarity', 0.0):.4f}")
        print(f"    Max Similarity     : {comparison_summary.get('max_similarity', 0.0):.4f}")
        print(f"    Median Similarity  : {comparison_summary.get('median_similarity', 0.0):.4f}")
        
        print(f"\n  🔢 DATASET STATISTICS:")
        print(f"    Total Samples      : {n_samples}")
        print(f"    Avg Count Run 1    : {comparison_summary.get('mean_count2', 0.0):.2f} ± {comparison_summary.get('std_count2', 0.0):.2f}")
        print(f"    Avg Count Run 2    : {comparison_summary.get('mean_count1', 0.0):.2f} ± {comparison_summary.get('std_count1', 0.0):.2f}")
        print(f"    Total Count Run 1  : {comparison_summary.get('total_count2', 0)}")
        print(f"    Total Count Run 2  : {comparison_summary.get('total_count1', 0)}")
        
        # Show comparison types distribution
        comparison_types = comparison_summary.get('comparison_types', {})
        if comparison_types:
            print(f"\n  📋 COMPARISON TYPES:")
            for comp_type, count in comparison_types.items():
                percentage = (count / n_samples) * 100 if n_samples > 0 else 0
                print(f"    {comp_type:15s}: {count:4d} ({percentage:5.1f}%)")

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Calculate stability between different quintuple extraction runs')
    parser.add_argument('--max-samples', '-m', type=int, default=None,
                       help='Maximum number of samples to process (for testing)')
    parser.add_argument('--data-path', '-d', type=str, default=None,
                       help='Path to the data file (overrides default)')
    return parser.parse_args()

async def main():
    """
    Main function to run the stability evaluation.
    """
    start_time = time.time()
    
    # Parse command line arguments
    args = parse_arguments()
    
    print("🎯 Starting Stability Evaluation")
    print("=" * 50)
    logger.info("Beginning stability evaluation")
    
    # Update configuration based on arguments
    if args.max_samples:
        global MAX_SAMPLES
        MAX_SAMPLES = args.max_samples
        print(f"🎯 Limited to {MAX_SAMPLES} samples (testing mode)")
        logger.info(f"Testing mode: limited to {MAX_SAMPLES} samples")
    
    if args.data_path:
        global DATA_PATH
        DATA_PATH = Path(args.data_path)
        print(f"🎯 Using custom data path: {DATA_PATH}")
        logger.info(f"Using custom data path: {DATA_PATH}")
    
    try:
        # Import ATOM modules
        try:
            from atom.llm_output_parsing.langchain_output_parser import LangchainOutputParser
            from langchain_openai import ChatOpenAI, OpenAIEmbeddings
            print("   ✅ ATOM modules imported successfully")
            logger.info("ATOM modules imported successfully")
        except ImportError as e:
            print(f"❌ Error importing ATOM modules: {e}")
            logger.error(f"Failed to import ATOM modules: {e}")
            return
        
        # Load data
        print(f"📁 Loading data from: {DATA_PATH}")
        logger.info(f"Loading dataset from {DATA_PATH}")
        try:
            df = pd.read_pickle(DATA_PATH)
            print(f"   ✅ Loaded {len(df)} samples")
            logger.info(f"Successfully loaded dataset with {len(df)} samples")
            
            # Show available columns for debugging
            available_cols = [col for col in df.columns if 'quintuple' in col.lower()]
            print(f"   📋 Available quintuple columns: {available_cols}")
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            logger.error(f"Failed to load dataset: {e}")
            return
        
        # Initialize language model components
        print("🤖 Initializing language model components...")
        logger.info("Initializing language model components")
        try:
            openai_api_key = "###"
            
            openai_llm_model = ChatOpenAI(
                api_key=openai_api_key,
                model="o3-mini",
                max_tokens=None,
                timeout=None,
                max_retries=2,
            )
            
            openai_embeddings_model = OpenAIEmbeddings(
                api_key=openai_api_key,
                model="text-embedding-3-large",
            )
            
            lg_kg_construction = LangchainOutputParser(
                llm_model=openai_llm_model,
                embeddings_model=openai_embeddings_model
            )
            print("   ✅ Language model components initialized")
            logger.info("Language model components initialized successfully")
        except Exception as e:
            print(f"❌ Error initializing language model: {e}")
            logger.error(f"Failed to initialize language model: {e}")
            return
        
        # Run stability evaluation
        print("🔍 Running stability evaluation...")
        logger.info("Starting stability evaluation")
        
        results = await evaluate_stability(
            df=df,
            lg_kg_construction=lg_kg_construction,
            max_samples=MAX_SAMPLES
        )
        
        if results is None:
            print("❌ Stability evaluation failed")
            return
        
        # Calculate summary statistics
        print("📊 Calculating summary statistics...")
        summary = calculate_stability_statistics(results)
        
        # Save results to JSON
        save_stability_results(results, summary, OUTPUT_JSON)
        
        # Print summary report
        print_stability_report(summary)
        
        elapsed_time = time.time() - start_time
        print("\n✨ Stability evaluation complete!")
        print(f"📊 Results saved to: {OUTPUT_JSON}")
        print(f"⏱️  Total time: {elapsed_time:.2f} seconds")
        logger.info(f"Stability evaluation completed successfully in {elapsed_time:.2f} seconds")
        
    except Exception as e:
        elapsed_time = time.time() - start_time
        logger.error(f"Error occurred after {elapsed_time:.2f} seconds: {str(e)}")
        print(f"❌ Error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    print("=" * 50)
    print("  STABILITY EVALUATION")
    print("=" * 50)
    asyncio.run(main())
