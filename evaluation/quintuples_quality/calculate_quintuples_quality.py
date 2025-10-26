"""
Quintuple Extraction Quality Evaluation

This script evaluates the quality of automatically extracted quintuples
against ground truth annotations. It uses semantic similarity (embeddings) to match extracted
entities and relations with ground truth, computing detailed metrics for both entity and relation extraction.

Usage:
    python calculate_quintuples_quality.py

Output:
    - JSON file with detailed quality metrics for different extraction approaches
    - Comparison of direct extraction vs. factoid-based extraction
"""

import asyncio
import json
import logging
import time
import argparse
import numpy as np
import pandas as pd
import pickle
import dateparser
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
import sys
from pathlib import Path

# Add the project root to Python path (same pattern as exhaustivity_evaluation_nyt.py)
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

print("🚀 Starting quintuples quality evaluation script...")
logger.info("Setting up configuration and API connections...")

# ============================================================================
# GLOBAL CONFIGURATION VARIABLES
# ============================================================================

# Data configuration
DATA_PATH = project_root / "datasets" / "nyt_news" / "2020_nyt_COVID_last_version_ready_quintuples_gpt41_from_factoids.pkl"
GOLD_COL = "quintuples_g_truth"
PREDICTED_COL_CASE1 = "quintuples_gpt41"
PREDICTED_COL_CASE2 = "quintuples_gpt41_from_factoids"

# Analysis parameters
SIMILARITY_THRESHOLD = 0.7
MAX_SAMPLES = None # Set to None for all samples, or integer for limit

# Output configuration
OUTPUT_JSON = project_root / "evaluation" / "quintuples_quality_results.json"
GOLD_EMBEDDINGS_CACHE = project_root / "evaluation" / "gold_quintuples_embeddings_cache.pkl"

# ============================================================================
# CORE FUNCTIONS
# ============================================================================

def save_gold_embeddings_cache(gold_embeddings, gold_quintuples, cache_path):
    """Save gold embeddings to cache file."""
    cache_data = {
        'embeddings': gold_embeddings,
        'quintuples': gold_quintuples,
        'timestamp': datetime.now().isoformat()
    }
    with open(cache_path, 'wb') as f:
        pickle.dump(cache_data, f)
    logger.info(f"Gold embeddings cached to {cache_path}")

def load_gold_embeddings_cache(cache_path):
    """Load gold embeddings from cache file."""
    try:
        with open(cache_path, 'rb') as f:
            cache_data = pickle.load(f)
        logger.info(f"Loaded cached gold embeddings from {cache_path}")
        return cache_data['embeddings'], cache_data['quintuples']
    except (FileNotFoundError, Exception) as e:
        logger.info(f"No valid cache found: {e}")
        return None, None

async def calculate_comprehensive_metrics(quintuples, gold_quintuples, lg_kg_construction, 
                                        gold_embeddings=None, threshold=0.7):
    """
    Calculate comprehensive metrics including MATCH, HALL, OM and their temporal variants.
    Uses the correct temporal analysis logic from the notebook.
    
    Args:
        quintuples: List of predicted quintuples (head, relation, tail, t_start, t_end)
        gold_quintuples: List of gold standard quintuples
        lg_kg_construction: Language model construction object for embeddings
        gold_embeddings: Pre-computed gold embeddings (optional)
        threshold: Similarity threshold for matching
        
    Returns:
        Dict with all metrics: MATCH, HALL, OM, MATCH_t, OM_t, HALL_t
    """
    logger.debug(f"Calculating metrics for {len(quintuples)} predicted vs {len(gold_quintuples)} gold quintuples")
    
    if not quintuples and not gold_quintuples:
        return {
            'MATCH': 0.0, 'HALL': 0.0, 'OM': 0.0, 'MATCH_t': 0.0, 'OM_t': 0.0, 'HALL_t': 0.0,
            'MATCH_count': 0, 'HALL_count': 0, 'OM_count': 0, 'MATCH_t_count': 0, 'OM_t_count': 0, 'HALL_t_count': 0,
            'total_predicted': 0, 'total_gold': 0
        }
    
    if not quintuples:
        # All gold quintuples are omitted
        total_gold = len(gold_quintuples)
        return {
            'MATCH': 0.0, 'HALL': 0.0, 'OM': 1.0, 'MATCH_t': 0.0, 'OM_t': 1.0, 'HALL_t': 0.0,
            'MATCH_count': 0, 'HALL_count': 0, 'OM_count': total_gold, 'MATCH_t_count': 0, 'OM_t_count': total_gold, 'HALL_t_count': 0,
            'total_predicted': 0, 'total_gold': total_gold
        }
    
    if not gold_quintuples:
        # All predicted quintuples are hallucinations
        total_predicted = len(quintuples)
        return {
            'MATCH': 0.0, 'HALL': 1.0, 'OM': 0.0, 'MATCH_t': 0.0, 'OM_t': 0.0, 'HALL_t': 0.0,
            'MATCH_count': 0, 'HALL_count': total_predicted, 'OM_count': 0, 'MATCH_t_count': 0, 'OM_t_count': 0, 'HALL_t_count': 0,
            'total_predicted': total_predicted, 'total_gold': 0
        }
    
    # Format quintuples as text for embedding
    quintuple_texts = [f"{q[0]} {q[1]} {q[2]}" for q in quintuples]
    gold_quintuple_texts = [f"{gq[0]} {gq[1]} {gq[2]}" for gq in gold_quintuples]
    
    # Calculate embeddings
    quintuple_embeddings = await lg_kg_construction.calculate_embeddings(text=quintuple_texts)
    
    # Use cached gold embeddings if available
    if gold_embeddings is not None:
        gold_quintuple_embeddings = gold_embeddings
    else:
        gold_quintuple_embeddings = await lg_kg_construction.calculate_embeddings(text=gold_quintuple_texts)
    
    # Convert to numpy arrays and ensure proper shape
    quintuple_embeddings = np.array(quintuple_embeddings)
    gold_quintuple_embeddings = np.array(gold_quintuple_embeddings)
    
    if quintuple_embeddings.ndim == 1:
        quintuple_embeddings = quintuple_embeddings.reshape(1, -1)
    if gold_quintuple_embeddings.ndim == 1:
        gold_quintuple_embeddings = gold_quintuple_embeddings.reshape(1, -1)
    
    # Compute similarity matrix
    similarity_matrix = cosine_similarity(quintuple_embeddings, gold_quintuple_embeddings)
    
    def is_empty_temporal(value):
        """Check if temporal value is empty (None, '', or 'None')"""
        return value is None or value == '' or str(value).lower() == 'none'
    
    def temporal_similar(pred_val, gold_val):
        """Check if temporal values are similar using dateparser for robust date comparison"""
        if is_empty_temporal(pred_val) and is_empty_temporal(gold_val):
            return True
        if is_empty_temporal(pred_val) or is_empty_temporal(gold_val):
            return False
        
        # Try to parse both dates using dateparser
        try:
            pred_date = dateparser.parse(str(pred_val).strip())
            gold_date = dateparser.parse(str(gold_val).strip())
            
            # If both dates parsed successfully, compare them
            if pred_date is not None and gold_date is not None:
                # Compare dates (ignoring time if only date is relevant)
                return pred_date.date() == gold_date.date()
            
            # If parsing failed, fall back to string comparison
            return str(pred_val).strip().lower() == str(gold_val).strip().lower()
            
        except (ValueError, TypeError, AttributeError):
            # If any error occurs during parsing, fall back to string comparison
            return str(pred_val).strip().lower() == str(gold_val).strip().lower()
    
    # Find best matches above threshold with proper one-to-one matching
    matches = []
    temporal_analysis = []
    match_t_count = 0
    om_t_count = 0
    hall_t_count = 0
    used_gold_indices = set()  # Track which gold quintuples have been matched
    
    # Create a list of (predicted_idx, gold_idx, similarity) for all valid matches
    potential_matches = []
    for i, quintuple in enumerate(quintuples):
        similarities = similarity_matrix[i]
        for j, similarity in enumerate(similarities):
            if similarity > threshold:
                potential_matches.append((i, j, similarity))
    
    # Sort by similarity (highest first) to ensure best matches are selected
    potential_matches.sort(key=lambda x: x[2], reverse=True)
    
    # Select matches ensuring one-to-one mapping
    matched_predicted_indices = set()
    for pred_idx, gold_idx, similarity in potential_matches:
        # Skip if either quintuple is already matched
        if pred_idx in matched_predicted_indices or gold_idx in used_gold_indices:
            continue
        
        # Accept this match
        quintuple = quintuples[pred_idx]
        matched_gold = gold_quintuples[gold_idx]
        matches.append(matched_gold)
        matched_predicted_indices.add(pred_idx)
        used_gold_indices.add(gold_idx)
        
        # Temporal analysis for this match - FIXED LOGIC
        pred_t_start = quintuple[3] if len(quintuple) > 3 else None
        pred_t_end = quintuple[4] if len(quintuple) > 4 else None
        gold_t_start = matched_gold[3] if len(matched_gold) > 3 else None
        gold_t_end = matched_gold[4] if len(matched_gold) > 4 else None
        
        # Check temporal matches
        t_start_match = temporal_similar(pred_t_start, gold_t_start)
        t_end_match = temporal_similar(pred_t_end, gold_t_end)
        
        # Classify this semantic match into exactly ONE temporal category
        temporal_category = None
        
        # Priority 1: MATCH_t - BOTH t_start AND t_end must match
        if t_start_match and t_end_match:
            temporal_category = 'MATCH_t'
            match_t_count += 1
        
        # Priority 2: OM_t - ANY predicted temporal missing but gold has temporal
        elif (is_empty_temporal(pred_t_start) and not is_empty_temporal(gold_t_start)) or \
             (is_empty_temporal(pred_t_end) and not is_empty_temporal(gold_t_end)):
            temporal_category = 'OM_t'
            om_t_count += 1
        
        # Priority 3: HALL_t - ANY predicted has temporal but gold is missing
        elif (not is_empty_temporal(pred_t_start) and is_empty_temporal(gold_t_start)) or \
             (not is_empty_temporal(pred_t_end) and is_empty_temporal(gold_t_end)):
            temporal_category = 'HALL_t'
            hall_t_count += 1
        
        # Priority 4: Partial matches and mismatches - treat as OM_t
        # (semantic match but temporal not perfect = omission of correct temporal)
        else:
            temporal_category = 'OM_t'
            om_t_count += 1
        
        temporal_analysis.append({
            'predicted_quintuple': quintuple,
            'matched_gold_quintuple': matched_gold,
            'similarity_score': similarity,
            'temporal_category': temporal_category,
            't_start_match': t_start_match,
            't_end_match': t_end_match,
            'pred_temporal': {'t_start': pred_t_start, 't_end': pred_t_end},
            'gold_temporal': {'t_start': gold_t_start, 't_end': gold_t_end}
        })
    
    # Calculate overall statistics
    match_count = len(matches)
    om_count = len(gold_quintuples) - match_count  # Omissions: gold quintuples not matched
    hall_count = len(quintuples) - match_count     # Hallucinations: predicted quintuples not matched
    
    total_predicted = len(quintuples)
    total_gold = len(gold_quintuples)
    
    # Calculate metrics as proportions
    MATCH = match_count / total_gold if total_gold > 0 else 0.0
    HALL = hall_count / total_predicted if total_predicted > 0 else 0.0
    OM = om_count / total_gold if total_gold > 0 else 0.0
    
    # Temporal variants - all normalized by total_gold for equation to hold
    MATCH_t = match_t_count / total_gold if total_gold > 0 else 0.0
    HALL_t = hall_t_count / total_gold if total_gold > 0 else 0.0
    OM_t = om_t_count / total_gold if total_gold > 0 else 0.0
    
    return {
        'MATCH': MATCH,
        'HALL': HALL,
        'OM': OM,
        'MATCH_t': MATCH_t,
        'OM_t': OM_t,
        'HALL_t': HALL_t,
        'MATCH_count': match_count,
        'HALL_count': hall_count,
        'OM_count': om_count,
        'MATCH_t_count': match_t_count,
        'OM_t_count': om_t_count,
        'HALL_t_count': hall_t_count,
        'total_predicted': total_predicted,
        'total_gold': total_gold
    }

async def evaluate_quintuples_quality(df, lg_kg_construction, threshold=0.7, max_samples=None):
    """
    Evaluate quintuples quality for both cases with comprehensive metrics.
    
    Args:
        df: DataFrame containing the data
        lg_kg_construction: Language model construction object
        threshold: Similarity threshold
        max_samples: Maximum number of samples to process
        
    Returns:
        Dictionary with results for both cases
    """
    print("🚀 Evaluating quintuples quality for both cases")
    logger.info(f"Starting quality evaluation with threshold {threshold}")
    
    # Check if columns exist
    required_cols = [GOLD_COL, PREDICTED_COL_CASE1, PREDICTED_COL_CASE2]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logger.error(f"Missing columns: {missing_cols}")
        return None
    
    # Filter valid rows (rows where all required columns have data)
    valid_indices = (df[GOLD_COL].notna() & 
                    df[PREDICTED_COL_CASE1].notna() & 
                    df[PREDICTED_COL_CASE2].notna())
    valid_df = df[valid_indices].copy()
    
    if max_samples:
        valid_df = valid_df.head(max_samples)
    
    if len(valid_df) == 0:
        logger.error("No valid data found")
        return None
    
    logger.info(f"Processing {len(valid_df)} valid samples")
    
    # Try to load cached gold embeddings
    cached_embeddings, cached_quintuples = load_gold_embeddings_cache(GOLD_EMBEDDINGS_CACHE)
    
    # Prepare gold embeddings cache
    gold_embeddings_dict = {}
    if cached_embeddings is not None and cached_quintuples is not None:
        # Create a mapping from quintuple text to embedding
        for i, gq in enumerate(cached_quintuples):
            gold_text = f"{gq[0]} {gq[1]} {gq[2]}"
            if i < len(cached_embeddings):
                gold_embeddings_dict[gold_text] = cached_embeddings[i]
        logger.info(f"Loaded {len(gold_embeddings_dict)} cached gold embeddings")
    
    results = {'case1': [], 'case2': []}
    all_gold_quintuples = []
    all_gold_texts = []
    
    # Collect all unique gold quintuples for caching
    for idx in valid_df.index:
        gold_quintuples = valid_df[GOLD_COL].loc[idx]
        if gold_quintuples:
            for gq in gold_quintuples:
                gold_text = f"{gq[0]} {gq[1]} {gq[2]}"
                if gold_text not in [f"{existing[0]} {existing[1]} {existing[2]}" for existing in all_gold_quintuples]:
                    all_gold_quintuples.append(gq)
                    all_gold_texts.append(gold_text)
    
    # Calculate embeddings for all unique gold quintuples not in cache
    uncached_gold_texts = [gt for gt in all_gold_texts if gt not in gold_embeddings_dict]
    if uncached_gold_texts:
        logger.info(f"Calculating embeddings for {len(uncached_gold_texts)} uncached gold quintuples")
        uncached_embeddings = await lg_kg_construction.calculate_embeddings(text=uncached_gold_texts)
        for i, text in enumerate(uncached_gold_texts):
            gold_embeddings_dict[text] = uncached_embeddings[i]
    
    # Save updated cache
    if uncached_gold_texts:
        all_cached_quintuples = []
        all_cached_embeddings = []
        for gq in all_gold_quintuples:
            gold_text = f"{gq[0]} {gq[1]} {gq[2]}"
            if gold_text in gold_embeddings_dict:
                all_cached_quintuples.append(gq)
                all_cached_embeddings.append(gold_embeddings_dict[gold_text])
        
        save_gold_embeddings_cache(all_cached_embeddings, all_cached_quintuples, GOLD_EMBEDDINGS_CACHE)
    
    # Process each row
    for row_idx, idx in enumerate(valid_df.index):
        if row_idx % 10 == 0:
            logger.info(f"Processing row {row_idx + 1}/{len(valid_df)}")
        
        gold_quintuples = valid_df[GOLD_COL].loc[idx]
        case1_quintuples = valid_df[PREDICTED_COL_CASE1].loc[idx]
        case2_quintuples = valid_df[PREDICTED_COL_CASE2].loc[idx]
        
        if not gold_quintuples:
            continue
        
        # Get gold embeddings for this row
        gold_texts_row = [f"{gq[0]} {gq[1]} {gq[2]}" for gq in gold_quintuples]
        gold_embeddings_row = [gold_embeddings_dict[gt] for gt in gold_texts_row if gt in gold_embeddings_dict]
        
        if len(gold_embeddings_row) != len(gold_quintuples):
            logger.warning(f"Embedding mismatch for row {idx}")
            gold_embeddings_row = None
        else:
            gold_embeddings_row = np.array(gold_embeddings_row)
        
        # Evaluate Case 1
        if case1_quintuples:
            case1_result = await calculate_comprehensive_metrics(
                quintuples=case1_quintuples,
                gold_quintuples=gold_quintuples,
                lg_kg_construction=lg_kg_construction,
                gold_embeddings=gold_embeddings_row,
                threshold=threshold
            )
            case1_result['row_idx'] = int(idx)
            results['case1'].append(case1_result)
        
        # Evaluate Case 2
        if case2_quintuples:
            case2_result = await calculate_comprehensive_metrics(
                quintuples=case2_quintuples,
                gold_quintuples=gold_quintuples,
                lg_kg_construction=lg_kg_construction,
                gold_embeddings=gold_embeddings_row,
                threshold=threshold
            )
            case2_result['row_idx'] = int(idx)
            results['case2'].append(case2_result)
    
    logger.info(f"Evaluation completed. Case1: {len(results['case1'])} samples, Case2: {len(results['case2'])} samples")
    return results

def calculate_summary_statistics(results):
    """
    Calculate summary statistics (mean and std) for all metrics.
    
    Args:
        results: Dictionary with results for both cases
        
    Returns:
        Dictionary with summary statistics
    """
    logger.info("Calculating summary statistics")
    
    summary = {}
    metrics = ['MATCH', 'HALL', 'OM', 'MATCH_t', 'OM_t', 'HALL_t']
    count_metrics = ['MATCH_count', 'HALL_count', 'OM_count', 'MATCH_t_count', 'OM_t_count', 'HALL_t_count', 'total_predicted', 'total_gold']
    
    for case_name, case_results in results.items():
        if not case_results:
            continue
        
        case_summary = {}
        
        # Calculate mean and std for proportion metrics
        for metric in metrics:
            values = [result[metric] for result in case_results if metric in result]
            if values:
                case_summary[f'{metric}_mean'] = float(np.mean(values))
                case_summary[f'{metric}_std'] = float(np.std(values))
            else:
                case_summary[f'{metric}_mean'] = 0.0
                case_summary[f'{metric}_std'] = 0.0
        
        # Calculate totals for count metrics
        for count_metric in count_metrics:
            values = [result[count_metric] for result in case_results if count_metric in result]
            if values:
                case_summary[f'{count_metric}_total'] = int(np.sum(values))
                case_summary[f'{count_metric}_mean'] = float(np.mean(values))
                case_summary[f'{count_metric}_std'] = float(np.std(values))
            else:
                case_summary[f'{count_metric}_total'] = 0
                case_summary[f'{count_metric}_mean'] = 0.0
                case_summary[f'{count_metric}_std'] = 0.0
        
        case_summary['n_samples'] = len(case_results)
        summary[case_name] = case_summary
    
    return summary

def save_results_to_json(results, summary, output_path):
    """
    Save results and summary to JSON file.
    
    Args:
        results: Dictionary with detailed results
        summary: Dictionary with summary statistics
        output_path: Path to save JSON file
    """
    logger.info(f"Saving results to JSON file: {output_path}")
    
    # Add metadata
    output_data = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'similarity_threshold': SIMILARITY_THRESHOLD,
            'data_path': str(DATA_PATH),
            'gold_column': GOLD_COL,
            'predicted_columns': {
                'case1': PREDICTED_COL_CASE1,
                'case2': PREDICTED_COL_CASE2
            },
            'total_samples': {
                'case1': len(results.get('case1', [])),
                'case2': len(results.get('case2', []))
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

def print_summary_report(summary):
    """Print a formatted summary report."""
    print("\n📊 QUINTUPLES QUALITY EVALUATION SUMMARY")
    print("=" * 80)
    
    for case_name, case_summary in summary.items():
        case_display = "GPT-4.1 Direct" if case_name == 'case1' else "GPT-4.1 from Factoids"
        print(f"\n🔍 {case_display}")
        print("-" * 60)
        
        # Print proportions with mean ± std
        print("  📊 PROPORTIONS (mean ± std):")
        metrics = ['MATCH', 'HALL', 'OM', 'MATCH_t', 'OM_t', 'HALL_t']
        for metric in metrics:
            mean_val = case_summary.get(f'{metric}_mean', 0.0)
            std_val = case_summary.get(f'{metric}_std', 0.0)
            print(f"    {metric:8s}: {mean_val:.4f} ± {std_val:.4f}")
        
        # Print total counts
        print("\n  🔢 TOTAL COUNTS (across all samples):")
        count_metrics = ['MATCH_count', 'HALL_count', 'OM_count', 'MATCH_t_count', 'OM_t_count', 'HALL_t_count']
        for count_metric in count_metrics:
            total_val = case_summary.get(f'{count_metric}_total', 0)
            metric_name = count_metric.replace('_count', '')
            print(f"    {metric_name:8s}: {total_val}")
        
        # Print verification equation
        total_match = case_summary.get('MATCH_count_total', 0)
        total_match_t = case_summary.get('MATCH_t_count_total', 0)
        total_om_t = case_summary.get('OM_t_count_total', 0)
        total_hall_t = case_summary.get('HALL_t_count_total', 0)
        temporal_sum = total_match_t + total_om_t + total_hall_t
        
        print("\n  ✅ VERIFICATION: MATCH = MATCH_t + OM_t + HALL_t")
        print(f"    {total_match} = {total_match_t} + {total_om_t} + {total_hall_t} = {temporal_sum}")
        if total_match == temporal_sum:
            print("    ✅ Equation verified!")
        else:
            print(f"    ⚠️  Equation mismatch! Difference: {total_match - temporal_sum}")
        
        # Print sample info
        n_samples = case_summary.get('n_samples', 0)
        total_predicted = case_summary.get('total_predicted_total', 0)
        total_gold = case_summary.get('total_gold_total', 0)
        print("\n  📝 DATASET INFO:")
        print(f"    Samples    : {n_samples}")
        print(f"    Total Pred : {total_predicted}")
        print(f"    Total Gold : {total_gold}")

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate quintuples quality metrics')
    parser.add_argument('--max-samples', '-m', type=int, default=None,
                       help='Maximum number of samples to process (for testing)')
    parser.add_argument('--threshold', '-t', type=float, default=0.7,
                       help='Similarity threshold for matching')
    return parser.parse_args()

async def main():
    """
    Main function to run the quintuples quality evaluation.
    """
    start_time = time.time()
    
    # Parse command line arguments
    args = parse_arguments()
    
    print("🎯 Starting Quintuples Quality Evaluation")
    print("=" * 50)
    logger.info("Beginning quintuples quality evaluation")
    
    if args.max_samples:
        global MAX_SAMPLES
        MAX_SAMPLES = args.max_samples
        print(f"🎯 Limited to {MAX_SAMPLES} samples (testing mode)")
        logger.info(f"Testing mode: limited to {MAX_SAMPLES} samples")
    
    if args.threshold != 0.7:  # Default threshold value
        global SIMILARITY_THRESHOLD
        SIMILARITY_THRESHOLD = args.threshold
        print(f"🎯 Using similarity threshold: {SIMILARITY_THRESHOLD}")
        logger.info(f"Using custom similarity threshold: {SIMILARITY_THRESHOLD}")
    
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
        
        # Run evaluation
        print("🔍 Running quality evaluation...")
        logger.info(f"Starting evaluation with threshold {SIMILARITY_THRESHOLD}")
        
        results = await evaluate_quintuples_quality(
            df=df,
            lg_kg_construction=lg_kg_construction,
            threshold=SIMILARITY_THRESHOLD,
            max_samples=MAX_SAMPLES
        )
        
        if results is None:
            print("❌ Evaluation failed")
            return
        
        # Calculate summary statistics
        print("📊 Calculating summary statistics...")
        summary = calculate_summary_statistics(results)
        
        # Save results to JSON
        save_results_to_json(results, summary, OUTPUT_JSON)
        
        # Print summary report
        print_summary_report(summary)
        
        elapsed_time = time.time() - start_time
        print("\n✨ Evaluation complete!")
        print(f"📊 Results saved to: {OUTPUT_JSON}")
        print(f"⏱️  Total time: {elapsed_time:.2f} seconds")
        logger.info(f"Evaluation completed successfully in {elapsed_time:.2f} seconds")
        
    except Exception as e:
        elapsed_time = time.time() - start_time
        logger.error(f"Error occurred after {elapsed_time:.2f} seconds: {str(e)}")
        print(f"❌ Error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    print("=" * 50)
    print("  QUINTUPLES QUALITY EVALUATION")
    print("=" * 50)
    asyncio.run(main())