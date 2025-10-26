"""
Factoid Exhaustivity Plot Generation

This script generates plots showing the exhaustivity (recall) of factoid extraction,
analyzing how well different LLM models maintain extraction quality as context size increases.

Usage:
    python plot_exhaustivity_factoids.py

Output:
    - PNG and PDF plots showing factoid extraction exhaustivity
    - JSON file with the results
    - Analysis of extraction quality across different models
"""

import asyncio
import json
import logging
import time
import argparse
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import dateparser
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
import sys
from pathlib import Path

# Add the project root to Python path
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

print("🚀 Starting factoids exhaustivity plot generation script...")
logger.info("Setting up configuration and API connections...")

# ============================================================================
# GLOBAL CONFIGURATION VARIABLES
# ============================================================================

# Models to evaluate (all available models - will be filtered for publication quality in plotting)
MODEL_NAMES = ['claude', 'gpt4o', 'mistral', 'o3mini', 'gpt41']

# Data configuration - adapted for factoids
DATA_PATH = project_root / "datasets" / "nyt_news" / "subset_2020_nyt_COVID_final_final.pkl"
PREDICTED_COL_TEMPLATE = "cumul_factoids_{}"
GOLD_COL = "cumul_factoids_g_truth"
TOKEN_COL = "cumul_lead_paragraph_observation_date_tokenc"

# Analysis parameters
SIMILARITY_THRESHOLD = 0.7
MAX_SAMPLES = None  # Set to None for all samples, or integer for limit

# Output configuration - specific to factoids
OUTPUT_JSON = project_root / "evaluation" / "exhaustivity_factoids_results.json"
OUTPUT_PLOT_PNG = project_root / "evaluation" / "exhaustivity_factoids_plot_publication.png"
OUTPUT_PLOT_PDF = project_root / "evaluation" / "exhaustivity_factoids_plot_publication.pdf"

# Cache configuration for gold truth embeddings
GOLD_EMBEDDINGS_CACHE = project_root / "evaluation" / "gold_factoids_embeddings_cache.pkl"

# Publication-quality plot settings
FIGURE_WIDTH = 4.8  # inches (wider to accommodate right-side legend)
FIGURE_HEIGHT = 2.8  # inches (maintain good aspect ratio)
DPI = 300

# Models for publication plot (all available models)
PUBLICATION_MODELS = ['claude', 'gpt4o', 'mistral', 'o3mini', 'gpt41']

# Publication color palette (colorblind-friendly)
COLORS = {
    'claude': '#1f77b4',    # Blue
    'gpt4o': '#ff7f0e',     # Orange  
    'mistral': '#2ca02c',   # Green
    'o3mini': '#d62728',    # Red
    'gpt41': '#9467bd'      # Purple
}

# Precise model names for legend display
MODEL_DISPLAY_NAMES = {
    'claude': 'claude-sonnet-4-20250514',
    'gpt4o': 'gpt-4o-2024-11-20',
    'mistral': 'mistral-large-latest',
    'o3mini': 'o3-mini-2025-01-31',
    'gpt41': 'gpt-4.1-2025-04-14'
}

# Font sizes for publication
FONT_SIZES = {
    'axis_labels': 13,      
    'tick_labels': 11,      
    'legend': 8,            # Reduced for compact right-side legend
    'title': 14
}


# ============================================================================
# CORE FUNCTIONS
# ============================================================================

async def compute_and_cache_gold_embeddings(df, lg_kg_construction, force_recalculate=False):
    """
    Compute and cache gold truth factoids embeddings to speed up evaluation.
    
    Args:
        df: DataFrame containing the data
        lg_kg_construction: Language model construction object for embeddings
        force_recalculate: Force recalculation even if cache exists
        
    Returns:
        Dictionary mapping row indices to gold factoids embeddings
    """
    cache_path = GOLD_EMBEDDINGS_CACHE
    
    # Check if cache exists and is valid
    if not force_recalculate and cache_path.exists():
        try:
            print("🔍 Loading cached gold truth embeddings...")
            logger.info(f"Loading cached gold embeddings from {cache_path}")
            
            with open(cache_path, 'rb') as f:
                cache_data = pickle.load(f)
            
            # Validate cache structure
            if ('embeddings' in cache_data and 'metadata' in cache_data and 
                cache_data['metadata'].get('data_path') == str(DATA_PATH)):
                
                embeddings_cache = cache_data['embeddings']
                print(f"   ✅ Loaded {len(embeddings_cache)} cached gold embeddings")
                logger.info(f"Successfully loaded {len(embeddings_cache)} gold embeddings from cache")
                return embeddings_cache
            else:
                print("   ⚠️  Cache structure invalid, will recompute")
                logger.warning("Cache structure is invalid, proceeding with recomputation")
        except Exception as e:
            print(f"   ⚠️  Error loading cache: {e}")
            logger.warning(f"Error loading cache: {e}, proceeding with recomputation")
    
    # Compute gold embeddings for all valid rows
    print("🔍 Computing gold truth embeddings for all rows...")
    logger.info("Computing and caching gold truth factoids embeddings")
    
    # Filter valid rows (same logic as in main evaluation)
    valid_indices = (df[GOLD_COL].notna() & df[TOKEN_COL].notna())
    valid_df = df[valid_indices].copy()
    
    print(f"   📊 Processing {len(valid_df)} rows with valid gold factoids")
    logger.info(f"Processing {len(valid_df)} valid rows for gold embeddings")
    
    embeddings_cache = {}
    batch_size = 50  # Process in batches to manage memory
    
    for batch_start in range(0, len(valid_df), batch_size):
        batch_end = min(batch_start + batch_size, len(valid_df))
        batch_indices = valid_df.index[batch_start:batch_end]
        
        print(f"   🔄 Processing batch {batch_start//batch_size + 1}/{(len(valid_df)-1)//batch_size + 1}")
        logger.debug(f"Processing batch {batch_start}-{batch_end}")
        
        # Collect all gold factoids texts for this batch
        batch_gold_texts = []
        batch_row_indices = []
        batch_factoid_counts = []
        
        for idx in batch_indices:
            gold_factoids = valid_df[GOLD_COL].loc[idx]
            if gold_factoids:
                gold_texts = [str(gf) for gf in gold_factoids]
                batch_gold_texts.extend(gold_texts)
                batch_row_indices.extend([idx] * len(gold_texts))
                batch_factoid_counts.append(len(gold_texts))
            else:
                batch_factoid_counts.append(0)
        
        if batch_gold_texts:
            # Compute embeddings for the batch
            batch_embeddings = await lg_kg_construction.calculate_embeddings(text=batch_gold_texts)
            batch_embeddings = np.array(batch_embeddings)
            
            # Distribute embeddings back to their respective rows
            embedding_idx = 0
            for i, idx in enumerate(batch_indices):
                count = batch_factoid_counts[i]
                if count > 0:
                    row_embeddings = batch_embeddings[embedding_idx:embedding_idx + count]
                    embeddings_cache[int(idx)] = row_embeddings
                    embedding_idx += count
    
    # Save to cache
    try:
        print("💾 Saving gold embeddings to cache...")
        logger.info(f"Saving gold embeddings cache to {cache_path}")
        
        cache_data = {
            'embeddings': embeddings_cache,
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'data_path': str(DATA_PATH),
                'total_rows': len(embeddings_cache),
                'embedding_model': 'text-embedding-3-large'
            }
        }
        
        with open(cache_path, 'wb') as f:
            pickle.dump(cache_data, f)
        
        print(f"   ✅ Cached {len(embeddings_cache)} gold embeddings")
        logger.info(f"Successfully cached {len(embeddings_cache)} gold embeddings")
        
    except Exception as e:
        print(f"   ⚠️  Warning: Failed to save cache: {e}")
        logger.warning(f"Failed to save embeddings cache: {e}")
    
    return embeddings_cache


def load_gold_embeddings_for_row(embeddings_cache, row_idx):
    """
    Load cached gold embeddings for a specific row.
    
    Args:
        embeddings_cache: Dictionary of cached embeddings
        row_idx: Row index to load embeddings for
        
    Returns:
        Numpy array of embeddings or None if not found
    """
    return embeddings_cache.get(int(row_idx))

async def find_matches_factoids_optimized(factoids, gold_factoids, lg_kg_construction, threshold=0.7, gold_embeddings=None):
    """
    Optimized function to find matches between factoids using embeddings and temporal analysis.
    
    Args:
        factoids: List of predicted factoids (strings with embedded temporal info)
        gold_factoids: List of gold standard factoids (strings with embedded temporal info)
        lg_kg_construction: Language model construction object for embeddings
        threshold: Similarity threshold for matching
        gold_embeddings: Pre-computed gold factoids embeddings (optional, for speed optimization)
        
    Returns:
        Dict with recall and recall_t metrics
    """
    logger.debug(f"Finding matches for {len(factoids)} predicted vs {len(gold_factoids)} gold factoids")
    
    if not factoids or not gold_factoids:
        logger.warning("Empty factoids or gold_factoids provided")
        return {'recall': 0.0, 'recall_t': 0.0}
    
    def extract_temporal_info_from_text(text):
        """Extract temporal information from text using dateparser"""
        if not text or not isinstance(text, str):
            return []
        
        try:
            # Find all potential date expressions in the text
            dates = []
            
            # Use dateparser to find temporal expressions
            # Split text into words and try to parse temporal expressions
            words = text.split()
            for i in range(len(words)):
                for j in range(i+1, min(i+10, len(words)+1)):  # Check up to 10-word phrases
                    phrase = ' '.join(words[i:j])
                    try:
                        parsed_date = dateparser.parse(phrase, settings={'PREFER_DAY_OF_MONTH': 'first'})
                        if parsed_date:
                            dates.append(parsed_date.date())
                    except (ValueError, TypeError, AttributeError):
                        continue
            
            # Remove duplicates and return
            return list(set(dates))
        except Exception as e:
            logger.debug(f"Error extracting temporal info from '{text}': {e}")
            return []
    
    # Factoids are already text strings
    factoid_texts = [str(f) for f in factoids]
    gold_factoid_texts = [str(gf) for gf in gold_factoids]
    
    # Calculate embeddings - use cached gold embeddings if available
    factoid_embeddings = await lg_kg_construction.calculate_embeddings(text=factoid_texts)
    
    if gold_embeddings is not None:
        # Use pre-computed gold embeddings
        gold_factoid_embeddings = gold_embeddings
        logger.debug("Using cached gold embeddings")
    else:
        # Compute gold embeddings (fallback)
        gold_factoid_embeddings = await lg_kg_construction.calculate_embeddings(text=gold_factoid_texts)
        logger.debug("Computing gold embeddings on-the-fly")
    
    # Convert to numpy arrays and ensure proper shape
    factoid_embeddings = np.array(factoid_embeddings)
    gold_factoid_embeddings = np.array(gold_factoid_embeddings)
    
    if factoid_embeddings.ndim == 1:
        factoid_embeddings = factoid_embeddings.reshape(1, -1)
    if gold_factoid_embeddings.ndim == 1:
        gold_factoid_embeddings = gold_factoid_embeddings.reshape(1, -1)
    
    # Compute similarity matrix
    similarity_matrix = cosine_similarity(factoid_embeddings, gold_factoid_embeddings)
    
    # Find matches and analyze temporal information
    # Use sets to track unique gold factoids that are matched
    matched_gold_indices = set()
    temporal_matched_gold_indices = set()
    
    def temporal_lists_overlap(pred_dates, gold_dates):
        """Check if any dates from predicted factoid overlap with gold factoid dates"""
        if not pred_dates and not gold_dates:
            return True  # Both have no temporal info
        if not pred_dates or not gold_dates:
            return False  # One has temporal info, the other doesn't
        
        # Check if any predicted date matches any gold date
        for pred_date in pred_dates:
            for gold_date in gold_dates:
                if pred_date == gold_date:
                    return True
        return False
    
    # Process each predicted factoid
    for i, factoid in enumerate(factoids):
        similarities = similarity_matrix[i]
        max_similarity_idx = np.argmax(similarities)
        max_similarity = similarities[max_similarity_idx]
        
        if max_similarity > threshold:
            # Track which gold factoid was matched (avoid double-counting)
            matched_gold_indices.add(max_similarity_idx)
            matched_gold = gold_factoids[max_similarity_idx]
            
            # Check temporal similarity by extracting dates from text
            pred_temporal_dates = extract_temporal_info_from_text(factoid)
            gold_temporal_dates = extract_temporal_info_from_text(matched_gold)
            
            if temporal_lists_overlap(pred_temporal_dates, gold_temporal_dates):
                temporal_matched_gold_indices.add(max_similarity_idx)
    
    # Calculate recall metrics based on unique gold factoids matched
    total_gold = len(gold_factoids)
    unique_gold_matches = len(matched_gold_indices)
    unique_temporal_gold_matches = len(temporal_matched_gold_indices)
    
    recall = unique_gold_matches / total_gold if total_gold > 0 else 0.0
    recall_t = unique_temporal_gold_matches / total_gold if total_gold > 0 else 0.0
    
    return {'recall': recall, 'recall_t': recall_t}


async def evaluate_models_by_token_count(df, model_names, lg_kg_construction, threshold=0.7, max_samples=None, gold_embeddings_cache=None):
    """
    Evaluate multiple models and return results by token count.
    
    Args:
        df: DataFrame containing the data
        model_names: List of model names to evaluate
        lg_kg_construction: Language model construction object
        threshold: Similarity threshold
        max_samples: Maximum number of samples to process per model
        gold_embeddings_cache: Pre-computed gold embeddings cache for speed optimization
        
    Returns:
        Dictionary with results for each model
    """
    print(f"🚀 Evaluating {len(model_names)} models: {model_names}")
    logger.info(f"Starting evaluation for {len(model_names)} models with threshold {threshold}")
    
    results = {}
    
    for model_name in model_names:
        print(f"📊 Processing model: {model_name.upper()}")
        logger.info(f"Processing model: {model_name}")
        
        predicted_col = PREDICTED_COL_TEMPLATE.format(model_name)
        
        # Check if columns exist
        if predicted_col not in df.columns:
            print(f"⚠️  Column {predicted_col} not found. Skipping {model_name}")
            logger.warning(f"Column {predicted_col} not found in dataframe. Skipping {model_name}")
            continue
            
        # Filter valid rows
        valid_indices = (df[predicted_col].notna() & 
                        df[GOLD_COL].notna() & 
                        df[TOKEN_COL].notna())
        valid_df = df[valid_indices].copy()
        
        if max_samples:
            valid_df = valid_df.head(max_samples)
        
        if len(valid_df) == 0:
            print(f"⚠️  No valid data for {model_name}")
            logger.warning(f"No valid data found for model {model_name}")
            continue
        
        logger.info(f"Processing {len(valid_df)} valid samples for {model_name}")
        model_results = []
        
        # Process each row
        for row_idx, idx in enumerate(valid_df.index):
            if row_idx % 10 == 0:  # Log progress every 10 rows
                logger.debug(f"Processing row {row_idx + 1}/{len(valid_df)} for {model_name}")
            factoids = valid_df[predicted_col].loc[idx]
            gold_factoids = valid_df[GOLD_COL].loc[idx]
            token_count = valid_df[TOKEN_COL].loc[idx]
            
            if not factoids or not gold_factoids:
                continue
                
            # Get cached gold embeddings for this row if available
            cached_gold_embeddings = None
            if gold_embeddings_cache:
                cached_gold_embeddings = load_gold_embeddings_for_row(gold_embeddings_cache, idx)
                
            # Calculate recall metrics
            result = await find_matches_factoids_optimized(
                factoids=factoids,
                gold_factoids=gold_factoids,
                lg_kg_construction=lg_kg_construction,
                threshold=threshold,
                gold_embeddings=cached_gold_embeddings
            )
            
            model_results.append({
                'token_count': int(token_count),
                'recall': float(result['recall']),
                'recall_t': float(result['recall_t']),
                'row_idx': int(idx)
            })
        
        results[model_name] = model_results
        print(f"   ✅ Processed {len(model_results)} samples")
        logger.info(f"Completed processing {model_name}: {len(model_results)} samples")
    
    logger.info(f"Evaluation completed for all models. Total results: {sum(len(v) for v in results.values())} samples")
    return results


def create_publication_exhaustivity_plot(results, model_names=None):
    """
    Create a publication-quality bar plot showing semantic and temporal recall by token count for factoids.
    
    Args:
        results: Dictionary with results for each model
        model_names: List of model names (defaults to PUBLICATION_MODELS for cleaner plot)
        
    Returns:
        matplotlib figure and axes objects
    """
    # Use publication models for cleaner plot if not specified
    if model_names is None:
        model_names = PUBLICATION_MODELS
    
    logger.info(f"Creating publication-quality factoids exhaustivity plot for models: {model_names}")
    
    # Set matplotlib parameters for publication quality
    plt.rcParams.update({
        'font.size': FONT_SIZES['tick_labels'],
        'axes.labelsize': FONT_SIZES['axis_labels'],
        'axes.titlesize': FONT_SIZES['title'],
        'legend.fontsize': FONT_SIZES['legend'],
        'xtick.labelsize': FONT_SIZES['tick_labels'],
        'ytick.labelsize': FONT_SIZES['tick_labels'],
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'text.usetex': False,  # Set to True if LaTeX is available
        'figure.dpi': DPI,
        'savefig.dpi': DPI,
        'axes.linewidth': 0.8,
        'grid.linewidth': 0.5,
        'lines.linewidth': 1.0
    })
    
    # Prepare data
    plot_data = []
    for model_name, model_results in results.items():
        for result in model_results:
            plot_data.append({
                'model': model_name.lower(),
                'token_count': result['token_count'],
                'recall': result['recall'],
                'recall_t': result['recall_t']
            })
    
    if not plot_data:
        logger.error("No data to plot!")
        return None, None
    
    df_plot = pd.DataFrame(plot_data)
    logger.info(f"Prepared plot data with {len(df_plot)} data points")
    
    # Group by token count and calculate means
    grouped = df_plot.groupby(['token_count', 'model']).agg({
        'recall': 'mean',
        'recall_t': 'mean'
    }).reset_index()
    
    # Get unique token counts and sort them - reduce density for publication
    unique_tokens = sorted(grouped['token_count'].unique())
    # Show every 3rd token count to reduce clutter and improve readability
    unique_tokens_reduced = unique_tokens[::3]
    logger.info(f"Plotting {len(unique_tokens_reduced)} token count points (reduced from {len(unique_tokens)})")
    
    n_models = len([m for m in model_names if m in results])
    
    # Create figure with publication dimensions
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT), dpi=DPI)
    
    # Set up bar positions
    x = np.arange(len(unique_tokens_reduced))
    width = 0.12  # Slightly reduced bar width to prevent touching
    
    # Plot bars for each model
    legend_elements = []
    for i, model_name in enumerate(model_names):
        if model_name not in results:
            continue
            
        model_data = grouped[grouped['model'] == model_name]
        
        recalls = []
        recalls_t = []
        
        for token_count in unique_tokens_reduced:
            token_data = model_data[model_data['token_count'] == token_count]
            if len(token_data) > 0:
                recalls.append(token_data['recall'].iloc[0])
                recalls_t.append(token_data['recall_t'].iloc[0])
            else:
                recalls.append(0)
                recalls_t.append(0)
        
        # Calculate bar positions
        x_pos = x + (i - (n_models-1)/2) * width
        
        # Get color for this model
        color = COLORS.get(model_name, '#666666')
        
        # Semantic recall bars (full bars with subtle background)
        bars_semantic = ax.bar(x_pos, recalls, width, 
                              color=color, alpha=0.85, 
                              edgecolor='black', linewidth=0.6,
                              zorder=2)
        
        # Temporal recall bars (overlaid with pattern and slight color variation)
        temporal_color = color  # Same base color
        bars_temporal = ax.bar(x_pos, recalls_t, width,
                              color=temporal_color, alpha=0.65,
                              edgecolor='black', linewidth=0.6,
                              hatch='///', zorder=3)
        
        # Add to legend with precise model names
        display_name = MODEL_DISPLAY_NAMES.get(model_name, model_name.upper())
        legend_elements.append((bars_semantic[0], f'{display_name} - Semantic'))
        legend_elements.append((bars_temporal[0], f'{display_name} - Temporal'))
    
    # Customize plot for publication
    ax.set_xlabel('Token count as context', fontsize=FONT_SIZES['axis_labels'], fontweight='bold')
    ax.set_ylabel('Exhaustivity', fontsize=FONT_SIZES['axis_labels'], fontweight='bold')
    ax.set_title('Exhaustivity of atomic facts', fontsize=FONT_SIZES['title'], fontweight='bold', pad=20)
    
    # Set y-axis range [0, 0.6] as requested
    ax.set_ylim(0, 0.6)
    
    # Add horizontal gridlines at specified intervals (improved visibility)
    gridlines = [0.1, 0.2, 0.3, 0.4, 0.5]
    for gridline in gridlines:
        ax.axhline(y=gridline, color='gray', linestyle='-', alpha=0.4, linewidth=0.6, zorder=1)
    
    # Add minor tick marks on y-axis
    ax.set_yticks([0.05, 0.15, 0.25, 0.35, 0.45, 0.55], minor=True)
    ax.tick_params(axis='y', which='minor', length=3, width=0.5)
    
    # Set x-axis with improved readability
    ax.set_xticks(x)
    
    # Format x-axis labels with scientific notation for readability
    def format_token_count(tc):
        if tc >= 10000:
            return f'{tc/1000:.1f}k'
        elif tc >= 1000:
            return f'{tc/1000:.1f}k'
        else:
            return f'{int(tc)}'
    
    ax.set_xticklabels([format_token_count(tc) for tc in unique_tokens_reduced], 
                       rotation=45, ha='right', fontsize=FONT_SIZES['tick_labels'])
    
    # Extend x-axis limits to use full plot width
    ax.set_xlim(-0.5, len(unique_tokens_reduced) - 0.5)
    
    # Create custom legend with single-column layout for right-side placement
    handles = []
    labels = []
    for handle, label in legend_elements:
        handles.append(handle)
        labels.append(label)
    
    # Place legend outside plot area on the right side with single-column layout
    ax.legend(handles, labels, bbox_to_anchor=(1.02, 1), loc='upper left', 
              fontsize=FONT_SIZES['legend'], frameon=True, fancybox=False, shadow=False,
              ncol=1, handletextpad=0.5, handlelength=1.2,
              framealpha=0.9, edgecolor='black', facecolor='white')
    
    # Increase axis border line width for better print quality
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
    
    # Remove top and right spines for cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Set grid below data
    ax.set_axisbelow(True)
    
    # Improve tick parameters for better print quality
    ax.tick_params(axis='both', which='major', width=1.0, length=4)
    ax.tick_params(axis='x', which='major', pad=8)  # Add padding for rotated labels
    
    # Adjust layout to accommodate right-side legend and rotated x-axis labels
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15, right=0.75)  # Extra space for rotated labels and right legend
    
    # Save both PNG and PDF formats
    logger.info("Saving factoids plot in multiple formats")
    plt.savefig(str(OUTPUT_PLOT_PNG), dpi=DPI, bbox_inches='tight', 
                facecolor='white', edgecolor='none', pad_inches=0.1)
    plt.savefig(str(OUTPUT_PLOT_PDF), dpi=DPI, bbox_inches='tight', 
                facecolor='white', edgecolor='none', format='pdf', pad_inches=0.1)
    
    print("📊 Factoids publication plot saved to:")
    print(f"   PNG: {OUTPUT_PLOT_PNG}")
    print(f"   PDF: {OUTPUT_PLOT_PDF}")
    logger.info(f"Factoids plot saved to {OUTPUT_PLOT_PNG} and {OUTPUT_PLOT_PDF}")
    
    return fig, ax


def load_existing_results(json_path):
    """
    Load existing results from JSON file if it exists.
    
    Args:
        json_path: Path to JSON results file
        
    Returns:
        Dictionary with results or None if file doesn't exist/invalid
    """
    try:
        if not Path(json_path).exists():
            logger.info(f"No existing results file found at {json_path}")
            return None
            
        with open(str(json_path), 'r') as f:
            data = json.load(f)
        
        # Validate the structure
        if 'results' not in data or 'metadata' not in data:
            logger.warning(f"Invalid JSON structure in {json_path}")
            return None
            
        # Check if metadata matches current configuration
        metadata = data['metadata']
        current_models = set(MODEL_NAMES)
        existing_models = set(metadata.get('model_names', []))
        
        if metadata.get('similarity_threshold') != SIMILARITY_THRESHOLD:
            logger.warning(f"Threshold mismatch: existing={metadata.get('similarity_threshold')}, current={SIMILARITY_THRESHOLD}")
            return None
            
        if not current_models.issubset(existing_models):
            missing_models = current_models - existing_models
            logger.warning(f"Missing models in existing results: {missing_models}")
            return None
            
        # Filter results to only include current models
        filtered_results = {model: data['results'][model] for model in MODEL_NAMES if model in data['results']}
        
        logger.info(f"✅ Loaded existing factoids results from {json_path}")
        logger.info(f"   Models: {list(filtered_results.keys())}")
        logger.info(f"   Total samples: {sum(len(v) for v in filtered_results.values())}")
        logger.info(f"   Timestamp: {metadata.get('timestamp', 'unknown')}")
        
        return filtered_results
        
    except Exception as e:
        logger.error(f"Error loading existing results from {json_path}: {e}")
        return None


def save_results_to_json(results, output_path):
    """
    Save results to JSON file for later analysis.
    
    Args:
        results: Dictionary with results
        output_path: Path to save JSON file
    """
    logger.info(f"Saving factoids results to JSON file: {output_path}")
    
    # Add metadata
    output_data = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'model_names': MODEL_NAMES,
            'similarity_threshold': SIMILARITY_THRESHOLD,
            'data_path': str(DATA_PATH),
            'total_samples': sum(len(v) for v in results.values()),
            'analysis_type': 'factoids'
        },
        'results': results
    }
    
    try:
        with open(str(output_path), 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"💾 Factoids results saved to: {output_path}")
        logger.info(f"Successfully saved {len(results)} model results to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save results to {output_path}: {e}")
        raise


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Generate exhaustivity plots for ATOM models - Factoids Analysis')
    parser.add_argument('--force-recalculate', '-f', action='store_true',
                       help='Force recalculation even if existing results are found')
    parser.add_argument('--max-samples', '-m', type=int, default=None,
                       help='Maximum number of samples to process per model (for testing)')
    return parser.parse_args()


async def main():
    """
    Main function to run the factoids exhaustivity analysis.
    """
    start_time = time.time()
    
    # Parse command line arguments
    args = parse_arguments()
    
    print("🎯 Starting Factoids Exhaustivity Analysis")
    print("=" * 50)
    logger.info("Beginning factoids exhaustivity plot generation analysis")
    
    if args.force_recalculate:
        print("🔄 Force recalculation mode enabled")
        logger.info("Force recalculation mode enabled - will skip existing results")
    
    if args.max_samples:
        global MAX_SAMPLES
        MAX_SAMPLES = args.max_samples
        print(f"🎯 Limited to {MAX_SAMPLES} samples per model (testing mode)")
        logger.info(f"Testing mode: limited to {MAX_SAMPLES} samples per model")
    
    try:
        # First, try to load existing results (unless force recalculation is enabled)
        results = None
        if not args.force_recalculate:
            print("🔍 Checking for existing factoids results...")
            logger.info(f"Looking for existing results in {OUTPUT_JSON}")
            results = load_existing_results(OUTPUT_JSON)
        
        if results is not None:
            print("   ✅ Found existing factoids results! Using cached data.")
            print(f"   📊 Loaded {len(results)} models with {sum(len(v) for v in results.values())} total samples")
            logger.info("Using existing results, skipping evaluation")
        else:
            print("   ⚠️  No existing factoids results found or force recalculation enabled. Running full evaluation...")
            logger.info("No existing results found or force recalculation enabled, proceeding with full evaluation")
            
            # Import ATOM modules
            try:
                from atom.llm_output_parsing.langchain_output_parser import LangchainOutputParser
                from langchain_openai import ChatOpenAI, OpenAIEmbeddings
                print("   ✅ ATOM modules imported successfully")
                logger.info("ATOM modules imported successfully")
            except ImportError as e:
                print(f"❌ Error importing ATOM modules: {e}")
                print(f"Current working directory: {Path.cwd()}")
                print(f"Project root: {project_root}")
                logger.error(f"Failed to import ATOM modules: {e}")
                print("Make sure you're running this from the correct directory")
                return
            
            # Load data
            print(f"📁 Loading data from: {DATA_PATH}")
            logger.info(f"Loading dataset from {DATA_PATH}")
            try:
                df = pd.read_pickle(DATA_PATH)
                print(f"   ✅ Loaded {len(df)} samples")
                logger.info(f"Successfully loaded dataset with {len(df)} samples")
                
                # Log dataset info
                logger.info(f"Dataset columns: {list(df.columns)}")
                logger.info(f"Available factoids models in dataset: {[col.replace('cumul_factoids_', '') for col in df.columns if col.startswith('cumul_factoids_') and col != GOLD_COL]}")
            except Exception as e:
                print(f"❌ Error loading data: {e}")
                logger.error(f"Failed to load dataset: {e}")
                return
            
            # Initialize language model components
            print("🤖 Initializing language model components...")
            logger.info("Initializing language model components")
            try:
                # API keys (same as in the working script)
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
            
            # Compute and cache gold embeddings first for speed optimization
            print("🚀 Computing/loading gold truth embeddings cache...")
            logger.info("Setting up gold embeddings cache for speed optimization")
            gold_embeddings_cache = await compute_and_cache_gold_embeddings(
                df=df, 
                lg_kg_construction=lg_kg_construction,
                force_recalculate=args.force_recalculate
            )
            
            # Run evaluation with cached gold embeddings
            print("🔍 Running factoids evaluation with cached gold embeddings...")
            logger.info(f"Starting evaluation with threshold {SIMILARITY_THRESHOLD}")
            if MAX_SAMPLES:
                logger.info(f"Limited to {MAX_SAMPLES} samples per model")
            
            results = await evaluate_models_by_token_count(
                df=df,
                model_names=MODEL_NAMES,
                lg_kg_construction=lg_kg_construction,
                threshold=SIMILARITY_THRESHOLD,
                max_samples=MAX_SAMPLES,
                gold_embeddings_cache=gold_embeddings_cache
            )
            
            # Save results to JSON
            save_results_to_json(results, OUTPUT_JSON)
        
        # Create and save publication-quality plot
        print("📊 Creating factoids publication-quality visualization...")
        logger.info("Creating factoids publication-quality plot")
        fig, ax = create_publication_exhaustivity_plot(results)
        
        if fig is not None:
            print("   ✅ Factoids plot created successfully")
            logger.info("Factoids plot created and saved successfully")
        else:
            print("   ⚠️  No factoids plot generated")
            logger.warning("No factoids plot was generated")
        
        elapsed_time = time.time() - start_time
        print("\n✨ Factoids Analysis complete!")
        print(f"📊 Results saved to: {OUTPUT_JSON}")
        print("🖼️  Publication plots saved to:")
        print(f"   PNG: {OUTPUT_PLOT_PNG}")
        print(f"   PDF: {OUTPUT_PLOT_PDF}")
        print(f"📈 Plot includes models: {PUBLICATION_MODELS}")
        print(f"⏱️  Total time: {elapsed_time:.2f} seconds")
        logger.info(f"Factoids analysis completed successfully in {elapsed_time:.2f} seconds")
        
    except Exception as e:
        elapsed_time = time.time() - start_time
        logger.error(f"Error occurred after {elapsed_time:.2f} seconds: {str(e)}")
        print(f"❌ Error occurred: {str(e)}")
        print("💡 Check the logs for more details.")
        raise


if __name__ == "__main__":
    print("=" * 50)
    print("  FACTOIDS EXHAUSTIVITY PLOT GENERATION FOR NYT COVID DATA")
    print("=" * 50)
    asyncio.run(main())
