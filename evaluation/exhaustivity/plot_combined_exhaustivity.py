"""
Combined Exhaustivity Plot Generation for Publication

This script generates publication-quality plots visualizing exhaustivity evaluation results
for both factoid extraction and quintuple extraction. It creates a combined figure with two
subplots showing recall metrics across different LLM models, using a colorblind-friendly
palette and professional formatting suitable for academic papers.

Usage:
    python plot_combined_exhaustivity.py

Output:
    - PNG and PDF files with publication-ready exhaustivity plots
    - Combined visualization of factoid and quintuple extraction quality
"""

import json
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys

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

print("🚀 Starting combined exhaustivity plot generation script...")
logger.info("Setting up combined plot configuration...")

# ============================================================================
# GLOBAL CONFIGURATION VARIABLES
# ============================================================================

# Input JSON files
FACTOIDS_JSON = project_root / "evaluation" / "exhaustivity_factoids_results.json"
QUINTUPLES_JSON = project_root / "evaluation" / "exhaustivity_results.json"

# Output configuration
OUTPUT_PLOT_PNG = project_root / "evaluation" / "combined_exhaustivity_plot_publication.png"
OUTPUT_PLOT_PDF = project_root / "evaluation" / "combined_exhaustivity_plot_publication.pdf"

# Publication-quality plot settings
FIGURE_WIDTH = 5.5  # inches (slightly wider for better readability)
FIGURE_HEIGHT = 7.0  # inches (taller to accommodate two subplots + bottom legend)
DPI = 300

# Models for publication plot
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
    'claude': 'claude-sonnet-4-2025-01-31',
    'gpt4o': 'gpt-4o-2024-11-20',
    'mistral': 'mistral-large-2411',
    'o3mini': 'o3-mini-2025-01-31',
    'gpt41': 'gpt-4.1-2025-04-14'
}

# Font sizes for publication
FONT_SIZES = {
    'axis_labels': 15,      
    'tick_labels':13,      
    'legend': 12,            # Increased for better readability
    'title': 16
}


# ============================================================================
# CORE FUNCTIONS
# ============================================================================

def load_results_from_json(json_path):
    """
    Load results from JSON file.
    
    Args:
        json_path: Path to JSON results file
        
    Returns:
        Dictionary with results or None if file doesn't exist
    """
    try:
        if not Path(json_path).exists():
            logger.error(f"Results file not found: {json_path}")
            return None
            
        with open(str(json_path), 'r') as f:
            data = json.load(f)
        
        if 'results' not in data:
            logger.error(f"Invalid JSON structure in {json_path}")
            return None
            
        results = data['results']
        logger.info(f"✅ Loaded results from {json_path}")
        logger.info(f"   Models: {list(results.keys())}")
        logger.info(f"   Total samples: {sum(len(v) for v in results.values())}")
        
        return results
        
    except Exception as e:
        logger.error(f"Error loading results from {json_path}: {e}")
        return None


def prepare_plot_data(results):
    """
    Prepare data for plotting from results.
    
    Args:
        results: Dictionary with results for each model
        
    Returns:
        Grouped pandas DataFrame ready for plotting
    """
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
        return None
    
    df_plot = pd.DataFrame(plot_data)
    
    # Group by token count and calculate means
    grouped = df_plot.groupby(['token_count', 'model']).agg({
        'recall': 'mean',
        'recall_t': 'mean'
    }).reset_index()
    
    return grouped


def plot_subplot_bars(ax, grouped, model_names, title_text, y_label=r"$R_{\text{MATCH}}$ and $R_{\text{MATCH}_t}$"):
    """
    Plot bars for a single subplot.
    
    Args:
        ax: Matplotlib axes object
        grouped: Grouped data for plotting
        model_names: List of model names to plot
        title_text: Title for the subplot
        y_label: Y-axis label
        
    Returns:
        List of legend elements
    """
    # Get unique token counts and reduce density for publication
    unique_tokens = sorted(grouped['token_count'].unique())
    unique_tokens_reduced = unique_tokens [::2] # Show every 4th token count for less crowding
    
    n_models = len([m for m in model_names if m in grouped['model'].values])
    
    # Set up bar positions
    x = np.arange(len(unique_tokens_reduced))
    width = 0.12  # Bar width
    
    # Plot bars for each model
    legend_elements = []
    for i, model_name in enumerate(model_names):
        if model_name not in grouped['model'].values:
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
        
        # Factual recall bars (full bars)
        bars_factual = ax.bar(x_pos, recalls, width, 
                             color=color, alpha=0.85, 
                             edgecolor='black', linewidth=1,
                             zorder=2)
        
        # Temporal recall bars (overlaid with pattern)
        bars_temporal = ax.bar(x_pos, recalls_t, width,
                              color=color, alpha=0.65,
                              edgecolor='black', linewidth=1,
                              hatch='//////', zorder=3)
        
        # Add to legend with just model names
        display_name = MODEL_DISPLAY_NAMES.get(model_name, model_name.upper())
        legend_elements.append((bars_factual[0], display_name))
    
    # Customize subplot
    ax.set_xlabel('Token count as context', fontsize=FONT_SIZES['axis_labels'], fontweight='bold')
    ax.set_ylabel(y_label, fontsize=FONT_SIZES['axis_labels'], fontweight='bold')
    ax.set_title(title_text, fontsize=FONT_SIZES['title'], fontweight='bold', pad=15)
    
    # Set y-axis range optimized for data
    ax.set_ylim(0, 1)  # Since data appears to max out around 0.6
    
    # Add improved horizontal gridlines
    gridlines = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    for gridline in gridlines:
        ax.axhline(y=gridline, color='gray', linestyle='-', alpha=0.3, linewidth=0.5, zorder=1)
    
    # Add minor tick marks on y-axis
    ax.set_yticks([0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1], minor=True)
    ax.tick_params(axis='y', which='minor', length=3, width=0.5)
    
    # Set x-axis with improved readability
    ax.set_xticks(x)
    
    # Format x-axis labels
    def format_token_count(tc):
        if tc >= 10000:
            return f'{tc/1000:.1f}k'
        elif tc >= 1000:
            return f'{tc/1000:.1f}k'
        else:
            return f'{int(tc)}'
    
    ax.set_xticklabels([format_token_count(tc) for tc in unique_tokens_reduced], 
                       rotation=35, ha='right', fontsize=FONT_SIZES['tick_labels'])  # Less aggressive rotation
    
    # Extend x-axis limits to use full plot width
    ax.set_xlim(-0.5, len(unique_tokens_reduced) - 0.5)
    
    # Add black borders around subplot (1.5-2pt stroke)
    for spine in ax.spines.values():
        spine.set_linewidth(1)  # 1.5-2pt border width
        spine.set_color('black')
        spine.set_visible(True)  # Ensure all borders are visible
    
    # Set grid below data
    ax.set_axisbelow(True)
    
    # Improve tick parameters
    ax.tick_params(axis='both', which='major', width=1.0, length=4)
    ax.tick_params(axis='x', which='major', pad=8)
    
    return legend_elements


def create_combined_exhaustivity_plot(factoids_results, quintuples_results):
    """
    Create a combined publication-quality plot with factoids on top and quintuples on bottom.
    
    Args:
        factoids_results: Dictionary with factoids results
        quintuples_results: Dictionary with quintuples results
        
    Returns:
        matplotlib figure and axes objects
    """
    logger.info("Creating combined exhaustivity plot")
    
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
        'text.usetex': False,
        'figure.dpi': DPI,
        'savefig.dpi': DPI,
        'axes.linewidth': 0.8,
        'grid.linewidth': 0.5,
        'lines.linewidth': 1.0
    })
    
    # Prepare data for both plots
    factoids_grouped = prepare_plot_data(factoids_results)
    quintuples_grouped = prepare_plot_data(quintuples_results)
    
    if factoids_grouped is None or quintuples_grouped is None:
        logger.error("Failed to prepare plot data")
        return None, None
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(FIGURE_WIDTH, FIGURE_HEIGHT), dpi=DPI)
    
    # Plot factoids (top subplot)
    legend_elements_factoids = plot_subplot_bars(
        ax1, factoids_grouped, PUBLICATION_MODELS, 
        "(a) Exhaustivity of atomic facts decomposition"
    )
    
    # Plot quintuples (bottom subplot)
    legend_elements_quintuples = plot_subplot_bars(
        ax2, quintuples_grouped, PUBLICATION_MODELS, 
        "(b) Exhaustivity of 5-tuples extraction"
    )
    
    # Create custom legend with models (filled) + factual/temporal symbols
    from matplotlib.patches import Rectangle
    from matplotlib.lines import Line2D
    import matplotlib.patches as mpatches
    
    # Create legend handles and labels
    legend_handles = []
    legend_labels = []
    
    # Define the order for models in legend
    ordered_model_labels = [
        MODEL_DISPLAY_NAMES["claude"],
        MODEL_DISPLAY_NAMES["gpt4o"], 
        MODEL_DISPLAY_NAMES["mistral"],
        MODEL_DISPLAY_NAMES["o3mini"],
        MODEL_DISPLAY_NAMES["gpt41"]
    ]
    
    ordered_model_keys = ["claude", "gpt4o", "mistral", "o3mini", "gpt41"]
    
    # Add model entries with filled colored rectangles
    for model_key, model_label in zip(ordered_model_keys, ordered_model_labels):
        color = COLORS.get(model_key, '#666666')
        
        # Create filled rectangle for each model
        model_patch = Rectangle((0, 0), 1, 1, facecolor=color, edgecolor='black', 
                               linewidth=1, alpha=0.85)
        legend_handles.append(model_patch)
        legend_labels.append(model_label)
    
    # Add factual symbol (transparent with stroke)
    factual_patch = Rectangle((0, 0), 1, 1, facecolor='none', edgecolor='black', 
                             linewidth=1.5, alpha=1.0)
    legend_handles.append(factual_patch)
    legend_labels.append('Factual')
    
    # Add temporal symbol (dashed and transparent)
    temporal_patch = Rectangle((0, 0), 1, 1, facecolor='none', edgecolor='black', 
                              linewidth=1.5, alpha=1.0, hatch='///')
    legend_handles.append(temporal_patch)
    legend_labels.append('Temporal')
    
    # Create the legend with 3 columns
    legend = fig.legend(legend_handles, legend_labels, 
                       loc='upper center', bbox_to_anchor=(0.5, 0.02),
                       fontsize=FONT_SIZES['legend'], frameon=True, 
                       fancybox=False, shadow=False,
                       ncol=3, columnspacing=1.0, handletextpad=0.8, 
                       handlelength=2.0, labelspacing=0.6,
                       framealpha=1.0, edgecolor='black', facecolor='white')
    
    # Set legend frame properties
    legend.get_frame().set_linewidth(1)
    legend.get_frame().set_edgecolor('black')
    
    # Adjust layout to accommodate the bottom legend (much better spacing)
    plt.tight_layout()
    plt.subplots_adjust(left=0.08, right=0.95, bottom=0.13, top=0.94, hspace=0.6)
    
    # Save both PNG and PDF formats
    logger.info("Saving combined plot in multiple formats")
    plt.savefig(str(OUTPUT_PLOT_PNG), dpi=DPI, bbox_inches='tight', 
                facecolor='white', edgecolor='none', pad_inches=0.1)
    plt.savefig(str(OUTPUT_PLOT_PDF), dpi=DPI, bbox_inches='tight', 
                facecolor='white', edgecolor='none', format='pdf', pad_inches=0.1)
    
    print("📊 Combined publication plot saved to:")
    print(f"   PNG: {OUTPUT_PLOT_PNG}")
    print(f"   PDF: {OUTPUT_PLOT_PDF}")
    logger.info(f"Combined plot saved to {OUTPUT_PLOT_PNG} and {OUTPUT_PLOT_PDF}")
    
    return fig, (ax1, ax2)


def main():
    """
    Main function to create the combined exhaustivity plot.
    """
    print("🎯 Starting Combined Exhaustivity Plot Generation")
    print("=" * 50)
    logger.info("Beginning combined exhaustivity plot generation")
    
    try:
        # Load factoids results
        print("📁 Loading factoids results...")
        factoids_results = load_results_from_json(FACTOIDS_JSON)
        if factoids_results is None:
            print(f"❌ Failed to load factoids results from {FACTOIDS_JSON}")
            return
        
        # Load quintuples results
        print("📁 Loading quintuples results...")
        quintuples_results = load_results_from_json(QUINTUPLES_JSON)
        if quintuples_results is None:
            print(f"❌ Failed to load quintuples results from {QUINTUPLES_JSON}")
            return
        
        # Create combined plot
        print("📊 Creating combined publication-quality visualization...")
        fig, axes = create_combined_exhaustivity_plot(factoids_results, quintuples_results)
        
        if fig is not None:
            print("   ✅ Combined plot created successfully")
            logger.info("Combined plot created and saved successfully")
        else:
            print("   ⚠️  No combined plot generated")
            logger.warning("No combined plot was generated")
        
        print("\n✨ Combined Plot Generation Complete!")
        print(f"📊 Factoids data: {sum(len(v) for v in factoids_results.values())} samples")
        print(f"📊 Quintuples data: {sum(len(v) for v in quintuples_results.values())} samples")
        print("🖼️  Combined plots saved to:")
        print(f"   PNG: {OUTPUT_PLOT_PNG}")
        print(f"   PDF: {OUTPUT_PLOT_PDF}")
        print(f"📈 Plot includes models: {PUBLICATION_MODELS}")
        logger.info("Combined plot generation completed successfully")
        
    except Exception as e:
        logger.error(f"Error occurred: {str(e)}")
        print(f"❌ Error occurred: {str(e)}")
        print("💡 Check the logs for more details.")
        raise


if __name__ == "__main__":
    print("=" * 50)
    print("  COMBINED EXHAUSTIVITY PLOT GENERATION")
    print("=" * 50)
    main()