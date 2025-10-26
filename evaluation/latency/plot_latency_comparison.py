#!/usr/bin/env python3
"""
Script to create a bar plot comparing latency between Graphiti and Atom.
"""

import json
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

# Global configuration
MAX_BARS_TO_DISPLAY = 11  # Reduce number of bars for better readability

def load_graphiti_data(json_path):
    """Load and process Graphiti latency data."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    results = []
    cumulative_latency = 0
    
    for batch in data['batch_results']:
        if batch['status'] == 'success' and batch['execution_time_seconds'] is not None:
            cumulative_latency += batch['execution_time_seconds']
            results.append({
                'total_factoids': batch['total_factoids_processed'],
                'cumulative_latency_seconds': cumulative_latency
            })
    
    return results

def load_atom_data(json_path):
    """Load and process Atom latency data."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    results = []
    cumulative_factoids = 0
    cumulative_total_latency = 0
    cumulative_api_latency = 0
    
    for batch in data:
        cumulative_factoids += batch['batch_size']
        cumulative_total_latency += batch['total_latency_seconds']
        
        # Calculate API call latency components for this batch
        batch_api_latency = (
            batch['api_latencies']['extraction_avg_seconds'] + 
            batch['api_latencies']['entity_embedding_avg_seconds'] + 
            batch['api_latencies']['relationship_embedding_avg_seconds']
        )
        cumulative_api_latency += batch_api_latency
        
        results.append({
            'total_factoids': cumulative_factoids,
            'cumulative_total_latency_seconds': cumulative_total_latency,
            'cumulative_api_latency_seconds': cumulative_api_latency
        })
    
    return results

def load_itext2kg_data(json_path):
    """Load and process iText2KG latency data."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    results = []
    cumulative_factoids = 0
    cumulative_latency = 0
    
    for batch in data:
        cumulative_factoids += batch['total_facts_processed']
        cumulative_latency += batch['total_latency_seconds']
        
        results.append({
            'total_factoids': cumulative_factoids,
            'cumulative_latency_seconds': cumulative_latency
        })
    
    return results

def create_latency_comparison_plot(graphiti_data, atom_data, itext2kg_data, max_factoids=None):
    """Create the latency comparison bar plot."""
    
    # Publication-quality settings
    FIGURE_WIDTH = 10
    FIGURE_HEIGHT = 8
    DPI = 300
    
    FONT_SIZES = {
        'axis_labels': 25,
        'tick_labels': 20,  # Slightly smaller for better fit
        'legend': 20,
        'title': 27
    }
    
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
        'axes.linewidth': 1.0,
        'grid.linewidth': 0.5,
        'lines.linewidth': 1.0
    })
    
    # Set up the figure
    fig, ax = plt.subplots(1, 1, figsize=(FIGURE_WIDTH, FIGURE_HEIGHT), dpi=DPI)
    
    # Publication color palette
    color_graphiti = '#1f77b4'  # Blue
    color_atom = '#d62728'      # Red
    color_itext2kg = '#2ca02c'  # Green
    
    # Prepare data for plotting
    graphiti_factoids = [d['total_factoids'] for d in graphiti_data]
    graphiti_hours = [d['cumulative_latency_seconds'] / 3600 for d in graphiti_data]
    
    atom_factoids = [d['total_factoids'] for d in atom_data]
    atom_total_hours = [d['cumulative_total_latency_seconds'] / 3600 for d in atom_data]
    atom_api_hours = [d['cumulative_api_latency_seconds'] / 3600 for d in atom_data]
    
    itext2kg_factoids = [d['total_factoids'] for d in itext2kg_data]
    itext2kg_hours = [d['cumulative_latency_seconds'] / 3600 for d in itext2kg_data]
    
    # Filter data to max_factoids if specified
    if max_factoids is not None:
        # Filter graphiti
        graphiti_mask = [f <= max_factoids for f in graphiti_factoids]
        graphiti_factoids = [f for f, m in zip(graphiti_factoids, graphiti_mask) if m]
        graphiti_hours = [h for h, m in zip(graphiti_hours, graphiti_mask) if m]
        
        # Filter atom
        atom_mask = [f <= max_factoids for f in atom_factoids]
        atom_factoids = [f for f, m in zip(atom_factoids, atom_mask) if m]
        atom_total_hours = [h for h, m in zip(atom_total_hours, atom_mask) if m]
        atom_api_hours = [h for h, m in zip(atom_api_hours, atom_mask) if m]
        
        # Filter itext2kg
        itext2kg_mask = [f <= max_factoids for f in itext2kg_factoids]
        itext2kg_factoids = [f for f, m in zip(itext2kg_factoids, itext2kg_mask) if m]
        itext2kg_hours = [h for h, m in zip(itext2kg_hours, itext2kg_mask) if m]
    
    # Sample data points to make the plot readable, with even spacing
    # Use numpy linspace for even spacing across the range
    num_samples = min(MAX_BARS_TO_DISPLAY, len(graphiti_factoids))
    
    if num_samples <= 1:
        sample_indices = [0] if len(graphiti_factoids) > 0 else []
    else:
        # Create evenly spaced indices, including first and last
        sample_positions = np.linspace(0, len(graphiti_factoids) - 1, num_samples)
        sample_indices = [int(round(pos)) for pos in sample_positions]
        # Remove duplicates while preserving order
        seen = set()
        sample_indices = [x for x in sample_indices if not (x in seen or seen.add(x))]
    
    graphiti_factoids_sample = [graphiti_factoids[i] for i in sample_indices]
    graphiti_hours_sample = [graphiti_hours[i] for i in sample_indices]
    
    # For Atom, find data points that match similar factoid counts
    atom_factoids_sample = []
    atom_total_hours_sample = []
    atom_api_hours_sample = []
    
    # For iText2KG, find data points that match similar factoid counts
    itext2kg_factoids_sample = []
    itext2kg_hours_sample = []
    
    # Find closest Atom and iText2KG data points to Graphiti factoid counts
    for g_factoids in graphiti_factoids_sample:
        # Atom
        if atom_factoids:
            closest_idx = min(range(len(atom_factoids)), 
                             key=lambda i: abs(atom_factoids[i] - g_factoids))
            atom_factoids_sample.append(atom_factoids[closest_idx])
            atom_total_hours_sample.append(atom_total_hours[closest_idx])
            atom_api_hours_sample.append(atom_api_hours[closest_idx])
        else:
            atom_factoids_sample.append(g_factoids)
            atom_total_hours_sample.append(0)
            atom_api_hours_sample.append(0)
        
        # iText2KG
        if itext2kg_factoids:
            closest_idx = min(range(len(itext2kg_factoids)), 
                             key=lambda i: abs(itext2kg_factoids[i] - g_factoids))
            itext2kg_factoids_sample.append(itext2kg_factoids[closest_idx])
            itext2kg_hours_sample.append(itext2kg_hours[closest_idx])
        else:
            itext2kg_factoids_sample.append(g_factoids)
            itext2kg_hours_sample.append(0)
    
    # Create bar positions
    x = np.arange(len(graphiti_factoids_sample))
    width = 0.25  # Narrower bars since we have 3 systems
    
    # Create bars in order: iText2KG (left), Graphiti (middle), Atom (right)
    
    # iText2KG - leftmost
    bars_itext2kg = ax.bar(x - width, itext2kg_hours_sample, width,
                          color=color_itext2kg, alpha=0.85,
                          edgecolor='black', linewidth=1, zorder=2)
    
    # Graphiti - middle
    bars_graphiti = ax.bar(x, graphiti_hours_sample, width, 
                          color=color_graphiti, alpha=0.85, 
                          edgecolor='black', linewidth=1, zorder=2)
    
    # Atom (total latency) - rightmost
    bars_atom_total = ax.bar(x + width, atom_total_hours_sample, width,
                            color=color_atom, alpha=0.85,
                            edgecolor='black', linewidth=1, zorder=2)
    
    # Overlay API call latency portion with hatching (same color as Atom)
    bars_atom_api = ax.bar(x + width, atom_api_hours_sample, width,
                          color=color_atom, alpha=0.65, 
                          hatch='//////', edgecolor='black', linewidth=1, zorder=3)
    
    # Customize the plot
    ax.set_xlabel('Number of atomic facts', fontsize=FONT_SIZES['axis_labels'], 
                  fontweight='bold')
    ax.set_ylabel('Processing time (hours)', fontsize=FONT_SIZES['axis_labels'], 
                  fontweight='bold')
    ax.set_title('Latency comparison', 
                fontsize=FONT_SIZES['title'], fontweight='bold', pad=20)
    
    # Format factoid counts with 'k' suffix
    def format_factoid_count(fc):
        if fc >= 1000:
            return f'{fc/1000:.1f}k'
        else:
            return f'{int(fc)}'
    
    # Set ALL x-axis ticks and labels explicitly
    ax.set_xticks(x)
    x_labels = [format_factoid_count(fc) for fc in graphiti_factoids_sample]
    ax.set_xticklabels(x_labels, rotation=45, ha='right', 
                       fontsize=FONT_SIZES['tick_labels'])
    
    # Force all ticks to be visible
    ax.tick_params(axis='x', which='major', length=4, width=1.0)
    
    # Extend x-axis limits to use full plot width
    ax.set_xlim(-0.5, len(x) - 0.5)
    
    # Set y-axis limit to 8 hours
    y_max_limit = 8
    y_ticks = np.arange(0, y_max_limit + 0.5, 0.5)  # Ticks every 0.5 hours
    ax.set_yticks(y_ticks)
    ax.set_ylim(0, y_max_limit)
    
    # Add minor tick marks on y-axis
    minor_ticks = []
    for i in range(len(y_ticks) - 1):
        minor_ticks.extend(np.linspace(y_ticks[i], y_ticks[i+1], 5)[1:-1])
    ax.set_yticks(minor_ticks, minor=True)
    ax.tick_params(axis='y', which='minor', length=3, width=0.5)
    
    # Format y-axis
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1f}'))
    
    # Add improved horizontal gridlines
    for y_val in y_ticks:
        ax.axhline(y=y_val, color='gray', linestyle='-', alpha=0.3, 
                   linewidth=0.5, zorder=1)
    
    # Create custom legend (order matches bar order: iText2KG, Graphiti, Atom)
    from matplotlib.patches import Rectangle
    
    legend_handles = []
    legend_labels = []
    
    # iText2KG
    itext2kg_patch = Rectangle((0, 0), 1, 1, facecolor=color_itext2kg, 
                              edgecolor='black', linewidth=1, alpha=0.85)
    legend_handles.append(itext2kg_patch)
    legend_labels.append('iText2KG (Total)')
    
    # Graphiti
    graphiti_patch = Rectangle((0, 0), 1, 1, facecolor=color_graphiti, 
                              edgecolor='black', linewidth=1, alpha=0.85)
    legend_handles.append(graphiti_patch)
    legend_labels.append('Graphiti (Total)')
    
    # Atom total
    atom_patch = Rectangle((0, 0), 1, 1, facecolor=color_atom, 
                           edgecolor='black', linewidth=1, alpha=0.85)
    legend_handles.append(atom_patch)
    legend_labels.append('Atom (Total)')
    
    # Atom API
    atom_api_patch = Rectangle((0, 0), 1, 1, facecolor=color_atom, 
                              edgecolor='black', linewidth=1, alpha=0.65, hatch='///')
    legend_handles.append(atom_api_patch)
    legend_labels.append('Atom (API Calls)')
    
    ax.legend(handles=legend_handles, labels=legend_labels,
              loc='upper left', frameon=True, fancybox=False, shadow=False,
              fontsize=FONT_SIZES['legend'], framealpha=1.0, 
              edgecolor='black', facecolor='white')
    
    # Set legend frame properties
    legend = ax.get_legend()
    legend.get_frame().set_linewidth(1)
    legend.get_frame().set_edgecolor('black')
    
    # Add black borders around plot (1pt stroke)
    for spine in ax.spines.values():
        spine.set_linewidth(1)
        spine.set_color('black')
        spine.set_visible(True)
    
    # Set grid below data
    ax.set_axisbelow(True)
    
    # Improve tick parameters
    ax.tick_params(axis='both', which='major', width=1.0, length=4)
    ax.tick_params(axis='x', which='major', pad=8)
    
    # Adjust layout to ensure labels aren't cut off
    plt.tight_layout()
    
    return fig

def main():
    """Main function to create the latency comparison plot."""
    
    # Define paths
    graphiti_path = Path('evaluation/atom_baselines/batch_latency_graphiti.json')
    atom_path = Path('evaluation/atom_baselines/batch_latency_atom.json')
    itext2kg_path = Path('batch_cache_itext2kg/batch_latency_stats.json')
    output_path = Path('evaluation/latency_comparison_plot.png')
    
    # Load data
    print("Loading Graphiti data...")
    graphiti_data = load_graphiti_data(graphiti_path)
    print(f"Loaded {len(graphiti_data)} Graphiti data points")
    if graphiti_data:
        print(f"Graphiti factoid range: {graphiti_data[0]['total_factoids']} to {graphiti_data[-1]['total_factoids']}")
    
    print("Loading Atom data...")
    atom_data = load_atom_data(atom_path)
    print(f"Loaded {len(atom_data)} Atom data points")
    if atom_data:
        print(f"Atom factoid range: {atom_data[0]['total_factoids']} to {atom_data[-1]['total_factoids']}")
    
    print("Loading iText2KG data...")
    itext2kg_data = load_itext2kg_data(itext2kg_path)
    print(f"Loaded {len(itext2kg_data)} iText2KG data points")
    if itext2kg_data:
        print(f"iText2KG factoid range: {itext2kg_data[0]['total_factoids']} to {itext2kg_data[-1]['total_factoids']}")
    
    # Calculate the common maximum number of atomic facts
    max_factoids_list = []
    if graphiti_data:
        max_factoids_list.append(graphiti_data[-1]['total_factoids'])
    if atom_data:
        max_factoids_list.append(atom_data[-1]['total_factoids'])
    if itext2kg_data:
        max_factoids_list.append(itext2kg_data[-1]['total_factoids'])
    
    max_factoids = min(max_factoids_list) if max_factoids_list else None
    print(f"Common maximum atomic facts: {max_factoids}")
    
    # Create plot
    print("Creating latency comparison plot...")
    fig = create_latency_comparison_plot(graphiti_data, atom_data, itext2kg_data, max_factoids)
    
    # Save plot
    fig.savefig(output_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none', pad_inches=0.1)
    print(f"Plot saved to: {output_path}")
    
    # Also save as PDF for publication
    pdf_path = output_path.with_suffix('.pdf')
    fig.savefig(pdf_path, bbox_inches='tight', 
                facecolor='white', edgecolor='none', format='pdf', pad_inches=0.1)
    print(f"PDF version saved to: {pdf_path}")
    
    plt.show()

if __name__ == "__main__":
    main()