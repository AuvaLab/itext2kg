#!/usr/bin/env python3
"""
Graphiti Latency Testing and Benchmarking

This script measures the latency performance of the Graphiti knowledge graph construction system
(a baseline approach) by processing batches of factoids and tracking timing metrics.
Usage:
    python test_graphiti.py
"""

import asyncio
import ast
import json
import os
import time
from datetime import datetime
from typing import Dict, List, Any

import pandas as pd
import dateparser
from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType
from graphiti_core.llm_client import OpenAIClient, LLMConfig

# =============================================================================
# CONFIGURATION - MODIFY THESE VARIABLES AS NEEDED
# =============================================================================

os.environ["OPENAI_API_KEY"] = "###"
os.environ["NEO4J_PASSWORD"] = "###"
# Dataset configuration
DATASET_PATH = "datasets/nyt_news/2020_nyt_COVID_last_version_ready_quintuples_gpt41_from_factoids_run3_run3.pkl"
FACTOIDS_COL = "factoids_g_truth"
DATE_COL = "date"

# Processing configuration
BATCH_SIZE = 40

# Output configuration
OUTPUT_FILE = "batch_latency_results.json"

# Database configuration (optional - can also use environment variables)
NEO4J_PASSWORD = None  # Set to your password or leave None to use environment variable

# =============================================================================
# END CONFIGURATION
# =============================================================================


def to_dictionary(df: pd.DataFrame, factoids_col: str, date_col: str) -> Dict[str, List[str]]:
    """
    Convert DataFrame to dictionary grouped by date.
    
    Args:
        df: Input DataFrame
        factoids_col: Column name containing factoids
        date_col: Column name containing dates
        
    Returns:
        Dictionary with dates as keys and factoids lists as values
    """
    # Handle string representation of lists
    if isinstance(df[factoids_col].iloc[0], str):
        df[factoids_col] = df[factoids_col].apply(lambda x: ast.literal_eval(x))
    
    # Group by date and sum factoids
    grouped_df = df.groupby(date_col)[factoids_col].sum().reset_index()
    
    return {
        str(date): factoids 
        for date, factoids in grouped_df.set_index(date_col)[factoids_col].to_dict().items()
    }


async def add_episodes_batch(graphiti: Graphiti, reference_date: str, episodes: List[str]) -> None:
    """
    Add a batch of episodes to Graphiti.
    
    Args:
        graphiti: Graphiti instance
        reference_date: Reference date for the episodes
        episodes: List of factoid episodes to add
    """
    for i, factoid in enumerate(episodes):
        await graphiti.add_episode(
            name=f"factoid_{i}_{reference_date}",
            episode_body=factoid,
            source=EpisodeType.text,
            source_description="Batch processed factoids",
            reference_time=dateparser.parse(reference_date),
        )


async def process_factoids_batch(
    graphiti: Graphiti, 
    factoids_dict: Dict[str, List[str]], 
    batch_size: int
) -> List[Dict[str, Any]]:
    """
    Process factoids in batches and measure latency.
    
    Args:
        graphiti: Graphiti instance
        factoids_dict: Dictionary of factoids grouped by date
        batch_size: Number of factoids to process in each batch
        
    Returns:
        List of results with latency measurements
    """
    results = []
    total_factoids_processed = 0
    
    for date, factoids in factoids_dict.items():
        print(f"Processing date: {date} with {len(factoids)} factoids")
        
        # Process factoids in batches
        for i in range(0, len(factoids), batch_size):
            batch = factoids[i:i + batch_size]
            batch_num = i // batch_size + 1
            
            try:
                print(f"  Processing batch {batch_num} with {len(batch)} factoids...")
                
                start_time = time.perf_counter()
                await add_episodes_batch(graphiti, date, batch)
                end_time = time.perf_counter()
                
                elapsed_time = end_time - start_time
                total_factoids_processed += len(batch)
                
                result = {
                    "date": date,
                    "batch_number": batch_num,
                    "batch_size": len(batch),
                    "total_factoids_processed": total_factoids_processed,
                    "execution_time_seconds": elapsed_time,
                    "factoids_per_second": len(batch) / elapsed_time if elapsed_time > 0 else 0,
                    "timestamp": datetime.now().isoformat(),
                    "status": "success"
                }
                
                results.append(result)
                print(f"    Completed in {elapsed_time:.2f}s ({len(batch) / elapsed_time:.2f} factoids/s)")
                
            except Exception as e:
                print(f"  Error processing batch {batch_num}: {e}")
                
                # Retry once
                print(f"  Retrying batch {batch_num}...")
                try:
                    start_time = time.perf_counter()
                    await add_episodes_batch(graphiti, date, batch)
                    end_time = time.perf_counter()
                    
                    elapsed_time = end_time - start_time
                    total_factoids_processed += len(batch)
                    
                    result = {
                        "date": date,
                        "batch_number": batch_num,
                        "batch_size": len(batch),
                        "total_factoids_processed": total_factoids_processed,
                        "execution_time_seconds": elapsed_time,
                        "factoids_per_second": len(batch) / elapsed_time if elapsed_time > 0 else 0,
                        "timestamp": datetime.now().isoformat(),
                        "status": "success_retry"
                    }
                    
                    results.append(result)
                    print(f"    Retry completed in {elapsed_time:.2f}s ({len(batch) / elapsed_time:.2f} factoids/s)")
                    
                except Exception as e_retry:
                    print(f"  Retry failed for batch {batch_num}: {e_retry}")
                    
                    result = {
                        "date": date,
                        "batch_number": batch_num,
                        "batch_size": len(batch),
                        "total_factoids_processed": total_factoids_processed,
                        "execution_time_seconds": None,
                        "factoids_per_second": None,
                        "timestamp": datetime.now().isoformat(),
                        "status": "error",
                        "error": str(e_retry)
                    }
                    
                    results.append(result)
    
    return results


def setup_graphiti() -> Graphiti:
    """
    Setup Graphiti instance with OpenAI client.
    
    Returns:
        Configured Graphiti instance
    """
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY environment variable not set")
    
    # Configure LLM
    llm_config = LLMConfig(
        model="gpt-4.1-2025-04-14",  # Using a more standard model name
        temperature=0.0,
        max_tokens=None,
    )
    
    # Initialize OpenAI client
    llm_client = OpenAIClient(config=llm_config)
    
    # Determine password to use
    password = NEO4J_PASSWORD if NEO4J_PASSWORD is not None else os.getenv("NEO4J_PASSWORD", "neo4j")
    
    # Initialize Graphiti with Neo4j connection
    graphiti = Graphiti(
        uri="bolt://localhost:7687",
        user="neo4j",
        password=password,
        llm_client=llm_client
    )
    
    return graphiti


def save_results(results: List[Dict[str, Any]], output_file: str) -> None:
    """
    Save results to JSON file.
    
    Args:
        results: List of result dictionaries
        output_file: Output file path
    """
    # Add summary statistics
    successful_results = [r for r in results if r["status"] in ["success", "success_retry"]]
    
    summary = {
        "total_batches": len(results),
        "successful_batches": len(successful_results),
        "failed_batches": len(results) - len(successful_results),
        "total_factoids": sum(r["batch_size"] for r in results),
        "total_execution_time": sum(r["execution_time_seconds"] for r in successful_results if r["execution_time_seconds"]),
        "average_factoids_per_second": sum(r["factoids_per_second"] for r in successful_results if r["factoids_per_second"]) / len(successful_results) if successful_results else 0,
        "processing_timestamp": datetime.now().isoformat()
    }
    
    output_data = {
        "summary": summary,
        "batch_results": results
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Results saved to {output_file}")


async def main():
    """Main function to process factoids in batches."""
    
    print("Starting batch factoids latency processor...")
    print("Configuration:")
    print(f"  Dataset path: {DATASET_PATH}")
    print(f"  Factoids column: {FACTOIDS_COL}")
    print(f"  Date column: {DATE_COL}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Output file: {OUTPUT_FILE}")
    
    try:
        # Load dataset
        print(f"\nLoading dataset from {DATASET_PATH}...")
        
        if DATASET_PATH.endswith('.csv'):
            df = pd.read_csv(DATASET_PATH)
        elif DATASET_PATH.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(DATASET_PATH)
        elif DATASET_PATH.endswith('.pkl'):
            df = pd.read_pickle(DATASET_PATH)
        else:
            raise ValueError("Unsupported file format. Use CSV, Excel, or Pickle files.")
        
        print(f"Dataset loaded: {len(df)} rows")
        
        # Validate columns
        if FACTOIDS_COL not in df.columns:
            raise ValueError(f"Column '{FACTOIDS_COL}' not found in dataset")
        if DATE_COL not in df.columns:
            raise ValueError(f"Column '{DATE_COL}' not found in dataset")
        
        # Convert to dictionary format
        print("Converting dataset to dictionary format...")
        factoids_dict = to_dictionary(df, FACTOIDS_COL, DATE_COL)
        
        total_factoids = sum(len(factoids) for factoids in factoids_dict.values())
        print(f"Total factoids to process: {total_factoids} across {len(factoids_dict)} dates")
        print(f"Batch size: {BATCH_SIZE}")
        
        # Setup Graphiti
        print("Setting up Graphiti...")
        graphiti = setup_graphiti()
        
        # Process factoids in batches
        print("Starting batch processing...")
        start_total = time.perf_counter()
        
        results = await process_factoids_batch(graphiti, factoids_dict, BATCH_SIZE)
        
        end_total = time.perf_counter()
        total_time = end_total - start_total
        
        print(f"\nProcessing completed in {total_time:.2f} seconds")
        print(f"Processed {len(results)} batches")
        
        # Save results
        print(f"Saving results to {OUTPUT_FILE}...")
        save_results(results, OUTPUT_FILE)
        
        # Print summary
        successful_batches = len([r for r in results if r["status"] in ["success", "success_retry"]])
        failed_batches = len(results) - successful_batches
        
        print("Summary:")
        print(f"  Successful batches: {successful_batches}")
        print(f"  Failed batches: {failed_batches}")
        print(f"  Total processing time: {total_time:.2f} seconds")
        print(f"  Average time per batch: {total_time / len(results) if results else 0:.2f} seconds")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
