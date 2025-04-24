import logging
import pickle
import numpy as np
import os
from multiprocessing import Pool
# Updated import path for newer langchain versions
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from itext2kg.utils import PubtatorProcessor
from itext2kg import iText2KG
import multiprocessing
from functools import partial
from tqdm import tqdm
import argparse # Import argparse
from langchain_openai import ChatOpenAI

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(process)d - %(levelname)s - %(message)s') # Added process ID to format


def process_pmid(pmid, pubtator_path, output_path, llm_model_name, embeddings_model_name):
    """
    Processes a single PMID using default Ollama connection settings.
    """
    try:
        # Initialize LLM and Embeddings using defaults
        logging.info(f"Processing PMID {pmid} using default Ollama endpoint.")


        # llm = ChatOllama(
        #     model=llm_model_name,
        #     temperature=0
        #      # base_url removed
        # )
        openai_api_key = "EMPTY"
        openai_api_base = "http://localhost:8000/v1"

        llm = ChatOpenAI(
            api_key=openai_api_key,
            base_url=openai_api_base,
            model = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
            temperature=0,
            
        )
        embeddings = OllamaEmbeddings(
            model=embeddings_model_name,
            base_url="http://127.0.0.1:11434" # Default Ollama base URL
        )

        # --- Check connection (Optional but Recommended) ---
        # If llm = True is used above, this block MUST be commented out or removed
        try:
            # A simple invoke to check if the model endpoint is responsive
            llm.invoke('hi', config={'max_new_tokens': 1})
            logging.debug(f"Successfully invoked LLM for PMID {pmid} using default endpoint.")
        except AttributeError as ae:
            if isinstance(llm, bool) and llm is True:
                logging.warning(f"LLM is set to True, skipping connection check for PMID {pmid}.")
            else:
                # Re-raise the error if it's not the expected case
                logging.error(f"AttributeError during LLM check for PMID {pmid}. Is LLM correctly initialized? Error: {ae}")
                return # Skip if LLM object doesn't have invoke (e.g., if llm=True)
        except Exception as invoke_err:
            logging.error(f"Failed to connect/invoke model on default endpoint for PMID {pmid}. Is Ollama running at default location? Error: {invoke_err}")
            return # Skip this pmid if connection fails


        # --- Rest of your processing logic ---
        itext2kg = iText2KG(llm_model=llm, embeddings_model=embeddings)
        pubtator_file = os.path.join(pubtator_path, f"{pmid}.txt") # Use os.path.join

        # Check if file exists before processing
        if not os.path.exists(pubtator_file):
            raise FileNotFoundError(f"PubTator file not found: {pubtator_file}")

        pubtator_process = PubtatorProcessor(pubtator_file, llm) # Pass the actual LLM instance
        semantic_blocks = pubtator_process.block
        properties_info = pubtator_process.properties_info
        pubtator_info = pubtator_process.pubtator_info

        # Ensure semantic_blocks is not empty before accessing index -1
        if not semantic_blocks:
            logging.warning(f"No semantic blocks found for PMID: {pmid}. Skipping graph build.")
            return

        # Handle potential KeyError if 'source' is missing
        source_info = properties_info.get('source', 'Unknown Source') # Provide default

        # Ensure the last block exists before assigning
        if semantic_blocks:
            pubtator_info['abstract'] = {'context': semantic_blocks[-1], 'source': source_info}
        else:
            # Handle cases where semantic_blocks might be empty after processing
            pubtator_info['abstract'] = {'context': '', 'source': source_info}
            logging.warning(f"Semantic blocks list was empty for PMID: {pmid} after processing. Abstract context will be empty.")

        # Build the graph
        try:
            kg1 = itext2kg.build_graph(
                sections=[semantic_blocks], # Expects a list of lists/blocks
                source=properties_info, # Pass the whole dict
                entities_info=pubtator_info,
                ent_threshold=0.9,
                rel_threshold=0.4
            )
        except Exception as kg_err: # Catch specific errors if possible
            logging.error(f"KG build error for PMID {pmid}: {kg_err}", exc_info=True) # Add traceback
            return

        # Save the graph
        output_file = os.path.join(output_path, f"{pmid}.pkl") # Use os.path.join
        with open(output_file, 'wb') as f:
            pickle.dump(kg1, f)
        logging.info(f"Successfully processed and saved PMID {pmid}")

    except FileNotFoundError:
        # Already logged potentially above, but catch again for safety
        logging.error(f"PubTator file not found for PMID: {pmid}")
    except Exception as e:
        logging.error(f"Unexpected error processing PMID {pmid}: {e}", exc_info=True) # Log traceback


def main():
    # --- Argument Parsing ---

    data_path = "/home/mindrank/fuli/itext2kg/Data/AD_pubtabor"
    output_path = "/home/mindrank/fuli/itext2kg/output_kg/AD"
    embed_model = "nomic-embed-text:latest"
    num_workers = 20
    llm_model = True
    logging.info(f"Using {num_workers} worker processes.")
    logging.info(f"Targeting default Ollama instance (usually http://127.0.0.1:11434).")


    # --- Prepare PMID List ---
    pmid_list = []
    skipped_existing = 0
    skipped_criteria = 0
    # Consider making this list configurable via args or a file if it grows large
    pmid_errors = ["29568244", "32603776", "15798005", "18591211", "18525126", "20489158"] # List of known error PMIDs

    os.makedirs(output_path, exist_ok=True) # Ensure output directory exists

    logging.info(f"Scanning data path: {data_path}")
    try:
        all_files = [f for f in os.listdir(data_path) if f.endswith('.txt')]
    except FileNotFoundError:
        logging.error(f"Data path not found: {data_path}")
        return # Exit if data path is invalid
    logging.info(f"Found {len(all_files)} .txt files.")

    for file_name in tqdm(all_files, desc="Scanning PMIDs"):
        pmid = file_name.split('.')[0]
        # Basic check if pmid is likely numeric (can be enhanced)
        if not pmid.isdigit():
            logging.warning(f"Skipping file with non-numeric name: {file_name}")
            continue

        output_file = os.path.join(output_path, f"{pmid}.pkl")
        data_file = os.path.join(data_path, file_name)

        if os.path.exists(output_file):
            skipped_existing += 1
            continue # Skip if output already exists

        if pmid in pmid_errors:
            logging.info(f"Skipping known error PMID: {pmid}")
            skipped_criteria +=1
            continue # Skip known error PMIDs

        try:
            with open(data_file, "r", encoding='utf-8') as f: # Specify encoding
                text_line = f.readlines()
                if len(text_line) < 2: # Check if file has at least 2 lines
                    logging.warning(f"Skipping {pmid}: File has less than 2 lines.")
                    skipped_criteria += 1
                    continue
                # Added check for '|' before splitting
                if '|' not in text_line[1]:
                    logging.warning(f"Skipping {pmid}: Second line has unexpected format (missing '|'): {text_line[1][:50]}...")
                    skipped_criteria += 1
                    continue
                abstract = text_line[1].split('|')[-1].strip() # Get abstract and strip whitespace

            # Check abstract length and keyword presence
            if len(abstract) > 20 and "alzheimer" in abstract.lower():
                pmid_list.append(pmid)
            else:
                if len(abstract) <= 20:
                    logging.debug(f"Skipping {pmid}: Abstract too short ({len(abstract)} chars).")
                if "alzheimer" not in abstract.lower():
                    logging.debug(f"Skipping {pmid}: 'alzheimer' not found in abstract.")
                skipped_criteria += 1
        except IndexError:
            logging.error(f"Error processing file {data_file}: IndexError (likely unexpected file format or empty lines).")
            skipped_criteria += 1
        except Exception as e:
            logging.error(f"Error reading or processing file {data_file}: {e}", exc_info=True)
            skipped_criteria += 1


    logging.info(f"Prepared PMID list: Total to process={len(pmid_list)}, Skipped (Existing Output)={skipped_existing}, Skipped (Criteria/Error/Format)={skipped_criteria}")

    if not pmid_list:
        logging.info("No PMIDs to process.")
        return

    # --- Prepare Task Arguments ---
    tasks = pmid_list


    process_func_with_fixed_args = partial(
        process_pmid,
        pubtator_path=data_path,
        output_path=output_path,
        llm_model_name=llm_model,
        embeddings_model_name=embed_model
        # base_url argument removed from partial call
    )

    # --- Run multiprocessing Pool ---
    logging.info(f"Starting processing {len(tasks)} PMIDs with {num_workers} workers...")
    # Use imap_unordered with tqdm for progress display
    # The worker function receives items from 'tasks' (which is pmid_list)
    with multiprocessing.Pool(num_workers) as pool:
        for _ in tqdm(pool.imap_unordered(process_func_with_fixed_args, tasks), total=len(tasks), desc="Processing PMIDs"):
            pass  # Result is not used, only iteration for progress bar

    logging.info("Processing finished.")


if __name__ == "__main__":
    main()