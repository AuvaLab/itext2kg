import logging
import pickle
import numpy as np
import os
from multiprocessing import Pool, Manager
from langchain_ollama import OllamaLLM
from langchain_ollama import OllamaEmbeddings
from itext2kg.utils import PubtatorProcessor
from itext2kg import iText2KG
import multiprocessing
from functools import partial
from tqdm import tqdm


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_pmid(pmid, pubtator_path, output_path, llm_model_name, embeddings_model_name):
    """
    Processes a single PMID.  Now takes model names instead of instances.
    """
    try:
        llm = OllamaLLM(
            model=llm_model_name, 
            temperature=0, 
            base_url="http://127.0.0.1:11443"
            )
        llm = True
        # embeddings = OllamaEmbeddings(
        #     model=embeddings_model_name, 
        #     base_url="http://127.0.0.1:11443"
        #     )
        
        llm.invoke('hi')
        itext2kg = iText2KG(llm_model=llm, embeddings_model=embeddings)
        pubtator_file = f"{pubtator_path}/{pmid}.txt"
        pubtator_process = PubtatorProcessor(pubtator_file, llm)
        semantic_blocks = pubtator_process.block
        properties_info = pubtator_process.properties_info
        pubtator_info = pubtator_process.pubtator_info

        pubtator_info['abstract'] = {'context': semantic_blocks[-1], 'source': properties_info['source']}

        # Build the graph
        try:
            kg1 = itext2kg.build_graph(
                sections=[semantic_blocks],
                source=properties_info,
                entities_info=pubtator_info,
                ent_threshold=0.9,
                rel_threshold=0.4
            )
        except:
            logging.error(f"kg error: {pmid}")
            return
        # Save the graph
        with open(f'{output_path}/{pmid}.pkl', 'wb') as f:
            pickle.dump(kg1, f)

    except FileNotFoundError:
        logging.error(f"PubTator file not found for PMID: {pmid}")


def main():
    DATA_PATH = "/home/mindrank/fuli/itext2kg/Data/AD_pubtabor"
    OUTPUT_PATH = "/home/mindrank/fuli/itext2kg/output_kg/AD"

    pmid_list = []
    for file_name in os.listdir(DATA_PATH):
        pmid = file_name.split('.')[0]
        if os.path.exists(f"{OUTPUT_PATH}/{pmid}.pkl") or pmid in ["29568244", "32603776", "15798005", "18591211", "18525126","20489158"]:
            pass
        else:
            with open(f"{DATA_PATH}/{pmid}.txt", "r") as f: #self.variable
                    text_line = f.readlines() # Strip AND filter out empty lines
                    abstract = text_line[1].split('|')[-1]
            if len(abstract) > 20 and "alzheimer" in abstract.lower():
                pmid_list.append(pmid)
    
    llm_model_name = "deepseek-r1:32b"
    embeddings_model_name = "nomic-embed-text:latest"
    process_func = partial(
        process_pmid, pubtator_path=DATA_PATH, 
        output_path=OUTPUT_PATH, llm_model_name=llm_model_name, 
        embeddings_model_name=embeddings_model_name
        ) 
    # Use imap_unordered with tqdm for progress display
    with multiprocessing.Pool(2) as pool:
        for _ in tqdm(pool.imap_unordered(process_func, pmid_list), total=len(pmid_list), desc="Processing PMIDs"):
            pass  # Result is not used, only iteration for progress bar

    print("Done")
    


if __name__ == "__main__":
    main()