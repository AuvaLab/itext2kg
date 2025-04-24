import json
import logging
import os
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, current_process  # 导入 current_process

from langchain_ollama import ChatOllama

from itext2kg.documents_distiller import DiseaseArticle
from itext2kg.documents_distiller import DocumentsDistiller  # Local import to fix cycle

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DATA_PATH = "/home/mindrank/fuli/itext2kg/Data/AD_pubtabor"
OUTPUT_PATH = "/home/mindrank/fuli/itext2kg/output_kg/AD"

def escape_curly_braces(text):
    return text.replace("{", "{{").replace("}", "}}")

def abstract_distiller(pmid):  # 修改函数签名
    try:
        with open(f"{DATA_PATH}/{pmid}.txt", "r") as f:
            text = f.readlines()
            title = text[0].strip().split('|')[-1]
            abstract = text[1].strip().split('|')[-1]
            title = escape_curly_braces(title)
            abstract = escape_curly_braces(abstract)
            context = f"Title: {title} Abstract: {abstract}"

        llm = True
        document_distiller = DocumentsDistiller(llm_model=llm)
        IE_query = """
            # DIRECTIVES:
            - As an experienced information extractor, your task is to extract biological entities from the provided bioinformatics context.
            - Only extract entities that are explicitly mentioned in the context; do not generate or create any new terms.
            - Extracted entities may include, but are not limited to, gene names, protein names, disease names, biological processes, pathways, molecular interactions, and other key bioinformatics terms.
            - If an entity is not clearly mentioned in the context, leave it blank and do not infer or generate non-existent information.
            - The output should only include the entities, excluding any non-entity content such as descriptive text or inferences.
            """

        distilled = document_distiller.distill(
            documents=[context], 
            IE_query=IE_query, 
            output_data_structure=DiseaseArticle)

        with open(f"{OUTPUT_PATH}_distiller/{pmid}.json", 'w') as f:
            json.dump(distilled, f, indent=4)

    except FileNotFoundError:
        logging.error(f"File not found: {DATA_PATH}/{pmid}.txt")
    except Exception as e:
        logging.exception(f"An error occurred while processing {pmid}: {e}")


# def init_worker():
#     global llm
#     llm = True
#     llm = ChatOllama(
#             model="deepseek-r1:32b",  # 或你的自定义模型
#             temperature=0,
#             base_url=f"http://127.0.0.1:{port}"  # 使用传入的端口
#         )
    

def main():
    os.makedirs(f"{OUTPUT_PATH}_distiller", exist_ok=True)

    pmid_list = []
    for file_name in os.listdir(DATA_PATH):
        pmid = file_name.split('.')[0]
        if not os.path.exists(f"/home/mindrank/fuli/itext2kg/output_kg/AD/{pmid}.pkl"):
            pmid_list.append(pmid)

    # pmid_list_chunk = np.array_split(pmid_list, 9)
    # pmid_list = pmid_list_chunk[int(chunk)]


    num_processes = 20
    with Pool(processes=num_processes) as pool:
        for _ in tqdm(pool.imap_unordered(abstract_distiller, pmid_list), total=len(pmid_list)):
            pass

if __name__ == '__main__':
    # import argparse
    # parser = argparse.ArgumentParser(
    #     prog='ProgramName',
    #     description='What the program does',
    #     epilog='Text at the bottom of help')
    # parser.add_argument('-i', '--chunk')
    # parser.add_argument('-p', '--port')  # 现在 -p 是基础端口
    # args = parser.parse_args()
    main()