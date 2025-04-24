
import logging
import time, re
from .schemas import DiseaseArticle
import re



class PubtatorProcessor:
    """
    A class for processing text and extracting information, using a pipeline of steps.
    """

    def __init__(self, pubtator_file, llm_model): #Take in as parameters, don't hardcode
        """Initializes the TextProcessor and processes the PubTator file."""
        self.pubtator_file = pubtator_file
        self.llm_model = llm_model  # Language Model
        self.context, self.PMID, self.pubtator_info, self.pubtator_distilled, self.species_info = self._process_pubtator() #Call helper during init.
        self.distilled = self._distiller_abstract()
        self.properties_info = self._construct_properties_info()
        self.block = self._contstruct_block()

    def escape_curly_braces(self, text):
        return text.replace("{", "{{").replace("}", "}}")

    def clean_string(self, s: str) -> str:
        """
        去除字符串中的双引号和特殊字符，仅保留字母、数字和空格。

        :param s: 原始字符串
        :return: 处理后的字符串
        """
        s = s.replace('"', '')  # 去除双引号
        s = re.sub(r'[\'\"@#$^&*(){}[\]<>:;.,!?/\\|+=~`]', '', s) # 仅保留字母、数字和空格
        return s.strip()  # 去除首尾空格
    
    def _process_pubtator(self): #Start every private function with _
        """Processes a PubTator file to extract context and entity information."""
        logging.info(f"Processing PubTator file: {self.pubtator_file}")

        try:
            with open(self.pubtator_file, "r") as f: #self.variable
                text_line = f.readlines() # Strip AND filter out empty lines

            PMID = text_line[0].split('|')[0]
            title = self.escape_curly_braces(text_line[0].strip().split('|')[-1])
            abstract = self.escape_curly_braces(text_line[1].strip().split('|')[-1])
            # abstract = self.clean_string(text_line[1].split('|')[-1])
            context = f"Title: {title} Abstract: {abstract}"

            pubtator_info = {}
            pubtator_distilled = {
                'disease': [],
                'gene': [],
                'variant': [],
                'cell_line': [],
                'chemical': [],
                'proteinmutation': [],
            }
            species_info = []
            seen_ids = set()

            for entity_line in text_line[2:]:
                entity_line = entity_line.strip().split("\t")
                if len(entity_line) == 6:
                    label = str(entity_line[4])
                    name = str(entity_line[3])
                    if label == 'Gene':
                        unique_ID = f"Gene ID:{entity_line[5]}"
                    else:
                        unique_ID = entity_line[5]

                    if name not in seen_ids:
                        if label == 'Species':
                            species_info.append(unique_ID)
                            seen_ids.add(name)

                        elif label.lower() in pubtator_distilled:
                            pubtator_distilled[label.lower()].append({label.lower(): name})
                            pubtator_info[name] = {"label": label.lower(), "unique_id": unique_ID}
                            seen_ids.add(name)
                    
                elif len(entity_line) == 5 and entity_line[4] != 'Species':
                    label = str(entity_line[4])
                    name = str(entity_line[3])
                    if name not in seen_ids:
                        seen_ids.add(name)
                        if label.lower() not in pubtator_distilled:
                            pubtator_distilled[label.lower()] = []
                            pubtator_distilled[label.lower()].append({label.lower(): name})
                            pubtator_info[name] = {"label": label.lower()}
                else:
                    # logging.warning(f"Skipping malformed entity line: {entity_line}")
                    pass
            return context, PMID, pubtator_info, pubtator_distilled, species_info

        except FileNotFoundError:
            logging.error(f"PubTator file not found: {self.pubtator_file}")
            return None, None, None, None, None
        except Exception as e:
            logging.error(f"Error processing PubTator file: {e}", exc_info=True)  #Include exception for easier debug
            return None, None, None, None, None

    def _distiller_abstract(self):
        MAX_RETRIES = 3
        RETRY_DELAY = 5  # seconds
        """Distills the abstract using a language model."""
        from itext2kg.documents_distiller import DocumentsDistiller #Local import to fix cycle
        
        for attempt in range(MAX_RETRIES):
            try:
                document_distiller = DocumentsDistiller(llm_model=self.llm_model)  # Pass in the LLM!
                IE_query = """
                    # DIRECTIVES:
                    - As an experienced information extractor, your task is to extract biological entities from the provided bioinformatics context.
                    - Only extract entities that are explicitly mentioned in the context; do not generate or create any new terms.
                    - Extracted entities may include, but are not limited to, gene names, protein names, disease names, biological processes, pathways, molecular interactions, and other key bioinformatics terms.
                    - If an entity is not clearly mentioned in the context, leave it blank and do not infer or generate non-existent information.
                    - The output should only include the entities, excluding any non-entity content such as descriptive text or inferences.
                    """
                
                distilled = document_distiller.distill(documents=[self.context], IE_query=IE_query, output_data_structure=DiseaseArticle)  # self.context not global
                # logging.info("Abstract distilled successfully.") #Log when its success
                return self._add_missing_entities(self._match_distilled_to_context(distilled))  # Call with self.

            except Exception as e:
                logging.error(f"Attempt {attempt + 1} failed: Error distilling abstract: {e}", exc_info=True)
                if attempt < MAX_RETRIES - 1:
                    logging.info(f"Retrying in {RETRY_DELAY} seconds...")
                    time.sleep(RETRY_DELAY)
                else:
                    logging.error("Max retries reached. Returning an empty dictionary.")
                    return {}  # Return an empty dictionary as a last resort

            # Should not reach here, but just in case
            logging.error("Unexpected exit from distillation loop. Returning an empty dictionary.")
        return {}
    

    def is_entity_in_context(self, entity, context): #Now a static
        """
        Checks if an entity exists verbatim (case-insensitive) in a given context.
        """
        if not isinstance(entity, str) or not isinstance(context, str):
            logging.warning(f"Invalid Input. It must be a string")
            return False # Return false
        entity_lower = entity.lower()
        context_lower = context.lower()
        return entity_lower in context_lower

    def _match_distilled_to_context(self, distilled):
        """Matches distilled information to the context."""
        distilled_ = {}
        if distilled is None or distilled == {} or not isinstance(distilled, dict):
            logging.info(f'distilled: {distilled}')
            return distilled_
        
        for key, value in distilled.items():
            if value and value != []:
                if isinstance(value, list):
                    distilled_[key] = []
                    for item in value:
                        for k, v in item.items():
                            if self.is_entity_in_context(v, self.context): #Now self.context
                                distilled_[key].append({key: v})
                else:
                    if self.is_entity_in_context(value, self.context): #Now self.context
                        distilled_[key] = value
        return distilled_ #Return the new value!

    def _add_missing_entities(self, abstract_distilled):
        """Compares abstract_distilled and pubtator_distilled and adds missing entities."""
        # miss_entities = []
        for entity_type, entities in self.pubtator_distilled.items(): #self.pubtator
            if entity_type not in abstract_distilled and entities != []:
                abstract_distilled[entity_type] = entities
                # miss_entities.append({entity_type: entities})
                # logging.info(f'Adding {entities} to the abstract')
            elif isinstance(entities, list):
                existing_names = set() #create all the strings with frozensets

                #Handle all the cases where they don't exist
                entity_type_key = entity_type[:-1] #Remove last character. Example diseases = disease

                if entity_type in abstract_distilled and isinstance(abstract_distilled[entity_type], list):
                    existing_names = {item.get(entity_type_key, '') for item in abstract_distilled[entity_type]} # Extract exisiting name.

                for entity in entities: # iterate through the existing ones in abstract_distilled, and append
                    entity_name = entity.get(entity_type, '') #Extract exisiting name
                    if entity_name and entity_name not in existing_names and len(entity_name)>0:
                        abstract_distilled[entity_type].append(entity) #Finally add the entity if the name doesn't match.
                        # miss_entities.append({entity_type: entity})
                        # logging.info(f'Adding {entity_name} to the abstract')

        return abstract_distilled #And return the new distilled version

    def _construct_properties_info(self):
        """Constructs a dictionary of properties information."""
        properties_info = {}
        for k, v in self.distilled.items(): #self.distilled now.
            if k not in ['disease', 'pathway', 'gene', 'metabolite', 'protein','processes', 'region','regulation','chemical'] and v != "" and v != None and v != []:
                if isinstance(v, list):
                    v_list = []
                    for item in v:
                        for _, value_ in item.items():
                            v_list.append(value_)
                    properties_info[k]=",".join([str(item) for item in v_list])
                else:
                    properties_info[k]=v
                    
        properties_info['source'] = f'PMID{self.PMID}' #Now self.variable
        if self.species_info != []: #Now self.property
            properties_info['species'] = ','.join(list(set(self.species_info))) #self.property
        
        return properties_info

    def _contstruct_block(self):
        """Constructs a block of text."""
        block = [
            f"{key} - {value}".replace("{", "[").replace("}", "]")
            for key, value in self.distilled.items() #self.variable
            if value !=[] and value != ""  and value is not None
            ]
        block.append(f'{self.context}') #self.variable
        return block
