from typing import List
from ..utils import LangchainOutputParser, RelationshipsExtractor
from ..utils import Matcher

class iRelationsExtractor():
    def __init__(self, openai_api_key:str, embeddings_model_name :str = "text-embedding-3-large", model_name:str = "gpt-4-turbo", temperature:float = 0, sleep_time:int=5) -> None:        
        self.langchain_output_parser =  LangchainOutputParser(openai_api_key=openai_api_key,
                                                       embeddings_model_name=embeddings_model_name,
                                                       model_name=model_name,
                                                       temperature=temperature,
                                                       sleep_time=sleep_time)  
        
    
    
    def __add_embeddings_as_property(self, entity:dict, property_name = "properties", embeddings_name = "embeddings", entity_name_name = "name", embeddings:bool = True):
        entity = entity.copy()
        
        entity[entity_name_name] = entity[entity_name_name].lower().replace("_", " ").replace("-", " ")
        entity[property_name] = {}
        
        if embeddings:
            entity[property_name][embeddings_name] = self.langchain_output_parser.calculate_embeddings(text=entity[entity_name_name])
            
        return entity
 
    
    
    def extract_relationships(self, context: str, keywords: List[str], embeddings: bool = True, property_name = "properties", entity_name_name = "name"):
        formatted_context = f"context : \n -- '{context}' \n keywords : \n -- {keywords}"
        IE_query = '''
        # Directives
        - Adhere completely to the provided entities list. 
        - Do not add any entity outside the provided list. 
        - Extract ONE predicate per subject and object.
        - ALL entities in the provided list should have a relation.
        '''
        
        relationships = self.langchain_output_parser.extract_information_as_json_for_context(output_data_structure = RelationshipsExtractor, context=formatted_context, IE_query=IE_query)
        print(relationships)
        
        if "relationships" not in relationships.keys() or relationships == None:
            print("we are retrying ....")
            self.extract_relationships(context=context, keywords=keywords, embeddings=embeddings, property_name=property_name, entity_name_name=entity_name_name)
        if not keywords:
            return []

        return list(map(lambda rel : self.__add_embeddings_as_property(entity = rel, embeddings=embeddings, property_name=property_name, entity_name_name=entity_name_name) , relationships["relationships"]))

    
    def relations_extraction_for_all_sections(self, sections:List[str], keywords, rel_threshold = 0.8):
        matcher = Matcher(threshold = rel_threshold)      
        print("[INFO] Extracting Entities from the Document ", 1)
        
        global_relationships = self.extract_relationships(context=sections[0], keywords = keywords)
        
        for i in range(1, len(sections)):
            print("[INFO] Extracting Entities from the Document ", i+1)
            entities = self.extract_relationships(context= sections[i], keywords=keywords)
            processed_relationships, global_relationships_ = matcher.process_lists(L1 = entities, L2=global_relationships, for_entity_or_relation="relation", threshold = rel_threshold)
            
            global_relationships.extend(processed_relationships)
        return global_relationships

        
    