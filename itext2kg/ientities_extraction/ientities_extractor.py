from typing import List
from ..utils import LangchainOutputParser, EntitiesExtractor
from ..utils import Matcher

class iEntitiesExtractor():
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
 
    
    
    def extract_entities(self, context: str, embeddings: bool = True, property_name = "properties", entity_name_name = "name"):
        
        entities = self.langchain_output_parser.extract_information_as_json_for_context(context=context, output_data_structure=EntitiesExtractor)
        print(entities)
        
        
        if "entities" not in entities.keys() or entities == None:
            print("Not formatted in the desired format, we are retrying ....")
            self.extract_entities(context=context, entities=entities, embeddings=embeddings, property_name=property_name, entity_name_name=entity_name_name)
        
        
        return list(map(lambda word : self.__add_embeddings_as_property(entity = word, embeddings=embeddings, property_name=property_name, entity_name_name=entity_name_name) , entities["entities"]))
    
    def extract_entities_for_all_sections(self, sections:List[str], ent_threshold = 0.8):
        matcher = Matcher(threshold = ent_threshold)      
        print("[INFO] Extracting Entities from the Document ", 1)
        global_entities = self.extract_entities(context=sections[0])
        
        for i in range(1, len(sections)):
            print("[INFO] Extracting Entities from the Document ", i+1)
            entities = self.extract_entities(context= sections[i])
            processed_entities, global_entities =  matcher.process_lists(list1 = entities, list2=global_entities, for_entity_or_relation="entity")
        return global_entities

        
    