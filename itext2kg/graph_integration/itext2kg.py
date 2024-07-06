import numpy as np
from typing import List
from ..ientities_extraction import iEntitiesExtractor
from ..irelation_extraction import iRelationsExtractor
from ..utils import Matcher, DataHandler

class iText2KG:
    def __init__(self, openai_api_key:str, embeddings_model_name :str = "text-embedding-3-large", model_name:str = "gpt-4-turbo", temperature:float = 0, sleep_time:int=5) -> None:        
        self.ientities_extractor =  iEntitiesExtractor(openai_api_key=openai_api_key,
                                                       embeddings_model_name=embeddings_model_name,
                                                       model_name=model_name,
                                                       temperature=temperature,
                                                       sleep_time=sleep_time) 
        
        self.irelations_extractor = iRelationsExtractor(openai_api_key=openai_api_key,
                                                       embeddings_model_name=embeddings_model_name,
                                                       model_name=model_name,
                                                       temperature=temperature,
                                                       sleep_time=sleep_time)

        self.data_handler = DataHandler()
        self.matcher = Matcher()
        
        
    def extract_entities_for_all_sections(self, sections:List[str], ent_threshold = 0.8):
              
        print("[INFO] Extracting Entities from the Document", 1)
        global_entities = self.ientities_extractor.extract_entities(context=sections[0])
        
        for i in range(1, len(sections)):
            print("[INFO] Extracting Entities from the Document", i+1)
            entities = self.ientities_extractor.extract_entities(context= sections[i])
            processed_entities, global_entities =  self.matcher.process_lists(list1 = entities, list2=global_entities, for_entity_or_relation="entity", threshold=ent_threshold)
        return global_entities

        
    def extract_relations_for_all_sections(self, sections:List[str], entities, rel_threshold = 0.8):
        print("[INFO] Extracting Entities from the Document", 1)
        
        global_relationships = self.irelations_extractor.extract_relations(context=sections[0], entities = entities)
        
        for i in range(1, len(sections)):
            print("[INFO] Extracting Entities from the Document", i+1)
            entities = self.irelations_extractor.extract_relations(context= sections[i], entities=entities)
            processed_relationships, global_relationships_ = self.matcher.process_lists(list1 = entities, list2=global_relationships, for_entity_or_relation="relation", threshold = rel_threshold)
            
            global_relationships.extend(processed_relationships)
        return global_relationships


    def build_graph(self, sections:List[str], ent_threshold:float = 0.7, rel_threshold:float = 0.7):
        print("[INFO] Extracting Entities from the Document", 1)
        global_entities = self.ientities_extractor.extract_entities(context=sections[0])
        print("[INFO] Extracting Relations from the Document", 1)
        global_relationships = self.irelations_extractor.extract_relations(context=sections[0], entities = list(map(lambda w:w["name"], global_entities)))
        
        
        for i in range(1, len(sections)):
            print("[INFO] Extracting Entities from the Document", i+1)
            entities = self.ientities_extractor.extract_entities(context= sections[i])
            
            processed_entities, global_entities = self.matcher.process_lists(list1 = entities, list2=global_entities, for_entity_or_relation="entity", threshold=ent_threshold)
            
            #relationships = relationship_extraction(context= sections[i], entities=list(map(lambda w:w["name"], processed_entities)))
            print("[INFO] Extracting Relations from the Document", i+1)
            relationships = self.irelations_extractor.extract_relations(context= sections, entities=list(map(lambda w:w["name"], processed_entities)))
            processed_relationships, global_relationships = self.matcher.process_lists(list1 = relationships, list2=global_relationships, for_entity_or_relation="relation", threshold=rel_threshold)
            
            #global_relationships.extend(processed_relationships)
            
        return self.data_handler.handle_data(global_entities, data_type="entity"), self.data_handler.handle_data(global_relationships, data_type="relation")
    