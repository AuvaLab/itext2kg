import numpy as np
from typing import List
from ..ientities_extraction import iEntitiesExtractor
from ..irelation_extraction import iRelationsExtractor
from ..utils import Matcher, DataHandler, LangchainOutputParser

class iText2KG:
    """
    A class designed to extract knowledge from text and structure it into a knowledge graph using
    entity and relationship extraction powered by language models.
    """
    def __init__(self, openai_api_key:str, embeddings_model_name :str = "text-embedding-3-large", model_name:str = "gpt-4-turbo", temperature:float = 0, sleep_time:int=5) -> None:        
        """
        Initializes the iText2KG with specified API key, models, and operational parameters.
        
        Args:
        openai_api_key (str): The API key for accessing OpenAI services.
        embeddings_model_name (str): The model name for text embeddings.
        model_name (str): The model name for the Chat API.
        temperature (float): The temperature setting for the Chat API's responses.
        sleep_time (int): The time to wait (in seconds) when encountering rate limits or errors.
        """
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
        self.langchain_output_parser = LangchainOutputParser(openai_api_key=openai_api_key,)
        
        
    def extract_entities_for_all_sections(self, sections:List[str], ent_threshold = 0.8):
        """
        Extracts and consolidates entities from all sections of a document.
        
        Args:
        sections (List[str]): A list of text sections from a document.
        ent_threshold (float): The threshold for considering two entities as identical.

        Returns:
        A list of consolidated entities.
        """
        print("[INFO] Extracting Entities from the Document", 1)
        global_entities = self.ientities_extractor.extract_entities(context=sections[0])
        
        for i in range(1, len(sections)):
            print("[INFO] Extracting Entities from the Document", i+1)
            entities = self.ientities_extractor.extract_entities(context= sections[i])
            processed_entities, global_entities =  self.matcher.process_lists(list1 = entities, list2=global_entities, for_entity_or_relation="entity", threshold=ent_threshold)
        return self.data_handler.handle_data(global_entities, data_type="entity")

        
    def extract_relations_for_all_sections(self, sections:List[str], entities, rel_threshold = 0.8):
        """
        Extracts and consolidates relationships from all sections of a document, considering isolated entities.
        
        Args:
        sections (List[str]): A list of text sections from a document.
        entities (List): A list of entities around which relationships are to be formed.
        rel_threshold (float): The threshold for considering two relationships as identical.

        Returns:
        A list of consolidated relationships.
        """
        entities_names = list(map(lambda entity:entity["name"], entities.copy()))
        print("[INFO] Extracting Relations from the Document", 1)
        
        global_relationships = self.irelations_extractor.extract_relations(context=sections[0], entities = entities_names)
        
        isolated_entities = self.data_handler.find_isolated_entities(global_entities=entities, relations=global_relationships)
        if isolated_entities:
            corrected_relations = self.irelations_extractor.extract_relations_for_isolated_entities(context=sections[0], isolated_entities=isolated_entities)
            global_relationships.extend(corrected_relations)
        
        global_relationships = self.data_handler.match_relations_with_isolated_entities(global_entities=entities, relations=global_relationships, matcher= lambda ent:self.matcher.find_match(ent, entities, match_type="entity", threshold=0.5), embedding_calculator= lambda ent:self.langchain_output_parser.calculate_embeddings(ent))

        
        for i in range(1, len(sections)):
            print("[INFO] Extracting Relations from the Document", i+1)
            entities = self.irelations_extractor.extract_relations(context= sections[i], entities=entities_names)
            processed_relationships, global_relationships_ = self.matcher.process_lists(list1 = entities, list2=global_relationships, for_entity_or_relation="relation", threshold = rel_threshold)
                        
            isolated_entities = self.data_handler.find_isolated_entities(global_entities=entities, relations=processed_relationships)
            if isolated_entities:
                corrected_relations = self.irelations_extractor.extract_relations_for_isolated_entities(context=sections[i], isolated_entities=isolated_entities)
                processed_relationships.extend(corrected_relations)
            
            processed_relationships = self.data_handler.match_relations_with_isolated_entities(global_entities=entities, relations=processed_relationships, matcher= lambda ent:self.matcher.find_match(ent, entities, match_type="entity", threshold=0.5), embedding_calculator= lambda ent:self.langchain_output_parser.calculate_embeddings(ent))
            
            global_relationships.extend(processed_relationships)
        #return self.data_handler.handle_data(global_relationships, data_type="relation")
        return global_relationships


    def build_graph(self, sections:List[str], ent_threshold:float = 0.7, rel_threshold:float = 0.7):
        """
        Builds a knowledge graph from text by extracting entities and relationships and then integrating them into a structured graph. This is the main function of the iText2KG class.
        
        Args:
        sections (List[str]): The sections of the document from which to extract information.
        ent_threshold (float): Entity match threshold for consolidating entities.
        rel_threshold (float): Relationship match threshold for consolidating relationships.

        Returns:
        Tuple[List, List]: A tuple containing lists of consolidated entities and relationships.
        """
        print("[INFO] Extracting Entities from the Document", 1)
        global_entities = self.ientities_extractor.extract_entities(context=sections[0])
        print("[INFO] Extracting Relations from the Document", 1)
        global_relationships = self.irelations_extractor.extract_relations(context=sections[0], entities = list(map(lambda w:w["name"], global_entities)))
        
        isolated_entities = self.data_handler.find_isolated_entities(global_entities=global_entities, relations=global_relationships)
        if isolated_entities:
            corrected_relations = self.irelations_extractor.extract_relations_for_isolated_entities(context=sections[0], isolated_entities=isolated_entities)
            global_relationships.extend(corrected_relations)
        
        global_relationships = self.data_handler.match_relations_with_isolated_entities(global_entities=global_entities, relations=global_relationships, matcher= lambda ent:self.matcher.find_match(ent, global_entities, match_type="entity", threshold=0.5), embedding_calculator= lambda ent:self.langchain_output_parser.calculate_embeddings(ent))
        
        for i in range(1, len(sections)):
            print("[INFO] Extracting Entities from the Document", i+1)
            entities = self.ientities_extractor.extract_entities(context= sections[i])
            
            processed_entities, global_entities = self.matcher.process_lists(list1 = entities, list2=global_entities, for_entity_or_relation="entity", threshold=ent_threshold)
            
            #relationships = relationship_extraction(context= sections[i], entities=list(map(lambda w:w["name"], processed_entities)))
            print("[INFO] Extracting Relations from the Document", i+1)
            relationships = self.irelations_extractor.extract_relations(context= sections, entities=list(map(lambda w:w["name"], processed_entities)))
            processed_relationships, _ = self.matcher.process_lists(list1 = relationships, list2=global_relationships, for_entity_or_relation="relation", threshold=rel_threshold)
            
            isolated_entities = self.data_handler.find_isolated_entities(global_entities=processed_entities, relations=processed_relationships)
            if isolated_entities:
                corrected_relations = self.irelations_extractor.extract_relations_for_isolated_entities(context=sections[i], isolated_entities=isolated_entities)
                processed_relationships.extend(corrected_relations)
            
            processed_relationships = self.data_handler.match_relations_with_isolated_entities(global_entities=processed_entities, relations=processed_relationships, matcher= lambda ent:self.matcher.find_match(ent, processed_entities, match_type="entity", threshold=0.5), embedding_calculator= lambda ent:self.langchain_output_parser.calculate_embeddings(ent))

            global_relationships.extend(processed_relationships)
            
        return self.data_handler.handle_data(global_entities, data_type="entity"), self.data_handler.handle_data(global_relationships, data_type="relation")
    