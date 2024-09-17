import numpy as np
from typing import List, Tuple
from ..ientities_extraction import iEntitiesExtractor
from ..irelations_extraction import iRelationsExtractor
from ..utils import Matcher, DataHandler, LangchainOutputParser

class iText2KG:
    """
    A class designed to extract knowledge from text and structure it into a knowledge graph using
    entity and relationship extraction powered by language models.
    """
    def __init__(self, llm_model, embeddings_model, sleep_time:int=5) -> None:        
        """
        Initializes the iText2KG with specified language model, embeddings model, and operational parameters.
        
        Args:
        llm_model: The language model instance to be used for extracting entities and relationships from text.
        embeddings_model: The embeddings model instance to be used for creating vector representations of extracted entities.
        sleep_time (int): The time to wait (in seconds) when encountering rate limits or errors. Defaults to 5 seconds.
        """
        self.ientities_extractor =  iEntitiesExtractor(llm_model=llm_model, 
                                                       embeddings_model=embeddings_model,
                                                       sleep_time=sleep_time) 
        
        self.irelations_extractor = iRelationsExtractor(llm_model=llm_model, 
                                                        embeddings_model=embeddings_model,
                                                        sleep_time=sleep_time)

        self.data_handler = DataHandler()
        self.matcher = Matcher()
        self.langchain_output_parser = LangchainOutputParser(llm_model=llm_model, embeddings_model=embeddings_model)
        
        
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


    def build_graph(self, sections:List[str], existing_global_entities:List[dict]=None, existing_global_relationships:List[dict]=None, ent_threshold:float = 0.7, rel_threshold:float = 0.7):
        """
        Builds a knowledge graph from text by extracting entities and relationships and then integrating them into a structured graph. This is the main function of the iText2KG class.
        
        Args:
        sections (List[str]): A list of strings where each string represents a section of the document 
                              from which entities and relationships will be extracted.
        existing_global_entities (List[dict], optional): A list of existing global entities to match 
                                                         against the newly extracted entities. Each 
                                                         entity is represented as a dictionary.
        existing_global_relationships (List[dict], optional): A list of existing global relationships 
                                                               to match against the newly extracted 
                                                               relationships. Each relationship is 
                                                               represented as a dictionary.
        ent_threshold (float, optional): The threshold for entity matching, used to consolidate entities 
                                         from different sections. Default is 0.7.
        rel_threshold (float, optional): The threshold for relationship matching, used to consolidate 
                                         relationships from different sections. Default is 0.7.

        Returns:
        Tuple[List, List]: A tuple containing lists of consolidated entities and relationships.
        """
        print("[INFO] Extracting Entities from the Document", 1)
        global_entities = self.ientities_extractor.extract_entities(context=sections[0])
        print("[INFO] Extracting Relations from the Document", 1)
        global_relationships = self.irelations_extractor.extract_relations(context=sections[0], entities = list(map(lambda w:w["name"], global_entities)))
        
        isolated_entities = self.data_handler.find_isolated_entities(global_entities=global_entities, relations=global_relationships)
        while isolated_entities:
            print("[INFO] The isolated entities are ", isolated_entities)
            corrected_relations = self.irelations_extractor.extract_relations_for_isolated_entities(context=sections[0], isolated_entities=list(map(lambda w:w["name"],isolated_entities)), local_non_isolated_entities=list(map(lambda w:w["name"],global_entities)))
            matched_corrected_relationships, _ = self.matcher.process_lists(list1 = corrected_relations, list2=global_relationships, for_entity_or_relation="relation", threshold=rel_threshold)
            global_relationships.extend(matched_corrected_relationships)
            # Re-evaluate isolated entities after extending global_relationships
            isolated_entities = self.data_handler.find_isolated_entities(global_entities=global_entities, relations=global_relationships)
        
        global_relationships = self.data_handler.match_relations_with_isolated_entities(global_entities=global_entities, relations=global_relationships, matcher= lambda ent:self.matcher.find_match(ent, global_entities, match_type="entity", threshold=0.5), embedding_calculator= lambda ent:self.langchain_output_parser.calculate_embeddings(ent))
        
        if existing_global_entities and existing_global_relationships:
            print(f"[INFO] Matching the Document {1} Entities and Relationships with the Existing Global Entities/Relations")
            global_entities, global_relationships = self.matcher.match_entities_and_update_relationships(entities1=global_entities,
                                                                 entities2=existing_global_entities,
                                                                 relationships1=global_relationships,
                                                                 relationships2=existing_global_relationships,
                                                                 ent_threshold=ent_threshold,
                                                                 rel_threshold=rel_threshold)        
        
        assert global_relationships != None, print("Warning", global_relationships)
        for i in range(1, len(sections)):
            print("[INFO] Extracting Entities from the Document", i+1)
            entities = self.ientities_extractor.extract_entities(context= sections[i])
            
            processed_entities, global_entities = self.matcher.process_lists(list1 = entities, list2=global_entities, for_entity_or_relation="entity", threshold=ent_threshold)
            
            #relationships = relationship_extraction(context= sections[i], entities=list(map(lambda w:w["name"], processed_entities)))
            print("[INFO] Extracting Relations from the Document", i+1)
            relationships = self.irelations_extractor.extract_relations(context= sections, entities=list(map(lambda w:w["name"], processed_entities)))
            processed_relationships, _ = self.matcher.process_lists(list1 = relationships, list2=global_relationships, for_entity_or_relation="relation", threshold=rel_threshold)
            
            isolated_entities = self.data_handler.find_isolated_entities(global_entities=processed_entities, relations=processed_relationships)
            while isolated_entities:
                print("[INFO] The isolated entities are ", isolated_entities)
                corrected_relations = self.irelations_extractor.extract_relations_for_isolated_entities(context=sections[i], isolated_entities=list(map(lambda w:w["name"],isolated_entities)), local_non_isolated_entities=list(map(lambda w:w["name"],processed_entities)))
                matched_corrected_relationships, _ = self.matcher.process_lists(list1 = corrected_relations, list2=global_relationships, for_entity_or_relation="relation", threshold=rel_threshold)
                processed_relationships.extend(matched_corrected_relationships)
                isolated_entities = self.data_handler.find_isolated_entities(global_entities=processed_entities, relations=processed_relationships)
            
            processed_relationships = self.data_handler.match_relations_with_isolated_entities(global_entities=processed_entities, relations=processed_relationships, matcher= lambda ent:self.matcher.find_match(ent, processed_entities, match_type="entity", threshold=0.5), embedding_calculator= lambda ent:self.langchain_output_parser.calculate_embeddings(ent))

            global_relationships.extend(processed_relationships)
            
        return self.data_handler.handle_data(global_entities, data_type="entity"), self.data_handler.handle_data(global_relationships, data_type="relation")