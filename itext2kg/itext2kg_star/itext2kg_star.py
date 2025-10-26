from typing import List
from itext2kg.itext2kg_star.irelations_extraction import SimpleDirectiRelationsExtractor
from itext2kg.itext2kg_star.graph_matching import Matcher
from itext2kg.llm_output_parsing.langchain_output_parser import LangchainOutputParser
from itext2kg.itext2kg_star.models import KnowledgeGraph
from itext2kg.logging_config import get_logger

logger = get_logger(__name__)

class iText2KG_Star:
    """
    A simplified class of iText2KG designed to extract knowledge from text by directly extracting relationships
    from context and deriving entities from those relationships. This approach is more efficient
    as it skips separate entity extraction and isolated entity handling.
    """
    def __init__(self, llm_model, embeddings_model, sleep_time:int=5) -> None:        
        """
        Initializes the SimpleiText2KG with specified language model and embeddings model.
        
        Args:
        llm_model: The language model instance to be used for extracting relationships directly from text.
        embeddings_model: The embeddings model instance to be used for creating vector representations.
        sleep_time (int): The time to wait (in seconds) when encountering rate limits or errors. Defaults to 5 seconds.
        """
        self.simple_irelations_extractor = SimpleDirectiRelationsExtractor(llm_model=llm_model, 
                                                                           embeddings_model=embeddings_model,
                                                                           sleep_time=sleep_time)

        self.matcher = Matcher()
        self.langchain_output_parser = LangchainOutputParser(llm_model=llm_model, embeddings_model=embeddings_model)


    async def build_graph(self, 
                    sections:List[str], 
                    existing_knowledge_graph:KnowledgeGraph=None, 
                    ent_threshold:float = 0.7, 
                    rel_threshold:float = 0.7, 
                    max_tries:int=5, 
                    entity_name_weight:float=0.6,
                    entity_label_weight:float=0.4,
                    observation_date:str=""
                    ) -> KnowledgeGraph:
        """
        Builds a knowledge graph from text by directly extracting relationships and deriving entities
        from those relationships. This simplified approach is more efficient and doesn't require
        separate entity extraction or isolated entity handling.

        Args:
        sections (List[str]): A list of strings where each string represents a section of the document.
        existing_knowledge_graph (KnowledgeGraph, optional): An existing knowledge graph to merge with. Default is None.
        ent_threshold (float, optional): The threshold for entity matching when merging sections. Default is 0.7.
        rel_threshold (float, optional): The threshold for relationship matching when merging sections. Default is 0.7.
        max_tries (int, optional): The maximum number of attempts to extract relationships. Defaults to 5.
        entity_name_weight (float): The weight of the entity name in matching. Default is 0.6.
        entity_label_weight (float): The weight of the entity label in matching. Default is 0.4.
        observation_date (str): Observation date to add to relationships. Defaults to "".

        Returns:
        KnowledgeGraph: A constructed knowledge graph with entities derived from relationships.
        """
        logger.info("------- Extracting Relations and Deriving Entities from Document %d", 1)
        global_entities, global_relationships = await self.simple_irelations_extractor.extract_relations_and_derive_entities(
            context=sections[0],
            max_tries=max_tries,
            entity_name_weight=entity_name_weight,
            entity_label_weight=entity_label_weight,
            observation_date=observation_date
        )
        
        for i in range(1, len(sections)):
            logger.info("------- Extracting Relations and Deriving Entities from Document %d", i+1)
            entities, relationships = await self.simple_irelations_extractor.extract_relations_and_derive_entities(
                context=sections[i],
                max_tries=max_tries,
                entity_name_weight=entity_name_weight,
                entity_label_weight=entity_label_weight,
                observation_date=observation_date
            )
            
            global_entities, global_relationships = self.matcher.match_entities_and_update_relationships(
                entities1=entities,
                entities2=global_entities,
                relationships1=relationships,
                relationships2=global_relationships,
                ent_threshold=ent_threshold,
                rel_threshold=rel_threshold
            )
        
        if existing_knowledge_graph:
            logger.info("------- Matching with Existing Knowledge Graph")
            global_entities, global_relationships = self.matcher.match_entities_and_update_relationships(
                entities1=global_entities,
                entities2=existing_knowledge_graph.entities,
                relationships1=global_relationships,
                relationships2=existing_knowledge_graph.relationships,
                ent_threshold=ent_threshold,
                rel_threshold=rel_threshold
            )    
        
        constructed_kg = KnowledgeGraph(entities=global_entities, relationships=global_relationships)
        constructed_kg.remove_duplicates_entities()
        constructed_kg.remove_duplicates_relationships()
         
        return constructed_kg