from typing import List
from itext2kg.llm_output_parsing.langchain_output_parser import LangchainOutputParser
from itext2kg.itext2kg_star.models import Entity, Relationship, KnowledgeGraph
from itext2kg.itext2kg_star.graph_matching.matcher import Matcher
from itext2kg.itext2kg_star.models.schemas import RelationshipsExtractor
from itext2kg.logging_config import get_logger

logger = get_logger(__name__)

class SimpleDirectiRelationsExtractor:
    """
    A class to extract relationships directly from context and derive entities from those relationships
    """
    def __init__(self, llm_model, embeddings_model, sleep_time:int=5) -> None:        
        """
        Initializes the SimpleDirectiRelationsExtractor with specified language model, embeddings model, and operational parameters.
        
        Args:
        llm_model: The language model instance used for extracting relationships directly from context.
        embeddings_model: The embeddings model instance used for generating vector representations of entities and relationships.
        sleep_time (int): The time to wait (in seconds) when encountering rate limits or errors. Defaults to 5 seconds.
        """
        self.langchain_output_parser =  LangchainOutputParser(llm_model=llm_model,
                                                              embeddings_model=embeddings_model,
                                                       sleep_time=sleep_time)
        self.matcher = Matcher()
    
    
    async def extract_relations_and_derive_entities(self, 
                          context: str, 
                          max_tries:int=5,
                          entity_name_weight:float=0.6,
                          entity_label_weight:float=0.4,
                          observation_date:str=""
                          ) -> tuple[List[Entity], List[Relationship]]:
        """
        Extract relationships directly from context and derive entities from those relationships.
        
        Args:
            context (str): The textual context from which relationships will be extracted.
            max_tries (int): The maximum number of attempts to extract relationships. Defaults to 5.
            entity_name_weight (float): The weight of the entity name, set to 0.6, indicating its
                                     relative importance in the overall evaluation process.
            entity_label_weight (float): The weight of the entity label, set to 0.4, reflecting its
                                      secondary significance in the evaluation process.
            observation_date (str): Observation date to add to relationships. Defaults to "".
        
        Returns:
            tuple[List[Entity], List[Relationship]]: A tuple containing derived entities and extracted relationships with embeddings.
        
        Raises:
            ValueError: If relationship extraction fails after multiple attempts.
        """
        formatted_context = f"# context : --\n'{context}'"
        IE_query = '''# DIRECTIVES : 
                        - Extract all meaningful relationships directly from the provided context.
                        - For each relationship, identify the start entity and end entity involved.
                        - Each entity should have a clear name and appropriate label/type.
                        - Avoid reflexive relations (entity relating to itself).
                        '''
                        
        tries = 0
        relationships = None
        
        while tries < max_tries:
            try:
                relationships_list = await self.langchain_output_parser.extract_information_as_json_for_context(
                    contexts=[formatted_context], output_data_structure=RelationshipsExtractor,
                    system_query=IE_query
                )

                if relationships_list and len(relationships_list) > 0:
                    relationships = relationships_list[0]  # Get the first (and only) result
                    if hasattr(relationships, 'relationships') and relationships.relationships:
                        break
                
            except Exception as e:
                logger.warning("Not Formatted in the desired format. Error occurred: %s. Retrying... (Attempt %d/%d)", e, tries + 1, max_tries)

            tries += 1
    
        if not relationships or not hasattr(relationships, 'relationships') or not relationships.relationships:
            raise ValueError("Failed to extract relationships after multiple attempts.")
        
        logger.debug("Extracted relationships: %s", relationships)
        
        entities = []
        converted_relationships = []
        
        for relationship in relationships.relationships:
            start_entity = Entity(label=relationship.startNode.label, name=relationship.startNode.name)
            start_entity.process()
            end_entity = Entity(label=relationship.endNode.label, name=relationship.endNode.name)
            end_entity.process()
            entities.append(start_entity)
            entities.append(end_entity)
            
            # Convert schema relationship to knowledge graph relationship
            kg_relationship = Relationship(
                startEntity=start_entity,
                endEntity=end_entity,
                name=relationship.name
            )
            converted_relationships.append(kg_relationship)
        
        kg = KnowledgeGraph(entities=entities, relationships=converted_relationships)
        kg.remove_duplicates_entities()
        await kg.embed_entities(
            entity_name_weight=entity_name_weight,
            entity_label_weight=entity_label_weight,
            embeddings_function=lambda x: self.langchain_output_parser.calculate_embeddings(x)
        )
        await kg.embed_relationships(
            embeddings_function=lambda x: self.langchain_output_parser.calculate_embeddings(x)
        )
        
        if observation_date:
            kg.add_observation_dates(observation_date=observation_date)
        
        return kg.entities, kg.relationships 