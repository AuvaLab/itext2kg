from typing import List
from itext2kg.llm_output_parsing.langchain_output_parser import LangchainOutputParser
from itext2kg.itext2kg_star.models import Entity, Relationship, KnowledgeGraph
from itext2kg.itext2kg_star.graph_matching.matcher import Matcher
from itext2kg.itext2kg_star.models.schemas import RelationshipsExtractor
from itext2kg.logging_config import get_logger

logger = get_logger(__name__)

class iRelationsExtractor:
    """
    A class to extract relationships between entities
    """
    def __init__(self, llm_model, embeddings_model, sleep_time:int=5) -> None:        
        """
        Initializes the iRelationsExtractor with specified language model, embeddings model, and operational parameters.
        
        Args:
        llm_model: The language model instance used for extracting relationships between entities.
        embeddings_model: The embeddings model instance used for generating vector representations of entities and relationships.
        sleep_time (int): The time to wait (in seconds) when encountering rate limits or errors. Defaults to 5 seconds.
        """
        self.langchain_output_parser =  LangchainOutputParser(llm_model=llm_model,
                                                              embeddings_model=embeddings_model,
                                                       sleep_time=sleep_time)
        self.matcher = Matcher()
    
    
    async def extract_relations(self, 
                          context: str, 
                          entities: List[Entity], 
                          isolated_entities_without_relations: List[Entity] = None,
                          max_tries:int=5,
                          entity_name_weight:float=0.6,
                          entity_label_weight:float=0.4,
                          ) -> List[Relationship]:
        """
        Extract relationships from a given context for specified entities and add embeddings. This method handles the invented entities.
        
        Args:
            context (str): The textual context from which relationships will be extracted.
            entities (List[Entity]): A list of Entity instances to be considered in the extraction.
            isolated_entities_without_relations (List[Entity], optional): A list of entities without existing relationships to include in the extraction. Defaults to None.
            max_tries (int): The maximum number of attempts to extract relationships. Defaults to 5.
            entity_name_weight (float): The weight of the entity name, set to 0.6, indicating its
                                     relative importance in the overall evaluation process.
            entity_label_weight (float): The weight of the entity label, set to 0.4, reflecting its
                                      secondary significance in the evaluation process.
        
        Returns:
            List[Relationship]: A list of extracted Relationship instances with embeddings.
        
        Raises:
            ValueError: If relationship extraction fails after multiple attempts.
        """
        # we would not give the LLM complex data structure as context to avoid the hallucination as much as possible
        entities_simplified = [f"{entity.name}:{entity.label}" for entity in entities]
        formatted_context = f"# context : --\n'{context}' \n# entities(adhere to it) :-- \n {entities_simplified}"
        IE_query = '''# DIRECTIVES : 
                        - Extract relationships between the provided entities based on the context.
                        - Adhere completely to the provided entities list.
                        - Do not change the name or label of the provided entities list.
                        - Do not add any entity outside the provided list.
                        - Avoid reflexive relations.
                        '''
                        
        if isolated_entities_without_relations:
            isolated_entities_without_relations_simplified = [(entity.name, entity.label) for entity in isolated_entities_without_relations]
            formatted_context = f"context :--\n'{context}'"
            IE_query = f'''
                    # Directives
                    - Based on the provided context, link the entities: \n {isolated_entities_without_relations_simplified} \n to the following entities: \n {entities_simplified}.
                    - Avoid reflexive relations.
                    '''
        tries = 0
        relationships = None
        curated_relationships:List[Relationship]= []
        
        while tries < max_tries:
            try:
                relationships_list = await self.langchain_output_parser.extract_information_as_json_for_context(
                    contexts=[formatted_context], output_data_structure=RelationshipsExtractor,
                    system_query=IE_query
                )

                if relationships_list and len(relationships_list) > 0:
                    relationships = relationships_list[0]  # Get the first (and only) result
                    if hasattr(relationships, 'relationships'):
                        break
                
            except Exception as e:
                logger.warning("Not Formatted in the desired format. Error occurred: %s. Retrying... (Attempt %d/%d)", e, tries + 1, max_tries)

            tries += 1
    
        if not relationships or not hasattr(relationships, 'relationships'):
            raise ValueError("Failed to extract relationships after multiple attempts.")
        logger.debug("Extracted relationships: %s", relationships)
        kg_llm_output = KnowledgeGraph(relationships=[], entities = entities)
        
        # -------- Verification of invented entities and matching to the closest ones from the input entities-------- #
        logger.info("Verification of invented entities")
        for relationship in relationships.relationships:
            startEntity = Entity(label=relationship.startNode.label, name=relationship.startNode.name)
            endEntity = Entity(label=relationship.endNode.label, name=relationship.endNode.name)
            
            startEntity.process()
            endEntity.process()
            
            startEntity_in_input_entities = kg_llm_output.get_entity(startEntity)
            endEntity_in_input_entities = kg_llm_output.get_entity(endEntity)
            
            if startEntity_in_input_entities is not None and endEntity_in_input_entities is not None :
                curated_relationships.append(Relationship(startEntity= startEntity_in_input_entities, 
                                      endEntity = endEntity_in_input_entities,
                                      name = relationship.name))
                
            elif startEntity_in_input_entities is None and endEntity_in_input_entities is None:
                logger.info("[INVENTED ENTITIES] The entities %s and %s are invented. Solving them ...", startEntity, endEntity)
                await startEntity.embed_Entity(embeddings_function=self.langchain_output_parser.calculate_embeddings, 
                                               entity_label_weight=entity_label_weight, 
                                               entity_name_weight=entity_name_weight)
                await endEntity.embed_Entity(embeddings_function=self.langchain_output_parser.calculate_embeddings,
                                             entity_label_weight=entity_label_weight,
                                             entity_name_weight=entity_name_weight)
                
                startEntity = self.matcher.find_match(obj1=startEntity, list_objects=entities, threshold=0.5)
                endEntity = self.matcher.find_match(obj1=endEntity, list_objects=entities, threshold=0.5)
                
                curated_relationships.append(Relationship(startEntity= startEntity, 
                                      endEntity = endEntity,
                                      name = relationship.name))
                
            elif startEntity_in_input_entities is None:
                logger.info("[INVENTED ENTITIES] The entity %s is invented. Solving it ...", startEntity)
                await startEntity.embed_Entity(embeddings_function=self.langchain_output_parser.calculate_embeddings,
                                               entity_label_weight=entity_label_weight,
                                               entity_name_weight=entity_name_weight)
                startEntity = self.matcher.find_match(obj1=startEntity, list_objects=entities, threshold=0.5)
                
                curated_relationships.append(Relationship(startEntity= startEntity, 
                                      endEntity = endEntity,
                                      name = relationship.name))
                
            elif endEntity_in_input_entities is None:
                logger.info("[INVENTED ENTITIES] The entity %s is invented. Solving it ...", endEntity)
                await endEntity.embed_Entity(embeddings_function=self.langchain_output_parser.calculate_embeddings,
                                             entity_label_weight=entity_label_weight,
                                             entity_name_weight=entity_name_weight)
                endEntity = self.matcher.find_match(obj1=endEntity, list_objects=entities, threshold=0.5)
                
                curated_relationships.append(Relationship(startEntity= startEntity, 
                                      endEntity = endEntity,
                                      name = relationship.name))
        
        kg = KnowledgeGraph(relationships = curated_relationships, entities=entities)
        await kg.embed_relationships(
            embeddings_function=lambda x: self.langchain_output_parser.calculate_embeddings(x)
            )
        return kg.relationships
    
    
    async def extract_verify_and_correct_relations(self,
                          context: str, 
                          entities: List[Entity],
                          rel_threshold:float = 0.7,
                          max_tries:int=5,
                          max_tries_isolated_entities:int=3,
                          entity_name_weight:float=0.6,
                          entity_label_weight:float=0.4,
                          observation_date:str="") -> List[Relationship]:
        """
        Extract, verify, and correct relationships between entities in the given context.

        Args:
            context (str): The textual context for extracting relationships.
            entities (List[Entity]): A list of Entity instances to consider.
            rel_threshold (float): The threshold for matching corrected relationships. Defaults to 0.7.
            max_tries (int): The maximum number of attempts to extract relationships. Defaults to 5.
            max_tries_isolated_entities (int): The maximum number of attempts to process isolated entities. Defaults to 3.
            entity_name_weight (float): The weight of the entity name, set to 0.6, indicating its
                                     relative importance in the overall evaluation process.
            entity_label_weight (float): The weight of the entity label, set to 0.4, reflecting its
                                      secondary significance in the evaluation process.
        
        Returns:
            List[Relationship]: A list of curated Relationship instances after verification and correction.
        """
        tries = 0
        isolated_entities_without_relations:List[Entity]= []
        curated_relationships = await self.extract_relations(context=context,
                                                             entities=entities,
                                                             max_tries=max_tries,
                                                             entity_name_weight=entity_name_weight,
                                                             entity_label_weight=entity_label_weight)
        
        # -------- Verification of isolated entities without relations and re-prompting the LLM accordingly-------- #   
        isolated_entities_without_relations = KnowledgeGraph(entities=entities, 
                                                             relationships=curated_relationships).find_isolated_entities()
        
        while tries < max_tries_isolated_entities and isolated_entities_without_relations:
            logger.info("[ISOLATED ENTITIES][TRY-%d] There are some isolated entities without relations %s. Solving them ...", tries+1, isolated_entities_without_relations)  
            corrected_relationships = await self.extract_relations(context = context, 
                                                                   entities=isolated_entities_without_relations,
                                                                   isolated_entities_without_relations=isolated_entities_without_relations,
                                                                   entity_name_weight=entity_name_weight,
                                                                   entity_label_weight=entity_label_weight)
            matched_corrected_relationships, _ = self.matcher.process_lists(list1 = corrected_relationships, list2=curated_relationships, threshold=rel_threshold)
            curated_relationships.extend(matched_corrected_relationships)
                
            isolated_entities_without_relations = KnowledgeGraph(entities=entities, relationships=corrected_relationships).find_isolated_entities()
            tries += 1
            
        temp_kg = KnowledgeGraph(relationships=curated_relationships)
        temp_kg.add_observation_dates(observation_date=observation_date)
        return temp_kg.relationships