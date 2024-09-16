from typing import List
from ..utils import LangchainOutputParser, RelationshipsExtractor, DataHandler

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
        
        self.data_handler = DataHandler()
    
    
    def extract_relations(self, context: str, entities: List[str], embeddings: bool = True, property_name = "properties", relation_name_key = "name"):
        """
        Extract relationships from a given context for specified entities and optionally add embeddings.
        
        Args:
        context (str): The textual context from which relationships will be extracted.
        entities (List[str]): A list of relation names to be considered in the extraction.
        embeddings (bool): A flag to determine whether to add embeddings to the extracted relationships.
        property_name (str): The property name under which embeddings will be stored in the relationship.
        relation_name_key (str): The key name for the relation's name in the relationship.
        
        Returns:
        List[dict]: A list of extracted relationships with optional embeddings.
        """
        formatted_context = f"context : \n -- '{context}' \n entities : \n -- {entities}"
        IE_query = '''
        # Directives
        - Extract relationships between the provided entities based on the context.
        - Adhere completely to the provided entities list. 
        - Do not add any entity outside the provided list. 
        '''
        
        relationships = self.langchain_output_parser.extract_information_as_json_for_context(output_data_structure = RelationshipsExtractor, context=formatted_context, IE_query=IE_query)
        print(relationships)
        
        if "relationships" not in relationships.keys() or relationships == None:
            print("we are retrying ....")
            self.extract_relations(context=context, entities=entities, embeddings=embeddings, property_name=property_name, relation_name_key=relation_name_key)
        if not entities:
            return []
        
        return self.data_handler.add_embeddings_as_property_batch(embeddings_function=lambda x:self.langchain_output_parser.calculate_embeddings(x), 
                                                                  items=relationships["relationships"],
                                                                  property_name=property_name,
                                                                  item_name_key=relation_name_key,
                                                                  embeddings=embeddings)    
    
    def extract_relations_for_isolated_entities(self, context: str, local_non_isolated_entities:List[str], isolated_entities:List[str], embeddings: bool = True, property_name = "properties", entity_name_key = "name"):
        """
        Extract and retry extraction for relations of isolated entities without initial relations.
        
        Args:
        context (str): The textual context from which relationships are re-extracted.
        isolated_entities (List[str]): A list of isolated entities to be re-examined.
        embeddings (bool): A flag to determine whether to add embeddings to the extracted relationships.
        property_name (str): The property name under which embeddings will be stored in the relationship.
        entity_name_name (str): The key name for the entity's name in the relationship.
        
        Returns:
        List[dict]: A list of re-extracted relationships with optional embeddings.
        """
        print("Some isolated entities without relations were detected ... trying to solve them!")
        formatted_context = f"context : \n -- '{context}'"
        
        IE_query = f'''
        # Directives
        The entities {isolated_entities} are not linked to the given entities {local_non_isolated_entities}. Extract any relation between these isolated entities and the given entities from the context.
        '''
        
        relationships = self.langchain_output_parser.extract_information_as_json_for_context(output_data_structure = RelationshipsExtractor, context=formatted_context, IE_query=IE_query)
        print(relationships)
        
        if "relationships" not in relationships.keys() or relationships == None:
            print("we are retrying ....")
            self.extract_relations_for_isolated_entities(context=context, isolated_entities=isolated_entities, embeddings=embeddings, property_name=property_name, entity_name_name=entity_name_key)


        return self.data_handler.add_embeddings_as_property_batch(embeddings_function=lambda x:self.langchain_output_parser.calculate_embeddings(x), 
                                                                  items=relationships["relationships"],
                                                                  property_name=property_name,
                                                                  item_name_key=entity_name_key,
                                                                  embeddings=embeddings)