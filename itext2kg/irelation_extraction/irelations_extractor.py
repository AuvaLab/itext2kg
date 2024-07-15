from typing import List
from ..utils import LangchainOutputParser, RelationshipsExtractor

class iRelationsExtractor:
    """
    A class to extract relationships between entities
    """
    def __init__(self, openai_api_key:str, embeddings_model_name :str = "text-embedding-3-large", model_name:str = "gpt-4-turbo", temperature:float = 0, sleep_time:int=5) -> None:        
        """
        Initializes the iRelationsExtractor with specified API key, models, and operational parameters.
        
        Args:
        openai_api_key (str): The API key for accessing OpenAI services.
        embeddings_model_name (str): The model name for text embeddings.
        model_name (str): The model name for the Chat API.
        temperature (float): The temperature setting for the Chat API's responses.
        sleep_time (int): The time to wait (in seconds) when encountering rate limits or errors.
        """
        self.langchain_output_parser =  LangchainOutputParser(openai_api_key=openai_api_key,
                                                       embeddings_model_name=embeddings_model_name,
                                                       model_name=model_name,
                                                       temperature=temperature,
                                                       sleep_time=sleep_time)  
        
    
    
    def __add_embeddings_as_property(self, entity:dict, property_name = "properties", embeddings_name = "embeddings", entity_name_name = "name", embeddings:bool = True):
        """
        Add embeddings as a property to the given entity dictionary.
        
        Args:
        entity (dict): The entity to which embeddings will be added.
        property_name (str): The key under which embeddings will be stored.
        embeddings_name (str): The name of the embeddings key.
        entity_name_name (str): The key name for the entity's name.
        embeddings (bool): A flag to determine whether to calculate embeddings.
        
        Returns:
        dict: The entity dictionary with added embeddings.
        """
        entity = entity.copy()
        
        entity[entity_name_name] = entity[entity_name_name].lower().replace("_", " ").replace("-", " ")
        entity[property_name] = {}
        
        if embeddings:
            entity[property_name][embeddings_name] = self.langchain_output_parser.calculate_embeddings(text=entity[entity_name_name])
            
        return entity
 
    
    
    def extract_relations(self, context: str, entities: List[str], embeddings: bool = True, property_name = "properties", entity_name_name = "name"):
        """
        Extract relationships from a given context for specified entities and optionally add embeddings.
        
        Args:
        context (str): The textual context from which relationships will be extracted.
        entities (List[str]): A list of entity names to be considered in the extraction.
        embeddings (bool): A flag to determine whether to add embeddings to the extracted relationships.
        property_name (str): The property name under which embeddings will be stored in the relationship.
        entity_name_name (str): The key name for the entity's name in the relationship.
        
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
            self.extract_relations(context=context, entities=entities, embeddings=embeddings, property_name=property_name, entity_name_name=entity_name_name)
        if not entities:
            return []

        return list(map(lambda rel : self.__add_embeddings_as_property(entity = rel, embeddings=embeddings, property_name=property_name, entity_name_name=entity_name_name) , relationships["relationships"]))
    
    
    def extract_relations_for_isolated_entities(self, context: str, isolated_entities:List[str], embeddings: bool = True, property_name = "properties", entity_name_name = "name"):
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
        The entities {isolated_entities} do not contain any relation. Try to re-extract the relation(s) for them from the context.
        '''
        
        relationships = self.langchain_output_parser.extract_information_as_json_for_context(output_data_structure = RelationshipsExtractor, context=formatted_context, IE_query=IE_query)
        print(relationships)
        
        if "relationships" not in relationships.keys() or relationships == None:
            print("we are retrying ....")
            self.extract_relations_for_isolated_entities(context=context, isolated_entities=isolated_entities, embeddings=embeddings, property_name=property_name, entity_name_name=entity_name_name)


        return list(map(lambda rel : self.__add_embeddings_as_property(entity = rel, embeddings=embeddings, property_name=property_name, entity_name_name=entity_name_name) , relationships["relationships"]))
