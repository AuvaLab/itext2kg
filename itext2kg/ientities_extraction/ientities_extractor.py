from ..utils import LangchainOutputParser, EntitiesExtractor

class iEntitiesExtractor():
    """
    A class to extract entities from text using natural language processing tools and embeddings.
    """
    def __init__(self, openai_api_key:str, embeddings_model_name :str = "text-embedding-3-large", model_name:str = "gpt-4-turbo", temperature:float = 0, sleep_time:int=5) -> None:        
        """
        Initializes the iEntitiesExtractor with specified API key, models, and operational parameters.
        
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
 
    
    
    def extract_entities(self, context: str, embeddings: bool = True, property_name = "properties", entity_name_name = "name"):
        """
        Extract entities from a given context and optionally add embeddings to each.
        
        Args:
        context (str): The textual context from which entities will be extracted.
        embeddings (bool): A flag to determine whether to add embeddings to the extracted entities.
        property_name (str): The property name under which embeddings will be stored in the entity.
        entity_name_name (str): The key name for the entity's name.
        
        Returns:
        List[dict]: A list of extracted entities with optional embeddings.
        """
        entities = self.langchain_output_parser.extract_information_as_json_for_context(context=context, output_data_structure=EntitiesExtractor)
        print(entities)
        
        
        if "entities" not in entities.keys() or entities == None:
            print("Not formatted in the desired format, we are retrying ....")
            self.extract_entities(context=context, entities=entities, embeddings=embeddings, property_name=property_name, entity_name_name=entity_name_name)
        
        
        return list(map(lambda word : self.__add_embeddings_as_property(entity = word, embeddings=embeddings, property_name=property_name, entity_name_name=entity_name_name) , entities["entities"]))
    

        
    