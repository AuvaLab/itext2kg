from ..utils import LangchainOutputParser, EntitiesExtractor, DataHandler

class iEntitiesExtractor():
    """
    A class to extract entities from text using natural language processing tools and embeddings.
    """
    def __init__(self, llm_model, embeddings_model, sleep_time:int=5) -> None:        
        """
        Initializes the iEntitiesExtractor with specified language model, embeddings model, and operational parameters.
        
        Args:
        llm_model: The language model instance to be used for extracting entities from text.
        embeddings_model: The embeddings model instance to be used for generating vector representations of text entities.
        sleep_time (int): The time to wait (in seconds) when encountering rate limits or errors. Defaults to 5 seconds.
        """
    
        self.langchain_output_parser =  LangchainOutputParser(llm_model=llm_model,
                                                              embeddings_model=embeddings_model,
                                                       sleep_time=sleep_time)  
        self.data_handler = DataHandler()
 
    
    def extract_entities(self, context: str, embeddings: bool = True, property_name = "properties", entity_name_key = "name"):
        """
        Extract entities from a given context and optionally add embeddings to each.
        
        Args:
        context (str): The textual context from which entities will be extracted.
        embeddings (bool): A flag to determine whether to add embeddings to the extracted entities.
        property_name (str): The property name under which embeddings will be stored in the entity.
        entity_name_key (str): The key name for the entity's name.
        
        Returns:
        List[dict]: A list of extracted entities with optional embeddings.
        """
        entities = self.langchain_output_parser.extract_information_as_json_for_context(context=context, output_data_structure=EntitiesExtractor)
        print(entities)
        
        
        if "entities" not in entities.keys() or entities == None:
            print("Not formatted in the desired format, we are retrying ....")
            self.extract_entities(context=context, entities=entities, embeddings=embeddings, property_name=property_name, entity_name_key=entity_name_key)
        
        return self.data_handler.add_embeddings_as_property_batch(embeddings_function=lambda x:self.langchain_output_parser.calculate_embeddings(x), 
                                                                  items=entities["entities"],
                                                                  property_name=property_name,
                                                                  item_name_key=entity_name_key,
                                                                  embeddings=embeddings)
    

        
    