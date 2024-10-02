from ..utils import LangchainOutputParser, EntitiesExtractor, DataHandler
from ..models import Entity, KnowledgeGraph

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
 
    
    def extract_entities(self, context: str, max_tries:int=5):
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
        tries = 0
        entities = None
        
 
        while tries < max_tries:
            try:
                entities = self.langchain_output_parser.extract_information_as_json_for_context(
                    context=context, output_data_structure=EntitiesExtractor
                )

                if entities and "entities" in entities.keys():
                    break
                
            except Exception as e:
                print(f"Not Formatted in the desired format. Error occurred: {e}. Retrying... (Attempt {tries + 1}/{max_tries})")

            tries += 1
    
        if not entities or "entities" not in entities:
            raise ValueError("Failed to extract entities after multiple attempts.")

        entities = [Entity(label=entity["label"], name = entity["name"]) 
                    for entity in entities["entities"]]
        
        kg = KnowledgeGraph(entities = entities, relationships=[])
        kg.embed_entities(
            embeddings_function=lambda x:self.langchain_output_parser.calculate_embeddings(x)
            )
        print(kg.entities)
        return kg.entities