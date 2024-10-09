from ..utils import LangchainOutputParser, EntitiesExtractor
from ..models import Entity, KnowledgeGraph
from typing import List
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
    
    def extract_entities(self, context: str, 
                         max_tries:int=5,
                         entity_name_weight:float=0.6,
                         entity_label_weight:float=0.4) -> List[Entity]:
        """
        Extract entities from a given context.
        
        Args:
            context (str): The textual context from which entities will be extracted.
            max_tries (int): The maximum number of attempts to extract entities. Defaults to 5.
            entity_name_weight (float): The weight of the entity name, set to 0.6, indicating its
                                     relative importance in the overall evaluation process.
            entity_label_weight (float): The weight of the entity label, set to 0.4, reflecting its
                                      secondary significance in the evaluation process.
        
        Returns:
            List[Entity]: A list of extracted entities with embeddings.
        
        Raises:
            ValueError: If entity extraction fails after the specified maximum number of attempts.
        
        """
        tries = 0
        entities = None
        IE_query  = '''
        # DIRECTIVES : 
        - Act like an experienced knowledge graph builder.
        '''
 
        while tries < max_tries:
            try:
                entities = self.langchain_output_parser.extract_information_as_json_for_context(
                    context=context, 
                    output_data_structure=EntitiesExtractor,
                    IE_query=IE_query
                )

                if entities and "entities" in entities.keys():
                    break
                
            except Exception as e:
                print(f"Not Formatted in the desired format. Error occurred: {e}. Retrying... (Attempt {tries + 1}/{max_tries})")

            tries += 1
    
        if not entities or "entities" not in entities:
            raise ValueError("Failed to extract entities after multiple attempts.")

        print (entities)
        entities = [Entity(label=entity["label"], name = entity["name"]) 
                    for entity in entities["entities"]]
        #print(entities)
        kg = KnowledgeGraph(entities = entities, relationships=[])
        kg.embed_entities(
            embeddings_function=lambda x:self.langchain_output_parser.calculate_embeddings(x),
            entity_label_weight=entity_label_weight,
            entity_name_weight=entity_name_weight
            )
        return kg.entities