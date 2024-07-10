from typing import List
from ..utils import LangchainOutputParser, RelationshipsExtractor
from ..utils import Matcher

class iRelationsExtractor():
    def __init__(self, openai_api_key:str, embeddings_model_name :str = "text-embedding-3-large", model_name:str = "gpt-4-turbo", temperature:float = 0, sleep_time:int=5) -> None:        
        self.langchain_output_parser =  LangchainOutputParser(openai_api_key=openai_api_key,
                                                       embeddings_model_name=embeddings_model_name,
                                                       model_name=model_name,
                                                       temperature=temperature,
                                                       sleep_time=sleep_time)  
        
    
    
    def __add_embeddings_as_property(self, entity:dict, property_name = "properties", embeddings_name = "embeddings", entity_name_name = "name", embeddings:bool = True):
        entity = entity.copy()
        
        entity[entity_name_name] = entity[entity_name_name].lower().replace("_", " ").replace("-", " ")
        entity[property_name] = {}
        
        if embeddings:
            entity[property_name][embeddings_name] = self.langchain_output_parser.calculate_embeddings(text=entity[entity_name_name])
            
        return entity
 
    
    
    def extract_relations(self, context: str, entities: List[str], embeddings: bool = True, property_name = "properties", entity_name_name = "name"):
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
    
    
    
    def extract_relations_for_isolated_entities(self, context: str, entities: List[str], relations_with_isolated_entities:List[str], embeddings: bool = True, property_name = "properties", entity_name_name = "name"):
        print("Some relations with isolated entities were detected ... trying to solve them!")
        formatted_context = f"context : \n -- '{context}' \n entities : \n -- {entities}"
        relations_with_isolated_entities_names = [f"{rel['startNode']} -> {rel['endNode']}" for rel in relations_with_isolated_entities]
        IE_query = f'''
        # Directives
        The relation {relations_with_isolated_entities_names} contains missed entities in the provided entities list. Try to re-extract a relation from the context based on the provided entities.
        '''
        
        relationships = self.langchain_output_parser.extract_information_as_json_for_context(output_data_structure = RelationshipsExtractor, context=formatted_context, IE_query=IE_query)
        print(relationships)
        
        if "relationships" not in relationships.keys() or relationships == None:
            print("we are retrying ....")
            self.extract_relations(context=context, entities=entities, relations_with_isolated_entities=relations_with_isolated_entities, embeddings=embeddings, property_name=property_name, entity_name_name=entity_name_name)
        if not entities:
            return []

        return list(map(lambda rel : self.__add_embeddings_as_property(entity = rel, embeddings=embeddings, property_name=property_name, entity_name_name=entity_name_name) , relationships["relationships"]))


        
    