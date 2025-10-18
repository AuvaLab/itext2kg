from pydantic import BaseModel, SkipValidation
from typing import Callable, Awaitable
import numpy as np
import re
from dateutil.parser import parse as parse_date

class EntityProperties(BaseModel):
    embeddings: SkipValidation[np.array]=None
    class Config:
        arbitrary_types_allowed = True
        
class RelationshipProperties(BaseModel):
    embeddings:SkipValidation[np.array]=None
    observation_dates:list[str]=[]
    class Config:
        arbitrary_types_allowed = True
    
class Entity(BaseModel):
    label:str = ""
    name:str = ""
    properties:EntityProperties = EntityProperties()
    
    def process(self):
        # Replace spaces, dashes, periods, and '&' in names with underscores or 'and'.
        self.label = re.sub(r'[^a-zA-Z0-9]', '_', self.label).replace("&", "and")
        self.name = self.name.lower().replace("_", " ").replace("-", " ").replace('"', " ").strip()
    
    async def embed_Entity(self,
                           embeddings_function:Callable[[str], Awaitable[np.array]],
                           entity_name_weight:float=0.6,
                           entity_label_weight:float=0.4)-> None:
        self.process()
        name_embedding = await embeddings_function(self.name)
        label_embedding = await embeddings_function(self.label)
        self.properties.embeddings = (
            entity_name_weight * name_embedding
            +
            entity_label_weight * label_embedding
        )
        
    def __eq__(self, other) -> bool:
        if isinstance(other, Entity):
            return self.name == other.name and self.label == other.label
        return False
    
    def __hash__(self) -> int:
        return hash((self.name, self.label))
    
    def __repr__(self):
        return f"Entity(name={self.name}, label={self.label}, properties={self.properties})"

class Relationship(BaseModel):
    startEntity:Entity = Entity()
    endEntity:Entity = Entity()
    name:str = ""
    properties:RelationshipProperties = RelationshipProperties()
    
    def process(self):
        # Replace spaces, dashes, periods, and '&' in names with underscores or 'and'.
        self.name = re.sub(r'[^a-zA-Z0-9]', '_', self.name).replace("&", "and")
            
    def embed_relationship(self, embeddings_function:Callable[[str], np.array]):
        self.process()
        self.properties.embeddings = embeddings_function(self.name)
    
    def combine_observation_dates(self, observation_dates:list[str])-> None:
        self.properties.observation_dates.extend(observation_dates)
        
    def __eq__(self, other) -> bool:
        if isinstance(other, Relationship):
            return (self.startEntity == other.startEntity
                    and self.endEntity == other.endEntity 
                    and self.name == other.name 
                    )
        return False
    
    def __hash__(self):
        return hash((self.name, self.startEntity, self.endEntity))

    def __repr__(self):
        return f"Relationship(name={self.name}, startEntity={self.startEntity}, endEntity={self.endEntity}, properties={self.properties})"
    

class KnowledgeGraph(BaseModel):
    entities:list[Entity]= []
    relationships:list[Relationship] = []
    
    async def embed_entities(self,
                             embeddings_function:Callable[[str], Awaitable[np.array]],
                             entity_name_weight:float=0.6,
                             entity_label_weight:float=0.4)-> None:
        self.remove_duplicates_entities()
        for Entity in self.entities:
            Entity.process()
        
        # Get embeddings for labels and names separately
        labels_embeddings = await embeddings_function([Entity.label for Entity in self.entities])
        names_embeddings = await embeddings_function([Entity.name for Entity in self.entities])
        
        entities_embeddings = (
            entity_label_weight * labels_embeddings
            +  
            entity_name_weight * names_embeddings
            )
        
        for Entity, embedding in zip(self.entities, entities_embeddings):
            Entity.properties.embeddings = embedding
            
        
    async def embed_relationships(self, embeddings_function:Callable[[str], Awaitable[np.array]])-> None:
        self.remove_duplicates_relationships()
        for relationship in self.relationships:
            relationship.process()
        
        relationships_embeddings = await embeddings_function([relationship.name for relationship in self.relationships])
        
        for relationship, embedding in zip(self.relationships, relationships_embeddings):
            relationship.properties.embeddings = embedding
    
    def get_entity(self, other_entity:Entity):
        for entity in self.entities:
            if entity == other_entity : 
                return entity
        return None
    
    def get_relationship(self, other_relationship:Relationship):
        for relationship in self.relationships:
            if relationship == other_relationship:
                return relationship
        return None
    
    def add_observation_dates(self, observation_date: str) -> None:
        if not observation_date or observation_date.strip() == "":
            return None
        
        parsed_date = parse_date(observation_date.strip())
        timestamp = parsed_date.timestamp()
        
        for relationship in self.relationships:
            relationship.properties.observation_dates.append(timestamp)
    
    
    def remove_duplicates_entities(self) -> None:
        """
        Remove duplicate entities (entities) by relying on the `__hash__` and `__eq__` methods of the `Entity` class.
        This will update the `entities` attribute by filtering out duplicates.
        """
        self.entities = list(set(self.entities))  # Using set to automatically remove duplicates based on hash and eq methods

    def remove_duplicates_relationships(self) -> None:
        """
        Remove duplicate relationships by relying on the `__hash__` and `__eq__` methods of the `Relationship` class.
        This will update the `relationships` attribute by filtering out duplicates.
        """
        self.relationships = list(set(self.relationships))  # Using set to automatically remove duplicates based on hash and eq methods
    
    def find_isolated_entities(self):
        relation_entities = set(rel.startEntity for rel in self.relationships) | set(rel.endEntity for rel in self.relationships)
        isolated_entities = [ent for ent in self.entities if ent not in relation_entities]
        return isolated_entities