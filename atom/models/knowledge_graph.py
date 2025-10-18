import numpy as np
from typing import Callable, List, Union, Optional, Awaitable
from pydantic import BaseModel, Field, ConfigDict
from atom.models.entity import Entity, EntityProperties
from atom.models.relationship import Relationship, RelationshipProperties

# -------------------------------------------
# Create a common base model class
# -------------------------------------------
class BaseModelWithConfig(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=False,
        extra="ignore"
    )

# -------------------------------------------
# KnowledgeGraph model
# -------------------------------------------
class KnowledgeGraph(BaseModelWithConfig):
    entities: List[Entity] = Field(default_factory=list)
    relationships: List[Relationship] = Field(default_factory=list)

    def is_empty(self) -> bool:
        return len(self.entities) == 0 and len(self.relationships) == 0
    def remove_duplicates_entities(self) -> None:
        self.entities = list(set(self.entities))

    async def embed_entities(self,
                             embeddings_function: Callable[[List[str]], Awaitable[np.ndarray]],
                             entity_name_weight: float = 0.8,
                             entity_label_weight: float = 0.2) -> None:
        self.remove_duplicates_entities()
        self.entities = list(map(lambda e: e.process(), self.entities))

        labels = [e.label for e in self.entities]
        names  = [e.name for e in self.entities]

        label_embeddings = await embeddings_function(labels)
        name_embeddings  = await embeddings_function(names)

        for e, le, ne in zip(self.entities, label_embeddings, name_embeddings):
            e.properties.embeddings = entity_label_weight * le + entity_name_weight * ne

    async def embed_relationships(self,
                                  embeddings_function: Callable[[List[str]], Awaitable[np.ndarray]]) -> None:
        self.relationships = list(map(lambda r: r.process(), self.relationships))

        names = [r.name for r in self.relationships]
        rel_embeddings = await embeddings_function(names)

        for r, emb in zip(self.relationships, rel_embeddings):
            r.properties.embeddings = emb

    def get_entity(self, other_entity: Entity) -> Optional[Entity]:
        """Finds and returns an entity using a fast dictionary lookup."""
        other_entity = other_entity.process()
        entity_dict = {e.__hash__(): e for e in self.entities}  # O(n) preprocessing, O(1) lookup
        return entity_dict.get(other_entity.__hash__())

    def get_relationship(self, other_relationship: Relationship) -> Optional[Relationship]:
        """Finds and returns a relationship using a fast dictionary lookup."""
        other_relationship = other_relationship.process()
        relationship_dict = {
            rel.__hash__(): rel for rel in self.relationships
        }
        return relationship_dict.get(other_relationship.__hash__())
    
    def add_timestamps_to_relationships(self, timestamps:Union[List[float], List[str]]) -> None:
        """Adds timestamps to relationships."""
        for rel in self.relationships:
            rel.combine_timestamps(timestamps=timestamps, temporal_aspect="timestamps")
    
    def add_sources_to_relationships(self, sources: List[str]) -> None:
        """Adds sources to relationships."""
        if self.relationships:
            for rel in self.relationships:
                rel.combine_sources(sources)

    def find_isolated_entities(self) -> List[Entity]:
        related_entities = set(r.startEntity for r in self.relationships) | \
                           set(r.endEntity   for r in self.relationships)
        return [ent for ent in self.entities if ent not in related_entities]

    
    def split_into_atomic_kgs(self) -> List['KnowledgeGraph']:
        """
        Splits this KnowledgeGraph into multiple atomic KnowledgeGraphs,
        where each atomic KG contains exactly one relationship and its associated entities.
        
        Returns:
            List[KnowledgeGraph]: A list of atomic KGs, each containing one relationship
                                and its startEntity and endEntity.
        
        """


        kgs = [KnowledgeGraph() for _ in range(len(self.relationships))]
        for i, relationship in enumerate(self.relationships):
            kgs[i].relationships = [relationship]
            kgs[i].entities = [relationship.startEntity, relationship.endEntity]
        return kgs

    @staticmethod
    def from_neo4j(graph_storage) -> 'KnowledgeGraph':
        """
        Builds a KnowledgeGraph by retrieving all data from Neo4j.
        
        Args:
            graph_storage: The graph storage instance with run_query_with_result method
            
        Returns:
            KnowledgeGraph: A KnowledgeGraph object populated with all data from Neo4j
        """
        entities = []
        relationships = []
        
        # Query to get all nodes with their properties
        nodes_query = "MATCH (n) RETURN n"
        node_records = graph_storage.run_query_with_result(nodes_query)
        
        # Build entities from nodes
        entity_dict = {}  # To map node identity to Entity objects for relationship building
        for record in node_records:
            node = record["n"]
            
            # Extract node properties
            properties = dict(node.items())
            
            # Handle embeddings if present
            embeddings = None
            if "embeddings" in properties:
                embeddings_str = properties.pop("embeddings")
                if embeddings_str:
                    try:
                        # Convert comma-separated string back to numpy array
                        embeddings = graph_storage.transform_str_list_to_embeddings(embeddings_str)
                    except:
                        embeddings = None
            
            # Create Entity
            entity = Entity(
                name=properties.get("name", ""),
                label=list(node.labels)[0] if node.labels else "",
                properties=EntityProperties(embeddings=embeddings if embeddings is not None else None)
            )
            
            # Store additional properties in the entity properties
            for key, value in properties.items():
                if key != "name" and hasattr(entity.properties, key):
                    setattr(entity.properties, key, value)
            
            entities.append(entity)
            entity_dict[node.element_id] = entity
        
        # Query to get all relationships with their properties
        rels_query = "MATCH (n)-[r]->(m) RETURN n, r, m"
        rel_records = graph_storage.run_query_with_result(rels_query)
        
        # Build relationships
        for record in rel_records:
            start_node = record["n"]
            rel = record["r"]
            end_node = record["m"]
            
            # Get corresponding entities
            start_entity = entity_dict.get(start_node.element_id)
            end_entity = entity_dict.get(end_node.element_id)
            
            if start_entity and end_entity:
                # Extract relationship properties
                rel_properties = dict(rel.items())
                
                # Handle embeddings if present
                embeddings = None
                if "embeddings" in rel_properties:
                    embeddings_str = rel_properties.pop("embeddings")
                    if embeddings_str:
                        try:
                            embeddings = graph_storage.transform_str_list_to_embeddings(embeddings_str)
                        except:
                            embeddings = None
                
                # Handle list properties (timestamps, sources, etc.)
                sources = rel_properties.pop("sources", [])
                timestamps = rel_properties.pop("timestamps", [])
                t_start = rel_properties.pop("t_start", [])
                t_end = rel_properties.pop("t_end", [])
                
                # Create RelationshipProperties
                rel_props = RelationshipProperties(
                    embeddings=embeddings if embeddings is not None else None,
                    sources=sources if isinstance(sources, list) else [],
                    timestamps=timestamps if isinstance(timestamps, list) else [],
                    t_start=t_start if isinstance(t_start, list) else [],
                    t_end=t_end if isinstance(t_end, list) else []
                )
                
                # Create Relationship
                relationship = Relationship(
                    name=rel.type,
                    startEntity=start_entity,
                    endEntity=end_entity,
                    properties=rel_props
                )
                
                relationships.append(relationship)
        
        return KnowledgeGraph(entities=entities, relationships=relationships)

    def __repr__(self) -> str:
        return (f"KnowledgeGraph("
                f"entities={self.entities!r}, "
                f"relationships={self.relationships!r})")