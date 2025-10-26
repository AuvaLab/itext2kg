from typing import List, Tuple, Protocol
from itext2kg.atom.models import Entity, Relationship

class GraphMatcherInterface(Protocol):
    """
    Protocol defining the interface for entity and relationship matching implementations.
    Implementations should handle matching using techniques like:
    - Name/label equivalences (exact match) 
    - Similarity metrics (e.g. cosine similarity)
    """

    def match_entities_and_update_relationships(
        self,
        entities_1: List["Entity"],
        entities_2: List["Entity"],
        relationships_1: List["Relationship"],
        relationships_2: List["Relationship"],
        rel_threshold: float = 0.8,
        ent_threshold: float = 0.8
    ) -> Tuple[List["Entity"], List["Relationship"]]:
        """
        Match and merge two sets of entities and relationships.

        Args:
            entities_1: First list of entities
            entities_2: Second list of entities
            relationships_1: First list of relationships
            relationships_2: Second list of relationships
            rel_threshold: Minimum similarity threshold for relationship matching
            ent_threshold: Minimum similarity threshold for entity matching

        Returns:
            Tuple containing:
            - global_entities: Merged list of unique entities
            - combined_relationships: Merged list of relationships with updated entity references
        """
        ...


