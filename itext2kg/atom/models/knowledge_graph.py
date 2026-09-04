import numpy as np
from typing import Callable, List, Union, Optional, Awaitable
from pydantic import BaseModel, Field, ConfigDict
from collections import defaultdict, deque
from itext2kg.atom.models.entity import Entity, EntityProperties
from itext2kg.atom.models.relationship import Relationship, RelationshipProperties

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
    
    def add_t_obs_to_relationships(self, t_obs:Union[List[float], List[str]]) -> None:
        """Adds t_obs to relationships."""
        for rel in self.relationships:
            rel.combine_timestamps(timestamps=t_obs, temporal_aspect="t_obs")
    
    def add_atomic_facts_to_relationships(self, atomic_facts: List[str]) -> None:
        """Adds atomic facts to relationships."""
        if self.relationships:
            for rel in self.relationships:
                rel.combine_atomic_facts(atomic_facts)

    def add_domains_to_relationships(self, domains: List[str]) -> None:
        """Adds domains to relationships."""
        if self.relationships:
            for rel in self.relationships:
                rel.combine_domains(domains)

    def find_isolated_entities(self) -> List[Entity]:
        related_entities = set(r.startEntity for r in self.relationships) | \
                           set(r.endEntity   for r in self.relationships)
        return [ent for ent in self.entities if ent not in related_entities]

    def extract_connecting_subgraph(self, triple_indices: List[int]) -> List[int]:
        """
        Returns the smallest set of relationship indices that connect all entities
        appearing in the given triples. Uses BFS shortest paths between entity pairs.

        Args:
            triple_indices: Indices of relationships (e.g. rare triples).

        Returns:
            Deduplicated list of relationship indices (rare triples + bridging edges).
        """
        if not self.relationships or not triple_indices:
            return list(set(triple_indices))

        # Entity-level adjacency: entity_name -> [(neighbor_entity_name, rel_index)]
        adj: dict = defaultdict(list)
        for idx, rel in enumerate(self.relationships):
            s, o = rel.startEntity.name, rel.endEntity.name
            adj[s].append((o, idx))
            adj[o].append((s, idx))

        # Entities that appear in the given triples
        seed_entities = set()
        for idx in triple_indices:
            if 0 <= idx < len(self.relationships):
                rel = self.relationships[idx]
                seed_entities.add(rel.startEntity.name)
                seed_entities.add(rel.endEntity.name)
        seed_list = sorted(seed_entities)

        result = set(triple_indices)
        # BFS between each pair of seed entities to get shortest path (as rel indices)
        for i in range(len(seed_list)):
            for j in range(i + 1, len(seed_list)):
                start, end = seed_list[i], seed_list[j]
                path_rels = self._bfs_entity_path(adj, start, end)
                result.update(path_rels)
        return sorted(result)

    @staticmethod
    def _bfs_entity_path(adj: dict, start: str, end: str) -> List[int]:
        """BFS from start to end on entity graph; returns list of relationship indices on path."""
        if start == end:
            return []
        if start not in adj or end not in adj:
            return []
        seen = {start}
        queue = deque([(start, [])])
        while queue:
            entity, path_rel_indices = queue.popleft()
            for neighbor, rel_idx in adj[entity]:
                if neighbor == end:
                    return path_rel_indices + [rel_idx]
                if neighbor not in seen:
                    seen.add(neighbor)
                    queue.append((neighbor, path_rel_indices + [rel_idx]))
        return []

    
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
                
                # Handle list properties with support for multiple naming conventions
                # Check for both old names (sources, timestamps, t_valid, t_invalid) 
                # and new names (atomic_facts, t_obs, t_start, t_end)
                atomic_facts = rel_properties.pop("sources", rel_properties.pop("atomic_facts", []))
                domains = rel_properties.pop("domains", [])
                t_obs = rel_properties.pop("timestamps", rel_properties.pop("t_obs", []))
                t_start = rel_properties.pop("t_valid", rel_properties.pop("t_start", []))
                t_end = rel_properties.pop("t_invalid", rel_properties.pop("t_end", []))
                
                # Create RelationshipProperties
                rel_props = RelationshipProperties(
                    embeddings=embeddings if embeddings is not None else None,
                    atomic_facts=atomic_facts if isinstance(atomic_facts, list) else [],
                    domains=domains if isinstance(domains, list) else [],
                    t_obs=t_obs if isinstance(t_obs, list) else [],
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


    def to_json(self, path, embeddings_path=None) -> None:
        """Persist graph structure as JSON; optionally write embeddings to an NPZ sidecar.

        Embeddings are excluded from the JSON payload. When ``embeddings_path`` is
        given, entity and relationship embedding vectors are stored as float32
        arrays aligned by index with the JSON entity/relationship lists.
        """
        from pathlib import Path
        import json

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        entities_payload = []
        entity_embeddings = []
        for e in self.entities:
            entities_payload.append({"name": e.name, "label": e.label})
            emb = e.properties.embeddings
            entity_embeddings.append(
                None if emb is None else np.asarray(emb, dtype=np.float32)
            )

        relationships_payload = []
        relationship_embeddings = []
        for rel in self.relationships:
            props = rel.properties
            relationships_payload.append(
                {
                    "name": rel.name,
                    "startEntity": {
                        "name": rel.startEntity.name,
                        "label": rel.startEntity.label,
                    },
                    "endEntity": {
                        "name": rel.endEntity.name,
                        "label": rel.endEntity.label,
                    },
                    "properties": {
                        "atomic_facts": list(props.atomic_facts or []),
                        "domains": list(getattr(props, "domains", None) or []),
                        "t_obs": list(props.t_obs or []),
                        "t_start": list(props.t_start or []),
                        "t_end": list(props.t_end or []),
                    },
                }
            )
            emb = props.embeddings
            relationship_embeddings.append(
                None if emb is None else np.asarray(emb, dtype=np.float32)
            )

        payload = {
            "schema_version": 1,
            "entities": entities_payload,
            "relationships": relationships_payload,
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        if embeddings_path is not None:
            embeddings_path = Path(embeddings_path)
            embeddings_path.parent.mkdir(parents=True, exist_ok=True)
            e_mask = np.array(
                [emb is not None for emb in entity_embeddings], dtype=bool
            )
            r_mask = np.array(
                [emb is not None for emb in relationship_embeddings], dtype=bool
            )
            if e_mask.any():
                dim = next(emb.shape[0] for emb in entity_embeddings if emb is not None)
                e_mat = np.zeros((len(entity_embeddings), dim), dtype=np.float32)
                for i, emb in enumerate(entity_embeddings):
                    if emb is not None:
                        e_mat[i] = emb
            else:
                e_mat = np.zeros((len(entity_embeddings), 0), dtype=np.float32)
            if r_mask.any():
                dim = next(
                    emb.shape[0] for emb in relationship_embeddings if emb is not None
                )
                r_mat = np.zeros((len(relationship_embeddings), dim), dtype=np.float32)
                for i, emb in enumerate(relationship_embeddings):
                    if emb is not None:
                        r_mat[i] = emb
            else:
                r_mat = np.zeros((len(relationship_embeddings), 0), dtype=np.float32)
            np.savez_compressed(
                embeddings_path,
                entity_embeddings=e_mat,
                relationship_embeddings=r_mat,
                entity_mask=e_mask,
                relationship_mask=r_mask,
            )

    @classmethod
    def from_json(cls, path, embeddings_path=None):
        """Load a KnowledgeGraph (or subclass) from JSON, optionally reattaching NPZ embeddings."""
        from pathlib import Path
        import json

        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        if int(data.get("schema_version", 1)) != 1:
            raise ValueError(
                f"Unsupported KnowledgeGraph schema_version: {data.get('schema_version')}"
            )

        from typing import get_args, get_origin

        rel_field = cls.model_fields.get("relationships")
        rel_cls = Relationship
        if rel_field is not None:
            ann = rel_field.annotation
            args = get_args(ann)
            if args:
                rel_cls = args[0]
        props_field = getattr(rel_cls, "model_fields", {}).get("properties")
        props_cls = RelationshipProperties
        if props_field is not None:
            props_cls = props_field.annotation

        entities = [
            Entity(name=e["name"], label=e.get("label", ""))
            for e in data.get("entities", [])
        ]
        entity_map = {(e.name, e.label): e for e in entities}

        relationships = []
        for rel in data.get("relationships", []):
            start_key = (rel["startEntity"]["name"], rel["startEntity"].get("label", ""))
            end_key = (rel["endEntity"]["name"], rel["endEntity"].get("label", ""))
            start = entity_map.get(start_key)
            end = entity_map.get(end_key)
            if start is None:
                start = Entity(name=start_key[0], label=start_key[1])
                entities.append(start)
                entity_map[start_key] = start
            if end is None:
                end = Entity(name=end_key[0], label=end_key[1])
                entities.append(end)
                entity_map[end_key] = end
            props_data = dict(rel.get("properties") or {})
            props_data.pop("embeddings", None)
            relationships.append(
                rel_cls(
                    name=rel.get("name", ""),
                    startEntity=start,
                    endEntity=end,
                    properties=props_cls(**props_data),
                )
            )

        kg = cls(entities=entities, relationships=relationships)

        if embeddings_path is not None:
            embeddings_path = Path(embeddings_path)
            if embeddings_path.exists():
                with np.load(embeddings_path) as arrays:
                    e_mat = arrays["entity_embeddings"]
                    r_mat = arrays["relationship_embeddings"]
                    e_mask = arrays["entity_mask"]
                    r_mask = arrays["relationship_mask"]
                for i, entity in enumerate(kg.entities):
                    if i < len(e_mask) and e_mask[i]:
                        entity.properties.embeddings = np.asarray(
                            e_mat[i], dtype=np.float32
                        )
                for i, rel in enumerate(kg.relationships):
                    if i < len(r_mask) and r_mask[i]:
                        rel.properties.embeddings = np.asarray(
                            r_mat[i], dtype=np.float32
                        )
        return kg

    def __repr__(self) -> str:
        return (f"KnowledgeGraph("
                f"entities={self.entities!r}, "
                f"relationships={self.relationships!r})")