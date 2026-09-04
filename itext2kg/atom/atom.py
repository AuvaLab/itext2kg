from itext2kg.atom.models import KnowledgeGraph, Entity, Relationship, RelationshipProperties
from itext2kg.atom.graph_matching import GraphMatcher
from itext2kg.llm_output_parsing import LangchainOutputParser
from itext2kg.atom.models.schemas import Relationship as RelationshipSchema
from itext2kg.atom.models.schemas import (
    RelationshipsExtractor,
    AtomicFact,
    DomainedFact,
    DomainedAtomicFact,
)
import concurrent.futures
from typing import List, Optional
from pathlib import Path
from itext2kg.atom.models.prompts import Prompt
from dateutil import parser
import asyncio
from itext2kg.logging_config import get_logger

logger = get_logger(__name__)

class Atom:
    def __init__(
        self,
        llm_model,
        embeddings_model,
        kg_store_dir: Path | str | None = None,
    ) -> None:
        """
        Initializes the ATOM with specified language model, embeddings model, and operational parameters.

        Args:
            llm_model: Chat model for extraction.
            embeddings_model: Embeddings model for entity/relation matching.
            kg_store_dir: Optional directory where built KGs are persisted as
                JSON (+ NPZ for embeddings). When set, each ``build_graph`` call
                writes ``{obs_timestamp}.json`` / ``.npz``, and multi-obs merges
                also write ``merged.json`` / ``merged.npz``.
        """
        self.matcher = GraphMatcher()
        self.llm_output_parser = LangchainOutputParser(llm_model=llm_model, embeddings_model=embeddings_model)
        self.kg_store_dir = Path(kg_store_dir) if kg_store_dir is not None else None

    async def extract_atomic_facts(
        self,
        texts: List[str],
        observation_timestamp: str,
        domains: List[str] | None = None,
    ) -> List[List[DomainedFact]]:
        """Extract atomic facts from texts.

        ``domains=[]`` (default): no domain classification — every returned
        ``DomainedFact.domain`` is empty and propagates empty downstream.

        ``domains`` non-empty: each fact is labeled with one of the allowed domains.
        """
        domains = list(domains or [])
        system_query = Prompt.atomic_facts_system_query(
            obs_timestamp=observation_timestamp,
            domains=domains,
        )
        if domains:
            results = await self.llm_output_parser.extract_information_as_json_for_context(
                output_data_structure=DomainedAtomicFact,
                contexts=texts,
                system_query=system_query,
            )
            out: List[List[DomainedFact]] = []
            for result in results:
                if result is None:
                    out.append([])
                    continue
                facts = getattr(result, "atomic_fact", None) or []
                out.append(list(facts))
            return out

        results = await self.llm_output_parser.extract_information_as_json_for_context(
            output_data_structure=AtomicFact,
            contexts=texts,
            system_query=system_query,
        )
        out = []
        for result in results:
            if result is None:
                out.append([])
                continue
            raw = getattr(result, "atomic_fact", None) or []
            out.append([DomainedFact(fact=str(f), domain="") for f in raw])
        return out

    async def extract_quintuples(self, atomic_facts: List[str], observation_timestamp: str) -> List[RelationshipsExtractor]:
        """
        Extracts relationships from atomic facts using the language model.
        """
        return await self.llm_output_parser.extract_information_as_json_for_context(
            output_data_structure=RelationshipsExtractor,
            contexts=atomic_facts,
            system_query=Prompt.temporal_system_query(observation_timestamp) + Prompt.EXAMPLES.value
        )

    def merge_two_kgs(self, kg1, kg2, rel_threshold:float=0.8, ent_threshold:float=0.8):
        """
        Merges two KGs using the same logic as the sequential approach above.
        Returns a single KnowledgeGraph.
        """
        updated_entities, updated_relationships = self.matcher.match_entities_and_update_relationships(
            entities_2=kg1.entities,
            relationships_2=kg1.relationships,
            entities_1=kg2.entities,
            relationships_1=kg2.relationships,
            rel_threshold=rel_threshold,
            ent_threshold=ent_threshold
        )
        return KnowledgeGraph(entities=updated_entities, relationships=updated_relationships)

    def parallel_atomic_merge(self, kgs: List[KnowledgeGraph], existing_kg: Optional[KnowledgeGraph] = None, rel_threshold: float = 0.8, ent_threshold: float = 0.8, max_workers: int = 4) -> KnowledgeGraph:
        """
        Merges a list of KnowledgeGraphs in parallel, reducing them pairwise.
        """
        if not kgs:
            if existing_kg and not existing_kg.is_empty():
                return existing_kg
            return KnowledgeGraph()
        # Keep merging until we have just one KG
        current = kgs
        while len(current) > 1:
            merged_results = []
            
            # Prepare pairs
            pairs = [(current[i], current[i+1]) 
                    for i in range(0, len(current) - 1, 2)]
            
            # If there's an odd KG out, keep it aside to append later
            leftover = current[-1] if len(current) % 2 == 1 else None
            
            # Merge pairs in parallel
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(self.merge_two_kgs, p[0], p[1], rel_threshold, ent_threshold) for p in pairs]
                for f in concurrent.futures.as_completed(futures):
                    merged_results.append(f.result())
            
            # Rebuild current list from newly merged KGs + leftover
            if leftover:
                merged_results.append(leftover)
            
            current = merged_results
        if existing_kg and not existing_kg.is_empty():
            return self.merge_two_kgs(current[0], existing_kg, rel_threshold, ent_threshold)
        return current[0]

    async def build_atomic_kg_from_quintuples(self, 
        relationships:list[RelationshipSchema], 
        entity_name_weight:float=0.8, 
        entity_label_weight:float=0.2,
        rel_threshold:float=0.8,
        ent_threshold:float=0.8,
        max_workers:int=8,
        ):
        embedded_relationships = []
        temp_kg = KnowledgeGraph(entities=[Entity(**rel.startNode.model_dump()) for rel in relationships] + [Entity(**rel.endNode.model_dump()) for rel in relationships])
        await temp_kg.embed_entities(embeddings_function=self.llm_output_parser.calculate_embeddings, entity_name_weight=entity_name_weight, entity_label_weight=entity_label_weight)

        for relationship in relationships:
            if relationship.t_start is None:
                relationship.t_start = []
            elif relationship.t_end is None:
                relationship.t_end = []
            
            start_entity = temp_kg.get_entity(Entity(**relationship.startNode.model_dump()))
            end_entity = temp_kg.get_entity(Entity(**relationship.endNode.model_dump()))
            
            # Handle the case where entities might not be found (though they should be)
            if start_entity is None or end_entity is None:
                raise ValueError(f"Could not find entities for relationship {relationship.name}")
            
            # Handle timestamp parsing with None checks and error handling
            t_start_timestamps = []
            if relationship.t_start:
                for ts in relationship.t_start:
                    try:
                        parsed_dt = parser.parse(ts)
                        if parsed_dt is not None:
                            t_start_timestamps.append(parsed_dt.timestamp())
                    except Exception as e:
                        logger.warning(f"Could not parse t_start timestamp '{ts}': {e}. Skipping this timestamp.")
                        # Keep the place empty by simply not adding anything to the list
                        continue
            
            t_end_timestamps = []
            if relationship.t_end:
                for ts in relationship.t_end:
                    try:
                        parsed_dt = parser.parse(ts)
                        if parsed_dt is not None:
                            t_end_timestamps.append(parsed_dt.timestamp())
                    except Exception as e:
                        logger.warning(f"Could not parse t_end timestamp '{ts}': {e}. Skipping this timestamp.")
                        # Keep the place empty by simply not adding anything to the list
                        continue
            
            embedded_relationships.append(Relationship(name=relationship.name, 
                                        startEntity=start_entity, 
                                        endEntity=end_entity,
                                        properties = RelationshipProperties(t_start=t_start_timestamps, 
                                                                            t_end=t_end_timestamps)))
            
        

        kg = KnowledgeGraph(entities=temp_kg.entities, relationships=embedded_relationships)
        await kg.embed_relationships(embeddings_function=self.llm_output_parser.calculate_embeddings)
        # this line is just to ensure there are no duplicates entities and relationships inside the same factoid.
        atomic_kgs = kg.split_into_atomic_kgs()
        
        return self.parallel_atomic_merge(
            kgs=atomic_kgs, 
            rel_threshold=rel_threshold, 
            ent_threshold=ent_threshold, 
            max_workers=max_workers)

    async def build_graph(self, 
                          atomic_facts:List[str],
                          obs_timestamp: str,
                          existing_knowledge_graph:KnowledgeGraph=None,
                          domains:List[str]|None=None,
                          ent_threshold:float = 0.8,
                          rel_threshold:float = 0.7,
                          entity_name_weight:float=0.8,
                          entity_label_weight:float=0.2,
                          max_workers:int=8,
                        ) -> KnowledgeGraph:
        domains = list(domains or [])
        system_query = Prompt.temporal_system_query(obs_timestamp=obs_timestamp)
        examples = Prompt.EXAMPLES.value
        logger.info("------- Extracting Quintuples---------")
        relationships = await self.llm_output_parser.extract_information_as_json_for_context(output_data_structure=RelationshipsExtractor, contexts=atomic_facts, system_query=system_query+examples)
        
        logger.info("------- Building Atomic KGs---------")
        
        atomic_kgs = await asyncio.gather(*list(map(
            self.build_atomic_kg_from_quintuples, 
            [relation.relationships for relation in relationships], 
            [entity_name_weight for _ in relationships], 
            [entity_label_weight for _ in relationships],
            [rel_threshold for _ in relationships],
            [ent_threshold for _ in relationships],
            [max_workers for _ in relationships])))

        logger.info("------- Adding Atomic Facts to Atomic KGs---------")
        if domains and len(domains) != len(atomic_facts):
            raise ValueError(
                f"domains length ({len(domains)}) must match atomic_facts length ({len(atomic_facts)})"
            )
        for i, (atomic_kg, fact) in enumerate(zip(atomic_kgs, atomic_facts)):
            atomic_kg.add_atomic_facts_to_relationships(atomic_facts=[fact])
            if domains:
                domain = domains[i]
                if domain:
                    atomic_kg.add_domains_to_relationships(domains=[domain])

        logger.info("------- Merging Atomic KGs---------")
        cleaned_atomic_kgs = [kg for kg in atomic_kgs if kg.relationships != []]
        merged_kg = self.parallel_atomic_merge(kgs=cleaned_atomic_kgs, 
        rel_threshold=rel_threshold, 
        ent_threshold=ent_threshold, 
        max_workers=max_workers
        )

        logger.info("------- Adding Observation Timestamp to Relationships---------")
        merged_kg.add_t_obs_to_relationships(t_obs=[obs_timestamp])
    
        if existing_knowledge_graph:
            global_entities, global_relationships = self.matcher.match_entities_and_update_relationships(entities_1=merged_kg.entities,
                                                                 entities_2=existing_knowledge_graph.entities,
                                                                 relationships_1=merged_kg.relationships,
                                                                 relationships_2=existing_knowledge_graph.relationships,
                                                                 ent_threshold=ent_threshold,
                                                                 rel_threshold=rel_threshold,
                                                                #  entity_name_weight=entity_name_weight,
                                                                #  entity_label_weight=entity_label_weight
                                                                 )    
        
            constructed_kg = KnowledgeGraph(entities=global_entities, relationships=global_relationships)
            self._persist_kg(constructed_kg, obs_timestamp)
            return constructed_kg
        self._persist_kg(merged_kg, obs_timestamp)
        return merged_kg
    
    async def build_graph_from_different_obs_times(self,
                                                   atomic_facts_with_obs_timestamps:dict,
                                                    existing_knowledge_graph:KnowledgeGraph=None,
                                                    domains_with_obs_timestamps:dict|None=None,
                                                    ent_threshold:float = 0.8,
                                                    rel_threshold:float = 0.7,
                                                    entity_name_weight:float=0.8,
                                                    entity_label_weight:float=0.2,
                                                    max_workers:int=8,
                                               ):
        domains_with_obs_timestamps = domains_with_obs_timestamps or {}
        kgs = await asyncio.gather(*[
                        self.build_graph(
                            atomic_facts=atomic_facts_with_obs_timestamps[timestamp], 
                            obs_timestamp=timestamp,
                            domains=domains_with_obs_timestamps.get(timestamp, []),
                            ent_threshold=ent_threshold,
                            rel_threshold=rel_threshold,
                            entity_name_weight=entity_name_weight,
                            entity_label_weight=entity_label_weight,
                            existing_knowledge_graph=None,
                        ) for timestamp in atomic_facts_with_obs_timestamps
                    ])
        if existing_knowledge_graph:
            merged = self.parallel_atomic_merge(kgs=[existing_knowledge_graph] + kgs, rel_threshold=rel_threshold, ent_threshold=ent_threshold, max_workers=max_workers)
        else:
            merged = self.parallel_atomic_merge(kgs=kgs, rel_threshold=rel_threshold, ent_threshold=ent_threshold, max_workers=max_workers)
        self._persist_kg(merged, "merged")
        return merged

    def _persist_kg(self, kg: KnowledgeGraph, label: str) -> None:
        """Write JSON (+ NPZ) under kg_store_dir when configured."""
        if self.kg_store_dir is None:
            return
        safe = str(label).replace("/", "_").replace(" ", "_")
        json_path = self.kg_store_dir / f"{safe}.json"
        npz_path = self.kg_store_dir / f"{safe}.npz"
        kg.to_json(json_path, embeddings_path=npz_path)
        logger.info("Persisted KG to %s (+ %s)", json_path, npz_path)
