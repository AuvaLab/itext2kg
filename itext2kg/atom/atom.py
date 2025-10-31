from itext2kg.atom.models import KnowledgeGraph, Entity, Relationship, RelationshipProperties
from itext2kg.atom.graph_matching import GraphMatcher
from itext2kg.llm_output_parsing import LangchainOutputParser
from itext2kg.atom.models.schemas import Relationship as RelationshipSchema
from itext2kg.atom.models.schemas import RelationshipsExtractor
import concurrent.futures
from typing import List, Optional
from itext2kg.atom.models.prompts import Prompt
from dateutil import parser
import asyncio
from itext2kg.logging_config import get_logger

logger = get_logger(__name__)

class Atom:
    def __init__(self, 
                 llm_model,
                 embeddings_model,
                 ) -> None:        
        """
        Initializes the ATOM with specified language model, embeddings model, and operational parameters.
        
        Args:
        matcher: The matcher instance to be used for matching entities and relationships.
        llm_output_parser: The language model instance to be used for extracting entities and relationships from text.
        """
        self.matcher = GraphMatcher()
        self.llm_output_parser = LangchainOutputParser(llm_model=llm_model, embeddings_model=embeddings_model)
    
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
                          ent_threshold:float = 0.8,
                          rel_threshold:float = 0.7,
                          entity_name_weight:float=0.8,
                          entity_label_weight:float=0.2,
                          max_workers:int=8,
                        ) -> KnowledgeGraph:
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
        for atomic_kg, fact in zip(atomic_kgs, atomic_facts):
            atomic_kg.add_atomic_facts_to_relationships(atomic_facts=[fact])

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
            return constructed_kg
        return merged_kg
    
    async def build_graph_from_different_obs_times(self,
                                                   atomic_facts_with_obs_timestamps:dict,
                                                    existing_knowledge_graph:KnowledgeGraph=None,
                                                    ent_threshold:float = 0.8,
                                                    rel_threshold:float = 0.7,
                                                    entity_name_weight:float=0.8,
                                                    entity_label_weight:float=0.2,
                                                    max_workers:int=8,
                                               ):
        kgs = await asyncio.gather(*[
                        self.build_graph(
                            atomic_facts=atomic_facts_with_obs_timestamps[timestamp], 
                            obs_timestamp=timestamp,
                            ent_threshold=ent_threshold,
                            rel_threshold=rel_threshold,
                            entity_name_weight=entity_name_weight,
                            entity_label_weight=entity_label_weight,
                            existing_knowledge_graph=None,
                        ) for timestamp in atomic_facts_with_obs_timestamps
                    ])
        if existing_knowledge_graph:
            return self.parallel_atomic_merge(kgs=[existing_knowledge_graph] + kgs, rel_threshold=rel_threshold, ent_threshold=ent_threshold, max_workers=max_workers)
        
        return self.parallel_atomic_merge(kgs=kgs, rel_threshold=rel_threshold, ent_threshold=ent_threshold, max_workers=max_workers)