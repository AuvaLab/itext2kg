import numpy as np
from sklearn.metrics.pairwise import cosine_similarity  # type: ignore
from typing import List, Tuple
from itext2kg.atom.models import Entity, Relationship, KnowledgeGraph
from itext2kg.atom.graph_matching.matcher_interface import GraphMatcherInterface
import logging

logger = logging.getLogger(__name__)

class GraphMatcher(GraphMatcherInterface):
    """
    Class to handle the matching and processing of entities or relations using:
      - Name/label equivalences (exact match)
      - High cosine similarity (batch-computed via matrix ops).
    """
    def __init__(self):
        pass

    def _batch_match_entities(
        self,
        entities1: List["Entity"],
        entities2: List["Entity"],
        threshold: float = 0.8
    ) -> Tuple[List["Entity"], List["Entity"]]:
        """
        Batch-match entities1 against entities2, returning:
        - matched_entities1: For each e1 in entities1, the best match in entities2 if above
            threshold, or e1 itself if no good match was found, preserving the same index order
            as in entities1.
        - global_entities: a union of matched_entities1 + entities2 with duplicates removed.
        """

        # matched_entities1[i] will store the chosen match for entities1[i].
        matched_entities1 = [None] * len(entities1)

        # We'll accumulate those from entities1 that still need embedding-based matching.
        # We'll store (index_in_entities1, entity) so we can fill matched_entities1 in the same order.
        to_match = []

        # 1) Exact matches by name+label first
        for i, e1 in enumerate(entities1):
            if e1 in entities2:
                # The e1 object matches exactly an entity in entities2 
                # (same name+label => __eq__)
                logger.info(f"Exact match for Entity: {e1.name}")
                matched_entities1[i] = e1  # or unify onto the actual object in entities2 if desired
            else:
                to_match.append((i, e1))

        # Identify the subset of entities2 that were not matched in the exact-match pass
        # Because we do not want to attempt embedding matching against those that have
        # already been "taken."
        already_matched_e2 = set()
        for i, e1 in enumerate(entities1):
            if matched_entities1[i] is not None:
                # Find the actual e2 reference that equals e1 (if we want to unify onto e2)
                # or we can just treat it as already matched. 
                for e2 in entities2:
                    if e2 == e1:
                        already_matched_e2.add(e2)
                        break

        unmatched_entities2 = [e2 for e2 in entities2 if e2 not in already_matched_e2]

        # 2) For those in 'to_match', do embedding-based matching
        if to_match and unmatched_entities2:
            # Build embedding matrices
            e1_embs = np.vstack([t[1].properties.embeddings for t in to_match])  # (N, d)
            e2_embs = np.vstack([u.properties.embeddings for u in unmatched_entities2])  # (M, d)
            sim_matrix = cosine_similarity(e1_embs, e2_embs)  # shape (N, M)

            # Pick for each e1 the best unmatched e2
            best_cols = sim_matrix.argmax(axis=1)
            best_scores = sim_matrix.max(axis=1)

            for (row_idx, col_idx, score) in zip(range(len(to_match)), best_cols, best_scores):
                i_in_e1 = to_match[row_idx][0]
                e1_obj  = to_match[row_idx][1]

                if score >= threshold:
                    best_match_e2 = unmatched_entities2[col_idx]
                    logger.info(f"Wohoo! Entity was matched --- [{e1_obj.name}:{e1_obj.label}] --merged --> [{best_match_e2.name}:{best_match_e2.label}] (score={score:.2f})")
                    # Fill matched_entities1 at the same index as e1
                    #
                    # If you prefer to unify onto the object from entities2 (so references
                    # all end up pointing to the same object), do:
                    matched_entities1[i_in_e1] = best_match_e2
                else:
                    # No sufficiently close match => keep e1
                    matched_entities1[i_in_e1] = e1_obj
        else:
            # No embedding matching needed, just fill them in as themselves
            for (i_in_e1, e1_obj) in to_match:
                matched_entities1[i_in_e1] = e1_obj

        # 3) Build a union of matched_entities1 + entities2, removing duplicates
        # matched_entities1 may contain references to entities2 objects if they were matched
        combined = matched_entities1 + entities2
        kg = KnowledgeGraph(entities=combined)
        kg.remove_duplicates_entities()

        return matched_entities1, kg.entities
    

    def _batch_match_relationships(
        self,
        rels1: List["Relationship"],
        rels2: List["Relationship"],
        threshold: float = 0.8
    ) -> List["Relationship"]:
        """
        For each Relationship in `rels1`, find the most similar Relationship in `rels2`
        by comparing the name embeddings (via cosine similarity).
        
        If the highest similarity >= `threshold`, we rename the `rels1[i]` relationship 
        to have the same name as the best-matching relationship from `rels2`.
        
        We keep `rels1[i].startEntity` and `rels1[i].endEntity` unchanged.
        
        Returns the updated `rels1`.
        """
        # Handle empty relationship lists
        if not rels1:
            return [], rels2
        if not rels2:
            return rels1, rels1
            
        # Gather embeddings from both sets
        r1_embs = np.vstack([r.properties.embeddings for r in rels1])
        r2_embs = np.vstack([r.properties.embeddings for r in rels2])

        # Compute matrix of cosine similarities: shape (len(rels1), len(rels2))
        sim_matrix = cosine_similarity(r1_embs, r2_embs)

        # For each row i, find the best column j
        best_cols = sim_matrix.argmax(axis=1)  # best match index in rels2
        best_scores = sim_matrix.max(axis=1)   # best match score

        # Track relationships that should be removed (to avoid modifying list during iteration)
        full_relationships_match = []

        # - Matching the names of relationships -------------------------------------
        # Rename relationships in rels1 if above threshold
        for i, rel1 in enumerate(rels1):
            if best_scores[i] >= threshold:
                j = best_cols[i]
                if rels1[i].name == rels2[j].name:
                    logger.info(f"Wohoo! Relation --- [{rels1[i].name}] -- exists already in the global relationships")
                else:
                    logger.info(f"Wohoo! Relation was matched --- [{rels1[i].name}] --merged --> [{rels2[j].name}] (score={best_scores[i]:.2f})")
                # Rename rels1[i] to the best matching name in rels2[j]
                rels1[i].name = rels2[j].name
                # We do NOT change startEntity/endEntity!
        
            # - Matching relationships by timestamps -------------------------------------
            if rel1 in rels2:
                kg = KnowledgeGraph(relationships=rels2)
                rel2 = kg.get_relationship(rel1)
                   
                rel2.combine_timestamps(timestamps=rel1.properties.t_obs, temporal_aspect="t_obs")
                rel2.combine_timestamps(timestamps=rel1.properties.t_start, temporal_aspect="t_start")
                rel2.combine_timestamps(timestamps=rel1.properties.t_end, temporal_aspect="t_end")
                rel2.combine_atomic_facts(rel1.properties.atomic_facts)
                
                # Track for removal instead of removing immediately
                full_relationships_match.append(rel1)

        # Remove all relationships that were marked for removal
        for rel_to_remove in full_relationships_match:
            rels1.remove(rel_to_remove)

        kg = KnowledgeGraph(relationships=rels1+rels2)
        return rels1, kg.relationships


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
        1) Batch-match 'entities_1' to 'entities_2' using matrix-based similarity.
        2) Batch-match 'relationships_1' to 'relationships_2'.
        3) Create a mapping from old entity objects to matched entity objects.
        4) Update the relationships in relationships_1 based on that mapping.
        5) Return (global_entities, updated_relationships_2).
        """
        # Copy so we don't modify inputs in-place
        e1 = entities_1.copy()
        e2 = entities_2.copy()
        r1 = relationships_1.copy()
        r2 = relationships_2.copy()

        # -- 1) Batch-match the entities --
        matched_e1, global_entities = self._batch_match_entities(e1, e2, threshold=ent_threshold)

        # -- 2) Batch-match the relationships --
        matched_r1, _ = self._batch_match_relationships(r1, r2, threshold=rel_threshold)

        # -- 3) Build a mapping from old Entity objects => matched Entity objects
        entity_name_mapping = {}
        for old_ent, new_ent in zip(e1, matched_e1):
            if old_ent is not new_ent:
                # old_ent => new_ent
                entity_name_mapping[old_ent] = new_ent
        # -- 4) Update matched_r1's start/end references using the mapping --
        def update_relationships(rel_list: List["Relationship"]) -> List["Relationship"]:
            updated = []
            for rel in rel_list:
                # model_copy or your own "deepcopy"-like approach
                new_rel = rel.model_copy()
                

                # If startEntity or endEntity was replaced, update it
                if rel.startEntity in entity_name_mapping:
                    new_rel.startEntity = entity_name_mapping[rel.startEntity]
                if rel.endEntity in entity_name_mapping:
                    new_rel.endEntity = entity_name_mapping[rel.endEntity]
                updated.append(new_rel)
            return updated

        updated_matched_r1 = update_relationships(matched_r1)

        # We incorporate these updated relationships into the global list
        # Because 'matched_r1' references objects that might have been changed
        # We'll unify them with 'global_rels' => remove duplicates by name
        combined_relationships = r2 + updated_matched_r1

        return global_entities, combined_relationships