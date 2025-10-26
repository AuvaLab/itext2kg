import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple, Union
from itext2kg.itext2kg_star.models import Entity, Relationship, KnowledgeGraph
from itext2kg.itext2kg_star.graph_matching.matcher_interface import GraphMatcherInterface
from itext2kg.logging_config import get_logger

logger = get_logger(__name__)

class Matcher(GraphMatcherInterface):
    """
    Class to handle the matching and processing of entities or relations based on cosine similarity or name matching.
    """
    def __init__(self):
        pass
    
    def find_match(self, obj1: Union[Entity, Relationship], list_objects: List[Union[Entity, Relationship]], threshold: float = 0.8) -> Union[Entity, Relationship]:
        """
        Find a matching Entity or Relationship object based on name or high cosine similarity.
        :param obj1: The Entity or Relationship to find matches for.
        :param list_objects: List of Entities or Relationships to match against.
        :param match_type: Type of match to perform, either 'entity' or 'relation'.
        :param threshold: Cosine similarity threshold.
        :return: The best match or the original object if no match is found.
        """
        name1 = obj1.name
        label1 = obj1.label if isinstance(obj1, Entity) else None
        emb1 = np.array(obj1.properties.embeddings).reshape(1, -1)
        best_match = None
        best_cosine_sim = threshold

        for obj2 in list_objects:
            name2 = obj2.name
            label2 = obj2.label if isinstance(obj2, Entity) else None
            emb2 = np.array(obj2.properties.embeddings).reshape(1, -1)

            if name1 == name2 and label1 == label2:
                return obj1
            cosine_sim = cosine_similarity(emb1, emb2)[0][0]
            if cosine_sim > best_cosine_sim:
                best_cosine_sim = cosine_sim
                best_match = obj2

        if best_match:
            if isinstance(obj1, Relationship):
                logger.info("Relation was matched --- [%s] --merged --> [%s]", obj1.name, best_match.name)
                obj1.name = best_match.name
                obj1.properties.embeddings = best_match.properties.embeddings
                
            elif isinstance(obj1, Entity):
                logger.info("Entity was matched --- [%s:%s] --merged--> [%s:%s]", obj1.name, obj1.label, best_match.name, best_match.label)
                return best_match

        return obj1

    def create_union_list(self, list1: List[Union[Entity, Relationship]], list2: List[Union[Entity, Relationship]]) -> List[Union[Entity, Relationship]]:
        """
        Create a union of two lists (Entity or Relationship objects), avoiding duplicates.
        If it's a relationship, matching will be based on the relationship's name.
        If it's a Entity, matching will be based on both the Entity's name and label.
        
        :param list1: First list of Entity or Relationship objects.
        :param list2: Second list of Entity or Relationship objects.
        :return: A unified list of Entity or Relationship objects.
        """
        union_list = list1.copy()

        # Store existing names and labels in the union list
        existing_Entity_key = {(obj.name, obj.label) for obj in union_list if isinstance(obj, Entity)}
        existing_relation_names = {obj.name for obj in union_list if isinstance(obj, Relationship)}

        for obj2 in list2:
            if isinstance(obj2, Entity):
                # For Entities, check both name and label to avoid duplicates
                if (obj2.name, obj2.label) not in existing_Entity_key:
                    union_list.append(obj2)
                    existing_Entity_key.add((obj2.name, obj2.label))

            elif isinstance(obj2, Relationship):
                # For relationships, check based on the name only
                if obj2.name not in existing_relation_names:
                    union_list.append(obj2)
                    existing_relation_names.add(obj2.name)

        return union_list


    def process_lists(self, 
                      list1: List[Union[Entity, Relationship]], 
                      list2: List[Union[Entity, Relationship]], 
                      threshold: float = 0.8
                      ) -> Tuple[List[Union[Entity, Relationship]], List[Union[Entity, Relationship]]]:
        """
        Process two lists to generate new lists based on specified conditions.
        :param list1: First list to process (local items).
        :param list2: Second list to be compared against (global items).
        :param for_entity_or_relation: Specifies whether the processing is for entities or relations.
        :return: (matched_local_items, new_global_items)
        """
        list3 = [self.find_match(obj1, list2, threshold=threshold) for obj1 in list1] #matched_local_items
        list4 = self.create_union_list(list3, list2) #new_global_items
        return list3, list(set(list4))
    
    
    def match_entities_and_update_relationships(
                                                self,
                                                entities1: List[Entity],
                                                entities2: List[Entity],
                                                relationships1: List[Relationship],
                                                relationships2: List[Relationship],
                                                rel_threshold: float = 0.8,
                                                ent_threshold: float = 0.8
                                            ) -> Tuple[List[Entity], List[Relationship]]:
        """
        Match two lists of entities (Entities) and update the relationships list accordingly.
        :param entities1: First list of entities to match.
        :param entities2: Second list of entities to match against.
        :param relationships1: First list of relationships to update.
        :param relationships2: Second list of relationships to compare.
        :param rel_threshold: Cosine similarity threshold for relationships.
        :param ent_threshold: Cosine similarity threshold for entities.
        :return: Updated entities list and relationships list.
        """
        # Step 1: Match the entities and relations from both lists
        matched_entities1, global_entities = self.process_lists(entities1, entities2, ent_threshold)
        matched_relations, _ = self.process_lists(relationships1, relationships2, rel_threshold)

        # Step 2: Create a mapping from old entity names to matched entity names
        entity_name_mapping = {
            entity: matched_entity 
            for entity, matched_entity in zip(entities1, matched_entities1) 
            if entity != matched_entity
        }

        # Step 3: Update relationships based on matched entities
        def update_relationships(relationships: List[Relationship]) -> List[Relationship]:
            updated_relationships = []
            for rel in relationships:
                tem_kg = KnowledgeGraph(relationships=relationships2)
                rel2 = tem_kg.get_relationship(rel)
                if rel2:
                    rel.combine_observation_dates(rel2.properties.observation_dates)
                updated_rel = rel.model_copy()  # Create a copy to modify
                # Update the 'startEntity' and 'endEntity' names with matched entity names
                if rel.startEntity in entity_name_mapping:
                    updated_rel.startEntity = entity_name_mapping[rel.startEntity]
                if rel.endEntity in entity_name_mapping:
                    updated_rel.endEntity = entity_name_mapping[rel.endEntity]
                updated_relationships.append(updated_rel)
            return updated_relationships

        # Step 4: Extend relationships2 with updated relationships
        relationships2.extend(update_relationships(matched_relations))

        return global_entities, relationships2