import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, List, Literal, Tuple

class Matcher:
    """
    Class to handle the matching and processing of entities or relations based on cosine similarity or name matching.
    """
    def __init__(self):
        pass
    
    def find_match(self, obj1: Dict, list_objects: List[Dict], match_type: Literal['entity', 'relation'], threshold: float = 0.8) -> Dict:
        """
        Find a matching object based on name or high cosine similarity.
        :param obj1: The object to find matches for.
        :param list_objects: List of objects to match against.
        :param match_type: Type of match to perform, either 'entity' or 'relation'.
        :return: The best match or the original object if no match is found.
        """
        name1 = obj1['name']
        emb1 = np.array(obj1['properties']['embeddings']).reshape(1, -1)
        best_match = None
        best_cosine_sim = threshold

        for obj2 in list_objects:
            name2 = obj2['name']
            emb2 = np.array(obj2['properties']['embeddings']).reshape(1, -1)

            if name1 == name2:
                return obj1

            cosine_sim = cosine_similarity(emb1, emb2)[0][0]
            if cosine_sim > best_cosine_sim:
                best_cosine_sim = cosine_sim
                best_match = obj2

        if best_match:
            if match_type == 'relation':
                print(f"[INFO] Wohoo ! Relation using embeddings is matched --- {obj1['name']} -merged--> {best_match['name']} ")
                obj1['name'] = best_match['name']
                obj1['properties']['embeddings'] = best_match['properties']['embeddings']
                
            elif match_type == 'entity':
                print(f"[INFO] Wohoo ! Entity using embeddings is matched --- {obj1['name']} -merged--> {best_match['name']} ")
                return best_match

        return obj1

    def create_union_list(self, list1: List[Dict], list2: List[Dict]) -> List[Dict]:
        """
        Create a union of two lists, avoiding duplicates.
        :param list1: First list of dictionaries.
        :param list2: Second list of dictionaries.
        :return: A unified list of dictionaries.
        """
        union_list = list1.copy()
        existing_names = {obj['name'] for obj in union_list}

        for obj2 in list2:
            if obj2['name'] not in existing_names:
                union_list.append(obj2)

        return union_list

    def process_lists(self, list1: List[Dict], list2: List[Dict], for_entity_or_relation: Literal['entity', 'relation'], threshold: float = 0.8) -> Tuple[List[Dict], List[Dict]]:
        """
        Process two lists to generate new lists based on specified conditions.
        :param list1: First list to process (local items).
        :param list2: Second list to be compared against (global items).
        :param for_entity_or_relation: Specifies whether the processing is for entities or relations.
        :return: Two processed lists.
        """
        list3 = [self.find_match(obj1, list2, for_entity_or_relation, threshold=threshold) for obj1 in list1] #matched_local_items
        list4 = self.create_union_list(list3, list2) #new_global_items
        return list3, list4
    
    
    def match_entities_and_update_relationships(
        self,
        entities1: List[Dict],
        entities2: List[Dict],
        relationships1: List[Dict],
        relationships2: List[Dict],
        rel_threshold: float = 0.8,
        ent_threshold: float = 0.8
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Match two lists of entities and update the relationships list accordingly.
        
        :param entities1: First list of entities to match.
        :param entities2: Second list of entities to match against.
        :param relationships1: First list of relationships to update.
        :param relationships2: Second list of relationships to compare.
        :param threshold: Threshold for cosine similarity matching.
        :return: Updated entities list and relationships list.
        """
        # Step 1: Match the entities from both lists
        matched_entities1, global_entities = self.process_lists(entities1, entities2, 'entity', ent_threshold)
        matched_relations, _ = self.process_lists(relationships1, relationships2, 'relation', rel_threshold)

        # Create a mapping from old entity names to matched entity names
        entity_name_mapping = {entity['name']: matched_entity['name'] for entity, matched_entity in zip(entities1, matched_entities1) if entity['name'] != matched_entity['name']}
        
        # Step 2: Update relationships based on matched entities
        def update_relationships(relationships: List[Dict]) -> List[Dict]:
            updated_relationships = []
            for rel in relationships:
                updated_rel = rel.copy()
                # Update the 'startNode' and 'endNode' with matched entity names
                if rel['startNode'] in entity_name_mapping:
                    updated_rel['startNode'] = entity_name_mapping[rel['startNode']]
                if rel['endNode'] in entity_name_mapping:
                    updated_rel['endNode'] = entity_name_mapping[rel['endNode']]
                updated_relationships.append(updated_rel)
            return updated_relationships
        relationships2.extend(update_relationships(matched_relations))
        
        return global_entities, relationships2

