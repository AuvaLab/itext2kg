import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, List, Literal, Tuple

class Matcher:
    """
    Class to handle the matching and processing of entities or relations based on cosine similarity or name matching.
    """
    def __init__(self, threshold: float = 0.8):
        """
        Initialize the Matcher with a similarity threshold.
        :param threshold: The cosine similarity threshold to consider a match.
        """
        self.threshold = threshold

    def find_match(self, obj1: Dict, list_objects: List[Dict], match_type: Literal['entity', 'relation']) -> Dict:
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
        best_cosine_sim = self.threshold

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

    def process_lists(self, list1: List[Dict], list2: List[Dict], for_entity_or_relation: Literal['entity', 'relation']) -> Tuple[List[Dict], List[Dict]]:
        """
        Process two lists to generate new lists based on specified conditions.
        :param list1: First list to process.
        :param list2: Second list to be compared against.
        :param for_entity_or_relation: Specifies whether the processing is for entities or relations.
        :return: Two processed lists.
        """
        list3 = [self.find_match(obj1, list2, for_entity_or_relation) for obj1 in list1]
        list4 = self.create_union_list(list3, list2)
        return list3, list4

