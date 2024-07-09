import pytest
import pickle
from itext2kg.utils import Matcher
import os

# Import current_entities.pkl
current_dir = os.path.dirname(os.path.abspath(__file__))
current_ent_path = os.path.join(current_dir, 'current_entities.pkl')
global_ent_path = os.path.join(current_dir, 'global_entities.pkl')

with open(current_ent_path, 'rb') as file:
    CURRENT_ENTITIES = pickle.load(file)

# Import global_entities.pkl
with open(global_ent_path, 'rb') as file:
    GLOBAL_ENTITIES = pickle.load(file)


GLOBAL_ENTITIES_FINAL_NAMES = ['John Doe',
 'Software Engineer',
 'XYZ Technologies',
 'San Francisco, CA',
 'Python',
 'Java',
 'B.Sc. in Computer Science',
 'Stanford University',
 'AWS',
 'Docker',
 'Kubernetes',
 'Problem Solving',
 'Team Collaboration',
 'Agile Practices',
 'Machine Learning',
 'Google Professional Data Engineer']

matcher = Matcher()

class Testor:            
    @pytest.mark.parametrize(
        "current_entities, global_entities",
        [(CURRENT_ENTITIES, GLOBAL_ENTITIES)],
        ids=["testing merging concepts"],
    )
    def test_merging_concepts(self, current_entities, global_entities):
        matched_entities, global_entities_final = matcher.process_lists(current_entities, global_entities, for_entity_or_relation='entity', threshold=0.5)
        assert(len(matched_entities) == len(current_entities))
        assert(len(global_entities_final) == len(GLOBAL_ENTITIES_FINAL_NAMES))
        assert([entity['name'] for entity in global_entities_final] == GLOBAL_ENTITIES_FINAL_NAMES)
