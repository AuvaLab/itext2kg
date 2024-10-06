import pytest
import pickle
from itext2kg.utils import Matcher
from itext2kg.models import Entity
import os

# Import current_entities.pkl
current_dir = os.path.dirname(os.path.abspath(__file__))
current_ent_path = os.path.join(current_dir, 'current_entities_.pkl')
global_ent_path = os.path.join(current_dir, 'global_entities_.pkl')

with open(current_ent_path, 'rb') as file:
    CURRENT_ENTITIES = pickle.load(file)

# Import global_entities.pkl
with open(global_ent_path, 'rb') as file:
    GLOBAL_ENTITIES = pickle.load(file)

GLOBAL_ENTITIES_FINAL = [
    Entity(name='Python', label='Skill'),
    Entity(name='AWS', label='Certifying Body'),
    Entity(name='San Francisco, CA', label='Location'),
    Entity(name='Kubernetes', label='Skill'),
    Entity(name='Problem Solving', label='Skill'),
    Entity(name='Team Collaboration', label='Skill'),
    Entity(name='B.Sc. in Computer Science', label='Education Degree'),
    Entity(name='Machine Learning', label='Skill'),
    Entity(name='Java', label='Skill'),
    Entity(name='Docker', label='Skill'),
    Entity(name='Agile Practices', label='Skill'),
    Entity(name='Google Professional Data Engineer', label='Certification'),
    Entity(name='Stanford University', label='Educational Institution'),
    Entity(name='John Doe', label='Person'),
    Entity(name='Software Engineer', label='Job Title'),
    Entity(name='XYZ Technologies', label='Company')
]

matcher = Matcher()

class Testor:            
    @pytest.mark.parametrize(
        "current_entities, global_entities",
        [(CURRENT_ENTITIES, GLOBAL_ENTITIES)],
        ids=["testing merging concepts"],
    )
    def test_merging_concepts(self, current_entities, global_entities):
        matched_entities, global_entities_final = matcher.process_lists(current_entities, global_entities, threshold=0.5)
        assert(len(matched_entities) == len(current_entities))
        assert(len(global_entities_final) == len(GLOBAL_ENTITIES_FINAL))
        assert(set(global_entities_final) == set(GLOBAL_ENTITIES_FINAL))
