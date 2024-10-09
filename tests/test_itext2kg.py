import pytest
import pickle
from unittest.mock import patch, MagicMock
from itext2kg import iText2KG
from itext2kg.utils import Matcher
import os


current_dir = os.path.dirname(os.path.abspath(__file__))
examples_of_entities_and_relations_path = os.path.join(current_dir, 'examples_of_entities_and_relations.pkl')

with open(examples_of_entities_and_relations_path, 'rb') as file:
    ENTITIES_AND_RELATIONS = pickle.load(file)


entities_1 = ENTITIES_AND_RELATIONS[0]
entities_2 = ENTITIES_AND_RELATIONS[1]
relations_1 = ENTITIES_AND_RELATIONS[2]
relations_2 = ENTITIES_AND_RELATIONS[3]
expected_entities = ENTITIES_AND_RELATIONS[4]
expected_relations = ENTITIES_AND_RELATIONS[5]

matcher = Matcher()

@pytest.fixture
def itext2kg():
    """Fixture to initialize the iText2KG instance with mock models."""
    # Create mock models
    llm_model = MagicMock()
    embeddings_model = MagicMock()
    
    # Return the iText2KG instance
    return iText2KG(llm_model=llm_model, embeddings_model=embeddings_model)

def test_build_graph_multiple_sections_merge(itext2kg):
    """Test build_graph method for multiple sections with merging entities and relationships."""
    
    # Mock the entity extraction for two sections
    with patch.object(itext2kg.ientities_extractor, 'extract_entities', side_effect=[entities_1, entities_2]) as mock_extract_entities:
        # Mock the relation extraction for two sections
        with patch.object(itext2kg.irelations_extractor, 'extract_verify_and_correct_relations', side_effect=[relations_1, relations_2]) as mock_extract_relations:
            
                
                # Call the method under test
                result_graph = itext2kg.build_graph(sections=[
                    "Elon Musk is the CEO of SpaceX. Tesla produces electric cars.",
                    "Elon Musk leads SpaceX as its chief executive officer. Tesla Inc. manufactures electric vehicles."
                ], rel_threshold = 0.6)
                
                # Check that the mock methods were called
                assert mock_extract_entities.call_count == 2
                assert mock_extract_relations.call_count == 2

                # Assert that the resulting knowledge graph matches the expected one (merged case)
                assert set(result_graph.entities) == set(expected_entities)
                expected_relations.extend(relations_1)
                assert set(result_graph.relationships) == set(expected_relations)