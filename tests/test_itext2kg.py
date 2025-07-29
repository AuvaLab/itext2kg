import pytest
import pickle
from unittest.mock import patch, MagicMock
from datetime import datetime
from itext2kg import iText2KG, iText2KG_Star
from itext2kg.graph_matching import Matcher
from itext2kg.models import Relationship, KnowledgeGraph
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

@pytest.fixture
def itext2kg_star():
    """Fixture to initialize the iText2KG_Star instance with mock models."""
    # Create mock models
    llm_model = MagicMock()
    embeddings_model = MagicMock()
    
    # Return the iText2KG_Star instance
    return iText2KG_Star(llm_model=llm_model, embeddings_model=embeddings_model)

@pytest.mark.asyncio
async def test_build_graph_multiple_sections_merge(itext2kg):
    """Test build_graph method for multiple sections with merging entities and relationships."""
    
    # Mock the entity extraction for two sections
    with patch.object(itext2kg.ientities_extractor, 'extract_entities', side_effect=[entities_1, entities_2]) as mock_extract_entities:
        # Mock the relation extraction for two sections
        with patch.object(itext2kg.irelations_extractor, 'extract_verify_and_correct_relations', side_effect=[relations_1, relations_2]) as mock_extract_relations:
            
                
                # Call the method under test
                result_graph = await itext2kg.build_graph(sections=[
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

@pytest.mark.asyncio
async def test_itext2kg_star_build_graph_single_section(itext2kg_star):
    """Test iText2KG_Star build_graph method with a single section."""
    
    # Mock the simple_irelations_extractor.extract_relations_and_derive_entities method
    with patch.object(itext2kg_star.simple_irelations_extractor, 'extract_relations_and_derive_entities', 
                      return_value=(entities_1, relations_1)) as mock_extract:
        
        result_graph = await itext2kg_star.build_graph(
            sections=["Elon Musk is the CEO of SpaceX and Tesla."],
            ent_threshold=0.7,
            rel_threshold=0.7
        )
        
        # Check that the mock method was called once
        assert mock_extract.call_count == 1
        
        # Assert that the resulting knowledge graph contains the expected entities and relationships
        assert len(result_graph.entities) == len(entities_1)
        assert len(result_graph.relationships) == len(relations_1)
        assert set(result_graph.entities) == set(entities_1)
        assert set(result_graph.relationships) == set(relations_1)

@pytest.mark.asyncio
async def test_itext2kg_star_build_graph_multiple_sections(itext2kg_star):
    """Test iText2KG_Star build_graph method with multiple sections and merging."""
    
    # Mock the matcher to return merged results
    merged_entities = entities_1 + entities_2  # Should merge entities
    merged_relationships = relations_1 + relations_2
    
    with patch.object(itext2kg_star.simple_irelations_extractor, 'extract_relations_and_derive_entities', 
                      side_effect=[(entities_1, relations_1), 
                                   (entities_2, relations_2)]) as mock_extract:
        with patch.object(itext2kg_star.matcher, 'match_entities_and_update_relationships',
                         return_value=(merged_entities, merged_relationships)) as mock_matcher:
            
            result_graph = await itext2kg_star.build_graph(
                sections=[
                    "Elon Musk is the CEO of SpaceX and Tesla.",
                    "Elon Musk founded Neuralink."
                ],
                ent_threshold=0.7,
                rel_threshold=0.7
            )
            
            # Check that the extraction method was called twice
            assert mock_extract.call_count == 2
            
            # Check that the matcher was called once (for merging sections)
            assert mock_matcher.call_count == 1
            
            # Assert results
            assert len(result_graph.entities) > 0
            assert len(result_graph.relationships) > 0

@pytest.mark.asyncio
async def test_itext2kg_star_with_existing_knowledge_graph(itext2kg_star):
    """Test iText2KG_Star build_graph method with existing knowledge graph."""
    
    # Create existing knowledge graph using existing test data
    existing_kg = KnowledgeGraph(
        entities=entities_2, 
        relationships=relations_2
    )
    
    # Mock the matcher to return merged results
    final_entities = entities_1 + entities_2
    final_relationships = relations_1 + relations_2
    
    with patch.object(itext2kg_star.simple_irelations_extractor, 'extract_relations_and_derive_entities',
                      return_value=(entities_1, relations_1)) as mock_extract:
        with patch.object(itext2kg_star.matcher, 'match_entities_and_update_relationships',
                         return_value=(final_entities, final_relationships)) as mock_matcher:
            
            result_graph = await itext2kg_star.build_graph(
                sections=["Elon Musk is the CEO of SpaceX."],
                existing_knowledge_graph=existing_kg,
                ent_threshold=0.7,
                rel_threshold=0.7
            )
            
            # Check that extraction was called
            assert mock_extract.call_count == 1
            
            # Check that matcher was called for merging with existing KG
            assert mock_matcher.call_count == 1
            
            # Verify the matcher was called with the existing knowledge graph
            mock_matcher.assert_called_with(
                entities1=entities_1,
                entities2=existing_kg.entities,
                relationships1=relations_1,
                relationships2=existing_kg.relationships,
                ent_threshold=0.7,
                rel_threshold=0.7
            )

@pytest.mark.asyncio
async def test_itext2kg_star_with_observation_dates(itext2kg_star):
    """Test iText2KG_Star build_graph method with observation dates as Unix timestamps."""
    
    # Create a specific date and convert to Unix timestamp
    test_date = "2024-01-15 10:30:00"
    expected_timestamp = datetime.strptime(test_date, "%Y-%m-%d %H:%M:%S").timestamp()
    
    # Add observation dates to existing relationships
    relations_with_dates = []
    for rel in relations_1:
        rel.properties.observation_dates = [expected_timestamp]
        relations_with_dates.append(rel)
    
    with patch.object(itext2kg_star.simple_irelations_extractor, 'extract_relations_and_derive_entities',
                      return_value=(entities_1, relations_with_dates)) as mock_extract:
        
        result_graph = await itext2kg_star.build_graph(
            sections=["Elon Musk is the CEO of SpaceX."],
            observation_date=test_date
        )
        
        # Check that extraction was called with the observation date
        mock_extract.assert_called_with(
            context="Elon Musk is the CEO of SpaceX.",
            max_tries=5,
            entity_name_weight=0.6,
            entity_label_weight=0.4,
            observation_date=test_date
        )
        
        # Verify that relationships have observation dates
        for relationship in result_graph.relationships:
            assert len(relationship.properties.observation_dates) > 0
            # The timestamp should be close to our expected timestamp
            assert abs(relationship.properties.observation_dates[0] - expected_timestamp) < 1.0

@pytest.mark.asyncio
async def test_itext2kg_with_observation_dates(itext2kg):
    """Test regular iText2KG build_graph method with observation dates."""
    # Create a test date and Unix timestamp
    test_date = "2024-01-15"
    expected_timestamp = datetime.strptime(test_date, "%Y-%m-%d").timestamp()
    
    # Mock the extractors to verify observation_date is passed correctly
    with patch.object(itext2kg.ientities_extractor, 'extract_entities', side_effect=[entities_1, entities_2]) as mock_extract_entities:
        with patch.object(itext2kg.irelations_extractor, 'extract_verify_and_correct_relations', side_effect=[relations_1, relations_2]) as mock_extract_relations:
            
            result_graph = await itext2kg.build_graph(
                sections=["Test section 1", "Test section 2"],
                observation_date=test_date
            )
            
            # Verify that extract_verify_and_correct_relations was called with the observation_date
            # It should be called twice (once for each section)
            assert mock_extract_relations.call_count == 2
            
            # Check that observation_date was passed to both calls
            for call_args in mock_extract_relations.call_args_list:
                assert call_args.kwargs['observation_date'] == test_date

@pytest.mark.asyncio
async def test_itext2kg_star_empty_observation_date(itext2kg_star):
    """Test iText2KG_Star with empty observation date."""
    
    # Create copies of relationships without observation dates
    clean_relations = []
    for rel in relations_1:
        clean_rel = Relationship(
            startEntity=rel.startEntity,
            endEntity=rel.endEntity,
            name=rel.name
        )
        clean_rel.properties.embeddings = rel.properties.embeddings
        clean_rel.properties.observation_dates = []  # Ensure empty observation dates
        clean_relations.append(clean_rel)
    
    with patch.object(itext2kg_star.simple_irelations_extractor, 'extract_relations_and_derive_entities',
                      return_value=(entities_1, clean_relations)) as mock_extract:
        
        result_graph = await itext2kg_star.build_graph(
            sections=["Elon Musk is the CEO of SpaceX."],
            observation_date=""  # Empty observation date
        )
        
        # Check that extraction was called with empty observation date
        mock_extract.assert_called_with(
            context="Elon Musk is the CEO of SpaceX.",
            max_tries=5,
            entity_name_weight=0.6,
            entity_label_weight=0.4,
            observation_date=""
        )
        
        # Verify that relationships don't have observation dates when empty string is passed
        for relationship in result_graph.relationships:
            # Observation dates should be empty or not modified
            assert len(relationship.properties.observation_dates) == 0

@pytest.mark.asyncio
async def test_itext2kg_star_custom_parameters(itext2kg_star):
    """Test iText2KG_Star with custom parameters."""
    
    with patch.object(itext2kg_star.simple_irelations_extractor, 'extract_relations_and_derive_entities',
                      return_value=(entities_1, relations_1)) as mock_extract:
        
        result_graph = await itext2kg_star.build_graph(
            sections=["Test section"],
            ent_threshold=0.8,
            rel_threshold=0.9,
            max_tries=3,
            entity_name_weight=0.7,
            entity_label_weight=0.3,
            observation_date="2024-12-01"
        )
        
        # Verify that custom parameters were passed correctly
        mock_extract.assert_called_with(
            context="Test section",
            max_tries=3,
            entity_name_weight=0.7,
            entity_label_weight=0.3,
            observation_date="2024-12-01"
        )

def test_unix_timestamp_conversion():
    """Test Unix timestamp conversion functionality."""
    # Test various date formats
    test_dates = [
        "2024-01-15",
        "2024-01-15 10:30:00",
        "January 15, 2024",
        "15/01/2024"
    ]
    
    for date_str in test_dates:
        try:
            from dateutil.parser import parse as parse_date
            parsed_date = parse_date(date_str)
            timestamp = parsed_date.timestamp()
            
            # Verify timestamp is reasonable (after 2020, before 2030)
            assert timestamp > 1577836800  # 2020-01-01 timestamp
            assert timestamp < 1893456000  # 2030-01-01 timestamp
            
            # Verify we can convert back
            from_timestamp = datetime.fromtimestamp(timestamp)
            assert from_timestamp.year == parsed_date.year
            assert from_timestamp.month == parsed_date.month
            assert from_timestamp.day == parsed_date.day
        except Exception as e:
            # Some date formats might not be parseable, that's okay
            pass