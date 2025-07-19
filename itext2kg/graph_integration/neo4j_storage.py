from neo4j import GraphDatabase
import numpy as np
from typing import List
from itext2kg.models import KnowledgeGraph
from itext2kg.logging_config import get_logger

logger = get_logger(__name__)

class Neo4jStorage:
    """
    A class to integrate and manage graph data in a Neo4j database.
    """
    def __init__(self, uri: str, username: str, password: str):
        """
        Initializes the Neo4jStorage with database connection parameters.
        
        Args:
            uri (str): URI for the Neo4j database.
            username (str): Username for database access.
            password (str): Password for database access.
        """
        self.uri = uri
        self.username = username
        self.password = password
        self.driver = self.connect()
        
    def connect(self):
        """
        Establishes a connection to the Neo4j database.
        
        Returns:
            A Neo4j driver instance for executing queries.
        """
        driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))
        logger.debug("Created Neo4j driver: %s", driver)
        return driver

    def run_query(self, query: str):
        """
        Runs a Cypher query against the Neo4j database.
        
        Args:
            query (str): The Cypher query to run.
        """
        session = self.driver.session()
        try:
            session.run(query)
        finally:
            session.close()
            
    @staticmethod
    def transform_embeddings_to_str_list(embeddings: np.ndarray) -> str:
        """
        Transforms a NumPy array of embeddings into a comma-separated string.
        
        Args:
            embeddings (np.array): An array of embeddings.
        
        Returns:
            str: A comma-separated string of embeddings.
        """
        if embeddings is None:
            return ""
        return ",".join(list(embeddings.astype("str")))
    
    @staticmethod
    def transform_str_list_to_embeddings(embeddings: str) -> np.ndarray:
        """
        Transforms a comma-separated string of embeddings back into a NumPy array.
        
        Args:
            embeddings (str): A comma-separated string of embeddings.
        
        Returns:
            np.array: A NumPy array of embeddings.
        """
        if embeddings is None:
            return ""
        return np.array(embeddings.split(",")).astype(np.float64)
    
    @staticmethod
    def escape_str(s: str) -> str:
        """
        Escapes double quotes in a string for safe insertion into a Cypher query.
        """
        return s.replace('"', '\\"')
    
    @staticmethod
    def format_value(value) -> str:
        """
        Converts a value to a string and escapes it for safe Cypher insertion.
        """
        return Neo4jStorage.escape_str(str(value))
    
    @staticmethod
    def format_property_value(key: str, value) -> str:
        """
        Formats a property value for safe Cypher insertion, handling different data types.
        
        Args:
            key (str): The property key name
            value: The property value to format
            
        Returns:
            str: A formatted string for Cypher query
        """
        if key == "embeddings":
            return f'"{Neo4jStorage.transform_embeddings_to_str_list(value)}"'
        elif isinstance(value, list):
            # Handle list properties properly for Neo4j
            if not value:  # Empty list
                return "[]"
            # Convert list items to strings and create Neo4j list syntax
            formatted_items = []
            for item in value:
                if isinstance(item, str):
                    # Escape quotes in string items
                    escaped_item = Neo4jStorage.escape_str(item)
                    formatted_items.append(f'"{escaped_item}"')
                else:
                    # For numbers, booleans, etc.
                    formatted_items.append(str(item))
            return f"[{', '.join(formatted_items)}]"
        else:
            # Handle scalar values (strings, numbers, etc.)
            return f'"{Neo4jStorage.format_value(value)}"'

    def run_query_with_result(self, query: str):
        """
        Runs a Cypher query against the Neo4j database and returns results.
        
        Args:
            query (str): The Cypher query to run.
        
        Returns:
            List of records from the query result.
        """
        session = self.driver.session()
        try:
            result = session.run(query)
            return [record for record in result]
        finally:
            session.close()

    def create_nodes(self, knowledge_graph: KnowledgeGraph) -> List[str]:
        """
        Constructs Cypher queries for creating nodes in the graph database from a KnowledgeGraph object.
        
        Args:
            knowledge_graph (KnowledgeGraph): The KnowledgeGraph object containing entities.
        
        Returns:
            List[str]: A list of Cypher queries for node creation.
        """
        queries = []
        for node in knowledge_graph.entities:
            # Escape the node name and label if needed.
            node_name = Neo4jStorage.format_value(node.name)
            node_label = node.label  # Assuming label is already valid
            
            properties = []
            for prop, value in node.properties.model_dump().items():
                if prop == "embeddings":
                    value_str = Neo4jStorage.transform_embeddings_to_str_list(value)
                else:
                    value_str = Neo4jStorage.format_value(value)
                # Build a SET clause for each property.
                properties.append(f'SET n.{prop.replace(" ", "_")} = "{value_str}"')

            query = f'MERGE (n:{node_label} {{name: "{node_name}"}}) ' + ' '.join(properties)
            queries.append(query)
        return queries

    def create_relationships(self, knowledge_graph: KnowledgeGraph) -> List[str]:
        """
        Constructs Cypher queries for creating relationships in the graph database from a KnowledgeGraph object.
        
        Args:
            knowledge_graph (KnowledgeGraph): The KnowledgeGraph object containing relationships.
        
        Returns:
            List[str]: A list of Cypher queries for relationship creation.
        """
        rels = []
        for rel in knowledge_graph.relationships:
            # Escape start and end node names.
            start_label = rel.startEntity.label
            start_name = Neo4jStorage.format_value(rel.startEntity.name)
            end_label = rel.endEntity.label
            end_name = Neo4jStorage.format_value(rel.endEntity.name)
            rel_name = rel.name  # Assuming relationship type is valid
            
            # Build property statements for setting all properties
            property_statements = []
            for key, value in rel.properties.model_dump().items():
                formatted_value = Neo4jStorage.format_property_value(key, value)
                property_key = key.replace(" ", "_")
                property_statements.append(f'r.{property_key} = {formatted_value}')
            
            # Build SET clause for properties
            set_clause = f'SET {", ".join(property_statements)}' if property_statements else ''
            
            # Use MERGE with only relationship name for uniqueness
            # ON MATCH SET will update existing relationships with new properties
            # This prefers incoming relationship properties over existing ones
            query = (
                f'MATCH (n:{start_label} {{name: "{start_name}"}}), '
                f'(m:{end_label} {{name: "{end_name}"}}) '
                f'MERGE (n)-[r:{rel_name}]->(m) '
                f'ON CREATE {set_clause} '
                f'ON MATCH {set_clause}'
            )
            rels.append(query)
            
        return rels


    def visualize_graph(self, knowledge_graph: KnowledgeGraph, parent_node_type: str = "Hadith") -> None:
        """
        Runs the necessary queries to visualize a graph structure from a KnowledgeGraph input.
        Also creates HAS_ENTITY relationships between existing nodes and knowledge graph entities.
        
        Args:
            knowledge_graph (KnowledgeGraph): The KnowledgeGraph object containing the graph structure.
            parent_node_type (str): The type of parent nodes to create HAS_ENTITY relationships with.
        """
        nodes = self.create_nodes(knowledge_graph=knowledge_graph)
        relationships = self.create_relationships(knowledge_graph=knowledge_graph)
        has_entity_relationships = self.create_has_entity_relationships(knowledge_graph=knowledge_graph, parent_node_type=parent_node_type)
        
        for node_query in nodes:
            self.run_query(node_query)

        for rel_query in relationships:
            self.run_query(rel_query)
            
        for has_entity_query in has_entity_relationships:
            self.run_query(has_entity_query)
