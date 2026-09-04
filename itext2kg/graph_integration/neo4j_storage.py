from neo4j import GraphDatabase
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from itext2kg.atom.models import KnowledgeGraph
from itext2kg.graph_integration.storage_interface import GraphStorageInterface
from itext2kg.logging_config import get_logger

logger = get_logger(__name__)

class Neo4jStorage(GraphStorageInterface):
    """
    A class to integrate and manage graph data in a Neo4j database.
    """
    def __init__(self, uri: str, username: str, password: str, database: Optional[str] = None):
        """
        Initializes the Neo4jStorage with database connection parameters.
        
        Args:
            uri (str): URI for the Neo4j database.
            username (str): Username for database access.
            password (str): Password for database access.
            database (Optional[str]): The name of the database to connect to. Defaults to None, which uses the default database.
        """
        self.uri = uri
        self.username = username
        self.password = password
        self.database = database 
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
        session = self.driver.session(database=self.database)
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
                elif isinstance(item, (int, float)):
                    # For numbers, don't add quotes
                    formatted_items.append(str(item))
                else:
                    # For other types, convert to string and escape
                    escaped_item = Neo4jStorage.escape_str(str(item))
                    formatted_items.append(f'"{escaped_item}"')
            return f"[{', '.join(formatted_items)}]"
        elif isinstance(value, (int, float)):
            # For numeric values, don't add quotes
            return str(value)
        else:
            # Handle scalar values (strings, etc.)
            return f'"{Neo4jStorage.format_value(value)}"'

    def run_query_with_result(self, query: str):
        """
        Runs a Cypher query against the Neo4j database and returns results.
        
        Args:
            query (str): The Cypher query to run.
        
        Returns:
            List of records from the query result.
        """
        session = self.driver.session(database=self.database)
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
            original_label = node.label
            node_label = Neo4jStorage.sanitize_label(node.label)
            
            # Log label sanitization for debugging
            if original_label != node_label:
                logger.info(f"Sanitized node label '{original_label}' to '{node_label}' for node '{node_name}'")
            
            properties = []
            for prop, value in node.properties.model_dump().items():
                if prop == "embeddings":
                    value_str = Neo4jStorage.transform_embeddings_to_str_list(value)
                    properties.append(f'SET n.{prop.replace(" ", "_")} = "{value_str}"')
                elif isinstance(value, (int, float)):
                    # For numeric values, don't add quotes
                    properties.append(f'SET n.{prop.replace(" ", "_")} = {value}')
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
            original_start_label = rel.startEntity.label
            original_end_label = rel.endEntity.label
            original_rel_name = rel.name
            
            start_label = Neo4jStorage.sanitize_label(rel.startEntity.label)
            start_name = Neo4jStorage.format_value(rel.startEntity.name)
            end_label = Neo4jStorage.sanitize_label(rel.endEntity.label)
            end_name = Neo4jStorage.format_value(rel.endEntity.name)
            rel_name = Neo4jStorage.sanitize_relationship_type(rel.name)
            
            # Log sanitization for debugging
            if original_start_label != start_label:
                logger.info(f"Sanitized start entity label '{original_start_label}' to '{start_label}'")
            if original_end_label != end_label:
                logger.info(f"Sanitized end entity label '{original_end_label}' to '{end_label}'")
            if original_rel_name != rel_name:
                logger.info(f"Sanitized relationship type '{original_rel_name}' to '{rel_name}'")
            
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

    @staticmethod
    def _serialize_properties(raw_properties: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize model_dump properties for Neo4j driver parameter binding."""
        properties: Dict[str, Any] = {}
        for prop, value in raw_properties.items():
            prop_key = prop.replace(" ", "_")
            if prop == "embeddings":
                if value is not None:
                    properties[prop_key] = Neo4jStorage.transform_embeddings_to_str_list(value)
                else:
                    properties[prop_key] = ""
            else:
                # Lists and scalars are passed as-is; the Neo4j driver handles them.
                properties[prop_key] = value
        return properties

    def _prepare_batched_nodes(self, knowledge_graph: KnowledgeGraph) -> Dict[str, List[Dict[str, Any]]]:
        """
        Prepares batched node data grouped by label for efficient UNWIND queries.
        
        Args:
            knowledge_graph (KnowledgeGraph): The KnowledgeGraph object containing entities.
            
        Returns:
            dict: Dictionary mapping sanitized labels to lists of node data dictionaries.
        """
        nodes_by_label: Dict[str, List[Dict[str, Any]]] = {}
        
        for entity in knowledge_graph.entities:
            sanitized_label = Neo4jStorage.sanitize_label(entity.label)
            node_data = {
                "name": entity.name,
                "properties": Neo4jStorage._serialize_properties(entity.properties.model_dump()),
            }
            
            if sanitized_label not in nodes_by_label:
                nodes_by_label[sanitized_label] = []
            nodes_by_label[sanitized_label].append(node_data)
        
        return nodes_by_label
    
    def _prepare_batched_relationships(self, knowledge_graph: KnowledgeGraph) -> Dict[str, List[Dict[str, Any]]]:
        """
        Prepares batched relationship data grouped by relationship type for efficient UNWIND queries.
        
        Args:
            knowledge_graph (KnowledgeGraph): The KnowledgeGraph object containing relationships.
            
        Returns:
            dict: Dictionary mapping sanitized relationship types to lists of relationship data dictionaries.
        """
        rels_by_type: Dict[str, List[Dict[str, Any]]] = {}
        
        for rel in knowledge_graph.relationships:
            sanitized_rel_type = Neo4jStorage.sanitize_relationship_type(rel.name)
            rel_data = {
                "startLabel": Neo4jStorage.sanitize_label(rel.startEntity.label),
                "startName": rel.startEntity.name,
                "endLabel": Neo4jStorage.sanitize_label(rel.endEntity.label),
                "endName": rel.endEntity.name,
                "properties": Neo4jStorage._serialize_properties(rel.properties.model_dump()),
            }
            
            if sanitized_rel_type not in rels_by_type:
                rels_by_type[sanitized_rel_type] = []
            rels_by_type[sanitized_rel_type].append(rel_data)
        
        return rels_by_type

    def visualize_graph(self, knowledge_graph: KnowledgeGraph, parent_node_type: str = "Hadith") -> None:
        """
        Runs the necessary queries to visualize a graph structure from a KnowledgeGraph input.
        Uses batched UNWIND queries for efficient bulk writes inside a single transaction.
        
        Args:
            knowledge_graph (KnowledgeGraph): The KnowledgeGraph object containing the graph structure.
            parent_node_type (str): Unused; kept for API compatibility.
        """
        nodes_by_label = self._prepare_batched_nodes(knowledge_graph)
        rels_by_type = self._prepare_batched_relationships(knowledge_graph)
        
        total_nodes = sum(len(nodes) for nodes in nodes_by_label.values())
        total_rels = sum(len(rels) for rels in rels_by_type.values())
        
        logger.info(
            "Preparing to write %s nodes and %s relationships using batched queries",
            total_nodes,
            total_rels,
        )
        
        with self.driver.session(database=self.database) as session:
            with session.begin_transaction() as tx:
                try:
                    for label, nodes in nodes_by_label.items():
                        query = f"""
                        UNWIND $nodes AS node
                        MERGE (n:{label} {{name: node.name}})
                        SET n += node.properties
                        """
                        tx.run(query, nodes=nodes)
                        logger.debug("Created %s nodes with label %s", len(nodes), label)
                    
                    # Labels/types must be static in Cypher, so regroup by combo.
                    rels_by_combo: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}
                    for rel_type, rels in rels_by_type.items():
                        for rel in rels:
                            combo_key = (rel["startLabel"], rel["endLabel"], rel_type)
                            if combo_key not in rels_by_combo:
                                rels_by_combo[combo_key] = []
                            rels_by_combo[combo_key].append(rel)
                    
                    for (start_label, end_label, rel_type), rels_group in rels_by_combo.items():
                        query = f"""
                        UNWIND $relationships AS rel
                        MATCH (start:{start_label} {{name: rel.startName}})
                        MATCH (end:{end_label} {{name: rel.endName}})
                        MERGE (start)-[r:{rel_type}]->(end)
                        SET r += rel.properties
                        """
                        tx.run(query, relationships=rels_group)
                        logger.debug(
                            "Created %s relationships of type %s between %s and %s",
                            len(rels_group),
                            rel_type,
                            start_label,
                            end_label,
                        )
                    
                    tx.commit()
                    logger.info(
                        "Successfully wrote %s nodes and %s relationships to Neo4j using batched queries",
                        total_nodes,
                        total_rels,
                    )
                except Exception as e:
                    logger.error("Error writing to Neo4j, transaction rolled back: %s", e)
                    raise

    @staticmethod
    def sanitize_label(label: str) -> str:
        """
        Sanitizes a label to be Neo4j compliant.
        Neo4j labels cannot start with numbers and must follow specific naming conventions.
        
        Args:
            label (str): The original label to sanitize
            
        Returns:
            str: A sanitized label that is Neo4j compliant
        """
        if not label:
            return "Entity"
        
        # Remove any non-alphanumeric characters except underscores
        sanitized = ''.join(c for c in label if c.isalnum() or c == '_')
        
        # If the label starts with a number, prefix it with 'L'
        if sanitized and sanitized[0].isdigit():
            sanitized = 'L' + sanitized
        
        # If the label is empty after sanitization, use a default
        if not sanitized:
            sanitized = "Entity"
            
        return sanitized
    
    @staticmethod
    def sanitize_relationship_type(rel_type: str) -> str:
        """
        Sanitizes a relationship type to be Neo4j compliant.
        Neo4j relationship types cannot start with numbers and must follow specific naming conventions.
        
        Args:
            rel_type (str): The original relationship type to sanitize
            
        Returns:
            str: A sanitized relationship type that is Neo4j compliant
        """
        if not rel_type:
            return "RELATES_TO"
        
        # Remove any non-alphanumeric characters except underscores
        sanitized = ''.join(c for c in rel_type if c.isalnum() or c == '_')
        
        # If the relationship type starts with a number, prefix it with 'R'
        if sanitized and sanitized[0].isdigit():
            sanitized = 'R' + sanitized
        
        # If the relationship type is empty after sanitization, use a default
        if not sanitized:
            sanitized = "RELATES_TO"
            
        return sanitized
    
    def get_sanitization_mapping(self, knowledge_graph: KnowledgeGraph) -> dict:
        """
        Returns a mapping of original labels/relationship types to their sanitized versions.
        Useful for understanding how labels will be transformed before database insertion.
        
        Args:
            knowledge_graph (KnowledgeGraph): The KnowledgeGraph object to analyze
            
        Returns:
            dict: A dictionary with 'labels' and 'relationships' keys containing the mappings
        """
        label_mapping = {}
        relationship_mapping = {}
        
        # Map entity labels
        for entity in knowledge_graph.entities:
            original = entity.label
            sanitized = Neo4jStorage.sanitize_label(original)
            if original != sanitized:
                label_mapping[original] = sanitized
        
        # Map relationship types
        for rel in knowledge_graph.relationships:
            original = rel.name
            sanitized = Neo4jStorage.sanitize_relationship_type(original)
            if original != sanitized:
                relationship_mapping[original] = sanitized
        
        return {
            'labels': label_mapping,
            'relationships': relationship_mapping
        }
