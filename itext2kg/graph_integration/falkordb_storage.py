from falkordb.falkordb import FalkorDB
from falkordb import Graph as FalkorGraph
import numpy as np
from typing import List, Dict, Optional, Any
from itext2kg.models import KnowledgeGraph
from itext2kg.logging_config import get_logger

logger = get_logger(__name__)

class FalkorDBStorage:
    """
    A class to integrate and manage graph data in a FalkorDB database using the official client.
    Enhanced to match Neo4j integration functionality.
    """
    def __init__(self, host: str, port: int, password: Optional[str], graph_name: str):
        """
        Initializes the FalkorDBStorage with host, port, and target graph name.

        Args:
            host (str): Hostname or IP of the FalkorDB server.
            port (int): Port for the FalkorDB server.
            password (str): Password for database access (if any).
            graph_name (str): Name of the graph to write to.
        """
        try:
            self.client = FalkorDB(host=host, port=port, password=password)
            logger.debug("Connected FalkorDB client to %s:%d", host, port)
            self.graph = FalkorGraph(self.client, graph_name)
            self.graph_name = graph_name
            logger.info("Initialized FalkorDB storage for graph: %s", graph_name)
        except Exception as e:
            logger.error("Failed to connect to FalkorDB at %s:%d - %s", host, port, e)
            raise

    def connect(self):
        """
        Returns the existing connection (FalkorDB maintains connection internally).
        
        Returns:
            The FalkorDB client instance.
        """
        return self.client

    def run_query(self, query: str):
        """
        Runs a Cypher query against the FalkorDB database.
        
        Args:
            query (str): The Cypher query to run.
            
        Returns:
            Query result from FalkorDB.
        """
        try:
            result = self.graph.query(query)
            logger.debug("Executed query successfully: %s", query[:100])
            return result
        except Exception as e:
            logger.error("Failed to execute query '%s': %s", query[:100], e)
            raise

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
        if embeddings is None or embeddings == "":
            return np.array([])
        return np.array(embeddings.split(",")).astype(np.float64)
    
    @staticmethod
    def escape_str(s: str) -> str:
        """
        Escapes double quotes and backslashes in a string for safe insertion into a Cypher query.
        """
        return s.replace('\\', '\\\\').replace('"', '\\"').replace("'", "\\'")
    
    @staticmethod
    def format_value(value: Any) -> str:
        """
        Converts a value to a string and escapes it for safe Cypher insertion.
        """
        return FalkorDBStorage.escape_str(str(value))
    
    @staticmethod
    def format_property_value(key: str, value: Any) -> str:
        """
        Formats a property value for safe Cypher insertion, handling different data types.
        
        Args:
            key (str): The property key name
            value: The property value to format
            
        Returns:
            str: A formatted string for Cypher query
        """
        if key == "embeddings":
            return f'"{FalkorDBStorage.transform_embeddings_to_str_list(value)}"'
        elif isinstance(value, list):
            if not value:  # Empty list
                return "[]"
            formatted_items = []
            for item in value:
                if isinstance(item, str):
                    escaped_item = FalkorDBStorage.escape_str(item)
                    formatted_items.append(f'"{escaped_item}"')
                elif isinstance(item, (int, float)):
                    formatted_items.append(str(item))
                else:
                    escaped_item = FalkorDBStorage.escape_str(str(item))
                    formatted_items.append(f'"{escaped_item}"')
            return f"[{', '.join(formatted_items)}]"
        elif isinstance(value, (int, float)):
            return str(value)
        elif isinstance(value, bool):
            return str(value).lower()
        else:
            return f'"{FalkorDBStorage.format_value(value)}"'

    def create_nodes(self, knowledge_graph: KnowledgeGraph) -> List[str]:
        """
        Creates or merges nodes in FalkorDB based on a KnowledgeGraph object.
        Returns list of executed queries for logging/debugging.

        Args:
            knowledge_graph (KnowledgeGraph): The graph with entities defined.
            
        Returns:
            List[str]: List of executed Cypher queries.
        """
        queries = []
        for node in knowledge_graph.entities:
            try:
                # Build properties for MERGE
                node_name = FalkorDBStorage.format_value(node.name)
                node_label = node.label
                
                # Build SET clauses for properties
                properties = []
                props = node.properties.model_dump()
                
                for prop, value in props.items():
                    if value is None:
                        continue
                    prop_key = prop.replace(" ", "_")
                    if prop == "embeddings":
                        value_str = FalkorDBStorage.transform_embeddings_to_str_list(value)
                        properties.append(f'SET n.{prop_key} = "{value_str}"')
                    elif isinstance(value, (int, float)):
                        properties.append(f'SET n.{prop_key} = {value}')
                    elif isinstance(value, list):
                        formatted_value = FalkorDBStorage.format_property_value(prop, value)
                        properties.append(f'SET n.{prop_key} = {formatted_value}')
                    else:
                        value_str = FalkorDBStorage.format_value(value)
                        properties.append(f'SET n.{prop_key} = "{value_str}"')

                # Build final query
                set_clause = ' '.join(properties) if properties else ''
                query = f'MERGE (n:{node_label} {{name: "{node_name}"}}) {set_clause}'
                
                self.run_query(query)
                queries.append(query)
                logger.debug("Created/merged node: %s:%s", node_label, node.name)
                
            except Exception as e:
                logger.error("Failed to create/merge node '%s': %s", node.name, e)
                raise
                
        return queries

    def create_relationships(self, knowledge_graph: KnowledgeGraph) -> List[str]:
        """
        Creates or merges relationships in FalkorDB based on a KnowledgeGraph object.
        Returns list of executed queries for logging/debugging.

        Args:
            knowledge_graph (KnowledgeGraph): The graph with relationships defined.
            
        Returns:
            List[str]: List of executed Cypher queries.
        """
        queries = []
        for rel in knowledge_graph.relationships:
            try:
                start = rel.startEntity
                end = rel.endEntity
                
                # Format node identifiers
                start_name = FalkorDBStorage.format_value(start.name)
                end_name = FalkorDBStorage.format_value(end.name)
                
                # Build property statements
                property_statements = []
                props = rel.properties.model_dump()
                
                for key, value in props.items():
                    if value is None:
                        continue
                    formatted_value = FalkorDBStorage.format_property_value(key, value)
                    property_key = key.replace(" ", "_")
                    property_statements.append(f'r.{property_key} = {formatted_value}')
                
                # Build SET clause
                set_clause = f'SET {", ".join(property_statements)}' if property_statements else ''
                
                # Create relationship with ON CREATE and ON MATCH clauses for property handling
                query = (
                    f'MATCH (a:{start.label} {{name: "{start_name}"}}), '
                    f'(b:{end.label} {{name: "{end_name}"}}) '
                    f'MERGE (a)-[r:{rel.name}]->(b) '
                    f'ON CREATE {set_clause} '
                    f'ON MATCH {set_clause}'
                ) if set_clause else (
                    f'MATCH (a:{start.label} {{name: "{start_name}"}}), '
                    f'(b:{end.label} {{name: "{end_name}"}}) '
                    f'MERGE (a)-[r:{rel.name}]->(b)'
                )
                
                self.run_query(query)
                queries.append(query)
                logger.debug("Created/merged relationship: %s -[%s]-> %s", 
                           start.name, rel.name, end.name)
                
            except Exception as e:
                logger.error("Failed to create/merge relationship '%s': %s", rel.name, e)
                raise
                
        return queries

    def visualize_graph(self, knowledge_graph: KnowledgeGraph, parent_node_type: str = "Document") -> None:
        """
        Runs the necessary queries to visualize a graph structure from a KnowledgeGraph input.
        Also creates HAS_ENTITY relationships between existing nodes and knowledge graph entities.
        
        Args:
            knowledge_graph (KnowledgeGraph): The KnowledgeGraph object containing the graph structure.
            parent_node_type (str): The type of parent nodes to create HAS_ENTITY relationships with.
        """
        logger.info("Visualizing graph '%s' in FalkorDB: %d nodes, %d relationships",
                    self.graph_name,
                    len(knowledge_graph.entities),
                    len(knowledge_graph.relationships))
        
        try:
            # Create nodes first
            node_queries = self.create_nodes(knowledge_graph)
            logger.info("Created %d nodes", len(node_queries))
            
            # Then create relationships
            rel_queries = self.create_relationships(knowledge_graph)
            logger.info("Created %d relationships", len(rel_queries))
            
            # Optionally create parent relationships
            if parent_node_type:
                self._create_parent_relationships(knowledge_graph, parent_node_type)
                
            logger.info("Graph visualization completed successfully")
            
        except Exception as e:
            logger.error("Failed to visualize graph: %s", e)
            raise

    def _create_parent_relationships(self, knowledge_graph: KnowledgeGraph, parent_node_type: str) -> None:
        """
        Creates HAS_ENTITY relationships between parent nodes and entities in the knowledge graph.
        
        Args:
            knowledge_graph (KnowledgeGraph): The knowledge graph containing entities
            parent_node_type (str): The label of parent nodes to link to
        """
        try:
            for entity in knowledge_graph.entities:
                entity_name = FalkorDBStorage.format_value(entity.name)
                query = (
                    f'MATCH (parent:{parent_node_type}), (entity:{entity.label} {{name: "{entity_name}"}}) '
                    f'MERGE (parent)-[r:HAS_ENTITY]->(entity)'
                )
                self.run_query(query)
                
            logger.debug("Created parent relationships with %s nodes", parent_node_type)
            
        except Exception as e:
            logger.warning("Failed to create parent relationships: %s", e)

    def clear_graph(self) -> None:
        """
        Clears all nodes and relationships from the current graph.
        """
        try:
            self.run_query("MATCH (n) DETACH DELETE n")
            logger.info("Cleared all data from graph: %s", self.graph_name)
        except Exception as e:
            logger.error("Failed to clear graph: %s", e)
            raise

    def get_graph_stats(self) -> Dict[str, int]:
        """
        Returns basic statistics about the current graph.
        
        Returns:
            Dict[str, int]: Dictionary containing node and relationship counts
        """
        try:
            # Get node count
            node_result = self.run_query("MATCH (n) RETURN count(n) as node_count")
            node_count = 0
            if hasattr(node_result, 'result_set') and node_result.result_set:
                node_count = node_result.result_set[0][0] if len(node_result.result_set) > 0 and len(node_result.result_set[0]) > 0 else 0
            
            # Get relationship count  
            rel_result = self.run_query("MATCH ()-[r]->() RETURN count(r) as rel_count")
            rel_count = 0
            if hasattr(rel_result, 'result_set') and rel_result.result_set:
                rel_count = rel_result.result_set[0][0] if len(rel_result.result_set) > 0 and len(rel_result.result_set[0]) > 0 else 0
            
            stats = {
                'nodes': int(node_count) if node_count is not None else 0,
                'relationships': int(rel_count) if rel_count is not None else 0
            }
            
            logger.info("Graph stats: %s", stats)
            return stats
            
        except Exception as e:
            logger.error("Failed to get graph statistics: %s", e)
            return {'nodes': 0, 'relationships': 0}

    def close(self) -> None:
        """
        Closes the database connection.
        """
        try:
            if hasattr(self.client, 'close'):
                self.client.close()
            logger.info("Closed FalkorDB connection")
        except Exception as e:
            logger.error("Error closing FalkorDB connection: %s", e)
