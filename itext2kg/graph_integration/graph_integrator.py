from neo4j import GraphDatabase
import numpy as np
from typing import List
from ..models import KnowledgeGraph
class GraphIntegrator:
    """
    A class to integrate and manage graph data in a Neo4j database.
    """
    def __init__(self, uri: str, username: str, password: str):
        """
        Initializes the GraphIntegrator with database connection parameters.
        
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
    def transform_embeddings_to_str_list(embeddings:np.array):
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
    def transform_str_list_to_embeddings(embeddings:List[str]):
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
    
    def create_nodes(self, knowledge_graph:KnowledgeGraph) -> List[str]:
        """
        Constructs Cypher queries for creating nodes in the graph database from a KnowledgeGraph object.
        
        Args:
        knowledge_graph (KnowledgeGraph): The KnowledgeGraph object containing entities.
        
        Returns:
        List[str]: A list of Cypher queries for node creation.
        """
        queries = []
        for node in knowledge_graph.entities:
            properties = []
            for prop, value in node.properties.model_dump().items():
                if prop == "embeddings":
                    value = GraphIntegrator.transform_embeddings_to_str_list(value)
                properties.append(f'SET n.{prop.replace(" ", "_")} = "{value}"')

            query = f'CREATE (n:{node.label} {{name: "{node.name}"}}) ' + ' '.join(properties)
            queries.append(query)
        return queries

    def create_relationships(self, knowledge_graph:KnowledgeGraph) -> list:
        """
        Constructs Cypher queries for creating relationships in the graph database from a KnowledgeGraph object.
        
        Args:
        kg (KnowledgeGraph): The KnowledgeGraph object containing relationships.
        
        Returns:
        List[str]: A list of Cypher queries for relationship creation.
        """
        rels = []
        for rel in knowledge_graph.relationships:
            property_statements = ' '.join(
            [f'SET r.{key.replace(" ", "_")} = "{value}"' 
             if key != "embeddings" 
             else f'SET r.{key.replace(" ", "_")} = "{GraphIntegrator.transform_embeddings_to_str_list(value)}"' 
             for key, value in rel.properties.model_dump().items()]
            )
            
            query = (
                f'MATCH (n:{rel.startEntity.label} {{name: "{rel.startEntity.name}"}}), '
                f'(m:{rel.endEntity.label} {{name: "{rel.endEntity.name}"}}) '
                f'MERGE (n)-[r:{rel.name}]->(m) {property_statements}'
            )
            rels.append(query)
            
        return rels
    

    def visualize_graph(self, knowledge_graph:KnowledgeGraph) -> None:
        """
        Runs the necessary queries to visualize a graph structure from a KnowledgeGraph input.
        
        Args:
        kg (KnowledgeGraph): The KnowledgeGraph object containing the graph structure.
        """
        self.connect()

        nodes, relationships = (
            self.create_nodes(knowledge_graph=knowledge_graph),
            self.create_relationships(knowledge_graph=knowledge_graph),
        )
        
        for node in nodes:
            self.run_query(node)

        for relation in relationships:
            self.run_query(relation)