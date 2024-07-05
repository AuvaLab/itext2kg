from neo4j import GraphDatabase
import numpy as np
from typing import List

class GraphIntegrator:
    def __init__(self, uri: str, username: str, password: str):
        self.uri = uri
        self.username = username
        self.password = password
        self.driver = self.connect()

    def connect(self):
        driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))
        return driver

    def run_query(self, query: str):
        session = self.driver.session()
        try:
            session.run(query)
        finally:
            session.close()
            
    @staticmethod
    def transform_embeddings_to_str_list(embeddings:np.array):
        if embeddings is None:
            return ""
        return ",".join(list(embeddings.astype("str")))
    
    @staticmethod
    def transform_str_list_to_embeddings(embeddings:List[str]):
        if embeddings is None:
            return ""
        return np.array(embeddings.split(",")).astype(np.float64)
    
    def create_nodes(self, json_graph:dict) -> List[str]:
        queries = []
        for node in json_graph["nodes"]:
            properties = []
            for prop, value in node["properties"].items():
                if prop == "embeddings":
                    value = GraphIntegrator.transform_embeddings_to_str_list(value)
                properties.append(f'SET n.{prop.replace(" ", "_")} = "{value}"')

            query = f'CREATE (n:{node["label"]} {{name: "{node["name"]}"}}) ' + ' '.join(properties)
            queries.append(query)
        return queries

    def create_relationships(self, json_graph:dict) -> list:
        rels = []
        for rel in json_graph["relationships"]:
            property_statements = ' '.join(
            [f'SET r.{key.replace(" ", "_")} = "{value}"' if key != "embeddings" else f'SET r.{key.replace(" ", "_")} = "{GraphIntegrator.transform_embeddings_to_str_list(value)}"' for key, value in rel.get("properties", {}).items()]
            )
            
            query = (f'MATCH (n {{name: "{rel["startNode"]}"}}), (m {{name: "{rel["endNode"]}"}}) '
                     f'MERGE (n)-[r:{rel["name"]}]->(m) {property_statements}')
            rels.append(query)
            
        return rels
    

    def visualize_graph(self, json_graph:dict) -> None:
        self.connect()

        nodes, relationships = (
            self.create_nodes(json_graph=json_graph),
            self.create_relationships(json_graph=json_graph),
        )

        for node in nodes:
            self.run_query(node)

        for relation in relationships:
            self.run_query(relation)