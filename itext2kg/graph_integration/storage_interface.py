from typing import Protocol
from itext2kg.atom.models import KnowledgeGraph

class GraphStorageInterface(Protocol):
    """
    Interface defining the contract for storage systems.
    This interface ensures consistent behavior across different storage implementations.
    """
    
    def visualize_graph(self, knowledge_graph: KnowledgeGraph) -> None:
        """
        Visualizes the knowledge graph.
        
        Args:
            graph (KnowledgeGraph): The knowledge graph to visualize.
        """
        ...