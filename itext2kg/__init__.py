from .documents_distiller import DocumentsDistiller
from .documents_distiller import DocumentsDistiller as DocumentDistiller  # Add alias for backward compatibility
from .graph_integration import Neo4jStorage as GraphIntegrator
from .graph_integration import Neo4jStorage, FalkorDBStorage  # Add both integrations
from .itext2kg import iText2KG
from .itext2kg_star import iText2KG_Star
from . import logging_config 

__all__ = ['DocumentsDistiller', 'DocumentDistiller', 'GraphIntegrator', 'Neo4jStorage', 'FalkorDBStorage', 'iText2KG', 'iText2KG_Star']