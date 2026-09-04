"""C-Unseen KG types: ATOM KnowledgeGraph plus analysis annotation flags."""

from typing import List

from pydantic import Field

from itext2kg.atom.models.knowledge_graph import KnowledgeGraph
from itext2kg.atom.models.relationship import Relationship, RelationshipProperties


class SignalProperties(RelationshipProperties):
    """ATOM edge properties plus C-Unseen analysis annotations."""

    rare: bool = False
    weak_signal_pred: bool = False
    already_corroborated: bool = False


class SignalRelationship(Relationship):
    properties: SignalProperties = Field(default_factory=SignalProperties)


class SignalKnowledgeGraph(KnowledgeGraph):
    relationships: List[SignalRelationship] = Field(default_factory=list)
