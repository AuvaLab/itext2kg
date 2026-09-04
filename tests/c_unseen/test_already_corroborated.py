"""Regression tests for already_corroborated bookkeeping in WeakSignalAlerter."""

from __future__ import annotations

from itext2kg.atom.models import Entity
from itext2kg.c_unseen.models.schemas import PastRareRef, WeakSignalExplanation
from itext2kg.c_unseen.models.signal_knowledge_graph import (
    SignalKnowledgeGraph,
    SignalProperties,
    SignalRelationship,
)
from itext2kg.c_unseen.weak_signal_alerter.alerter import WeakSignalAlerter


def _entity(name: str) -> Entity:
    return Entity(name=name, label="org")


def _rel(
    start: Entity,
    end: Entity,
    *,
    rare: bool = False,
    already_corroborated: bool = False,
    weak_signal_pred: bool = False,
) -> SignalRelationship:
    return SignalRelationship(
        name="relates_to",
        startEntity=start,
        endEntity=end,
        properties=SignalProperties(
            rare=rare,
            already_corroborated=already_corroborated,
            weak_signal_pred=weak_signal_pred,
        ),
    )


def _kg(rels: list[SignalRelationship]) -> SignalKnowledgeGraph:
    entities: dict[tuple[str, str], Entity] = {}
    for rel in rels:
        for ent in (rel.startEntity, rel.endEntity):
            entities[(ent.name, ent.label)] = ent
    return SignalKnowledgeGraph(
        entities=list(entities.values()),
        relationships=rels,
    )


def test_open_rare_indices_skips_already_corroborated() -> None:
    a, b, c = _entity("A"), _entity("B"), _entity("C")
    kg = _kg(
        [
            _rel(a, b, rare=True),
            _rel(b, c, rare=True, already_corroborated=True),
            _rel(a, c, rare=False),
        ]
    )
    assert WeakSignalAlerter._open_rare_indices(kg) == [0]


def test_mark_already_corroborated_consumes_current_weak_rares() -> None:
    a, b = _entity("A"), _entity("B")
    kg = _kg([_rel(a, b, rare=True), _rel(a, b, rare=True)])
    # Bridging subgraph maps display index 0 -> kg index 1
    current_bridging = [1]
    WeakSignalAlerter._mark_already_corroborated(
        kg,
        previous_kgs=[],
        previous_labels=[],
        current_bridging=current_bridging,
        explanations=[],
        weak_idx_set={0},
    )
    assert kg.relationships[0].properties.already_corroborated is False
    assert kg.relationships[1].properties.already_corroborated is True


def test_mark_already_corroborated_consumes_cited_past_rares() -> None:
    a, b = _entity("A"), _entity("B")
    past = _kg([_rel(a, b, rare=True), _rel(a, b, rare=True)])
    current = _kg([_rel(a, b, rare=True)])
    explanation = WeakSignalExplanation(
        index=[0],
        past_rare=[PastRareRef(snapshot="2015", index=[1])],
    )
    WeakSignalAlerter._mark_already_corroborated(
        current,
        previous_kgs=[past],
        previous_labels=["2015"],
        current_bridging=[0],
        explanations=[explanation],
        weak_idx_set=set(),
    )
    assert past.relationships[0].properties.already_corroborated is False
    assert past.relationships[1].properties.already_corroborated is True
    assert current.relationships[0].properties.already_corroborated is False


def test_mark_skips_non_rare_current_bridging_edges() -> None:
    a, b = _entity("A"), _entity("B")
    kg = _kg([_rel(a, b, rare=False)])
    WeakSignalAlerter._mark_already_corroborated(
        kg,
        previous_kgs=[],
        previous_labels=[],
        current_bridging=[0],
        explanations=[],
        weak_idx_set={0},
    )
    assert kg.relationships[0].properties.already_corroborated is False
