from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from itext2kg.llm_output_parsing.langchain_output_parser import LangchainOutputParser

from itext2kg.atom.models.relationship import Relationship

from ..logging_utils import save_snapshot_trace
from ..models.prompts import (
    render_whole_snapshot_rare_prompt,
    WHOLE_SNAPSHOT_RARE_DETECTION_PROMPT,
)
from ..models.schemas import WholeSnapshotRareResult
from ..models.signal_knowledge_graph import SignalKnowledgeGraph


def _parse_time_values(values) -> list[datetime]:
    dates: list[datetime] = []
    for value in list(values or []):
        if value in (None, "", 0, 0.0):
            continue
        if isinstance(value, (int, float)):
            try:
                dates.append(datetime.fromtimestamp(float(value), tz=timezone.utc))
            except (OSError, OverflowError, ValueError, TypeError):
                continue
            continue
        text = str(value).strip()
        if not text:
            continue
        try:
            from dateutil import parser as date_parser

            parsed = date_parser.parse(text)
        except Exception:
            continue
        if parsed is not None:
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            dates.append(parsed)
    return dates


def _format_bound(values, pick: str) -> str:
    dates = _parse_time_values(values)
    if not dates:
        return ""
    chosen = min(dates) if pick == "start" else max(dates)
    return chosen.strftime("%Y-%m-%d")


def format_quintuple(rel: Relationship) -> str:
    """Serialize one DTKG edge as (name: type) --> predicate (t_start, t_end) --> (name: type)."""
    start = rel.startEntity
    end = rel.endEntity
    t_start = _format_bound(getattr(rel.properties, "t_start", []) or [], "start")
    t_end = _format_bound(getattr(rel.properties, "t_end", []) or [], "end")
    s_lbl = start.label or "entity"
    o_lbl = end.label or "entity"
    return (
        f"({start.name}: {s_lbl}) --> {rel.name} ({t_start}, {t_end}) --> "
        f"({end.name}: {o_lbl})"
    )


class RareElementsDetector:
    def __init__(
        self,
        llm_output_parser: LangchainOutputParser,
        reports_dir: Path | None = None,
    ) -> None:
        self.parser = llm_output_parser
        self.reports_dir = reports_dir

    @staticmethod
    def _format_indexed_triple(i: int, rel: Relationship) -> str:
        dom = (
            Counter(rel.properties.domains).most_common(1)[0][0]
            if rel.properties.domains
            else "n/a"
        )
        return f"[{i}] {format_quintuple(rel)} | domain={dom}"

    @classmethod
    def format_indexed_triples(cls, relationships: list[Relationship]) -> str:
        return "\n".join(
            cls._format_indexed_triple(i, rel) for i, rel in enumerate(relationships)
        )

    async def detect(
        self,
        kg: SignalKnowledgeGraph,
        snapshot_label: str | None = None,
        central_entity_name: str | None = None,
    ) -> SignalKnowledgeGraph:
        return await self._detect_on_whole_snapshot(
            kg,
            snapshot_label=snapshot_label or "unknown",
            central_entity_name=central_entity_name or "n/a",
        )

    async def _detect_on_whole_snapshot(
        self,
        kg: SignalKnowledgeGraph,
        snapshot_label: str,
        central_entity_name: str,
    ) -> SignalKnowledgeGraph:
        if not kg.relationships:
            save_snapshot_trace(
                self.reports_dir,
                "whole_snapshot_rare",
                snapshot_label,
                prompt_text="(empty snapshot — no triples)",
                response=None,
                flagged=[],
                note="No relationships in snapshot.",
            )
            return kg

        numbered_triples = self.format_indexed_triples(kg.relationships)
        variables = {
            "snapshot_label": snapshot_label,
            "central_entity": central_entity_name,
            "numbered_triples": numbered_triples,
        }
        prompt_text = render_whole_snapshot_rare_prompt(**variables)

        results: list[WholeSnapshotRareResult] = await self.parser.batch_structured_chat_calls(
            prompt_template=WHOLE_SNAPSHOT_RARE_DETECTION_PROMPT,
            variables_list=[variables],
            output_data_structure=WholeSnapshotRareResult,
        )

        result = results[0] if results else WholeSnapshotRareResult()
        flagged: list[tuple[int, str]] = []
        rare_idx_set: set[int] = set(result.rare_indices)

        for entry in result.explanations:
            rare_idx_set.update(entry.index)

        for idx in sorted(rare_idx_set):
            if 0 <= idx < len(kg.relationships):
                kg.relationships[idx].properties.rare = True

        for entry in result.explanations:
            group_label = ", ".join(str(i) for i in entry.index)
            for idx in entry.index:
                if 0 <= idx < len(kg.relationships):
                    flagged.append(
                        (
                            idx,
                            f"group=[{group_label}] {self._format_indexed_triple(idx, kg.relationships[idx])}",
                        )
                    )

        for idx in sorted(rare_idx_set - {i for e in result.explanations for i in e.index}):
            if 0 <= idx < len(kg.relationships):
                flagged.append((idx, self._format_indexed_triple(idx, kg.relationships[idx])))

        save_snapshot_trace(
            self.reports_dir,
            "whole_snapshot_rare",
            snapshot_label,
            prompt_text=prompt_text,
            response=result,
            flagged=flagged,
        )
        return kg
