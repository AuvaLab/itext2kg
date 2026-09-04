from pathlib import Path

from itext2kg.llm_output_parsing.langchain_output_parser import LangchainOutputParser

from ..logging_utils import save_snapshot_trace
from ..models.prompts import (
    render_weak_signal_bridging_prompt,
    WEAK_SIGNAL_BRIDGING_PROMPT,
)
from ..models.schemas import WeakSignalBridgingResult, WeakSignalExplanation
from ..models.signal_knowledge_graph import SignalKnowledgeGraph
from ..rare_elements_detector.detector import RareElementsDetector


class WeakSignalAlerter:
    def __init__(
        self,
        llm_output_parser: LangchainOutputParser,
        reports_dir: Path | None = None,
    ) -> None:
        self.parser = llm_output_parser
        self.reports_dir = reports_dir

    async def alert(
        self,
        kg: SignalKnowledgeGraph,
        previous_kgs: list[SignalKnowledgeGraph],
        snapshot_label: str | None = None,
        previous_labels: list[str] | None = None,
    ) -> SignalKnowledgeGraph:
        return await self._alert_on_bridging_subgraph(
            kg,
            previous_kgs,
            snapshot_label=snapshot_label or "unknown",
            previous_labels=previous_labels or [],
        )

    @staticmethod
    def _open_rare_indices(kg: SignalKnowledgeGraph) -> list[int]:
        """Rare triples that have not yet been consumed by a weak-signal corroboration."""
        return [
            j
            for j, r in enumerate(kg.relationships)
            if r.properties.rare and not r.properties.already_corroborated
        ]

    @staticmethod
    def _mark_already_corroborated(
        kg: SignalKnowledgeGraph,
        previous_kgs: list[SignalKnowledgeGraph],
        previous_labels: list[str],
        current_bridging: list[int],
        explanations: list[WeakSignalExplanation],
        weak_idx_set: set[int],
    ) -> None:
        """Consume cited past rares and current rare triples that fired as weak signals."""
        label_to_kg = {
            previous_labels[i] if i < len(previous_labels) else str(i): prev_kg
            for i, prev_kg in enumerate(previous_kgs)
        }

        for j in weak_idx_set:
            if 0 <= j < len(current_bridging):
                kg_idx = current_bridging[j]
                rel = kg.relationships[kg_idx]
                if rel.properties.rare:
                    rel.properties.already_corroborated = True

        for entry in explanations:
            for past_ref in entry.past_rare:
                past_kg = label_to_kg.get(past_ref.snapshot)
                if past_kg is None:
                    continue
                for idx in past_ref.index:
                    if 0 <= idx < len(past_kg.relationships):
                        rel = past_kg.relationships[idx]
                        if rel.properties.rare:
                            rel.properties.already_corroborated = True

    @staticmethod
    def _render_bridging_block(
        label: str,
        kg: SignalKnowledgeGraph,
        bridging_indices: list[int],
        *,
        zero_based: bool = False,
    ) -> str:
        lines = [f"--- Snapshot {label} ---"]
        for j, rel_idx in enumerate(bridging_indices):
            if 0 <= rel_idx < len(kg.relationships):
                display_idx = j if zero_based else rel_idx
                lines.append(
                    RareElementsDetector._format_indexed_triple(
                        display_idx, kg.relationships[rel_idx]
                    )
                )
        return "\n".join(lines)

    async def _alert_on_bridging_subgraph(
        self,
        kg: SignalKnowledgeGraph,
        previous_kgs: list[SignalKnowledgeGraph],
        snapshot_label: str,
        previous_labels: list[str],
    ) -> SignalKnowledgeGraph:
        past_blocks: list[str] = []
        for i, prev_kg in enumerate(previous_kgs):
            label = previous_labels[i] if i < len(previous_labels) else str(i)
            rare_idx = self._open_rare_indices(prev_kg)
            if not rare_idx:
                continue
            bridging_idx = prev_kg.extract_connecting_subgraph(rare_idx)
            if bridging_idx:
                past_blocks.append(
                    self._render_bridging_block(label, prev_kg, bridging_idx, zero_based=False)
                )

        current_rare = [i for i, r in enumerate(kg.relationships) if r.properties.rare]
        current_bridging = kg.extract_connecting_subgraph(current_rare)

        if not current_bridging:
            save_snapshot_trace(
                self.reports_dir,
                "weak_signal_bridging",
                snapshot_label,
                prompt_text="(no rare triples in current snapshot — skipping LLM call)",
                response=None,
                flagged=[],
                note="No rare triples in current snapshot.",
            )
            return kg

        current_bridging_text = self._render_bridging_block(
            snapshot_label,
            kg,
            current_bridging,
            zero_based=True,
        )
        past_bridging_subgraphs = (
            "\n\n".join(past_blocks) if past_blocks else "(no prior bridging subgraphs)"
        )

        variables = {
            "snapshot_label": snapshot_label,
            "past_bridging_subgraphs": past_bridging_subgraphs,
            "current_bridging_triples": current_bridging_text,
        }
        prompt_text = render_weak_signal_bridging_prompt(**variables)

        if not past_blocks:
            save_snapshot_trace(
                self.reports_dir,
                "weak_signal_bridging",
                snapshot_label,
                prompt_text=prompt_text,
                response=None,
                flagged=[],
                note="No prior snapshots with rare triples — skipping LLM call.",
            )
            return kg

        results: list[WeakSignalBridgingResult] = await self.parser.batch_structured_chat_calls(
            prompt_template=WEAK_SIGNAL_BRIDGING_PROMPT,
            variables_list=[variables],
            output_data_structure=WeakSignalBridgingResult,
        )

        result = results[0] if results else WeakSignalBridgingResult()
        flagged: list[tuple[int, str]] = []
        weak_idx_set: set[int] = set(result.weak_signal_indices)

        for entry in result.explanations:
            weak_idx_set.update(entry.index)

        for j in sorted(weak_idx_set):
            if 0 <= j < len(current_bridging):
                kg_idx = current_bridging[j]
                kg.relationships[kg_idx].properties.weak_signal_pred = True

        self._mark_already_corroborated(
            kg,
            previous_kgs,
            previous_labels,
            current_bridging,
            result.explanations,
            weak_idx_set,
        )

        explained_indices = {j for entry in result.explanations for j in entry.index}
        for entry in result.explanations:
            group_label = ", ".join(str(i) for i in entry.index)
            past_label = "; ".join(
                f"{ref.snapshot}:{ref.index}" for ref in entry.past_rare
            ) or "(none)"
            for j in entry.index:
                if 0 <= j < len(current_bridging):
                    kg_idx = current_bridging[j]
                    flagged.append(
                        (
                            kg_idx,
                            f"group=[{group_label}] past_rare=[{past_label}] bridging={j} "
                            f"{RareElementsDetector._format_indexed_triple(j, kg.relationships[kg_idx])}",
                        )
                    )

        for j in sorted(weak_idx_set - explained_indices):
            if 0 <= j < len(current_bridging):
                kg_idx = current_bridging[j]
                flagged.append(
                    (
                        kg_idx,
                        RareElementsDetector._format_indexed_triple(
                            j, kg.relationships[kg_idx]
                        ),
                    )
                )

        save_snapshot_trace(
            self.reports_dir,
            "weak_signal_bridging",
            snapshot_label,
            prompt_text=prompt_text,
            response=result,
            flagged=flagged,
        )
        return kg
