from pathlib import Path

from itext2kg.llm_output_parsing.langchain_output_parser import LangchainOutputParser

from .models.signal_knowledge_graph import SignalKnowledgeGraph
from .rare_elements_detector import RareElementsDetector
from .weak_signal_alerter import WeakSignalAlerter


class CUnseen:
    def __init__(
        self,
        llm_model,
        embeddings_model,
        reports_dir: Path | str | None = None,
    ) -> None:
        self.llm_output_parser = LangchainOutputParser(
            llm_model=llm_model,
            embeddings_model=embeddings_model,
        )
        self.reports_dir = Path(reports_dir) if reports_dir is not None else None
        self.detector = RareElementsDetector(
            self.llm_output_parser,
            reports_dir=self.reports_dir,
        )
        self.alerter = WeakSignalAlerter(
            self.llm_output_parser,
            reports_dir=self.reports_dir,
        )

    async def process_snapshot(
        self,
        kg: SignalKnowledgeGraph,
        previous_kgs: list[SignalKnowledgeGraph] | None = None,
        snapshot_label: str | None = None,
        central_entity_name: str | None = None,
        previous_labels: list[str] | None = None,
    ) -> SignalKnowledgeGraph:
        await self.detector.detect(
            kg,
            snapshot_label=snapshot_label,
            central_entity_name=central_entity_name,
        )
        if previous_kgs:
            await self.alerter.alert(
                kg,
                previous_kgs,
                snapshot_label=snapshot_label,
                previous_labels=previous_labels,
            )
        return kg

    async def process_snapshots(
        self,
        kgs: list[SignalKnowledgeGraph],
        snapshot_labels: list[str] | None = None,
        central_entity_name: str | None = None,
    ) -> list[SignalKnowledgeGraph]:
        labels = snapshot_labels or [str(i) for i in range(len(kgs))]
        for i, kg in enumerate(kgs):
            await self.process_snapshot(
                kg,
                previous_kgs=kgs[:i],
                snapshot_label=labels[i] if i < len(labels) else str(i),
                central_entity_name=central_entity_name,
                previous_labels=labels[:i],
            )
        return kgs
