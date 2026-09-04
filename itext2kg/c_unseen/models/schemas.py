from typing import List, Literal
from pydantic import BaseModel, Field


# ---------------- C-Unseen Rare/Weak Signal Structured Outputs --------------------------- #


class RareIndexExplanation(BaseModel):
    index: list[int]
    why_rare: str = ""
    monitor_priority: Literal["low", "medium", "high"] = "low"


class WholeSnapshotRareResult(BaseModel):
    baseline_pattern: str = ""
    rare_indices: list[int] = Field(default_factory=list)
    explanations: list[RareIndexExplanation] = Field(default_factory=list)


class PastRareRef(BaseModel):
    snapshot: str = Field(
        description=(
            "Past snapshot label as shown in the prompt header "
            "(e.g. '2015' from '--- Snapshot 2015 ---')."
        ),
    )
    index: list[int] = Field(
        default_factory=list,
        description=(
            "KG relationship indices as displayed in that past bridging block "
            "(the numbers in brackets). Cite only the past rare triples whose "
            "tension this weak signal develops — not connective bridging edges."
        ),
    )


class WeakSignalExplanation(BaseModel):
    index: list[int] = Field(
        description=(
            "One or more 0-based indices into the current bridging subgraph that "
            "constitute this weak signal."
        ),
    )
    past_rare: list[PastRareRef] = Field(
        default_factory=list,
        description=(
            "Past rare triples whose underlying tension this current weak signal "
            "develops. Each entry names a snapshot label and the displayed KG "
            "indices of the rare triples in that past block."
        ),
    )
    theme: str = ""
    explanation: str = ""


class WeakSignalBridgingResult(BaseModel):
    weak_signal_indices: list[int] = Field(default_factory=list)
    explanations: list[WeakSignalExplanation] = Field(default_factory=list)
