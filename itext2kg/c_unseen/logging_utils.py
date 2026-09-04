from pathlib import Path
from typing import Any

from pydantic import BaseModel


def save_snapshot_trace(
    reports_dir: Path | None,
    kind: str,
    snapshot_label: str,
    prompt_text: str,
    response: BaseModel | None,
    flagged: list[tuple[int, str]],
    note: str = "",
) -> Path | None:
    """
    Write evaluation/reports/{kind}_{snapshot_label}.txt with PROMPT / RESPONSE / FLAGGED sections.

    If reports_dir is None, skip writing and return None.
    """
    if reports_dir is None:
        return None

    reports_dir.mkdir(parents=True, exist_ok=True)
    safe_label = snapshot_label.replace("/", "_").replace(" ", "_")
    out_path = reports_dir / f"{kind}_{safe_label}.txt"

    sections: list[str] = ["=== PROMPT ===", prompt_text, ""]

    if note:
        sections.extend(["=== NOTE ===", note, ""])

    sections.append("=== RESPONSE ===")
    if response is not None:
        sections.append(response.model_dump_json(indent=2))
    else:
        sections.append("(no response)")
    sections.append("")

    sections.append("=== FLAGGED ===")
    if flagged:
        sections.extend(f"[{idx}] {triple}" for idx, triple in flagged)
    else:
        sections.append("(none)")

    out_path.write_text("\n".join(sections), encoding="utf-8")
    return out_path
