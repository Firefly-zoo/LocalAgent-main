from __future__ import annotations

from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]


def _resolve_image_path(path_like: str) -> Path:
    resolved = Path(path_like)
    if resolved.exists():
        return resolved
    if not resolved.is_absolute():
        candidate = REPO_ROOT / resolved
        if candidate.exists():
            return candidate
    return resolved


def verify_local_structure(
    image_1_path: str,
    image_2_path: str,
    box_1: list[float],
    box_2: list[float],
    focus_hint: str | None = None,
) -> dict[str, Any]:
    try:
        from evidence_tools import compare_region_structure
    except ModuleNotFoundError:
        from plugins.evidence_tools import compare_region_structure

    output = compare_region_structure(
        image_1_path=str(_resolve_image_path(image_1_path)),
        image_2_path=str(_resolve_image_path(image_2_path)),
        box_1=box_1,
        box_2=box_2,
        focus_hint=focus_hint,
    )

    status = str(output.get("status", "")).strip().lower()
    if status != "success":
        return output

    verdict = str(output.get("verdict", "uncertain")).strip().lower()
    if verdict == "compatible":
        structure_verdict = "supports_same"
    elif verdict == "conflicting":
        structure_verdict = "supports_different"
    else:
        structure_verdict = "inconclusive"

    return {
        "status": "success",
        "tool_kind": "structure",
        "structure_verdict": structure_verdict,
        "structure_confidence": output.get("confidence", "low"),
        "same_score": output.get("same_score"),
        "different_score": output.get("different_score"),
        "shape_similarity": output.get("shape_match"),
        "layout_consistency": output.get("pattern_match"),
        "contrast_gap": output.get("contrast_gap"),
        "conflict_points": output.get("key_conflicts", []),
        "observation": output.get("evidence", ""),
        "box_1": output.get("box_1"),
        "box_2": output.get("box_2"),
        "focus_hint": output.get("focus_hint", ""),
        "raw_verdict": verdict,
    }
