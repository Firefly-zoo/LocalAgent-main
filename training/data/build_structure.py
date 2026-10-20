#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
PLUGIN_DIR = PROJECT_ROOT / "plugins"
if str(PLUGIN_DIR) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(PLUGIN_DIR))

from structure_schema import TWIN_DINO_STRUCTURE_TOOL_FAMILY_SCHEMA_V1  # noqa: E402


def _default_input_root() -> Path:
    return REPO_ROOT / "data" / "training" / "dino_patch_mnn"


def _rewrite_jsonl(src: Path, dst: Path) -> int:
    dst.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    tool_schema_json = json.dumps(TWIN_DINO_STRUCTURE_TOOL_FAMILY_SCHEMA_V1, ensure_ascii=False)
    with src.open("r", encoding="utf-8") as reader, dst.open("w", encoding="utf-8") as writer:
        for line in reader:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            row["tools"] = tool_schema_json
            writer.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Build dino+structure tool-family RL dataset by rewriting only the tool schema field of an existing DINO RL dataset.")
    parser.add_argument("--input-root", type=Path, default=_default_input_root())
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "data" / "training",
    )
    args = parser.parse_args()

    src_root = args.input_root.resolve()
    dst_root = args.output_root.resolve()
    src_tool_dir = src_root / "ms_swift" / "tool"
    dst_tool_dir = dst_root / "ms_swift" / "tool"

    outputs = {
        "input_root": str(src_root),
        "output_root": str(dst_root),
        "rewritten_files": {},
    }

    for name in ["train.jsonl", "val.jsonl", "val_balanced.jsonl"]:
        src = src_tool_dir / name
        if not src.exists():
            continue
        dst = dst_tool_dir / name
        outputs["rewritten_files"][name] = {
            "src": str(src),
            "dst": str(dst),
            "count": _rewrite_jsonl(src, dst),
        }

    summary_path = dst_root / "build_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(outputs, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(outputs, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
