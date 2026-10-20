#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]  # mmbench/
_REPO_ROOT = PROJECT_ROOT.parent                     # project root (LocalAgent/)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _resolve_swift_root() -> Path:
    candidates = []
    env_root = os.environ.get("SWIFT_SRC_ROOT")
    if env_root:
        candidates.append(Path(env_root))
    candidates.extend([
        _REPO_ROOT / "ms-swift",
        _REPO_ROOT.parent / "ms-swift",
    ])
    for candidate in candidates:
        candidate = candidate.resolve()
        if (candidate / "swift").is_dir():
            return candidate
    raise FileNotFoundError("Could not locate ms-swift source root containing `swift/`.")


SWIFT_SRC_ROOT = _resolve_swift_root()
if str(SWIFT_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SWIFT_SRC_ROOT))

os.environ.setdefault("LOG_LEVEL", "WARNING")

from src.evaluator_cached import evaluate_fgvqa_swift_fixed_cache  # noqa: E402
from src.runtime_eval import load_yaml_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate twin_identity_cot on FGVQA with fixed image cache keys.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--adapter", default="", help="Optional LoRA adapter path override.")
    parser.add_argument("--work-dir", default="", help="Optional work directory override.")
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    config["with_tools"] = False
    if args.adapter:
        config["adapters"] = [args.adapter]
    if args.work_dir:
        config["work_dir"] = args.work_dir

    result = evaluate_fgvqa_swift_fixed_cache(config)
    print(result)


if __name__ == "__main__":
    main()
