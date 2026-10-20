#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "completed":
        raise ValueError(f"Evaluation is not complete: {path}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate released Pair and Mul accuracy metrics.")
    parser.add_argument("--pair", type=Path, required=True)
    parser.add_argument("--mul", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    pair = _load(args.pair)
    mul = _load(args.mul)
    result = {
        "pair": {"n": int(pair["n"]), "accuracy": float(pair["acc_plain"])},
        "mul": {"n": int(mul["n"]), "accuracy": float(mul["acc_plain"])},
        "mean_accuracy": (float(pair["acc_plain"]) + float(mul["acc_plain"])) / 2.0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
