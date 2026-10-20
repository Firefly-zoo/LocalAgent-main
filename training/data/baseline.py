#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


TRAINING_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = TRAINING_ROOT.parent
PLUGIN_DIR = TRAINING_ROOT / "plugins"
for import_root in (REPO_ROOT, PLUGIN_DIR):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))


def _resolve_swift_root() -> Path:
    candidates: list[Path] = []
    env_root = os.environ.get("SWIFT_SRC_ROOT")
    if env_root:
        candidates.append(Path(env_root))
    candidates.append(REPO_ROOT / "ms-swift")
    for candidate in candidates:
        candidate = candidate.resolve()
        if (candidate / "swift").is_dir():
            return candidate
    raise FileNotFoundError("Could not locate ms-swift source root containing `swift/`.")


SWIFT_SRC_ROOT = _resolve_swift_root()
if str(SWIFT_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SWIFT_SRC_ROOT))

os.environ.setdefault("LOG_LEVEL", "WARNING")

from swift.infer_engine.protocol import InferRequest, RequestConfig  # noqa: E402
from swift.infer_engine.vllm_engine import VllmEngine  # noqa: E402
from image_utils import decode_images  # noqa: E402
from src.prompts import build_twin_direct_messages  # noqa: E402
from answer_utils import ANSWER_TAG_PATTERN, extract_final_answer  # noqa: E402

QUESTION_FALLBACK = (
    "Are these images of the same or identical products?\n"
    "Minor differences caused by background, lighting, viewpoint, crop, or photography are allowed.\n"
    "Focus on the main object and judge whether the core identity remains the same."
)

STOP_LINE_PATTERNS = (
    re.compile(r"^\s*Reason step by step in <think>.*$", re.IGNORECASE),
    re.compile(r"^\s*Then end with exactly one final decision:.*$", re.IGNORECASE),
    re.compile(r"^\s*<answer>\s*(yes|no)\s*</answer>\s*$", re.IGNORECASE),
    re.compile(r"^\s*You are a fine-grained visual comparison assistant\.\s*$", re.IGNORECASE),
    re.compile(r"^\s*Start with a <global_judgment> block\.\s*$", re.IGNORECASE),
    re.compile(r"^\s*Inside <global_judgment>.*$", re.IGNORECASE),
    re.compile(r"^\s*After <global_judgment>.*$", re.IGNORECASE),
    re.compile(r"^\s*After each tool response.*$", re.IGNORECASE),
    re.compile(r"^\s*Use at most \d+ tool calls? in total\.\s*$", re.IGNORECASE),
    re.compile(r"^\s*Use normalized coordinates.*$", re.IGNORECASE),
    re.compile(r"^\s*Always use image ids .*", re.IGNORECASE),
    re.compile(r"^\s*The only available tool is .*", re.IGNORECASE),
    re.compile(r"^\s*Write tool calls in this exact JSON form:.*$", re.IGNORECASE),
    re.compile(r"^\s*End with exactly one final decision.*$", re.IGNORECASE),
)


def _stringify_messages(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    for message in messages:
        role = str(message.get("role", "") or "").strip()
        if not role:
            continue
        content = message.get("content", "")
        text = str(content or "").strip()
        if text:
            normalized.append({"role": role, "content": text})
    return normalized


def _first_user_content(prompt: Any) -> str:
    if hasattr(prompt, "tolist"):
        prompt = prompt.tolist()
    if not isinstance(prompt, list):
        return str(prompt or "")
    for message in prompt:
        if isinstance(message, dict) and str(message.get("role", "")).lower() == "user":
            content = message.get("content", "")
            return content if isinstance(content, str) else str(content)
    if prompt and isinstance(prompt[0], dict):
        content = prompt[0].get("content", "")
        return content if isinstance(content, str) else str(content)
    return ""


def _clean_question(raw_text: str) -> str:
    text = str(raw_text or "").replace("<image>", "").strip()
    if not text:
        return QUESTION_FALLBACK
    kept_lines: list[str] = []
    for line in text.splitlines():
        if any(pattern.match(line) for pattern in STOP_LINE_PATTERNS):
            break
        kept_lines.append(line)
    cleaned = "\n".join(kept_lines).strip()
    return cleaned or QUESTION_FALLBACK


def _build_direct_prompt(prompt: Any) -> list[dict[str, str]]:
    question = _clean_question(_first_user_content(prompt))
    return [{"role": "user", "content": question}]


def _iter_split_rows(parquet_path: Path, read_batch_size: int, limit: int | None) -> Any:
    parquet_file = pq.ParquetFile(parquet_path)
    yielded = 0
    for batch in parquet_file.iter_batches(batch_size=max(1, int(read_batch_size))):
        for row in batch.to_pylist():
            if limit is not None and yielded >= limit:
                return
            yield row
            yielded += 1


def _build_record(split: str, row: dict[str, Any], completion: str) -> dict[str, Any]:
    extra_info = row.get("extra_info") or {}
    reward_model = row.get("reward_model") or {}
    source_subset = str(extra_info.get("source_subset") or "twin")
    ground_truth = str(reward_model.get("ground_truth") or "").strip().lower()
    prediction = extract_final_answer(completion)
    return {
        "sample_id": str(extra_info.get("id") or ""),
        "split": split,
        "data_source": str(extra_info.get("original_data_source") or row.get("data_source") or "twin"),
        "source_subset": source_subset,
        "tool_label": str(extra_info.get("tool_label") or ""),
        "ground_truth": ground_truth,
        "prediction": prediction,
        "is_correct": prediction == ground_truth if prediction is not None else None,
        "answer_tag_present": bool(ANSWER_TAG_PATTERN.search(completion or "")),
        "completion": completion,
    }


def _summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(records)
    valid = [record for record in records if record.get("prediction") in {"yes", "no"}]
    correct = [record for record in valid if record.get("is_correct") is True]
    return {
        "n": total,
        "acc": (len(correct) / total) if total else None,
        "pred_valid_rate": (len(valid) / total) if total else None,
        "answer_tag_rate": (
            sum(bool(record.get("answer_tag_present")) for record in records) / total if total else None
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run no-tool direct baseline inference on TWIN cot parquet splits and export baseline_aware jsonl.",
    )
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--metrics-json", type=Path, default=None)
    parser.add_argument("--base-model", type=str, required=True)
    parser.add_argument("--adapter-path", type=str, default="")
    parser.add_argument("--splits", type=str, default="train,val")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--request-chunk-size", type=int, default=4)
    parser.add_argument("--read-batch-size", type=int, default=16)
    parser.add_argument("--limit-per-split", type=int, default=None)
    parser.add_argument("--no-system-prompt", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_root = args.input_root.resolve()
    split_names = [item.strip() for item in str(args.splits).split(",") if item.strip()]
    if not split_names:
        raise ValueError("No valid splits were provided.")

    adapters = [args.adapter_path] if str(args.adapter_path or "").strip() else []
    engine_kwargs = {
        "model_id_or_path": args.base_model,
        "adapters": adapters,
        "gpu_memory_utilization": float(args.gpu_memory_utilization),
        "enforce_eager": bool(args.enforce_eager),
    }
    if int(args.tensor_parallel_size) > 1:
        engine_kwargs["tensor_parallel_size"] = int(args.tensor_parallel_size)
    engine = VllmEngine(**engine_kwargs)

    request_kwargs: dict[str, Any] = {
        "max_tokens": int(args.max_new_tokens),
        "temperature": float(args.temperature),
        "top_p": float(args.top_p),
        "stream": False,
        "n": 1,
    }
    if args.top_k is not None:
        request_kwargs["top_k"] = int(args.top_k)
    request_config = RequestConfig(**request_kwargs)

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if args.metrics_json is not None:
        args.metrics_json.parent.mkdir(parents=True, exist_ok=True)

    all_records: list[dict[str, Any]] = []
    by_split_records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_source_records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    chunk_size = max(1, int(args.request_chunk_size))
    use_system_prompt = not bool(args.no_system_prompt)

    with args.output_jsonl.open("w", encoding="utf-8") as out_f:
        for split in split_names:
            parquet_path = input_root / f"{split}_cot.parquet"
            if not parquet_path.is_file():
                raise FileNotFoundError(f"Missing split parquet: {parquet_path}")

            pending_rows: list[dict[str, Any]] = []
            processed = 0
            for row in _iter_split_rows(
                parquet_path=parquet_path,
                read_batch_size=args.read_batch_size,
                limit=args.limit_per_split,
            ):
                pending_rows.append(row)
                if len(pending_rows) < chunk_size:
                    continue

                infer_requests = []
                for pending_row in pending_rows:
                    decoded_images = decode_images(pending_row.get("images") or [])
                    raw_messages = build_twin_direct_messages(
                        _build_direct_prompt(pending_row.get("prompt") or []),
                        use_system_prompt=use_system_prompt,
                    )
                    infer_requests.append(
                        InferRequest(
                            messages=_stringify_messages(raw_messages),
                            tools=None,
                            images=decoded_images,
                        )
                    )

                responses = engine.infer(infer_requests, request_config=request_config)
                for pending_row, response in zip(pending_rows, responses):
                    completion = str(response.choices[0].message.content or "")
                    record = _build_record(split, pending_row, completion)
                    out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    all_records.append(record)
                    by_split_records[split].append(record)
                    by_source_records[str(record["data_source"])].append(record)
                    processed += 1
                out_f.flush()
                os.fsync(out_f.fileno())
                print(f"[twin-baseline-direct] split={split} processed={processed}")
                pending_rows = []

            if pending_rows:
                infer_requests = []
                for pending_row in pending_rows:
                    decoded_images = decode_images(pending_row.get("images") or [])
                    raw_messages = build_twin_direct_messages(
                        _build_direct_prompt(pending_row.get("prompt") or []),
                        use_system_prompt=use_system_prompt,
                    )
                    infer_requests.append(
                        InferRequest(
                            messages=_stringify_messages(raw_messages),
                            tools=None,
                            images=decoded_images,
                        )
                    )

                responses = engine.infer(infer_requests, request_config=request_config)
                for pending_row, response in zip(pending_rows, responses):
                    completion = str(response.choices[0].message.content or "")
                    record = _build_record(split, pending_row, completion)
                    out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    all_records.append(record)
                    by_split_records[split].append(record)
                    by_source_records[str(record["data_source"])].append(record)
                    processed += 1
                out_f.flush()
                os.fsync(out_f.fileno())
                print(f"[twin-baseline-direct] split={split} processed={processed}")

    metrics = {
        "input_root": str(input_root),
        "output_jsonl": str(args.output_jsonl.resolve()),
        "base_model": args.base_model,
        "adapter_path": str(args.adapter_path or ""),
        "splits": split_names,
        "use_system_prompt": use_system_prompt,
        "overall": _summarize(all_records),
        "by_split": {split: _summarize(records) for split, records in sorted(by_split_records.items())},
        "by_source": {source: _summarize(records) for source, records in sorted(by_source_records.items())},
        "prediction_histogram": dict(sorted(Counter(str(record.get("prediction")) for record in all_records).items())),
    }
    if args.metrics_json is not None:
        with args.metrics_json.open("w", encoding="utf-8") as handle:
            json.dump(metrics, handle, ensure_ascii=False, indent=2)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
