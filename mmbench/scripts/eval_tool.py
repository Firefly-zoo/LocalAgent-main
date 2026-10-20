#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[1]  # mmbench/
_REPO_ROOT = PROJECT_ROOT.parent                     # project root (LocalAgent/)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _resolve_swift_root() -> Path:
    candidates: list[Path] = []
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

from swift.infer_engine.protocol import InferRequest, RequestConfig  # noqa: E402
from swift.infer_engine.vllm_engine import VllmEngine  # noqa: E402

from plugins.geometry_tool import measure_local_geometry_consistency  # noqa: E402
from plugins.appearance_tool import measure_dino_patch_mnn  # noqa: E402
from plugins.tool_schema import TWIN_DINO_XFEAT_TOOL_FAMILY_SCHEMA_V1  # noqa: E402
from src.cache_utils import materialize_images_with_source_sample_key  # noqa: E402
from src.evaluator import (  # noqa: E402
    TOOL_CALL_PATTERN,
    TOOL_RESPONSE_PATTERN,
    _build_final_answer_reminder,
    _build_messages,
    _build_metrics_payload,
    _count_tool_calls,
    _empty_bucket,
    _extract_completion_and_rollout_info,
    _extract_extra,
    _extract_ground_truth,
    _extract_pair_id,
    _extract_plain_prediction,
    _extract_question,
    _extract_question_type,
    _extract_sample_id,
    _extract_source,
    _iter_parquet_rows,
    _tool_call_json_and_legal_counts,
    _update_bucket,
)
from src.runtime_eval import load_yaml_config  # noqa: E402


ALLOWED_TOOL_NAMES = {"measure_dino_patch_mnn", "measure_local_geometry_consistency"}


def _empty_latency_bucket() -> dict[str, float]:
    return {
        "n": 0.0,
        "latency_seconds_total": 0.0,
    }


def _update_latency_bucket(bucket: dict[str, float], latency_seconds: float) -> None:
    bucket["n"] += 1.0
    bucket["latency_seconds_total"] += float(max(0.0, latency_seconds))


def _summarize_latency_bucket(bucket: dict[str, float]) -> dict[str, float]:
    n = int(bucket["n"])
    latency_seconds_total = float(bucket["latency_seconds_total"])
    avg_latency_seconds = latency_seconds_total / n if n else 0.0
    return {
        "n": n,
        "latency_seconds_total": latency_seconds_total,
        "avg_latency_seconds": avg_latency_seconds,
        "avg_latency_ms": avg_latency_seconds * 1000.0,
    }


def _normalize_question_scope(scope: str | None) -> str:
    value = str(scope or "all").strip().lower()
    if value in {"all", "*"}:
        return "all"
    if value in {"par", "pairwise", "pair"}:
        return "pairwise"
    if value in {"mul", "mc", "multiple_choice", "multiple-choice"}:
        return "multiple_choice"
    raise ValueError(f"Unsupported question scope: {scope}")


def _scope_matches(question_type: str, scope: str) -> bool:
    if scope == "all":
        return True
    return question_type == scope


def _resolve_image_from_args(images: list[str], args: dict[str, Any], *, index_key: str, path_key: str) -> str:
    if path_key in args:
        return str(args[path_key])
    image_index = int(args[index_key])
    if image_index < 0 or image_index >= len(images):
        raise IndexError(f"`{index_key}` out of range: {image_index}")
    return str(images[image_index])


def _execute_tool(req: dict[str, Any], args: dict[str, Any], tool_name: str) -> dict[str, Any]:
    images = list(req.get("images", []))
    image_1_path = _resolve_image_from_args(images, args, index_key="image_1_index", path_key="image_1_path")
    image_2_path = _resolve_image_from_args(images, args, index_key="image_2_index", path_key="image_2_path")

    if tool_name == "measure_dino_patch_mnn":
        output = measure_dino_patch_mnn(
            image_1_path=image_1_path,
            image_2_path=image_2_path,
            box_1=args.get("box_1"),
            box_2=args.get("box_2"),
            focus_hint=args.get("focus_hint"),
        )
        return {"status": "success", "tool": tool_name, "output": output}

    if tool_name == "measure_local_geometry_consistency":
        output = measure_local_geometry_consistency(
            image_1_path=image_1_path,
            image_2_path=image_2_path,
            box_1=args.get("box_1"),
            box_2=args.get("box_2"),
            focus_hint=args.get("focus_hint"),
        )
        return {"status": "success", "tool": tool_name, "output": output}

    raise ValueError(f"Unsupported tool name: {tool_name}")


def _tool_family_infer(engine: VllmEngine, config: dict[str, Any], infer_requests: list[dict[str, Any]]) -> list[dict[str, Any]]:
    request_kwargs: dict[str, Any] = {
        "max_tokens": int(config.get("max_new_tokens", 1024)),
        "temperature": float(config.get("temperature", 0.0)),
        "top_p": float(config.get("top_p", 1.0)),
        "stream": False,
        "n": 1,
    }
    top_k = config.get("top_k")
    if top_k is not None:
        request_kwargs["top_k"] = int(top_k)
    stop = config.get("stop")
    if stop is not None:
        request_kwargs["stop"] = stop
    else:
        request_kwargs["stop"] = ["</tool_call>"]

    request_config = RequestConfig(**request_kwargs)
    max_tool_calls = max(0, int(config.get("max_tool_calls", 1)))
    max_turns = int(config.get("max_turns", max_tool_calls + 2))

    results: list[dict[str, Any]] = []
    for req in infer_requests:
        results.append({
            "messages": list(req["messages"]),
            "rollout_infos": {
                "tool_call_count": 0,
                "has_tool_response": False,
                "num_turns": 1,
                "tool_budget_exhausted": False,
                "tool_sequence": [],
            },
        })

    active_indices = list(range(len(infer_requests)))
    for turn in range(max_turns):
        if not active_indices:
            break

        batched_requests: list[InferRequest] = []
        for idx in active_indices:
            req = infer_requests[idx]
            tool_count = int(results[idx]["rollout_infos"].get("tool_call_count", 0))
            tools = req.get("tools") if tool_count < max_tool_calls else None
            batched_requests.append(InferRequest(messages=results[idx]["messages"], tools=tools, images=req.get("images", [])))

        responses = engine.infer(batched_requests, request_config=request_config)
        next_active: list[int] = []

        for i, idx in enumerate(active_indices):
            req = infer_requests[idx]
            content = responses[i].choices[0].message.content or ""
            if "<tool_call>" in content and "</tool_call>" not in content:
                content += "</tool_call>"
            results[idx]["messages"].append({"role": "assistant", "content": content})

            question_type = str(req.get("question_type", "pairwise"))
            tool_count = int(results[idx]["rollout_infos"].get("tool_call_count", 0))
            remaining_tool_budget = max(0, max_tool_calls - tool_count)
            call_match = TOOL_CALL_PATTERN.search(content)

            if call_match and remaining_tool_budget > 0:
                try:
                    payload = json.loads(call_match.group(1))
                    tool_name = str(payload.get("name", "")).strip()
                    arguments = payload.get("arguments", {})
                    if tool_name not in ALLOWED_TOOL_NAMES:
                        raise ValueError(f"Unsupported tool name: {tool_name}")
                    tool_response_payload = _execute_tool(req, arguments, tool_name)
                    if str(tool_response_payload.get("status", "")).strip().lower() == "success":
                        results[idx]["rollout_infos"]["tool_sequence"].append(tool_name)
                except Exception as exc:
                    tool_response_payload = {"status": "error", "error": str(exc)}

                tool_response = "\n<tool_response>" + json.dumps(tool_response_payload, ensure_ascii=False) + "</tool_response>"
                results[idx]["messages"][-1]["content"] += tool_response
                results[idx]["rollout_infos"]["tool_call_count"] += 1
                results[idx]["rollout_infos"]["has_tool_response"] = True
                results[idx]["rollout_infos"]["num_turns"] += 1
                results[idx]["rollout_infos"]["tool_budget_exhausted"] = True
                results[idx]["rollout_infos"]["last_tool"] = tool_response_payload.get("tool")
                last_status = str(tool_response_payload.get("status", "")).strip().lower()
                if last_status == "success":
                    nested = tool_response_payload.get("output")
                    if isinstance(nested, dict):
                        nested_status = str(nested.get("status", "")).strip().lower()
                        if nested_status:
                            last_status = nested_status
                results[idx]["rollout_infos"]["last_tool_status"] = last_status
                results[idx]["rollout_infos"]["last_tool_exec_status"] = str(tool_response_payload.get("status", "")).strip().lower()

                if turn + 1 < max_turns:
                    results[idx]["messages"].append({"role": "user", "content": _build_final_answer_reminder(question_type)})
                    next_active.append(idx)
            elif call_match:
                results[idx]["rollout_infos"]["tool_budget_exhausted"] = True
                if turn + 1 < max_turns:
                    results[idx]["messages"].append({"role": "user", "content": _build_final_answer_reminder(question_type)})
                    next_active.append(idx)

        active_indices = next_active

    for idx in range(len(results)):
        results[idx]["rollout_infos"]["num_turns"] = sum(1 for message in results[idx]["messages"] if message.get("role") == "assistant")
        full_content = "\n".join(message["content"] for message in results[idx]["messages"] if message.get("role") == "assistant")
        results[idx]["messages"].append({"role": "assistant", "content": full_content})

    return results


def _iter_filtered_rows(dataset_path: str, limit: int | None, default_question_type: str, question_scope: str) -> Iterable[dict[str, Any]]:
    for row in _iter_parquet_rows(dataset_path, limit=limit):
        answer = _extract_ground_truth(row)
        question_type = _extract_question_type(row, answer=answer, default_question_type=default_question_type)
        if _scope_matches(question_type, question_scope):
            yield row


def _count_expected_rows(dataset_path: str, limit: int | None, default_question_type: str, question_scope: str) -> int:
    if question_scope == "all" and limit is None:
        parquet_file = pq.ParquetFile(Path(dataset_path).resolve())
        return parquet_file.metadata.num_rows
    return sum(1 for _ in _iter_filtered_rows(dataset_path, limit, default_question_type, question_scope))


def evaluate_fgvqa_twin_dino_xfeat_tool_family_free(config: dict[str, Any]) -> dict[str, Any]:
    work_dir = Path(config.get("work_dir", "outputs/fgvqa_eval_twin_dino_xfeat_tool_family_free")).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    image_cache_dir = work_dir / "image_cache"

    predictions_path = work_dir / "predictions.jsonl"
    metrics_path = work_dir / "metrics.json"
    partial_metrics_path = work_dir / "metrics_partial.json"
    resolved_config_path = work_dir / "resolved_config.json"
    progress_every = int(config.get("progress_every", 50))
    request_chunk_size = max(1, int(config.get("request_chunk_size", 8)))
    question_scope = _normalize_question_scope(config.get("question_scope", "all"))

    resolved_config = dict(config)
    for key in ("dataset_path", "work_dir"):
        value = resolved_config.get(key)
        if value:
            resolved_config[key] = str(Path(value).resolve())
    resolved_config["backend"] = "ms_swift_vllm_local_twin_dino_xfeat_tool_family_free"
    resolved_config["question_scope"] = question_scope
    resolved_config["tools"] = TWIN_DINO_XFEAT_TOOL_FAMILY_SCHEMA_V1
    resolved_config_path.write_text(json.dumps(resolved_config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    base_model = config.get("base_model", "")
    assert base_model, "base_model is required in config"
    adapters = config.get("adapters", [])
    engine = VllmEngine(
        model_id_or_path=base_model,
        adapters=adapters,
        gpu_memory_utilization=float(config.get("gpu_memory_utilization", 0.9)),
    )

    system_prompt_path = config.get("system_prompt_file", "")
    system_prompt = Path(system_prompt_path).read_text(encoding="utf-8").strip() if system_prompt_path else None
    default_question_type = str(config.get("default_question_type", "pairwise"))
    tools = TWIN_DINO_XFEAT_TOOL_FAMILY_SCHEMA_V1 if bool(config.get("with_tools", True)) else None

    start_time = datetime.now(timezone.utc)
    started_at = start_time.isoformat()
    dataset_path = str(Path(config["dataset_path"]).resolve())
    limit = config.get("limit")
    expected_total = _count_expected_rows(
        dataset_path=dataset_path,
        limit=int(limit) if limit is not None else None,
        default_question_type=default_question_type,
        question_scope=question_scope,
    )

    total_bucket = _empty_bucket()
    by_source_buckets: dict[str, dict[str, float]] = {}
    by_question_type_buckets: dict[str, dict[str, float]] = {}
    by_source_question_type_buckets: dict[str, dict[str, dict[str, float]]] = {}
    latency_buckets = {
        "overall": _empty_latency_bucket(),
        "tool_called": _empty_latency_bucket(),
        "no_tool": _empty_latency_bucket(),
    }

    pending_rows: list[dict[str, Any]] = []
    pending_requests: list[dict[str, Any]] = []

    def flush_chunk(handle) -> None:
        nonlocal pending_rows, pending_requests
        if not pending_rows:
            return
        chunk_start_time = time.perf_counter()
        responses = _tool_family_infer(engine, config, pending_requests)
        chunk_elapsed_seconds = time.perf_counter() - chunk_start_time
        if len(responses) != len(pending_rows):
            raise RuntimeError(f"tool-family infer response count mismatch: got {len(responses)}, expected {len(pending_rows)}")
        per_request_latency_seconds = chunk_elapsed_seconds / max(1, len(pending_rows))

        for row, item in zip(pending_rows, responses):
            question_type = str(row["question_type"])
            completion, rollout_infos = _extract_completion_and_rollout_info(item if isinstance(item, dict) else {})
            prediction = _extract_plain_prediction(completion, question_type=question_type)
            answer = str(row["answer"]).strip().lower()
            is_correct = prediction == answer

            tool_calls = int(rollout_infos.get("tool_call_count", _count_tool_calls(completion)))
            tool_response_used = bool(rollout_infos.get("has_tool_response", bool(TOOL_RESPONSE_PATTERN.search(completion))))
            turns_used = int(rollout_infos.get("num_turns", max(1, tool_calls + 1)))
            tool_json_ok, tool_legal_ok = _tool_call_json_and_legal_counts(completion, allowed_tool_names=ALLOWED_TOOL_NAMES)

            source = str(row["dataset_source"])
            source_bucket = by_source_buckets.setdefault(source, _empty_bucket())
            question_bucket = by_question_type_buckets.setdefault(question_type, _empty_bucket())
            source_question_bucket = by_source_question_type_buckets.setdefault(source, {}).setdefault(question_type, _empty_bucket())

            kwargs = {
                "is_correct": is_correct,
                "prediction": prediction,
                "completion": completion,
                "tool_calls": tool_calls,
                "tool_response_used": tool_response_used,
                "tool_json_ok": tool_json_ok,
                "tool_legal_ok": tool_legal_ok,
                "turns_used": turns_used,
            }
            _update_bucket(total_bucket, **kwargs)
            _update_bucket(source_bucket, **kwargs)
            _update_bucket(question_bucket, **kwargs)
            _update_bucket(source_question_bucket, **kwargs)
            _update_latency_bucket(latency_buckets["overall"], per_request_latency_seconds)
            _update_latency_bucket(
                latency_buckets["tool_called" if tool_calls > 0 else "no_tool"],
                per_request_latency_seconds,
            )

            record = {
                "sample_id": row["sample_id"],
                "pair_id": row["pair_id"],
                "dataset_source": source,
                "question_type": question_type,
                "question": row["question"],
                "ground_truth": answer,
                "prediction": prediction,
                "is_correct": is_correct,
                "completion": completion,
                "num_images": len(row["images"]),
                "tool_calls": tool_calls,
                "tool_response_used": tool_response_used,
                "tool_json_ok": tool_json_ok,
                "tool_legal_ok": tool_legal_ok,
                "turns_used": turns_used,
                "latency_seconds": per_request_latency_seconds,
                "latency_ms": per_request_latency_seconds * 1000.0,
                "latency_attribution": "chunk_amortized",
                "rollout_infos": rollout_infos,
                "extra": row.get("extra"),
                "raw_response": item,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

        handle.flush()
        os.fsync(handle.fileno())

        processed = int(total_bucket["n"])
        if processed > 0 and processed % progress_every == 0:
            elapsed_seconds = (datetime.now(timezone.utc) - start_time).total_seconds()
            partial_metrics = _build_metrics_payload(
                total_bucket=total_bucket,
                by_source=by_source_buckets,
                by_question_type=by_question_type_buckets,
                by_source_question_type=by_source_question_type_buckets,
                processed=processed,
                elapsed_seconds=elapsed_seconds,
                expected_total=expected_total,
                started_at=started_at,
                config=resolved_config,
                status="running",
            )
            partial_metrics["latency_report"] = {
                "overall": _summarize_latency_bucket(latency_buckets["overall"]),
                "tool_called": _summarize_latency_bucket(latency_buckets["tool_called"]),
                "no_tool": _summarize_latency_bucket(latency_buckets["no_tool"]),
                "note": "Per-sample latency is amortized from chunk wall-clock time. Use request_chunk_size=1 for exact per-sample timing.",
            }
            partial_metrics_path.write_text(json.dumps(partial_metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        pending_rows = []
        pending_requests = []

    limit_value = int(limit) if limit is not None else None
    with predictions_path.open("w", encoding="utf-8") as handle:
        for row_index, row in enumerate(_iter_filtered_rows(dataset_path=dataset_path, limit=limit_value, default_question_type=default_question_type, question_scope=question_scope)):
            sample_id = _extract_sample_id(row, index=row_index)
            pair_id = _extract_pair_id(row, sample_id=sample_id)
            question = _extract_question(row)
            answer = _extract_ground_truth(row)
            question_type = _extract_question_type(row, answer=answer, default_question_type=default_question_type)
            dataset_source = _extract_source(row)
            extra = _extract_extra(row)
            normalized_row = {
                "sample_id": sample_id,
                "pair_id": pair_id,
                "dataset_source": dataset_source,
                "question_type": question_type,
                "question": question,
                "answer": answer,
                "images": row.get("images", []),
                "extra": extra,
            }
            image_paths = materialize_images_with_source_sample_key(normalized_row, image_cache_dir=image_cache_dir)
            messages = _build_messages(question=question, num_images=len(image_paths), system_prompt=system_prompt)
            infer_request = {
                "messages": messages,
                "images": image_paths,
                "question_type": question_type,
                "uuid": sample_id,
                "tools": tools,
            }
            pending_rows.append(normalized_row)
            pending_requests.append(infer_request)
            if len(pending_rows) >= request_chunk_size:
                flush_chunk(handle)
        flush_chunk(handle)

    final_elapsed_seconds = (datetime.now(timezone.utc) - start_time).total_seconds()
    metrics = _build_metrics_payload(
        total_bucket=total_bucket,
        by_source=by_source_buckets,
        by_question_type=by_question_type_buckets,
        by_source_question_type=by_source_question_type_buckets,
        processed=int(total_bucket["n"]),
        elapsed_seconds=final_elapsed_seconds,
        expected_total=expected_total,
        started_at=started_at,
        config=resolved_config,
        status="completed",
    )
    metrics["latency_report"] = {
        "overall": _summarize_latency_bucket(latency_buckets["overall"]),
        "tool_called": _summarize_latency_bucket(latency_buckets["tool_called"]),
        "no_tool": _summarize_latency_bucket(latency_buckets["no_tool"]),
        "note": "Per-sample latency is amortized from chunk wall-clock time. Use request_chunk_size=1 for exact per-sample timing.",
    }
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    partial_metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--adapter", default="")
    parser.add_argument("--question-scope", default="")
    parser.add_argument("--work-dir", default="")
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    if args.adapter:
        config["adapters"] = [args.adapter]
    if args.question_scope:
        config["question_scope"] = _normalize_question_scope(args.question_scope)
    if args.work_dir:
        config["work_dir"] = args.work_dir

    result = evaluate_fgvqa_twin_dino_xfeat_tool_family_free(config)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
