from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
import uuid
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pyarrow.parquet as pq
from PIL import Image

import sys
import os
from tqdm import tqdm
from swift.infer_engine.vllm_engine import VllmEngine
from swift.infer_engine.protocol import RequestConfig, InferRequest

from src.prompts import get_twin_direct_system_prompt, get_twin_tool_system_prompt
from src.tools import compare_regions

TOOL_CALL_PATTERN = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.IGNORECASE | re.DOTALL)
TOOL_RESPONSE_PATTERN = re.compile(r"<tool_response>\s*(.*?)\s*</tool_response>", re.IGNORECASE | re.DOTALL)
FINAL_ANSWER_PATTERN = re.compile(r"<final_answer>\s*(yes|no|0|1|2|3)\s*</final_answer>", re.IGNORECASE | re.DOTALL)
ANSWER_TAG_PATTERN = re.compile(r"<answer>\s*(yes|no|0|1|2|3)\s*</answer>", re.IGNORECASE | re.DOTALL)
PLAIN_ANSWER_PATTERN = re.compile(r"\b(yes|no|0|1|2|3)\b", re.IGNORECASE)
TERMINAL_TOOL_CALL_PATTERN = re.compile(r"\s*<tool_call>\s*.*?\s*</tool_call>\s*$", re.IGNORECASE | re.DOTALL)

VALID_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _iter_parquet_rows(path: str, limit: int | None = None) -> Iterable[dict[str, Any]]:
    parquet_file = pq.ParquetFile(path)
    yielded = 0
    for batch in parquet_file.iter_batches(batch_size=1):
        for row in batch.to_pylist():
            yield row
            yielded += 1
            if limit is not None and yielded >= limit:
                return


def _default_compare_regions_tools() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "compare_regions",
                "description": (
                    "Compare two cropped regions from two images and return "
                    "side-by-side visual evidence."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "img1": {"type": "integer", "description": "Index of image 1 (0-based)."},
                        "img1_left": {"type": "number", "description": "Normalized left in [0,1]."},
                        "img1_top": {"type": "number", "description": "Normalized top in [0,1]."},
                        "img1_right": {"type": "number", "description": "Normalized right in [0,1]."},
                        "img1_bottom": {"type": "number", "description": "Normalized bottom in [0,1]."},
                        "img2": {"type": "integer", "description": "Index of image 2 (0-based)."},
                        "img2_left": {"type": "number", "description": "Normalized left in [0,1]."},
                        "img2_top": {"type": "number", "description": "Normalized top in [0,1]."},
                        "img2_right": {"type": "number", "description": "Normalized right in [0,1]."},
                        "img2_bottom": {"type": "number", "description": "Normalized bottom in [0,1]."},
                    },
                    "required": [
                        "img1",
                        "img1_left",
                        "img1_top",
                        "img1_right",
                        "img1_bottom",
                        "img2",
                        "img2_left",
                        "img2_top",
                        "img2_right",
                        "img2_bottom",
                    ],
                },
            },
        }
    ]


def _resolve_system_prompt(config: dict[str, Any], system_prompt_style: str) -> str:
    inline_prompt = config.get("system_prompt")
    if isinstance(inline_prompt, str) and inline_prompt.strip():
        return inline_prompt

    prompt_file = config.get("system_prompt_file")
    if isinstance(prompt_file, str) and prompt_file.strip():
        return Path(prompt_file).read_text(encoding="utf-8")

    if system_prompt_style == "tool":
        return get_twin_tool_system_prompt()
    return get_twin_direct_system_prompt()


def _as_non_empty_str(value: Any) -> str | None:
    if isinstance(value, str):
        candidate = value.strip()
        return candidate if candidate else None
    return None


def _extract_question(row: dict[str, Any]) -> str:
    direct = _as_non_empty_str(row.get("question"))
    if direct is not None:
        return direct

    prompt = row.get("prompt")
    if isinstance(prompt, list):
        for message in reversed(prompt):
            if not isinstance(message, dict):
                continue
            if str(message.get("role", "")).lower() != "user":
                continue
            content = _as_non_empty_str(message.get("content"))
            if content is not None:
                return content
    elif isinstance(prompt, str):
        content = _as_non_empty_str(prompt)
        if content is not None:
            return content

    return "Are these images the same object?"


def _extract_ground_truth(row: dict[str, Any]) -> str:
    for key in ("answer", "ground_truth", "label", "target"):
        text = _as_non_empty_str(row.get(key))
        if text is not None:
            return text.lower()

    reward_model = row.get("reward_model")
    if isinstance(reward_model, dict):
        text = _as_non_empty_str(reward_model.get("ground_truth"))
        if text is not None:
            return text.lower()

    return ""


def _extract_question_type(row: dict[str, Any], answer: str, default_question_type: str) -> str:
    question_type = _as_non_empty_str(row.get("question_type"))
    if question_type is not None:
        return question_type
    if answer in {"yes", "no"}:
        return "pairwise"
    if answer in {"0", "1", "2", "3"}:
        return "multiple_choice"
    return default_question_type


def _extract_sample_id(row: dict[str, Any], index: int) -> str:
    for key in ("sample_id", "pair_id", "id"):
        text = _as_non_empty_str(row.get(key))
        if text is not None:
            return text
    return f"sample_{index}"


def _extract_pair_id(row: dict[str, Any], sample_id: str) -> str:
    text = _as_non_empty_str(row.get("pair_id"))
    return text if text is not None else sample_id


def _extract_source(row: dict[str, Any]) -> str:
    for key in ("dataset_source", "data_source", "source"):
        text = _as_non_empty_str(row.get(key))
        if text is not None:
            return text
    return "unknown"


def _extract_extra(row: dict[str, Any]) -> Any:
    if "extra" in row:
        return row.get("extra")
    if "extra_info" in row:
        return row.get("extra_info")
    return None


def _resolve_tools(config: dict[str, Any]) -> list[dict[str, Any]]:
    tools = config.get("tools")
    if tools is None:
        return _default_compare_regions_tools()
    if isinstance(tools, list):
        return tools
    if isinstance(tools, str):
        tools = tools.strip()
        if not tools:
            return _default_compare_regions_tools()
        # Try JSON string first, then file path.
        if tools.startswith("[") or tools.startswith("{"):
            parsed = json.loads(tools)
            return parsed if isinstance(parsed, list) else [parsed]
        tools_path = Path(tools)
        parsed = json.loads(tools_path.read_text(encoding="utf-8"))
        return parsed if isinstance(parsed, list) else [parsed]
    raise ValueError(f"Unsupported tools value type: {type(tools)}")


def _materialize_images(
    row: dict[str, Any],
    image_cache_dir: Path,
) -> list[str]:
    image_cache_dir.mkdir(parents=True, exist_ok=True)
    sample_id = _as_non_empty_str(row.get("sample_id")) or _as_non_empty_str(row.get("id")) or "sample"
    image_paths: list[str] = []
    for image_idx, image_payload in enumerate(row.get("images", [])):
        source_path = str(image_payload.get("path", ""))
        ext = Path(source_path).suffix.lower()
        if ext not in VALID_IMAGE_EXTS:
            ext = ".jpg"
        out_path = image_cache_dir / f"{sample_id}_{image_idx}{ext}"
        if not out_path.exists():
            image_bytes = image_payload.get("bytes")
            if image_bytes is None:
                raise ValueError(f"Row {sample_id} image[{image_idx}] has no bytes payload.")
            out_path.write_bytes(image_bytes)
        image_paths.append(str(out_path))
    return image_paths


def _build_messages(
    question: str,
    num_images: int,
    system_prompt: str | None,
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    if isinstance(system_prompt, str) and system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt})

    image_prefix = "\n".join(["<image>"] * max(0, int(num_images)))
    user_text = f"{image_prefix}\n{question}".strip()
    messages.append({"role": "user", "content": user_text})
    return messages


def _build_final_answer_reminder(question_type: str) -> str:
    if question_type == "multiple_choice":
        return (
            "Tool budget exhausted. Use the available evidence and answer with exactly one final "
            "<answer>0</answer>, <answer>1</answer>, <answer>2</answer>, or <answer>3</answer>. "
            "Do not output any further <tool_call>."
        )
    return (
        "Tool budget exhausted. Use the available evidence and answer with exactly one final "
        "<answer>yes</answer> or <answer>no</answer>. Do not output any further <tool_call>."
    )

def _local_swift_infer(engine: VllmEngine, config: dict[str, Any], infer_requests: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rc = {
        "max_tokens": int(config.get("max_new_tokens", 4096)),
        "temperature": float(config.get("temperature", 0.0)),
        "top_p": float(config.get("top_p", 1.0)),
        "stream": False,
        "n": 1,
    }
    
    top_k = config.get("top_k")
    if top_k is not None:
        rc["top_k"] = int(top_k)
    stop = config.get("stop")
    if stop is not None:
        rc["stop"] = stop
    else:
        rc["stop"] = ["</tool_call>"]

    request_config = RequestConfig(**rc)
    max_tool_calls = max(0, int(config.get("max_tool_calls", 3)))
    max_turns = int(config.get("max_turns", max_tool_calls + 2))
    
    results = []
    for req in infer_requests:
        results.append({
            "messages": list(req["messages"]),
            "rollout_infos": {
                "tool_call_count": 0,
                "has_tool_response": False,
                "num_turns": 1,
                "tool_budget_exhausted": False,
            }
        })
        
    active_indices = list(range(len(infer_requests)))
    
    for turn in range(max_turns):
        if not active_indices:
            break
            
        swift_reqs = []
        for idx in active_indices:
            req = infer_requests[idx]
            tool_count = int(results[idx]["rollout_infos"].get("tool_call_count", 0))
            tools = req.get("tools") if tool_count < max_tool_calls else None
            images = req.get("images", [])
            swift_reqs.append(InferRequest(messages=results[idx]["messages"], tools=tools, images=images))
            
        resp_list = engine.infer(swift_reqs, request_config=request_config)
        
        next_active = []
        for i, idx in enumerate(active_indices):
            req = infer_requests[idx]
            content = resp_list[i].choices[0].message.content
            question_type = str(req.get("question_type", "pairwise"))
            tool_count = int(results[idx]["rollout_infos"].get("tool_call_count", 0))
            remaining_tool_budget = max(0, max_tool_calls - tool_count)
            
            if "<tool_call>" in content and "</tool_call>" not in content:
                content += "</tool_call>"
                
            results[idx]["messages"].append({"role": "assistant", "content": content})
            
            call_match = TOOL_CALL_PATTERN.search(content)
            if call_match and remaining_tool_budget > 0:
                try:
                    payload = json.loads(call_match.group(1))
                    args = payload.get("arguments", {})
                    
                    img1_idx = int(args.get("img1", 0))
                    img2_idx = int(args.get("img2", 1))
                    
                    images_paths = req.get("images", [])
                    img1 = Image.open(images_paths[img1_idx]).convert("RGB") if img1_idx < len(images_paths) else None
                    img2 = Image.open(images_paths[img2_idx]).convert("RGB") if img2_idx < len(images_paths) else None
                    
                    if img1 and img2:
                        out_img = compare_regions(
                            img1, float(args.get("img1_left", 0)), float(args.get("img1_top", 0)), float(args.get("img1_right", 1)), float(args.get("img1_bottom", 1)),
                            img2, float(args.get("img2_left", 0)), float(args.get("img2_top", 0)), float(args.get("img2_right", 1)), float(args.get("img2_bottom", 1))
                        )
                        tmp_path = os.path.join(tempfile.gettempdir(), f"tool_out_{uuid.uuid4().hex}.jpg")
                        out_img.save(tmp_path)
                        req.setdefault("images", []).append(tmp_path)
                        
                        user_content = "<image>\n<tool_response>Visual evidence extracted from compare_regions.</tool_response>"
                        if tool_count + 1 >= max_tool_calls:
                            user_content += "\n" + _build_final_answer_reminder(question_type)
                        results[idx]["messages"].append({
                            "role": "user",
                            "content": user_content,
                        })
                        results[idx]["rollout_infos"]["tool_call_count"] += 1
                        results[idx]["rollout_infos"]["has_tool_response"] = True
                        results[idx]["rollout_infos"]["num_turns"] += 1
                        if results[idx]["rollout_infos"]["tool_call_count"] >= max_tool_calls:
                            results[idx]["rollout_infos"]["tool_budget_exhausted"] = True
                        next_active.append(idx)
                    else:
                        user_content = "<tool_response>Failed to load image indices.</tool_response>"
                        if tool_count + 1 >= max_tool_calls:
                            user_content += "\n" + _build_final_answer_reminder(question_type)
                        results[idx]["messages"].append({
                            "role": "user",
                            "content": user_content,
                        })
                        results[idx]["rollout_infos"]["tool_call_count"] += 1
                        results[idx]["rollout_infos"]["has_tool_response"] = True
                        results[idx]["rollout_infos"]["num_turns"] += 1
                        if results[idx]["rollout_infos"]["tool_call_count"] >= max_tool_calls:
                            results[idx]["rollout_infos"]["tool_budget_exhausted"] = True
                        next_active.append(idx)
                        
                except Exception as e:
                    user_content = f"<tool_response>Error executing tool: {str(e)}</tool_response>"
                    if tool_count + 1 >= max_tool_calls:
                        user_content += "\n" + _build_final_answer_reminder(question_type)
                    results[idx]["messages"].append({
                        "role": "user",
                        "content": user_content,
                    })
                    results[idx]["rollout_infos"]["tool_call_count"] += 1
                    results[idx]["rollout_infos"]["has_tool_response"] = True
                    results[idx]["rollout_infos"]["num_turns"] += 1
                    if results[idx]["rollout_infos"]["tool_call_count"] >= max_tool_calls:
                        results[idx]["rollout_infos"]["tool_budget_exhausted"] = True
                    next_active.append(idx)
            elif call_match:
                results[idx]["messages"][-1]["content"] = TERMINAL_TOOL_CALL_PATTERN.sub("", content).rstrip()
                results[idx]["rollout_infos"]["tool_budget_exhausted"] = True
                if turn + 1 < max_turns:
                    results[idx]["messages"].append({
                        "role": "user",
                        "content": _build_final_answer_reminder(question_type),
                    })
                    next_active.append(idx)

        active_indices = next_active
        
    # Collapse messages into one completion string for the evaluation metric parser
    for idx in range(len(results)):
        results[idx]["rollout_infos"]["num_turns"] = sum(
            1 for msg in results[idx]["messages"] if msg.get("role") == "assistant"
        )
        full_content = "\n".join(
            msg["content"] for msg in results[idx]["messages"] if msg["role"] == "assistant"
        )
        results[idx]["messages"].append({"role": "assistant", "content": full_content})
        
    return results


def _extract_completion_and_rollout_info(item: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    rollout_infos = item.get("rollout_infos")
    if not isinstance(rollout_infos, dict):
        rollout_infos = {}

    messages = item.get("messages")
    if isinstance(messages, list):
        for message in reversed(messages):
            if isinstance(message, dict) and message.get("role") == "assistant":
                content = message.get("content")
                if isinstance(content, str):
                    return content.strip(), rollout_infos

    response = item.get("response")
    if isinstance(response, dict):
        choices = response.get("choices")
        if isinstance(choices, list) and choices:
            message = choices[0].get("message", {})
            content = message.get("content", "")
            if isinstance(content, str):
                return content.strip(), rollout_infos

    choices = item.get("choices")
    if isinstance(choices, list) and choices:
        message = choices[0].get("message", {})
        content = message.get("content", "")
        if isinstance(content, str):
            return content.strip(), rollout_infos

    return "", rollout_infos


def _has_terminal_tool_call_without_answer(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    if FINAL_ANSWER_PATTERN.search(stripped) or ANSWER_TAG_PATTERN.search(stripped):
        return False
    last_open = stripped.lower().rfind("<tool_call>")
    if last_open < 0:
        return False
    last_close = stripped.lower().rfind("</tool_call>")
    if last_close < last_open:
        return True
    tail = stripped[last_close + len("</tool_call>"):].strip()
    return not tail


def _extract_plain_prediction(text: str, question_type: str) -> str | None:
    final_matches = FINAL_ANSWER_PATTERN.findall(text)
    if final_matches:
        return final_matches[-1].lower()

    answer_matches = ANSWER_TAG_PATTERN.findall(text)
    if answer_matches:
        return answer_matches[-1].lower()

    if _has_terminal_tool_call_without_answer(text):
        return None

    plain_matches = PLAIN_ANSWER_PATTERN.findall(text)
    if not plain_matches:
        return None

    candidate = plain_matches[-1].lower()
    if question_type == "pairwise":
        return candidate if candidate in {"yes", "no"} else None
    if question_type == "multiple_choice":
        return candidate if candidate in {"0", "1", "2", "3"} else None
    return candidate


def _count_tool_calls(text: str) -> int:
    return len(TOOL_CALL_PATTERN.findall(text))


def _tool_call_json_and_legal_counts(text: str, allowed_tool_names: set[str]) -> tuple[int, int]:
    json_ok = 0
    legal_ok = 0
    for raw_tool_call in TOOL_CALL_PATTERN.findall(text):
        try:
            payload = json.loads(raw_tool_call)
        except Exception:
            continue
        json_ok += 1

        if not isinstance(payload, dict):
            continue
        name = payload.get("name")
        arguments = payload.get("arguments")
        if not isinstance(name, str) or not isinstance(arguments, dict):
            continue
        if allowed_tool_names and name not in allowed_tool_names:
            continue

        legal = True
        for key, value in arguments.items():
            if key.endswith(("left", "top", "right", "bottom")):
                if not isinstance(value, (int, float)) or not (0.0 <= float(value) <= 1.0):
                    legal = False
                    break
            elif key in {"img1", "img2", "image_1_index", "image_2_index"}:
                if not isinstance(value, int) or value < 0:
                    legal = False
                    break
            elif key in {"image_1_path", "image_2_path"}:
                if not isinstance(value, str) or not value:
                    legal = False
                    break
        if legal:
            legal_ok += 1
    return json_ok, legal_ok


def _empty_bucket() -> dict[str, float]:
    return {
        "n": 0,
        "correct": 0,
        "pred_valid": 0,
        "final_answer_tag": 0,
        "answer_tag": 0,
        "tool_call_samples": 0,
        "tool_response_samples": 0,
        "tool_call_ge2_samples": 0,
        "tool_call_total": 0,
        "tool_json_ok": 0,
        "tool_legal_ok": 0,
        "turns_total": 0.0,
    }


def _update_bucket(
    bucket: dict[str, float],
    *,
    is_correct: bool,
    prediction: str | None,
    completion: str,
    tool_calls: int,
    tool_response_used: bool,
    tool_json_ok: int,
    tool_legal_ok: int,
    turns_used: int,
) -> None:
    bucket["n"] += 1
    bucket["correct"] += 1 if is_correct else 0
    bucket["pred_valid"] += 1 if prediction is not None else 0
    bucket["final_answer_tag"] += 1 if FINAL_ANSWER_PATTERN.search(completion) else 0
    bucket["answer_tag"] += 1 if ANSWER_TAG_PATTERN.search(completion) else 0
    bucket["tool_call_samples"] += 1 if tool_calls > 0 else 0
    bucket["tool_response_samples"] += 1 if tool_response_used else 0
    bucket["tool_call_ge2_samples"] += 1 if tool_calls >= 2 else 0
    bucket["tool_call_total"] += tool_calls
    bucket["tool_json_ok"] += tool_json_ok
    bucket["tool_legal_ok"] += tool_legal_ok
    bucket["turns_total"] += float(max(1, turns_used))


def _summarize_bucket(bucket: dict[str, float]) -> dict[str, float]:
    n = int(bucket["n"])
    tool_call_total = float(bucket["tool_call_total"])
    return {
        "n": n,
        "acc_plain": float(bucket["correct"] / n) if n else 0.0,
        "pred_valid_rate_plain": float(bucket["pred_valid"] / n) if n else 0.0,
        "final_answer_tag_rate": float(bucket["final_answer_tag"] / n) if n else 0.0,
        "answer_tag_rate": float(bucket["answer_tag"] / n) if n else 0.0,
        "tool_call_rate": float(bucket["tool_call_samples"] / n) if n else 0.0,
        "tool_response_rate": float(bucket["tool_response_samples"] / n) if n else 0.0,
        "tool_call_ge2_rate": float(bucket["tool_call_ge2_samples"] / n) if n else 0.0,
        "avg_tool_calls": float(tool_call_total / n) if n else 0.0,
        "tool_json_ratio": float(bucket["tool_json_ok"] / tool_call_total) if tool_call_total else 0.0,
        "tool_legal_ratio": float(bucket["tool_legal_ok"] / tool_call_total) if tool_call_total else 0.0,
        "avg_turns_used": float(bucket["turns_total"] / n) if n else 0.0,
    }


def _build_metrics_payload(
    *,
    total_bucket: dict[str, float],
    by_source: dict[str, dict[str, float]],
    by_question_type: dict[str, dict[str, float]],
    by_source_question_type: dict[str, dict[str, dict[str, float]]],
    processed: int,
    elapsed_seconds: float,
    expected_total: int | None,
    started_at: str,
    config: dict[str, Any],
    status: str,
) -> dict[str, Any]:
    estimated_total_seconds = None
    estimated_remaining_seconds = None
    eta = None
    if expected_total and processed > 0 and elapsed_seconds > 0:
        estimated_total_seconds = elapsed_seconds * expected_total / processed
        estimated_remaining_seconds = max(0.0, estimated_total_seconds - elapsed_seconds)
        eta_ts = time.time() + estimated_remaining_seconds
        eta = datetime.fromtimestamp(eta_ts, tz=timezone.utc).isoformat()

    return {
        "status": status,
        "started_at": started_at,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "num_samples": processed,
        "elapsed_seconds": elapsed_seconds,
        "samples_per_second": float(processed / elapsed_seconds) if elapsed_seconds > 0 else 0.0,
        "estimated_total_seconds": estimated_total_seconds,
        "estimated_remaining_seconds": estimated_remaining_seconds,
        "eta": eta,
        **_summarize_bucket(total_bucket),
        "config": config,
        "by_source": {key: _summarize_bucket(bucket) for key, bucket in sorted(by_source.items())},
        "by_question_type": {key: _summarize_bucket(bucket) for key, bucket in sorted(by_question_type.items())},
        "by_source_question_type": {
            source: {
                question_type: _summarize_bucket(bucket)
                for question_type, bucket in sorted(question_types.items())
            }
            for source, question_types in sorted(by_source_question_type.items())
        },
    }


def evaluate_fgvqa_swift(config: dict[str, Any]) -> dict[str, Any]:
    work_dir = Path(config.get("work_dir", "outputs/fgvqa_eval_rollout")).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    image_cache_dir = work_dir / "image_cache"

    predictions_path = work_dir / "predictions.jsonl"
    metrics_path = work_dir / "metrics.json"
    partial_metrics_path = work_dir / "metrics_partial.json"
    resolved_config_path = work_dir / "resolved_config.json"
    progress_every = int(config.get("progress_every", 10))
    request_chunk_size = int(config.get("request_chunk_size", 8))
    request_chunk_size = max(1, request_chunk_size)

    resolved_config = dict(config)
    for key in ("dataset_path", "work_dir"):
        value = resolved_config.get(key)
        if value:
            resolved_config[key] = str(Path(value).resolve())
    resolved_config["backend"] = "ms_swift_vllm_local"
    resolved_config_path.write_text(
        json.dumps(resolved_config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    
    base_model = config.get("base_model", "")
    assert base_model, "base_model is required in config"
    adapters = config.get("adapters", [])
    
    print(f"Loading VllmEngine with model={base_model}, adapters={adapters}")
    engine = VllmEngine(
        model_id_or_path=base_model,
        adapters=adapters,
        gpu_memory_utilization=0.9,
    )

    tools = _resolve_tools(config) if bool(config.get("with_tools", True)) else None
    if "allowed_tool_names" in config and config.get("allowed_tool_names") is not None:
        allowed_tool_names = {str(name) for name in config.get("allowed_tool_names", [])}
    elif tools:
        allowed_tool_names = {
            str(tool.get("function", {}).get("name"))
            for tool in tools
            if isinstance(tool, dict) and isinstance(tool.get("function"), dict)
        }
    else:
        allowed_tool_names = set()

    prepend_system_prompt = bool(config.get("prepend_system_prompt", config.get("use_system_prompt", False)))
    system_prompt_style = str(config.get("system_prompt_style", "tool"))
    system_prompt = _resolve_system_prompt(config, system_prompt_style) if prepend_system_prompt else None
    default_question_type = str(config.get("default_question_type", "pairwise"))

    start_time = time.perf_counter()
    started_at = datetime.now(timezone.utc).isoformat()
    dataset_path = Path(config["dataset_path"]).resolve()
    parquet_file = pq.ParquetFile(dataset_path)
    dataset_total = parquet_file.metadata.num_rows
    limit = config.get("limit")
    expected_total = min(dataset_total, int(limit)) if limit is not None else dataset_total

    total_bucket = _empty_bucket()
    by_source_buckets: dict[str, dict[str, float]] = {}
    by_question_type_buckets: dict[str, dict[str, float]] = {}
    by_source_question_type_buckets: dict[str, dict[str, dict[str, float]]] = {}

    pending_rows: list[dict[str, Any]] = []
    pending_requests: list[dict[str, Any]] = []

    def flush_chunk(handle) -> None:
        nonlocal pending_rows, pending_requests
        if not pending_rows:
            return
        responses = _local_swift_infer(engine, config, pending_requests)
        if len(responses) != len(pending_rows):
            raise RuntimeError(
                f"swift _local_swift_infer response count mismatch: got {len(responses)}, expected {len(pending_rows)}"
            )

        for row, item in zip(pending_rows, responses):
            question_type = str(row["question_type"])
            completion, rollout_infos = _extract_completion_and_rollout_info(item if isinstance(item, dict) else {})

            prediction = _extract_plain_prediction(completion, question_type=question_type)
            answer = str(row["answer"]).strip().lower()
            is_correct = prediction == answer

            tool_calls = int(rollout_infos.get("tool_call_count", _count_tool_calls(completion)))
            tool_response_used = bool(
                rollout_infos.get("has_tool_response", bool(TOOL_RESPONSE_PATTERN.search(completion)))
            )
            turns_used = int(rollout_infos.get("num_turns", max(1, tool_calls + 1)))
            tool_json_ok, tool_legal_ok = _tool_call_json_and_legal_counts(
                completion,
                allowed_tool_names=allowed_tool_names,
            )

            source = str(row["dataset_source"])
            source_bucket = by_source_buckets.setdefault(source, _empty_bucket())
            question_bucket = by_question_type_buckets.setdefault(question_type, _empty_bucket())
            source_question_bucket = by_source_question_type_buckets.setdefault(source, {}).setdefault(
                question_type,
                _empty_bucket(),
            )

            _update_bucket(
                total_bucket,
                is_correct=is_correct,
                prediction=prediction,
                completion=completion,
                tool_calls=tool_calls,
                tool_response_used=tool_response_used,
                tool_json_ok=tool_json_ok,
                tool_legal_ok=tool_legal_ok,
                turns_used=turns_used,
            )
            _update_bucket(
                source_bucket,
                is_correct=is_correct,
                prediction=prediction,
                completion=completion,
                tool_calls=tool_calls,
                tool_response_used=tool_response_used,
                tool_json_ok=tool_json_ok,
                tool_legal_ok=tool_legal_ok,
                turns_used=turns_used,
            )
            _update_bucket(
                question_bucket,
                is_correct=is_correct,
                prediction=prediction,
                completion=completion,
                tool_calls=tool_calls,
                tool_response_used=tool_response_used,
                tool_json_ok=tool_json_ok,
                tool_legal_ok=tool_legal_ok,
                turns_used=turns_used,
            )
            _update_bucket(
                source_question_bucket,
                is_correct=is_correct,
                prediction=prediction,
                completion=completion,
                tool_calls=tool_calls,
                tool_response_used=tool_response_used,
                tool_json_ok=tool_json_ok,
                tool_legal_ok=tool_legal_ok,
                turns_used=turns_used,
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
                "rollout_infos": rollout_infos,
                "extra": row.get("extra"),
                "raw_response": item,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()

            processed = int(total_bucket["n"])
            if processed % progress_every == 0:
                elapsed_seconds = time.perf_counter() - start_time
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
                partial_metrics_path.write_text(
                    json.dumps(partial_metrics, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )

        pending_rows = []
        pending_requests = []

    with predictions_path.open("w", encoding="utf-8") as handle:
        import tqdm
        for row_index, row in enumerate(tqdm.tqdm(_iter_parquet_rows(config["dataset_path"], limit=limit), total=expected_total, desc="Evaluating")):
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

            image_paths = _materialize_images(normalized_row, image_cache_dir=image_cache_dir)
            messages = _build_messages(
                question=question,
                num_images=len(image_paths),
                system_prompt=system_prompt,
            )
            infer_request: dict[str, Any] = {
                "messages": messages,
                "images": image_paths,
                "question_type": question_type,
                "uuid": sample_id,
            }
            if tools is not None:
                infer_request["tools"] = tools

            pending_rows.append(normalized_row)
            pending_requests.append(infer_request)
            if len(pending_rows) >= request_chunk_size:
                flush_chunk(handle)

        flush_chunk(handle)

    final_elapsed_seconds = time.perf_counter() - start_time
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
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    partial_metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return metrics
