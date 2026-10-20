#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
PLUGIN_DIR = PROJECT_ROOT / "plugins"
if str(PLUGIN_DIR) not in sys.path:
    sys.path.insert(0, str(PLUGIN_DIR))

from appearance_schema import TWIN_DINO_PATCH_MNN_TOOL_SCHEMA_V1  # noqa: E402


def _default_input_root() -> Path:
    return REPO_ROOT / "data" / "twin" / "rft_baseline_aware_pool"


DEFAULT_INPUT_ROOT = _default_input_root()
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "data" / "training" / "dino_patch_mnn"
DEFAULT_HARD_SUBSET_PATTERNS = ("pos_real_neg", "pos_gen_neg")

QUESTION_FALLBACK = (
    "Are these two images showing the same identity?\n"
    "Minor differences caused by background, lighting, viewpoint, crop, or photography are allowed.\n"
    "Focus on the main object and judge whether the core identity remains the same."
)

STOP_LINE_PATTERNS = (
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
    re.compile(r"^\s*<answer>\s*(yes|no)\s*</answer>\s*$", re.IGNORECASE),
)

ANSWER_TAG_PATTERN = re.compile(r"<answer>\s*(yes|no)\s*</answer>", re.IGNORECASE | re.DOTALL)
PREDICTION_FIELDS = (
    "prediction",
    "pred",
    "answer",
    "predicted_answer",
    "pred_answer",
    "final_prediction",
    "model_prediction",
    "vlm_prediction_direct",
    "baseline_prediction",
    "method_prediction",
)
CORRECTNESS_FIELDS = (
    "is_correct",
    "correct",
    "baseline_is_correct",
    "method_is_correct",
    "is_correct_direct",
    "is_correct_plain",
)


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


def _image_suffix(path_like: str) -> str:
    suffix = Path(path_like).suffix.lower()
    if suffix in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
        return suffix
    return ".jpg"


def _materialize_image(
    image_payload: Any,
    output_image_dir: Path,
    sample_id: str,
    image_idx: int,
    overwrite: bool,
) -> str:
    output_image_dir.mkdir(parents=True, exist_ok=True)
    if not isinstance(image_payload, dict):
        raise TypeError(f"Unsupported image payload type: {type(image_payload)}")

    raw_path = str(image_payload.get("path", "") or "")
    raw_bytes = image_payload.get("bytes")

    if isinstance(raw_bytes, (bytes, bytearray)):
        out_path = output_image_dir / f"{sample_id}_{image_idx}{_image_suffix(raw_path)}"
        if overwrite or not out_path.exists():
            with open(out_path, "wb") as handle:
                handle.write(raw_bytes)
        return str(out_path.resolve())

    raise ValueError(f"Image payload for sample '{sample_id}' has neither bytes nor a supported path.")


def _build_user_content(question: str, num_images: int) -> str:
    image_tokens = "\n".join("<image>" for _ in range(max(num_images, 0)))
    return f"{image_tokens}\n{question}" if image_tokens else question


def _copy_parquet(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_idx, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_idx}") from exc
            if isinstance(item, dict):
                records.append(item)
    return records


def _parse_patterns(raw_text: str | None) -> tuple[str, ...]:
    if raw_text is None:
        return DEFAULT_HARD_SUBSET_PATTERNS
    parts = [item.strip() for item in str(raw_text).split(",")]
    return tuple(item for item in parts if item)


def _is_hard_subset(source_subset: str, hard_patterns: tuple[str, ...]) -> bool:
    text = str(source_subset or "").strip().lower()
    return any(pattern in text for pattern in hard_patterns)


def _normalize_yes_no(value: Any) -> str | None:
    text = str(value or "").strip().lower()
    if not text:
        return None
    if text in {"yes", "no"}:
        return text
    match = ANSWER_TAG_PATTERN.search(text)
    if match:
        return match.group(1).lower()
    return None


def _normalize_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in {0, 1}:
        return bool(value)
    text = str(value or "").strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return None


def _candidate_record_keys(record: dict[str, Any]) -> list[str]:
    candidates: list[str] = []
    for key in ("sample_id", "sample_uid", "pair_id", "id", "uid"):
        value = record.get(key)
        if value:
            candidates.append(str(value).strip())
    extra_info = record.get("extra_info")
    if isinstance(extra_info, dict):
        value = extra_info.get("id")
        if value:
            candidates.append(str(value).strip())
    deduped: list[str] = []
    seen: set[str] = set()
    for value in candidates:
        if value and value not in seen:
            deduped.append(value)
            seen.add(value)
    return deduped


def _candidate_prediction_containers(record: dict[str, Any]) -> list[dict[str, Any]]:
    containers = [record]
    for key in ("parsed_json", "prediction_json", "parsed", "result"):
        value = record.get(key)
        if isinstance(value, dict):
            containers.append(value)
    return containers


def _extract_baseline_prediction(record: dict[str, Any]) -> str | None:
    for container in _candidate_prediction_containers(record):
        for key in PREDICTION_FIELDS:
            value = _normalize_yes_no(container.get(key))
            if value:
                return value
    for key in ("completion", "raw_completion", "response", "text"):
        value = _normalize_yes_no(record.get(key))
        if value:
            return value
    return None


def _extract_baseline_is_correct(record: dict[str, Any], ground_truth: str) -> bool | None:
    for container in _candidate_prediction_containers(record):
        for key in CORRECTNESS_FIELDS:
            value = _normalize_bool(container.get(key))
            if value is not None:
                return value
    baseline_prediction = _extract_baseline_prediction(record)
    truth = _normalize_yes_no(ground_truth)
    if baseline_prediction and truth:
        return baseline_prediction == truth
    return None


def _load_baseline_index(path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    records = _load_jsonl(path)
    index: dict[str, dict[str, Any]] = {}
    stats = Counter()
    for record in records:
        keys = _candidate_record_keys(record)
        if not keys:
            stats["unkeyed_records"] += 1
            continue
        stats["records_with_keys"] += 1
        for key in keys:
            if key in index:
                stats["duplicate_keys"] += 1
            index[key] = record
            stats["indexed_keys"] += 1
    stats["input_records"] = len(records)
    stats["unique_keys"] = len(index)
    return index, dict(sorted(stats.items()))


def _sample_tool_label(rng: random.Random, *, likely_prob: float, no_tool_prob: float) -> str:
    u = rng.random()
    if u < likely_prob:
        return "likely_tool"
    if u < likely_prob + no_tool_prob:
        return "no_tool"
    return "mixed"


def _derive_tool_label(
    *,
    rng: random.Random,
    sample_id: str,
    ground_truth: str,
    source_subset: str,
    force_tool_label: str | None,
    tool_label_policy: str,
    hard_patterns: tuple[str, ...],
    hard_likely_prob: float,
    hard_no_tool_prob: float,
    easy_likely_prob: float,
    easy_no_tool_prob: float,
    baseline_index: dict[str, dict[str, Any]] | None,
    baseline_missing_policy: str,
    baseline_wrong_likely_prob: float,
    baseline_wrong_no_tool_prob: float,
    baseline_correct_likely_prob: float,
    baseline_correct_no_tool_prob: float,
) -> tuple[str, bool, str | None]:
    is_hard = _is_hard_subset(source_subset, hard_patterns)
    if force_tool_label:
        return str(force_tool_label), is_hard, None

    policy = str(tool_label_policy).strip().lower()
    if policy == "hard_random":
        if is_hard:
            return (
                _sample_tool_label(rng, likely_prob=hard_likely_prob, no_tool_prob=hard_no_tool_prob),
                True,
                None,
            )
        return (
            _sample_tool_label(rng, likely_prob=easy_likely_prob, no_tool_prob=easy_no_tool_prob),
            False,
            None,
        )

    if policy != "baseline_aware":
        raise ValueError(f"Unsupported tool_label_policy: {tool_label_policy}")

    baseline_record = (baseline_index or {}).get(sample_id)
    baseline_status = "missing"
    if baseline_record is not None:
        baseline_is_correct = _extract_baseline_is_correct(baseline_record, ground_truth)
        if baseline_is_correct is True:
            baseline_status = "correct"
        elif baseline_is_correct is False:
            baseline_status = "wrong"
        else:
            baseline_status = "unknown"

    if baseline_status == "wrong":
        return (
            _sample_tool_label(
                rng,
                likely_prob=baseline_wrong_likely_prob,
                no_tool_prob=baseline_wrong_no_tool_prob,
            ),
            is_hard,
            baseline_status,
        )
    if baseline_status == "correct":
        return (
            _sample_tool_label(
                rng,
                likely_prob=baseline_correct_likely_prob,
                no_tool_prob=baseline_correct_no_tool_prob,
            ),
            is_hard,
            baseline_status,
        )
    if baseline_missing_policy == "likely_tool":
        return "likely_tool", is_hard, baseline_status
    if baseline_missing_policy == "no_tool":
        return "no_tool", is_hard, baseline_status
    if baseline_missing_policy == "mixed":
        return "mixed", is_hard, baseline_status
    if is_hard:
        return (
            _sample_tool_label(rng, likely_prob=hard_likely_prob, no_tool_prob=hard_no_tool_prob),
            True,
            baseline_status,
        )
    return (
        _sample_tool_label(rng, likely_prob=easy_likely_prob, no_tool_prob=easy_no_tool_prob),
        False,
        baseline_status,
    )


def _validate_prob_pair(name: str, likely_prob: float, no_tool_prob: float) -> None:
    if likely_prob < 0 or no_tool_prob < 0:
        raise ValueError(f"{name} probabilities must be non-negative.")
    if likely_prob + no_tool_prob > 1.0:
        raise ValueError(f"{name} probabilities must sum to <= 1.0.")


def _convert_split(
    parquet_path: Path,
    output_jsonl: Path,
    output_image_dir: Path,
    max_rows: int,
    parquet_batch_size: int,
    overwrite_images: bool,
    force_tool_label: str | None,
    tool_label_policy: str,
    hard_patterns: tuple[str, ...],
    hard_likely_prob: float,
    hard_no_tool_prob: float,
    easy_likely_prob: float,
    easy_no_tool_prob: float,
    baseline_index: dict[str, dict[str, Any]] | None,
    baseline_missing_policy: str,
    baseline_wrong_likely_prob: float,
    baseline_wrong_no_tool_prob: float,
    baseline_correct_likely_prob: float,
    baseline_correct_no_tool_prob: float,
    seed: int,
) -> dict[str, Any]:
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    source_counts = Counter()
    subset_counts = Counter()
    tool_label_counts = Counter()
    image_count_distribution = Counter()
    hardness_counts = Counter()
    tool_label_by_hardness = Counter()
    tool_label_by_gt = Counter()
    baseline_status_counts = Counter()
    tool_label_by_baseline_status = Counter()

    rng = random.Random(seed)
    num_rows = 0

    with open(output_jsonl, "w", encoding="utf-8") as writer:
        parquet_file = pq.ParquetFile(parquet_path)
        stop = False
        for batch in parquet_file.iter_batches(batch_size=max(1, int(parquet_batch_size))):
            for row in batch.to_pylist():
                if max_rows > 0 and num_rows >= max_rows:
                    stop = True
                    break

                extra_info = row.get("extra_info") or {}
                reward_model = row.get("reward_model") or {}
                sample_id = str(extra_info.get("id") or f"{parquet_path.stem}_{num_rows}")
                source_subset = str(extra_info.get("source_subset") or "twin")
                ground_truth = str(reward_model.get("ground_truth") or "").strip().lower()
                tool_label, is_hard, baseline_status = _derive_tool_label(
                    rng=rng,
                    sample_id=sample_id,
                    ground_truth=ground_truth,
                    source_subset=source_subset,
                    force_tool_label=force_tool_label,
                    tool_label_policy=tool_label_policy,
                    hard_patterns=hard_patterns,
                    hard_likely_prob=hard_likely_prob,
                    hard_no_tool_prob=hard_no_tool_prob,
                    easy_likely_prob=easy_likely_prob,
                    easy_no_tool_prob=easy_no_tool_prob,
                    baseline_index=baseline_index,
                    baseline_missing_policy=baseline_missing_policy,
                    baseline_wrong_likely_prob=baseline_wrong_likely_prob,
                    baseline_wrong_no_tool_prob=baseline_wrong_no_tool_prob,
                    baseline_correct_likely_prob=baseline_correct_likely_prob,
                    baseline_correct_no_tool_prob=baseline_correct_no_tool_prob,
                )
                data_source = str(extra_info.get("original_data_source") or row.get("data_source") or "twin")

                images = row.get("images") or []
                image_paths = [
                    _materialize_image(payload, output_image_dir, sample_id, image_idx, overwrite_images)
                    for image_idx, payload in enumerate(images)
                ]

                question = _clean_question(_first_user_content(row.get("prompt")))
                record = {
                    "messages": [{"role": "user", "content": _build_user_content(question, len(image_paths))}],
                    "images": image_paths,
                    "ground_truth": ground_truth,
                    "tool_label": tool_label,
                    "sample_id": sample_id,
                    "data_source": data_source,
                    "source_subset": source_subset,
                    "hard_flag": bool(is_hard),
                    "tools": json.dumps(TWIN_DINO_PATCH_MNN_TOOL_SCHEMA_V1, ensure_ascii=False),
                }
                writer.write(json.dumps(record, ensure_ascii=False) + "\n")

                source_counts[data_source] += 1
                subset_counts[source_subset] += 1
                tool_label_counts[tool_label] += 1
                image_count_distribution[str(len(image_paths))] += 1
                hardness_counts["hard" if is_hard else "non_hard"] += 1
                tool_label_by_hardness[f'{"hard" if is_hard else "non_hard"}::{tool_label}'] += 1
                tool_label_by_gt[f"{ground_truth}::{tool_label}"] += 1
                if baseline_status is not None:
                    baseline_status_counts[baseline_status] += 1
                    tool_label_by_baseline_status[f"{baseline_status}::{tool_label}"] += 1
                num_rows += 1
            if stop:
                break

    return {
        "parquet": str(parquet_path),
        "jsonl": str(output_jsonl),
        "images": str(output_image_dir),
        "num_rows": num_rows,
        "source_counts": dict(sorted(source_counts.items())),
        "subset_counts": dict(sorted(subset_counts.items())),
        "tool_label_counts": dict(sorted(tool_label_counts.items())),
        "image_count_distribution": dict(sorted(image_count_distribution.items())),
        "hardness_counts": dict(sorted(hardness_counts.items())),
        "tool_label_by_hardness": dict(sorted(tool_label_by_hardness.items())),
        "tool_label_by_gt": dict(sorted(tool_label_by_gt.items())),
        "baseline_status_counts": dict(sorted(baseline_status_counts.items())),
        "tool_label_by_baseline_status": dict(sorted(tool_label_by_baseline_status.items())),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build twin DINO patch-MNN ms-swift dataset.")
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--max-rows", type=int, default=0, help="Optional cap per split. 0 means use all rows.")
    parser.add_argument("--parquet-batch-size", type=int, default=128)
    parser.add_argument("--overwrite-images", action="store_true")
    parser.add_argument(
        "--force-tool-label",
        type=str,
        choices=["likely_tool", "mixed", "no_tool"],
        default=None,
        help="Override every sample's tool_label with a single value.",
    )
    parser.add_argument(
        "--tool-label-policy",
        type=str,
        choices=["hard_random", "baseline_aware"],
        default="hard_random",
        help="How to derive tool_label. baseline_aware routes from a baseline predictions jsonl and falls back when missing.",
    )
    parser.add_argument(
        "--baseline-predictions-jsonl",
        type=Path,
        default=None,
        help="Optional baseline predictions file used by baseline_aware policy.",
    )
    parser.add_argument(
        "--baseline-missing-policy",
        type=str,
        choices=["hard_random", "likely_tool", "mixed", "no_tool"],
        default="hard_random",
        help="Fallback tool-label behavior when baseline_aware cannot find or parse a baseline result.",
    )
    parser.add_argument(
        "--hard-subset-patterns",
        type=str,
        default=",".join(DEFAULT_HARD_SUBSET_PATTERNS),
        help="Comma-separated patterns used to identify hard subsets.",
    )
    parser.add_argument("--hard-likely-prob", type=float, default=0.75)
    parser.add_argument("--hard-no-tool-prob", type=float, default=0.05)
    parser.add_argument("--easy-likely-prob", type=float, default=0.15)
    parser.add_argument("--easy-no-tool-prob", type=float, default=0.50)
    parser.add_argument("--baseline-wrong-likely-prob", type=float, default=0.65)
    parser.add_argument("--baseline-wrong-no-tool-prob", type=float, default=0.10)
    parser.add_argument("--baseline-correct-likely-prob", type=float, default=0.20)
    parser.add_argument("--baseline-correct-no-tool-prob", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_root = args.input_root
    output_root = args.output_root
    tool_root = output_root / "ms_swift" / "tool"
    hard_patterns = _parse_patterns(args.hard_subset_patterns)
    _validate_prob_pair("hard_random.hard", args.hard_likely_prob, args.hard_no_tool_prob)
    _validate_prob_pair("hard_random.easy", args.easy_likely_prob, args.easy_no_tool_prob)
    _validate_prob_pair("baseline_aware.wrong", args.baseline_wrong_likely_prob, args.baseline_wrong_no_tool_prob)
    _validate_prob_pair(
        "baseline_aware.correct",
        args.baseline_correct_likely_prob,
        args.baseline_correct_no_tool_prob,
    )

    baseline_index: dict[str, dict[str, Any]] | None = None
    baseline_index_stats: dict[str, Any] = {}
    if args.tool_label_policy == "baseline_aware":
        if args.baseline_predictions_jsonl is None:
            raise ValueError("--baseline-predictions-jsonl is required when --tool-label-policy=baseline_aware")
        baseline_index, baseline_index_stats = _load_baseline_index(args.baseline_predictions_jsonl.resolve())

    _copy_parquet(input_root / "train_tool.parquet", output_root / "train_tool.parquet")
    _copy_parquet(input_root / "val_tool.parquet", output_root / "val_tool.parquet")

    train_summary = _convert_split(
        parquet_path=input_root / "train_tool.parquet",
        output_jsonl=tool_root / "train.jsonl",
        output_image_dir=tool_root / "images" / "train",
        max_rows=args.max_rows,
        parquet_batch_size=args.parquet_batch_size,
        overwrite_images=args.overwrite_images,
        force_tool_label=args.force_tool_label,
        tool_label_policy=args.tool_label_policy,
        hard_patterns=hard_patterns,
        hard_likely_prob=args.hard_likely_prob,
        hard_no_tool_prob=args.hard_no_tool_prob,
        easy_likely_prob=args.easy_likely_prob,
        easy_no_tool_prob=args.easy_no_tool_prob,
        baseline_index=baseline_index,
        baseline_missing_policy=args.baseline_missing_policy,
        baseline_wrong_likely_prob=args.baseline_wrong_likely_prob,
        baseline_wrong_no_tool_prob=args.baseline_wrong_no_tool_prob,
        baseline_correct_likely_prob=args.baseline_correct_likely_prob,
        baseline_correct_no_tool_prob=args.baseline_correct_no_tool_prob,
        seed=args.seed,
    )
    val_summary = _convert_split(
        parquet_path=input_root / "val_tool.parquet",
        output_jsonl=tool_root / "val.jsonl",
        output_image_dir=tool_root / "images" / "val",
        max_rows=args.max_rows,
        parquet_batch_size=args.parquet_batch_size,
        overwrite_images=args.overwrite_images,
        force_tool_label=args.force_tool_label,
        tool_label_policy=args.tool_label_policy,
        hard_patterns=hard_patterns,
        hard_likely_prob=args.hard_likely_prob,
        hard_no_tool_prob=args.hard_no_tool_prob,
        easy_likely_prob=args.easy_likely_prob,
        easy_no_tool_prob=args.easy_no_tool_prob,
        baseline_index=baseline_index,
        baseline_missing_policy=args.baseline_missing_policy,
        baseline_wrong_likely_prob=args.baseline_wrong_likely_prob,
        baseline_wrong_no_tool_prob=args.baseline_wrong_no_tool_prob,
        baseline_correct_likely_prob=args.baseline_correct_likely_prob,
        baseline_correct_no_tool_prob=args.baseline_correct_no_tool_prob,
        seed=args.seed + 1,
    )

    ms_swift_manifest = {
        "train_jsonl": str((tool_root / "train.jsonl").resolve()),
        "val_jsonl": str((tool_root / "val.jsonl").resolve()),
        "train_size": train_summary["num_rows"],
        "val_size": val_summary["num_rows"],
    }
    tool_root.mkdir(parents=True, exist_ok=True)
    with open(tool_root / "manifest.json", "w", encoding="utf-8") as handle:
        json.dump(ms_swift_manifest, handle, ensure_ascii=False, indent=2)

    source_counts = Counter(train_summary["source_counts"])
    source_counts.update(val_summary["source_counts"])
    subset_counts = Counter(train_summary["subset_counts"])
    subset_counts.update(val_summary["subset_counts"])
    tool_label_counts = Counter(train_summary["tool_label_counts"])
    tool_label_counts.update(val_summary["tool_label_counts"])
    image_count_distribution = Counter(train_summary["image_count_distribution"])
    image_count_distribution.update(val_summary["image_count_distribution"])
    hardness_counts = Counter(train_summary["hardness_counts"])
    hardness_counts.update(val_summary["hardness_counts"])
    tool_label_by_hardness = Counter(train_summary["tool_label_by_hardness"])
    tool_label_by_hardness.update(val_summary["tool_label_by_hardness"])
    tool_label_by_gt = Counter(train_summary["tool_label_by_gt"])
    tool_label_by_gt.update(val_summary["tool_label_by_gt"])
    baseline_status_counts = Counter(train_summary["baseline_status_counts"])
    baseline_status_counts.update(val_summary["baseline_status_counts"])
    tool_label_by_baseline_status = Counter(train_summary["tool_label_by_baseline_status"])
    tool_label_by_baseline_status.update(val_summary["tool_label_by_baseline_status"])

    sampling_stats = {
        "seed": args.seed,
        "val_ratio": args.val_ratio,
        "parquet_batch_size": args.parquet_batch_size,
        "force_tool_label": args.force_tool_label,
        "tool_label_policy": args.tool_label_policy,
        "baseline_predictions_jsonl": (
            str(args.baseline_predictions_jsonl.resolve()) if args.baseline_predictions_jsonl else None
        ),
        "baseline_missing_policy": args.baseline_missing_policy,
        "baseline_index_stats": baseline_index_stats,
        "hard_subset_patterns": list(hard_patterns),
        "hard_likely_prob": args.hard_likely_prob,
        "hard_no_tool_prob": args.hard_no_tool_prob,
        "easy_likely_prob": args.easy_likely_prob,
        "easy_no_tool_prob": args.easy_no_tool_prob,
        "baseline_wrong_likely_prob": args.baseline_wrong_likely_prob,
        "baseline_wrong_no_tool_prob": args.baseline_wrong_no_tool_prob,
        "baseline_correct_likely_prob": args.baseline_correct_likely_prob,
        "baseline_correct_no_tool_prob": args.baseline_correct_no_tool_prob,
        "total": train_summary["num_rows"] + val_summary["num_rows"],
        "train_size": train_summary["num_rows"],
        "val_size": val_summary["num_rows"],
        "source_counts": dict(sorted(source_counts.items())),
        "subset_counts": dict(sorted(subset_counts.items())),
        "tool_label_counts": dict(sorted(tool_label_counts.items())),
        "image_count_distribution": dict(sorted(image_count_distribution.items())),
        "hardness_counts": dict(sorted(hardness_counts.items())),
        "tool_label_by_hardness": dict(sorted(tool_label_by_hardness.items())),
        "tool_label_by_gt": dict(sorted(tool_label_by_gt.items())),
        "baseline_status_counts": dict(sorted(baseline_status_counts.items())),
        "tool_label_by_baseline_status": dict(sorted(tool_label_by_baseline_status.items())),
        "splits": {
            "train": train_summary,
            "val": val_summary,
        },
    }
    with open(output_root / "sampling_stats.json", "w", encoding="utf-8") as handle:
        json.dump(sampling_stats, handle, ensure_ascii=False, indent=2)

    print(json.dumps(sampling_stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
