#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import random
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


TRAINING_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = TRAINING_ROOT.parent
DATA_SOURCE = "glab-caltech/TWIN"
ABILITY = "spatial_reasoning"


def _first_existing(candidates: list[Path]) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _resolve_twin_source_parquet() -> Path:
    env_path = os.environ.get("TWIN_SOURCE_PARQUET")
    candidates: list[Path] = []
    if env_path:
        candidates.append(Path(env_path))
    candidates.append(REPO_ROOT / "data" / "twin" / "train-00000-of-00001.parquet")
    return _first_existing(candidates)


def _resolve_twin_images_root() -> Path:
    env_path = os.environ.get("TWIN_IMAGES_ROOT")
    candidates: list[Path] = []
    if env_path:
        candidates.append(Path(env_path))
    candidates.append(REPO_ROOT / "data" / "twin" / "images")
    return _first_existing(candidates)

DEFAULT_SOURCE_PARQUET = _resolve_twin_source_parquet()
DEFAULT_IMAGES_ROOT = _resolve_twin_images_root()
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "data" / "twin" / "rft_baseline_aware_pool"


def _base_question() -> str:
    return (
        "Are these images of the same or identical products?\n"
        "Minor changes caused by context, background, lighting, viewpoint, or photography are allowed, "
        "but the core product attributes should remain consistent.\n"
        "Focus on the primary product if multiple products appear.\n"
        "Judge consistency using overall shape, color, material, texture, structure, branding, and distinctive parts."
    )


def build_prompt(style: str) -> str:
    stem = _base_question()
    if style == "cot":
        instruction = (
            "Reason step by step in <think>...</think>.\n"
            "Then end with exactly one final decision:\n"
            "<answer>yes</answer>\n"
            "<answer>no</answer>"
        )
    else:
        instruction = (
            "You are a fine-grained visual comparison assistant.\n"
            "Start with a <global_judgment> block.\n"
            "Inside <global_judgment>, give your current yes/no judgment, name the 1-2 most discriminative visual "
            "features, and state which region or feature you want to verify first.\n"
            "After <global_judgment>, you may output one <tool_call>...</tool_call> to verify evidence.\n"
            "Use at most 3 tool calls in total.\n"
            "After each tool response, continue your reasoning using plain text only. Summarize the returned visual "
            "evidence and state whether it supports or weakens your current judgment. Do not use an <observe> tag.\n"
            "Use normalized coordinates in [0,1]. Never output pixel coordinates.\n"
            "Always use image ids img1 and img2.\n"
            "The only available tool is compare_regions, which shows a region from image 1 and a region from image 2 "
            "side by side for comparing overall shape, color, texture, structure, branding, and distinctive details.\n"
            "Write tool calls in this exact JSON form:\n"
            "<tool_call>{\"name\":\"compare_regions\",\"arguments\":{\"img1_left\":0.2,\"img1_top\":0.2,"
            "\"img1_right\":0.8,\"img1_bottom\":0.8,\"img2_left\":0.2,\"img2_top\":0.2,\"img2_right\":0.8,"
            "\"img2_bottom\":0.8}}</tool_call>\n"
            "End with exactly one final decision and output nothing after it:\n"
            "<answer>yes</answer>\n"
            "<answer>no</answer>"
        )
    return f"{stem}\n\n{instruction}"


def _hamilton_allocate(counts: dict[str, int], total: int) -> dict[str, int]:
    if total < 0:
        raise ValueError("total must be non-negative")
    if total == 0:
        return {key: 0 for key in counts}

    overall = sum(counts.values())
    if overall < total:
        raise ValueError(f"Not enough rows to allocate {total} samples from {overall} rows")
    if overall == 0:
        raise ValueError("Cannot allocate from an empty distribution")

    exact = {key: total * value / overall for key, value in counts.items()}
    base = {key: min(counts[key], math.floor(value)) for key, value in exact.items()}
    assigned = sum(base.values())

    remainders = sorted(
        ((exact[key] - base[key], counts[key] - base[key], key) for key in counts),
        key=lambda item: (-item[0], -item[1], item[2]),
    )

    remaining = total - assigned
    idx = 0
    while remaining > 0:
        if idx >= len(remainders):
            idx = 0
        _, _, key = remainders[idx]
        if base[key] < counts[key]:
            base[key] += 1
            remaining -= 1
        idx += 1
    return base


def _sample_group(df: pd.DataFrame, total: int, rng: random.Random) -> pd.DataFrame:
    source_counts = df["source_json"].value_counts().sort_index().to_dict()
    source_alloc = _hamilton_allocate(source_counts, total)

    pieces: list[pd.DataFrame] = []
    for source_json in sorted(source_alloc):
        source_take = source_alloc[source_json]
        source_df = df[df["source_json"] == source_json]
        super_counts = source_df["supercategory"].value_counts().sort_index().to_dict()
        super_alloc = _hamilton_allocate(super_counts, source_take)
        for supercategory in sorted(super_alloc):
            take = super_alloc[supercategory]
            group_df = source_df[source_df["supercategory"] == supercategory]
            if len(group_df) < take:
                raise ValueError(
                    f"Not enough rows for source_json={source_json}, supercategory={supercategory}: "
                    f"need {take}, got {len(group_df)}"
                )
            pieces.append(group_df.sample(n=take, random_state=rng.randrange(0, 2**31 - 1)))

    sampled = pd.concat(pieces, ignore_index=True)
    sampled = sampled.sample(frac=1.0, random_state=rng.randrange(0, 2**31 - 1)).reset_index(drop=True)
    return sampled


def sample_twin_rows(source_parquet: Path, total_samples: int, seed: int) -> pd.DataFrame:
    df = pd.read_parquet(source_parquet).reset_index(drop=False).rename(columns={"index": "row_index"})
    if "pair_id" not in df.columns:
        raise ValueError("Source parquet is missing required column 'pair_id'")
    return _sample_group(df, total_samples, random.Random(seed))


def _split_train_val(sampled_df: pd.DataFrame, val_ratio: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError("val_ratio must be in [0, 1)")
    if val_ratio == 0.0:
        return sampled_df.reset_index(drop=True), sampled_df.iloc[0:0].copy()

    val_size = max(1, round(len(sampled_df) * val_ratio))
    val_df = _sample_group(sampled_df, val_size, random.Random(seed + 1))
    val_ids = set(val_df["row_index"].tolist())
    train_df = sampled_df[~sampled_df["row_index"].isin(val_ids)].copy().reset_index(drop=True)
    return train_df, val_df.reset_index(drop=True)


def resolve_local_image(images_root: Path, rel_path: str) -> Path:
    normalized = Path(rel_path)
    direct = images_root / normalized
    if direct.exists():
        return direct
    by_name = images_root / normalized.name
    if by_name.exists():
        return by_name
    raise FileNotFoundError(f"Image not found for {rel_path}: tried {direct} and {by_name}")


def _tool_label_for_row(row: pd.Series) -> str:
    return "mixed" if row["source_json"] == "pos_pos.json" else "likely_tool"


def _build_record(row: pd.Series, style: str, images_root: Path) -> dict[str, Any]:
    image_1_local = resolve_local_image(images_root, row["image_1"])
    image_2_local = resolve_local_image(images_root, row["image_2"])
    return {
        "data_source": DATA_SOURCE,
        "prompt": [{"role": "user", "content": build_prompt(style)}],
        "images": [
            {"bytes": image_1_local.read_bytes(), "path": row["image_1"]},
            {"bytes": image_2_local.read_bytes(), "path": row["image_2"]},
        ],
        "ability": ABILITY,
        "reward_model": {"style": "rule", "ground_truth": str(row["answer"]).strip().lower()},
        "extra_info": {
            "id": f"twin_{row['row_index']}_{row['source_json'].replace('.json', '')}",
            "original_data_source": DATA_SOURCE,
            "source_subset": f"twin/{row['source_json'].replace('.json', '')}/{row['supercategory']}/{row['category']}",
            "tool_label": _tool_label_for_row(row),
            "train_style": style,
        },
    }


def build_records(sampled_df: pd.DataFrame, style: str, images_root: Path) -> list[dict[str, Any]]:
    return [_build_record(row, style=style, images_root=images_root) for _, row in sampled_df.iterrows()]


def build_sampling_stats(
    *,
    sampled_df: pd.DataFrame,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    total_available: int,
    seed: int,
    val_ratio: float,
) -> dict[str, Any]:
    subset_counter = Counter(
        f"twin/{row['source_json'].replace('.json', '')}/{row['supercategory']}/{row['category']}"
        for _, row in sampled_df.iterrows()
    )
    return {
        "seed": seed,
        "val_ratio": val_ratio,
        "total": int(len(sampled_df)),
        "train_size": int(len(train_df)),
        "val_size": int(len(val_df)),
        "total_available": total_available,
        "source_counts": dict(
            sorted(Counter(row["source_json"].replace(".json", "") for _, row in sampled_df.iterrows()).items())
        ),
        "subset_counts": dict(sorted(subset_counter.items())),
        "tool_label_counts": dict(sorted(Counter(_tool_label_for_row(row) for _, row in sampled_df.iterrows()).items())),
        "image_count_distribution": {"2": int(len(sampled_df))},
        "val_source_counts": dict(
            sorted(Counter(row["source_json"].replace(".json", "") for _, row in val_df.iterrows()).items())
        ),
    }


def write_stats(stats: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_parquet_chunked(
    *,
    rows_df: Any,
    style: str,
    images_root: Path,
    output_path: Path,
    chunk_rows: int,
) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer: pq.ParquetWriter | None = None
    written = 0
    try:
        for start in range(0, len(rows_df), max(1, int(chunk_rows))):
            chunk_df = rows_df.iloc[start:start + max(1, int(chunk_rows))]
            records = build_records(chunk_df, style=style, images_root=images_root)
            table = pa.Table.from_pylist(records)
            if writer is None:
                writer = pq.ParquetWriter(output_path, table.schema)
            writer.write_table(table)
            written += len(records)
    finally:
        if writer is not None:
            writer.close()
    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a larger joint TWIN pool for baseline-aware routing with aligned tool/cot parquet splits.",
    )
    parser.add_argument("--source-parquet", type=Path, default=DEFAULT_SOURCE_PARQUET)
    parser.add_argument("--images-root", type=Path, default=DEFAULT_IMAGES_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--total-samples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--chunk-rows", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    source_parquet = args.source_parquet.resolve()
    images_root = args.images_root.resolve()

    sampled_df = sample_twin_rows(
        source_parquet=source_parquet,
        total_samples=int(args.total_samples),
        seed=int(args.seed),
    )
    train_df, val_df = _split_train_val(
        sampled_df=sampled_df,
        val_ratio=float(args.val_ratio),
        seed=int(args.seed),
    )

    train_tool_path = output_root / "train_tool.parquet"
    val_tool_path = output_root / "val_tool.parquet"
    train_cot_path = output_root / "train_cot.parquet"
    val_cot_path = output_root / "val_cot.parquet"

    written_train_tool = _write_parquet_chunked(
        rows_df=train_df,
        style="tool",
        images_root=images_root,
        output_path=train_tool_path,
        chunk_rows=args.chunk_rows,
    )
    written_val_tool = _write_parquet_chunked(
        rows_df=val_df,
        style="tool",
        images_root=images_root,
        output_path=val_tool_path,
        chunk_rows=args.chunk_rows,
    )
    written_train_cot = _write_parquet_chunked(
        rows_df=train_df,
        style="cot",
        images_root=images_root,
        output_path=train_cot_path,
        chunk_rows=args.chunk_rows,
    )
    written_val_cot = _write_parquet_chunked(
        rows_df=val_df,
        style="cot",
        images_root=images_root,
        output_path=val_cot_path,
        chunk_rows=args.chunk_rows,
    )

    sampling_stats = build_sampling_stats(
        sampled_df=sampled_df,
        train_df=train_df,
        val_df=val_df,
        total_available=int(pq.read_table(source_parquet, columns=["pair_id"]).num_rows),
        seed=int(args.seed),
        val_ratio=float(args.val_ratio),
    )
    sampling_stats["chunk_rows"] = int(args.chunk_rows)
    sampling_stats["styles_written"] = {
        "train_tool": written_train_tool,
        "val_tool": written_val_tool,
        "train_cot": written_train_cot,
        "val_cot": written_val_cot,
    }
    write_stats(sampling_stats, output_root / "sampling_stats.json")

    manifest = {
        "dataset_name": "twin_baseline_aware_pool",
        "source_parquet": str(source_parquet),
        "images_root": str(images_root),
        "output_root": str(output_root),
        "total_samples": int(args.total_samples),
        "seed": int(args.seed),
        "val_ratio": float(args.val_ratio),
        "chunk_rows": int(args.chunk_rows),
        "files": {
            "train_tool_parquet": str(train_tool_path),
            "val_tool_parquet": str(val_tool_path),
            "train_cot_parquet": str(train_cot_path),
            "val_cot_parquet": str(val_cot_path),
            "sampling_stats": str((output_root / "sampling_stats.json").resolve()),
        },
    }
    output_root.mkdir(parents=True, exist_ok=True)
    with open(output_root / "manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)

    print(json.dumps({"manifest": manifest, "sampling_stats": sampling_stats}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
