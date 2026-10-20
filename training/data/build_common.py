#!/usr/bin/env python3
"""
Shared utilities for building sampled TWIN RFT datasets in VERL parquet format.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parents[1]
DEFAULT_SOURCE_PARQUET = REPO_ROOT / "data" / "twin" / "train-00000-of-00001.parquet"
DEFAULT_IMAGES_ROOT = REPO_ROOT / "data" / "twin" / "images"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "twin" / "rft"
DEFAULT_TOTAL_SAMPLES = 5000
DEFAULT_SEED = 42
DEFAULT_VAL_RATIO = 0.1

DATA_SOURCE = "glab-caltech/TWIN"
ABILITY = "spatial_reasoning"


def build_arg_parser(default_style: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=f"Build sampled TWIN RFT dataset ({default_style}).",
    )
    parser.add_argument(
        "--source-parquet",
        type=Path,
        default=DEFAULT_SOURCE_PARQUET,
        help="Path to the source TWIN parquet.",
    )
    parser.add_argument(
        "--images-root",
        type=Path,
        default=DEFAULT_IMAGES_ROOT,
        help="Root directory for local TWIN images.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for output parquet and stats.",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default=f"train_{default_style}.parquet",
        help="Output parquet filename.",
    )
    parser.add_argument(
        "--stats-name",
        type=str,
        default="sampling_stats.json",
        help="Output stats filename.",
    )
    parser.add_argument(
        "--total-samples",
        type=int,
        default=DEFAULT_TOTAL_SAMPLES,
        help="Total number of sampled rows before train/val split.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed for deterministic sampling.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=DEFAULT_VAL_RATIO,
        help="Held-out validation ratio, matching the peer dataset convention.",
    )
    parser.add_argument(
        "--val-output-name",
        type=str,
        default=f"val_{default_style}.parquet",
        help="Validation parquet filename.",
    )
    parser.add_argument(
        "--style",
        choices=["cot", "tool"],
        default=default_style,
        help="Prompt style to build.",
    )
    return parser


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


def _hamilton_allocate(counts: Dict[str, int], total: int) -> Dict[str, int]:
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
        _, capacity, key = remainders[idx]
        if base[key] < counts[key]:
            base[key] += 1
            remaining -= 1
        idx += 1

    return base


def _sample_group(df: pd.DataFrame, total: int, rng: random.Random) -> pd.DataFrame:
    source_counts = df["source_json"].value_counts().sort_index().to_dict()
    source_alloc = _hamilton_allocate(source_counts, total)

    pieces: List[pd.DataFrame] = []
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
            sample_seed = rng.randrange(0, 2**31 - 1)
            pieces.append(group_df.sample(n=take, random_state=sample_seed))

    sampled = pd.concat(pieces, ignore_index=True)
    shuffle_seed = rng.randrange(0, 2**31 - 1)
    sampled = sampled.sample(frac=1.0, random_state=shuffle_seed).reset_index(drop=True)
    return sampled


def sample_twin_rows(source_parquet: Path, total_samples: int, seed: int) -> pd.DataFrame:
    df = (
        pd.read_parquet(source_parquet)
        .reset_index(drop=False)
        .rename(columns={"index": "row_index"})
    )
    if "pair_id" not in df.columns:
        raise ValueError("Source parquet is missing required column 'pair_id'")
    rng = random.Random(seed)
    return _sample_group(df, total_samples, rng)


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
    source_json = row["source_json"]
    if source_json == "pos_pos.json":
        return "mixed"
    return "likely_tool"


def _build_record(row: pd.Series, style: str, images_root: Path) -> dict:
    image_1_local = resolve_local_image(images_root, row["image_1"])
    image_2_local = resolve_local_image(images_root, row["image_2"])

    record = {
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
    return record


def build_records(sampled_df: pd.DataFrame, style: str, images_root: Path) -> List[dict]:
    return [_build_record(row, style=style, images_root=images_root) for _, row in sampled_df.iterrows()]


def write_records(records: Iterable[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(list(records))
    pq.write_table(table, output_path)


def _split_train_val(sampled_df: pd.DataFrame, val_ratio: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError("val_ratio must be in [0, 1)")
    if val_ratio == 0.0:
        return sampled_df.reset_index(drop=True), sampled_df.iloc[0:0].copy()

    val_size = max(1, round(len(sampled_df) * val_ratio))
    rng = random.Random(seed + 1)
    val_df = _sample_group(sampled_df, val_size, rng)
    val_ids = set(val_df["row_index"].tolist())
    train_df = sampled_df[~sampled_df["row_index"].isin(val_ids)].copy().reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    return train_df, val_df


def build_sampling_stats(
    sampled_df: pd.DataFrame,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    total_available: int,
    seed: int,
    val_ratio: float,
) -> dict:
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
            sorted(
                Counter(
                    row["source_json"].replace(".json", "")
                    for _, row in sampled_df.iterrows()
                ).items()
            )
        ),
        "subset_counts": dict(sorted(subset_counter.items())),
        "tool_label_counts": dict(sorted(Counter(_tool_label_for_row(row) for _, row in sampled_df.iterrows()).items())),
        "image_count_distribution": {"2": int(len(sampled_df))},
        "val_source_counts": dict(
            sorted(
                Counter(
                    row["source_json"].replace(".json", "")
                    for _, row in val_df.iterrows()
                ).items()
            )
        ),
    }


def write_stats(stats: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def run_builder(style: str, args: argparse.Namespace) -> None:
    sampled_df = sample_twin_rows(
        source_parquet=args.source_parquet,
        total_samples=args.total_samples,
        seed=args.seed,
    )
    train_df, val_df = _split_train_val(
        sampled_df=sampled_df,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    output_path = args.output_dir / args.output_name
    val_output_path = args.output_dir / args.val_output_name
    stats_path = args.output_dir / args.stats_name

    train_records = build_records(
        sampled_df=train_df,
        style=style,
        images_root=args.images_root,
    )
    val_records = build_records(
        sampled_df=val_df,
        style=style,
        images_root=args.images_root,
    )
    write_records(train_records, output_path)
    write_records(val_records, val_output_path)
    write_stats(
        build_sampling_stats(
            sampled_df=sampled_df,
            train_df=train_df,
            val_df=val_df,
            total_available=int(pd.read_parquet(args.source_parquet, columns=["pair_id"]).shape[0]),
            seed=args.seed,
            val_ratio=args.val_ratio,
        ),
        stats_path,
    )

    print(f"Wrote {len(train_records)} train {style} rows to {output_path}")
    print(f"Wrote {len(val_records)} val {style} rows to {val_output_path}")
    print(f"Wrote sampling stats to {stats_path}")
