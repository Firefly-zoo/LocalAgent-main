from __future__ import annotations

import re
from pathlib import Path
from typing import Any


VALID_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}


def _as_non_empty_str(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return text


def _sanitize_token(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-")
    return sanitized or "unknown"


def build_cache_prefix(row: dict[str, Any]) -> str:
    source = _sanitize_token(_as_non_empty_str(row.get("dataset_source")) or "unknown_source")
    sample_id = _sanitize_token(
        _as_non_empty_str(row.get("sample_id"))
        or _as_non_empty_str(row.get("id"))
        or _as_non_empty_str(row.get("pair_id"))
        or "sample"
    )
    return f"{source}__{sample_id}"


def materialize_images_with_source_sample_key(
    row: dict[str, Any],
    image_cache_dir: Path,
) -> list[str]:
    image_cache_dir.mkdir(parents=True, exist_ok=True)
    cache_prefix = build_cache_prefix(row)
    image_paths: list[str] = []
    for image_idx, image_payload in enumerate(row.get("images", [])):
        source_path = str(image_payload.get("path", ""))
        ext = Path(source_path).suffix.lower()
        if ext not in VALID_IMAGE_EXTS:
            ext = ".jpg"
        out_path = image_cache_dir / f"{cache_prefix}_{image_idx}{ext}"
        if not out_path.exists():
            image_bytes = image_payload.get("bytes")
            if image_bytes is None:
                raise ValueError(f"Row {cache_prefix} image[{image_idx}] has no bytes payload.")
            out_path.write_bytes(image_bytes)
        image_paths.append(str(out_path))
    return image_paths
