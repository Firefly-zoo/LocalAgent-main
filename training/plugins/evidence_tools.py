from __future__ import annotations

import colorsys
import os
import re
from functools import lru_cache
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageOps


MAX_SIDE = 768
LOWRES_SIZE = 24
COLOR_SAMPLE_SIZE = 96


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _safe_unit(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-8:
        return np.zeros_like(vec)
    return vec / norm


def _load_image(path: str) -> Image.Image:
    image = Image.open(path)
    image = ImageOps.exif_transpose(image).convert("RGB")
    longest = max(image.size)
    if longest <= MAX_SIDE:
        return image
    scale = MAX_SIDE / float(longest)
    new_size = (max(32, int(round(image.width * scale))), max(32, int(round(image.height * scale))))
    return image.resize(new_size, Image.BICUBIC)


def _normalize_box(box: Iterable[float]) -> tuple[float, float, float, float]:
    values = [float(v) for v in box]
    if len(values) != 4:
        raise ValueError("Region box must contain exactly four numbers.")
    if max(values) > 1.0:
        values = [v / 1000.0 for v in values]
    x1, y1, x2, y2 = values
    x1 = _clamp01(x1)
    y1 = _clamp01(y1)
    x2 = _clamp01(x2)
    y2 = _clamp01(y2)
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Invalid normalized box with non-positive width or height.")
    return x1, y1, x2, y2


def _crop_box(image: Image.Image, box: tuple[float, float, float, float]) -> Image.Image:
    w, h = image.size
    x1 = max(0, min(w - 1, int(round(box[0] * w))))
    y1 = max(0, min(h - 1, int(round(box[1] * h))))
    x2 = max(x1 + 1, min(w, int(round(box[2] * w))))
    y2 = max(y1 + 1, min(h, int(round(box[3] * h))))
    return image.crop((x1, y1, x2, y2))


def _normalize_token(token: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "", token).lower()


def _tokenize_text(text: str) -> list[str]:
    raw_tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9\\-_/]{1,}", text or "")
    tokens = []
    for token in raw_tokens:
        norm = _normalize_token(token)
        if len(norm) >= 2:
            tokens.append(norm)
    return tokens


@lru_cache(maxsize=4)
def _easyocr_reader(model_dir: str, gpu: bool):
    import easyocr  # type: ignore

    kwargs: dict[str, Any] = {"gpu": gpu}
    if model_dir:
        kwargs["model_storage_directory"] = model_dir
        kwargs["download_enabled"] = False
    try:
        return easyocr.Reader(["en"], **kwargs)
    except TypeError:
        kwargs.pop("download_enabled", None)
        return easyocr.Reader(["en"], **kwargs)


def compare_region_text(
    image_1_path: str,
    image_2_path: str,
    box_1: Iterable[float],
    box_2: Iterable[float],
    focus_hint: str | None = None,
    model_dir: str | None = None,
    gpu: bool | None = None,
) -> dict[str, Any]:
    model_dir = model_dir or os.environ.get("EASYOCR_MODEL_DIR", "")
    if gpu is None:
        gpu = os.environ.get("EASYOCR_GPU", "true").strip().lower() not in {"0", "false", "no"}

    box_1_norm = _normalize_box(box_1)
    box_2_norm = _normalize_box(box_2)
    image_1 = _load_image(image_1_path)
    image_2 = _load_image(image_2_path)
    crop_1 = np.asarray(_crop_box(image_1, box_1_norm))
    crop_2 = np.asarray(_crop_box(image_2, box_2_norm))

    reader = _easyocr_reader(str(model_dir), bool(gpu))
    lines_1 = [str(x).strip() for x in reader.readtext(crop_1, detail=0, paragraph=False) if str(x).strip()]
    lines_2 = [str(x).strip() for x in reader.readtext(crop_2, detail=0, paragraph=False) if str(x).strip()]
    text_1 = " | ".join(lines_1)
    text_2 = " | ".join(lines_2)
    tokens_1 = _tokenize_text(" ".join(lines_1))
    tokens_2 = _tokenize_text(" ".join(lines_2))

    overlap = sorted(set(tokens_1) & set(tokens_2))
    conflict = sorted((set(tokens_1) ^ set(tokens_2)))
    informative_overlap = [tok for tok in overlap if len(tok) >= 3]
    informative_conflict = [tok for tok in conflict if len(tok) >= 3]

    has_text_1 = bool(tokens_1)
    has_text_2 = bool(tokens_2)
    if informative_overlap and not informative_conflict:
        verdict = "same"
        confidence = "high" if len(informative_overlap) >= 2 else "medium"
    elif informative_conflict and not informative_overlap:
        verdict = "different"
        confidence = "high" if len(informative_conflict) >= 2 else "medium"
    elif not has_text_1 and not has_text_2:
        verdict = "no_text"
        confidence = "low"
    else:
        verdict = "uncertain"
        confidence = "low"

    same_score = _clamp01(0.30 * min(3, len(informative_overlap)) + 0.10 * float(has_text_1 and has_text_2))
    different_score = _clamp01(0.24 * min(3, len(informative_conflict)) + 0.08 * float(has_text_1 != has_text_2))
    if verdict == "different":
        evidence = "Visible text differs across the two selected regions."
    elif verdict == "same":
        evidence = "Visible text matches across the two selected regions."
    elif verdict == "no_text":
        evidence = "No reliable visible text was detected in either selected region."
    else:
        evidence = "Visible text evidence is weak or mixed in the selected regions."

    return {
        "status": "success",
        "tool_kind": "text",
        "verdict": verdict,
        "confidence": confidence,
        "same_score": round(float(same_score), 4),
        "different_score": round(float(different_score), 4),
        "text_1": text_1,
        "text_2": text_2,
        "matched_tokens": informative_overlap[:12],
        "conflicting_tokens": informative_conflict[:12],
        "has_text_1": bool(has_text_1),
        "has_text_2": bool(has_text_2),
        "token_count_1": int(len(tokens_1)),
        "token_count_2": int(len(tokens_2)),
        "box_1": [round(v, 4) for v in box_1_norm],
        "box_2": [round(v, 4) for v in box_2_norm],
        "focus_hint": str(focus_hint or ""),
        "evidence": evidence,
    }


def _sample_pixels(image: Image.Image) -> np.ndarray:
    image = image.resize((COLOR_SAMPLE_SIZE, COLOR_SAMPLE_SIZE), Image.BILINEAR)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    return arr.reshape(-1, 3)


def _rgb_to_name(rgb: Iterable[float]) -> str:
    r, g, b = [float(x) for x in rgb]
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    if v <= 0.12:
        return "black"
    if s <= 0.12 and v >= 0.88:
        return "white"
    if s <= 0.16:
        return "gray"
    if 0.06 <= h < 0.12 and v < 0.65:
        return "brown"
    if h < 0.04 or h >= 0.96:
        return "red"
    if h < 0.08:
        return "orange"
    if h < 0.16:
        return "yellow"
    if h < 0.42:
        return "green"
    if h < 0.54:
        return "cyan"
    if h < 0.72:
        return "blue"
    if h < 0.84:
        return "purple"
    return "pink"


def _dominant_color_names(image: Image.Image) -> list[str]:
    pixels = _sample_pixels(image)
    counts: dict[str, int] = {}
    for pixel in pixels:
        name = _rgb_to_name(pixel)
        counts[name] = counts.get(name, 0) + 1
    return [name for name, _ in sorted(counts.items(), key=lambda x: x[1], reverse=True)[:3]]


def compare_region_color_attributes(
    image_1_path: str,
    image_2_path: str,
    box_1: Iterable[float],
    box_2: Iterable[float],
    focus_hint: str | None = None,
) -> dict[str, Any]:
    box_1_norm = _normalize_box(box_1)
    box_2_norm = _normalize_box(box_2)
    image_1 = _crop_box(_load_image(image_1_path), box_1_norm)
    image_2 = _crop_box(_load_image(image_2_path), box_2_norm)
    names_1 = _dominant_color_names(image_1)
    names_2 = _dominant_color_names(image_2)

    shared = sorted(set(names_1) & set(names_2))
    only_1 = sorted(set(names_1) - set(names_2))
    only_2 = sorted(set(names_2) - set(names_1))

    same_score = _clamp01(len(shared) / 3.0 + 0.08 * float(len(shared) >= 2))
    different_score = _clamp01((len(only_1) + len(only_2)) / 6.0)
    if len(shared) >= 2:
        verdict = "compatible"
        confidence = "medium"
    elif len(shared) == 0 and names_1 and names_2:
        verdict = "conflicting"
        confidence = "medium"
    else:
        verdict = "weak"
        confidence = "low"

    if verdict == "compatible":
        evidence = "The two selected regions share similar dominant color attributes."
    elif verdict == "conflicting":
        evidence = "The two selected regions show conflicting dominant color attributes."
    else:
        evidence = "Color evidence from the selected regions is weak or mixed."

    return {
        "status": "success",
        "tool_kind": "color",
        "verdict": verdict,
        "confidence": confidence,
        "same_score": round(float(same_score), 4),
        "different_score": round(float(different_score), 4),
        "colors_1": names_1,
        "colors_2": names_2,
        "shared_attributes": shared,
        "conflicting_attributes": sorted(only_1 + only_2),
        "box_1": [round(v, 4) for v in box_1_norm],
        "box_2": [round(v, 4) for v in box_2_norm],
        "focus_hint": str(focus_hint or ""),
        "evidence": evidence,
    }


def _to_gray_array(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("L"), dtype=np.float32) / 255.0


def _resize_gray(gray: np.ndarray, size: int) -> np.ndarray:
    image = Image.fromarray(np.clip(gray * 255.0, 0.0, 255.0).astype(np.uint8), mode="L")
    image = image.resize((size, size), Image.BILINEAR)
    return np.asarray(image, dtype=np.float32) / 255.0


def _edge_features(gray: np.ndarray) -> tuple[np.ndarray, float]:
    gy, gx = np.gradient(gray)
    magnitude = np.sqrt(gx * gx + gy * gy)
    orientation = (np.arctan2(gy, gx) + np.pi) / (2.0 * np.pi)
    bins = np.floor(orientation * 8.0).astype(np.int32) % 8
    hist = np.bincount(bins.reshape(-1), weights=magnitude.reshape(-1), minlength=8).astype(np.float32)
    return _safe_unit(hist), float(np.mean(magnitude))


def _structure_descriptor(image: Image.Image) -> dict[str, Any]:
    gray = _to_gray_array(image)
    lowres = _resize_gray(gray, LOWRES_SIZE)
    lowres = lowres - float(lowres.mean())
    std = float(lowres.std())
    if std > 1e-6:
        lowres = lowres / std
    else:
        lowres = np.zeros_like(lowres)
    edge_hist, edge_density = _edge_features(gray)
    return {
        "lowres": lowres.reshape(-1),
        "edge_hist": edge_hist,
        "edge_density": edge_density,
        "contrast": float(gray.std()),
    }


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-8:
        return 0.0
    return float(np.clip(np.dot(a, b) / denom, -1.0, 1.0))


def compare_region_structure(
    image_1_path: str,
    image_2_path: str,
    box_1: Iterable[float],
    box_2: Iterable[float],
    focus_hint: str | None = None,
) -> dict[str, Any]:
    box_1_norm = _normalize_box(box_1)
    box_2_norm = _normalize_box(box_2)
    crop_1 = _crop_box(_load_image(image_1_path), box_1_norm)
    crop_2 = _crop_box(_load_image(image_2_path), box_2_norm)
    desc_1 = _structure_descriptor(crop_1)
    desc_2 = _structure_descriptor(crop_2)

    struct_sim = max(0.0, _cosine_similarity(desc_1["lowres"], desc_2["lowres"]))
    edge_sim = max(0.0, _cosine_similarity(desc_1["edge_hist"], desc_2["edge_hist"]))
    contrast_gap = abs(float(desc_1["contrast"]) - float(desc_2["contrast"]))
    density_gap = abs(float(desc_1["edge_density"]) - float(desc_2["edge_density"]))

    same_score = _clamp01(0.62 * struct_sim + 0.30 * edge_sim - 0.08 * min(1.0, contrast_gap * 4.0))
    different_score = _clamp01(
        0.46 * (1.0 - struct_sim) + 0.30 * (1.0 - edge_sim) + 0.14 * min(1.0, contrast_gap * 4.0)
        + 0.10 * min(1.0, density_gap * 4.0)
    )

    if struct_sim >= 0.74 and edge_sim >= 0.68:
        verdict = "compatible"
        confidence = "high" if struct_sim >= 0.82 else "medium"
    elif struct_sim <= 0.48 and edge_sim <= 0.52:
        verdict = "conflicting"
        confidence = "high" if struct_sim <= 0.35 else "medium"
    else:
        verdict = "uncertain"
        confidence = "low"

    key_conflicts: list[str] = []
    if struct_sim <= 0.50:
        key_conflicts.append("local shape topology mismatch")
    if edge_sim <= 0.50:
        key_conflicts.append("edge layout mismatch")
    if contrast_gap >= 0.10:
        key_conflicts.append("local contrast pattern differs")

    if verdict == "compatible":
        evidence = "The selected regions are structurally compatible."
    elif verdict == "conflicting":
        evidence = "The selected regions show structural incompatibility."
    else:
        evidence = "Structural evidence from the selected regions is mixed."

    return {
        "status": "success",
        "tool_kind": "structure",
        "verdict": verdict,
        "confidence": confidence,
        "same_score": round(float(same_score), 4),
        "different_score": round(float(different_score), 4),
        "shape_match": round(float(struct_sim), 4),
        "pattern_match": round(float(edge_sim), 4),
        "contrast_gap": round(float(contrast_gap), 4),
        "key_conflicts": key_conflicts,
        "box_1": [round(v, 4) for v in box_1_norm],
        "box_2": [round(v, 4) for v in box_2_norm],
        "focus_hint": str(focus_hint or ""),
        "evidence": evidence,
    }
