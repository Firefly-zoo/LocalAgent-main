from __future__ import annotations

import os
import statistics
from functools import lru_cache
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageOps


REPO_ROOT = Path(__file__).resolve().parents[2]


def _resolve_image_path(path_like: str) -> Path:
    resolved = Path(path_like)
    if resolved.exists():
        return resolved
    if not resolved.is_absolute():
        candidate = REPO_ROOT / resolved
        if candidate.exists():
            return candidate
    return resolved


def _normalize_box(value: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    try:
        vals = [float(v) for v in value]
    except Exception:
        return None
    if max(vals) > 1.0:
        vals = [v / 1000.0 for v in vals]
    x1, y1, x2, y2 = vals
    x1 = max(0.0, min(1.0, x1))
    y1 = max(0.0, min(1.0, y1))
    x2 = max(0.0, min(1.0, x2))
    y2 = max(0.0, min(1.0, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _box_area(box: tuple[float, float, float, float]) -> float:
    return max(0.0, (box[2] - box[0]) * (box[3] - box[1]))


class ImageStat:
    def __init__(self, image: Image.Image):
        arr = image.resize((32, 32), Image.BICUBIC)
        px = list(arr.getdata())
        chans = list(zip(*px))
        self.mean = [sum(ch) / max(1, len(ch)) for ch in chans]


def _boxed_full(
    image: Image.Image,
    box: tuple[float, float, float, float],
    dim_alpha: float,
    border_width: int,
) -> Image.Image:
    image = image.convert("RGB")
    w, h = image.size
    overlay = Image.new("RGB", (w, h), (128, 128, 128))
    blended = Image.blend(image, overlay, dim_alpha)
    x1 = max(0, min(w - 1, int(round(box[0] * w))))
    y1 = max(0, min(h - 1, int(round(box[1] * h))))
    x2 = max(x1 + 1, min(w, int(round(box[2] * w))))
    y2 = max(y1 + 1, min(h, int(round(box[3] * h))))
    patch = image.crop((x1, y1, x2, y2))
    blended.paste(patch, (x1, y1))
    draw = ImageDraw.Draw(blended)
    for offset in range(max(1, border_width)):
        draw.rectangle((x1 - offset, y1 - offset, x2 + offset, y2 + offset), outline=(255, 64, 64))
    return blended


def _canonical_crop(
    image: Image.Image,
    box: tuple[float, float, float, float],
    context_ratio: float,
    canvas_size: int,
) -> Image.Image:
    image = image.convert("RGB")
    w, h = image.size
    x1 = box[0] * w
    y1 = box[1] * h
    x2 = box[2] * w
    y2 = box[3] * h
    bw = x2 - x1
    bh = y2 - y1
    padx = bw * context_ratio
    pady = bh * context_ratio
    ix1 = max(0, int(round(x1 - padx)))
    iy1 = max(0, int(round(y1 - pady)))
    ix2 = min(w, int(round(x2 + padx)))
    iy2 = min(h, int(round(y2 + pady)))
    crop = image.crop((ix1, iy1, max(ix1 + 1, ix2), max(iy1 + 1, iy2)))

    canvas = Image.new("RGB", (canvas_size, canvas_size), tuple(int(v) for v in ImageStat(crop).mean))
    fitted = ImageOps.contain(crop, (canvas_size, canvas_size), Image.BICUBIC)
    ox = (canvas_size - fitted.width) // 2
    oy = (canvas_size - fitted.height) // 2
    canvas.paste(fitted, (ox, oy))
    return canvas


def _rgb_autocontrast(image: Image.Image) -> Image.Image:
    return ImageOps.autocontrast(image.convert("RGB"))


def _gray_equalized_rgb(image: Image.Image) -> Image.Image:
    gray = ImageOps.grayscale(image)
    gray = ImageOps.equalize(gray)
    return Image.merge("RGB", (gray, gray, gray))


def _flip(image: Image.Image) -> Image.Image:
    return ImageOps.mirror(image.convert("RGB"))


def _safe_round(value: float | None, digits: int = 6) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def _measurement_quality(score_main: float, score_dispersion: float) -> str:
    if score_main >= 0.28 and score_dispersion <= 0.08:
        return "high"
    if score_main >= 0.14 and score_dispersion <= 0.14:
        return "medium"
    return "low"


def _score_profile(score_main: float) -> str:
    if score_main >= 0.32:
        return "strong_agreement"
    if score_main >= 0.20:
        return "moderate_agreement"
    if score_main >= 0.10:
        return "weak_agreement"
    return "very_weak_agreement"


def _resolve_device(requested: str) -> str:
    requested = str(requested or "cuda").strip().lower()
    import torch

    if requested.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return requested


@lru_cache(maxsize=4)
def _get_runtime(model_path: str, device_name: str):
    import torch
    import torch.nn.functional as F
    from transformers import AutoImageProcessor, AutoModel

    resolved_device = _resolve_device(device_name)
    processor = AutoImageProcessor.from_pretrained(model_path, use_fast=False)
    model = AutoModel.from_pretrained(model_path)
    model = model.to(resolved_device)
    model.eval()
    return processor, model, resolved_device, torch, F


def _encode_variants(images: list[Image.Image], *, model_path: str, device_name: str, batch_size: int):
    processor, model, device, torch, F = _get_runtime(model_path, device_name)
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(images), max(1, batch_size)):
            batch = images[start:start + max(1, batch_size)]
            inputs = processor(images=batch, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            last_hidden = model(**inputs).last_hidden_state[:, 1:]
            last_hidden = F.normalize(last_hidden, dim=-1)
            outputs.extend(last_hidden.unbind(0))
    return outputs, torch


def _patch_mnn_ratio(patch_1: Any, patch_2: Any, threshold: float, torch_module) -> float:
    sim = patch_1 @ patch_2.transpose(0, 1)
    nn12 = sim.argmax(dim=1)
    nn21 = sim.argmax(dim=0)
    idx = torch_module.arange(sim.shape[0], device=sim.device)
    mutual = nn21[nn12] == idx
    mutual_scores = sim[idx, nn12]
    valid_mutual = mutual & (mutual_scores >= threshold)
    return float(valid_mutual.float().mean().detach().cpu().item())


def measure_dino_patch_mnn(
    image_1_path: str,
    image_2_path: str,
    box_1: list[float],
    box_2: list[float],
    focus_hint: str | None = None,
) -> dict[str, Any]:
    normalized_box_1 = _normalize_box(box_1)
    normalized_box_2 = _normalize_box(box_2)
    if normalized_box_1 is None or normalized_box_2 is None:
        return {
            "status": "invalid_request",
            "error": "Both `box_1` and `box_2` must be valid normalized [x1, y1, x2, y2] boxes.",
        }

    box_area_1 = _box_area(normalized_box_1)
    box_area_2 = _box_area(normalized_box_2)
    min_box_area = float(os.getenv("TWIN_DINO_PATCH_MNN_MIN_BOX_AREA", "0.002"))
    max_box_area = float(os.getenv("TWIN_DINO_PATCH_MNN_MAX_BOX_AREA", "0.40"))
    if box_area_1 < min_box_area or box_area_2 < min_box_area:
        return {
            "status": "invalid_request",
            "error": "Boxes are too small for reliable local measurement.",
            "box_area_1": _safe_round(box_area_1, 4),
            "box_area_2": _safe_round(box_area_2, 4),
        }
    if box_area_1 > max_box_area or box_area_2 > max_box_area:
        return {
            "status": "invalid_request",
            "error": "Boxes are too large. Use a more local component-level region.",
            "box_area_1": _safe_round(box_area_1, 4),
            "box_area_2": _safe_round(box_area_2, 4),
        }

    model_path = os.getenv("TWIN_DINO_PATCH_MNN_MODEL_PATH", str(REPO_ROOT / "models" / "dinov2-small-local"))
    device_name = os.getenv("TWIN_DINO_PATCH_MNN_DEVICE", "cuda")
    batch_size = int(os.getenv("TWIN_DINO_PATCH_MNN_BATCH_SIZE", "8"))
    dim_alpha = float(os.getenv("TWIN_DINO_PATCH_MNN_BOXED_DIM_ALPHA", "0.45"))
    border_width = int(os.getenv("TWIN_DINO_PATCH_MNN_BOXED_BORDER_WIDTH", "3"))
    context_ratio = float(os.getenv("TWIN_DINO_PATCH_MNN_CANONICAL_CONTEXT_RATIO", "0.10"))
    canvas_size = int(os.getenv("TWIN_DINO_PATCH_MNN_CANONICAL_CANVAS_SIZE", "448"))
    mnn_threshold = float(os.getenv("TWIN_DINO_PATCH_MNN_MNN_THRESHOLD", "0.60"))

    img1 = Image.open(_resolve_image_path(image_1_path))
    img2 = Image.open(_resolve_image_path(image_2_path))
    img1 = ImageOps.exif_transpose(img1).convert("RGB")
    img2 = ImageOps.exif_transpose(img2).convert("RGB")

    boxed1 = _boxed_full(img1, normalized_box_1, dim_alpha, border_width)
    boxed2 = _boxed_full(img2, normalized_box_2, dim_alpha, border_width)
    canon1 = _canonical_crop(img1, normalized_box_1, context_ratio, canvas_size)
    canon2 = _canonical_crop(img2, normalized_box_2, context_ratio, canvas_size)

    variants_1 = {
        "boxed_base": boxed1,
        "boxed_color": _rgb_autocontrast(boxed1),
        "boxed_gray": _gray_equalized_rgb(boxed1),
        "boxed_flip": _flip(boxed1),
        "canonical_base": canon1,
        "canonical_color": _rgb_autocontrast(canon1),
        "canonical_gray": _gray_equalized_rgb(canon1),
        "canonical_flip": _flip(canon1),
    }
    variants_2 = {
        "boxed_base": boxed2,
        "boxed_color": _rgb_autocontrast(boxed2),
        "boxed_gray": _gray_equalized_rgb(boxed2),
        "boxed_flip": _flip(boxed2),
        "canonical_base": canon2,
        "canonical_color": _rgb_autocontrast(canon2),
        "canonical_gray": _gray_equalized_rgb(canon2),
        "canonical_flip": _flip(canon2),
    }
    variant_names = list(variants_1.keys())

    encoded_1, torch_module = _encode_variants(
        [variants_1[name] for name in variant_names],
        model_path=model_path,
        device_name=device_name,
        batch_size=batch_size,
    )
    encoded_2, _ = _encode_variants(
        [variants_2[name] for name in variant_names],
        model_path=model_path,
        device_name=device_name,
        batch_size=batch_size,
    )

    variant_scores: dict[str, float] = {}
    for index, name in enumerate(variant_names):
        variant_scores[name] = _patch_mnn_ratio(
            encoded_1[index],
            encoded_2[index],
            threshold=mnn_threshold,
            torch_module=torch_module,
        )

    all_scores = [float(variant_scores[name]) for name in variant_names]
    all_sorted = sorted(all_scores, reverse=True)
    top2 = all_sorted[:2] if len(all_sorted) >= 2 else all_sorted
    score_main = float(sum(top2) / len(top2)) if top2 else 0.0
    score_median = float(statistics.median(all_scores)) if all_scores else 0.0
    score_dispersion = float(statistics.pstdev(all_scores)) if len(all_scores) > 1 else 0.0
    stable_high_views = sum(1 for score in all_scores if score >= max(0.10, mnn_threshold - 0.20))
    top_views = sorted(
        [{"name": name, "score": _safe_round(score)} for name, score in variant_scores.items()],
        key=lambda item: float(item["score"] or 0.0),
        reverse=True,
    )[:2]

    return {
        "status": "success",
        "tool_name": "measure_dino_patch_mnn",
        "box_valid": True,
        "measurement_quality": _measurement_quality(score_main, score_dispersion),
        "score_profile": _score_profile(score_main),
        "score_main": _safe_round(score_main),
        "score_median": _safe_round(score_median),
        "score_dispersion": _safe_round(score_dispersion),
        "stable_high_views": int(stable_high_views),
        "box_area_1": _safe_round(box_area_1, 4),
        "box_area_2": _safe_round(box_area_2, 4),
        "patch_count_1": int(encoded_1[0].shape[0]) if encoded_1 else 0,
        "patch_count_2": int(encoded_2[0].shape[0]) if encoded_2 else 0,
        "mnn_threshold": _safe_round(mnn_threshold, 4),
        "focus_hint": str(focus_hint or "").strip(),
        "top_views": top_views,
        "variant_scores": {name: _safe_round(value) for name, value in variant_scores.items()},
    }
