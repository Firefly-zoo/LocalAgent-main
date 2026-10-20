from __future__ import annotations

import os
import statistics
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps
from skimage.feature import match_descriptors
from skimage.measure import ransac
from skimage.transform import ProjectiveTransform


PROJECT_ROOT = Path(__file__).resolve().parents[1]  # mmbench/
_REPO_ROOT = PROJECT_ROOT.parent                     # project root (LocalAgent/)

try:
    from local_measure import (
        MAX_BOX_AREA,
        MIN_BOX_AREA,
        _box_area,
        _clamp01,
        _extract_orb,
        _load_image,
        _normalize_box,
        _prepare_region,
        _resize_pair,
    )
except ModuleNotFoundError:
    from plugins.local_measure import (
        MAX_BOX_AREA,
        MIN_BOX_AREA,
        _box_area,
        _clamp01,
        _extract_orb,
        _load_image,
        _normalize_box,
        _prepare_region,
        _resize_pair,
    )


def _image_stat_mean(image: Image.Image) -> tuple[int, int, int]:
    arr = np.asarray(image.resize((32, 32), Image.BICUBIC), dtype=np.float32)
    mean = arr.reshape(-1, 3).mean(axis=0)
    return tuple(int(round(float(v))) for v in mean)


def _resolve_image_path(path_like: str) -> Path:
    resolved = Path(path_like)
    if resolved.exists():
        return resolved
    raw = str(path_like)
    if "/mmbench/" in raw:
        suffix = raw.split("/mmbench/", 1)[1]
        candidate = PROJECT_ROOT / suffix
        if candidate.exists():
            return candidate
    return resolved


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

    canvas = Image.new("RGB", (canvas_size, canvas_size), _image_stat_mean(crop))
    fitted = ImageOps.contain(crop, (canvas_size, canvas_size), Image.BICUBIC)
    ox = (canvas_size - fitted.width) // 2
    oy = (canvas_size - fitted.height) // 2
    canvas.paste(fitted, (ox, oy))
    return canvas


def _variant_images(image: Image.Image) -> dict[str, Image.Image]:
    gray = ImageOps.grayscale(image)
    equalized = ImageOps.equalize(gray)
    return {
        "canonical_base": image.convert("RGB"),
        "canonical_equalized": Image.merge("RGB", (equalized, equalized, equalized)),
        "canonical_flip": ImageOps.mirror(image.convert("RGB")),
    }


def _crop_to_array(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def _crop_to_uint8(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _grid_coverage(points_xy: np.ndarray, grid_size: int, width: int, height: int) -> float:
    if points_xy.size == 0 or width <= 1 or height <= 1:
        return 0.0
    xs = np.clip((points_xy[:, 0] / float(width)) * grid_size, 0, grid_size - 1e-6)
    ys = np.clip((points_xy[:, 1] / float(height)) * grid_size, 0, grid_size - 1e-6)
    cells = {(int(y), int(x)) for x, y in zip(xs, ys)}
    return float(len(cells) / float(grid_size * grid_size))


def _extract_ransac_residual_quality(
    model: Any,
    inliers: Any,
    src: np.ndarray,
    dst: np.ndarray,
    *,
    grid_size: int,
    width_1: int,
    height_1: int,
    width_2: int,
    height_2: int,
) -> tuple[int, float, float, float] | None:
    if inliers is None:
        return None
    try:
        model_ok = bool(model)
    except Exception:
        model_ok = False
    if not model_ok:
        return None

    inlier_count = int(np.sum(inliers))
    if inlier_count <= 0:
        return None

    src_inliers = src[inliers]
    dst_inliers = dst[inliers]
    if src_inliers.size == 0 or dst_inliers.size == 0:
        return None

    try:
        projected = model(src_inliers)
    except Exception:
        return None
    if projected is None:
        return None
    projected = np.asarray(projected, dtype=np.float32)
    if projected.shape != dst_inliers.shape:
        return None
    if not np.isfinite(projected).all() or not np.isfinite(dst_inliers).all():
        return None

    residuals = np.linalg.norm(projected - dst_inliers, axis=1)
    if residuals.size == 0 or not np.isfinite(residuals).all():
        return None
    median_residual = float(np.median(residuals))
    src_cov = _grid_coverage(src_inliers, grid_size, width_1, height_1)
    dst_cov = _grid_coverage(dst_inliers, grid_size, width_2, height_2)
    coverage = float((src_cov + dst_cov) * 0.5)
    residual_quality = _clamp01(1.0 - (median_residual / 12.0))
    return inlier_count, median_residual, coverage, residual_quality


def _estimate_geometry_metrics_orb(
    gray_1: np.ndarray,
    mask_1: np.ndarray,
    gray_2: np.ndarray,
    mask_2: np.ndarray,
    *,
    grid_size: int,
) -> dict[str, Any]:
    kp_1, desc_1 = _extract_orb(gray_1, mask_1)
    kp_2, desc_2 = _extract_orb(gray_2, mask_2)
    out = {
        "keypoint_count_1": int(len(kp_1)),
        "keypoint_count_2": int(len(kp_2)),
        "raw_match_count": 0,
        "inlier_match_count": 0,
        "inlier_ratio": 0.0,
        "coverage": 0.0,
        "residual_quality": 0.0,
        "variant_score": 0.0,
        "ransac_success": False,
    }
    if len(kp_1) < 4 or len(kp_2) < 4 or len(desc_1) < 4 or len(desc_2) < 4:
        return out

    try:
        matches = match_descriptors(desc_1, desc_2, metric="hamming", cross_check=True, max_ratio=0.9)
    except Exception:
        return out
    if matches.size == 0:
        return out

    src = kp_1[matches[:, 0]][:, ::-1]
    dst = kp_2[matches[:, 1]][:, ::-1]
    out["raw_match_count"] = int(len(matches))

    if len(matches) < 4:
        return out

    try:
        model, inliers = ransac(
            (src, dst),
            ProjectiveTransform,
            min_samples=4,
            residual_threshold=4.0,
            max_trials=200,
        )
    except Exception:
        return out

    extracted = _extract_ransac_residual_quality(
        model,
        inliers,
        src,
        dst,
        grid_size=grid_size,
        width_1=gray_1.shape[1],
        height_1=gray_1.shape[0],
        width_2=gray_2.shape[1],
        height_2=gray_2.shape[0],
    )
    if extracted is None:
        return out

    inlier_count, _median_residual, coverage, residual_quality = extracted
    norm_count = _clamp01(inlier_count / 24.0)
    inlier_ratio = float(inlier_count / max(1, len(matches)))
    variant_score = _clamp01(
        0.40 * inlier_ratio
        + 0.25 * norm_count
        + 0.20 * coverage
        + 0.15 * residual_quality
    )

    out["inlier_match_count"] = inlier_count
    out["inlier_ratio"] = float(inlier_ratio)
    out["coverage"] = float(coverage)
    out["residual_quality"] = float(residual_quality)
    out["variant_score"] = float(variant_score)
    out["ransac_success"] = True
    return out


def _resolve_accelerated_features_root() -> Path | None:
    env_root = os.getenv("TWIN_LOCAL_GEOM_XFEAT_ROOT", "").strip()
    candidates: list[Path] = []
    if env_root:
        candidates.append(Path(env_root))
    candidates.append(_REPO_ROOT / "accelerated_features")
    candidates.append(PROJECT_ROOT / "accelerated_features")
    for candidate in candidates:
        if (candidate / "modules" / "xfeat.py").exists():
            return candidate
    return None


def _resolve_xfeat_weights(root: Path) -> str:
    env_weights = os.getenv("TWIN_LOCAL_GEOM_XFEAT_WEIGHTS", "").strip()
    if env_weights:
        return env_weights
    return str(root / "weights" / "xfeat.pt")


@lru_cache(maxsize=2)
def _get_xfeat_runtime():
    root = _resolve_accelerated_features_root()
    if root is None:
        raise ModuleNotFoundError("accelerated_features root not found")

    root_str = str(root.resolve())
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    import torch  # noqa: WPS433
    from modules.xfeat import XFeat  # type: ignore  # noqa: WPS433

    weights = _resolve_xfeat_weights(root)
    top_k = int(os.getenv("TWIN_LOCAL_GEOM_XFEAT_TOP_K", "512"))
    detection_threshold = float(os.getenv("TWIN_LOCAL_GEOM_XFEAT_DETECTION_THRESHOLD", "0.03"))
    matcher = XFeat(weights=weights, top_k=top_k, detection_threshold=detection_threshold)
    min_cossim = float(os.getenv("TWIN_LOCAL_GEOM_XFEAT_MIN_COSSIM", "0.75"))
    return matcher, torch, min_cossim


def _estimate_geometry_metrics_xfeat(
    image_1: np.ndarray,
    image_2: np.ndarray,
    *,
    grid_size: int,
) -> dict[str, Any]:
    matcher, torch, min_cossim = _get_xfeat_runtime()
    out = {
        "keypoint_count_1": 0,
        "keypoint_count_2": 0,
        "raw_match_count": 0,
        "inlier_match_count": 0,
        "inlier_ratio": 0.0,
        "coverage": 0.0,
        "residual_quality": 0.0,
        "variant_score": 0.0,
        "ransac_success": False,
    }

    with torch.inference_mode():
        feat_1 = matcher.detectAndCompute(image_1, top_k=matcher.top_k)[0]
        feat_2 = matcher.detectAndCompute(image_2, top_k=matcher.top_k)[0]

    kp_1 = feat_1["keypoints"].detach().cpu().numpy()
    kp_2 = feat_2["keypoints"].detach().cpu().numpy()
    desc_1 = feat_1["descriptors"]
    desc_2 = feat_2["descriptors"]
    out["keypoint_count_1"] = int(len(kp_1))
    out["keypoint_count_2"] = int(len(kp_2))
    if len(kp_1) < 4 or len(kp_2) < 4 or len(desc_1) < 4 or len(desc_2) < 4:
        return out

    with torch.inference_mode():
        idx0, idx1 = matcher.match(desc_1, desc_2, min_cossim=min_cossim)
    idx0_np = idx0.detach().cpu().numpy()
    idx1_np = idx1.detach().cpu().numpy()
    if idx0_np.size == 0 or idx1_np.size == 0:
        return out

    src = kp_1[idx0_np]
    dst = kp_2[idx1_np]
    out["raw_match_count"] = int(len(idx0_np))
    if len(src) < 4 or len(dst) < 4:
        return out

    try:
        model, inliers = ransac(
            (src, dst),
            ProjectiveTransform,
            min_samples=4,
            residual_threshold=4.0,
            max_trials=200,
        )
    except Exception:
        return out

    extracted = _extract_ransac_residual_quality(
        model,
        inliers,
        src,
        dst,
        grid_size=grid_size,
        width_1=image_1.shape[1],
        height_1=image_1.shape[0],
        width_2=image_2.shape[1],
        height_2=image_2.shape[0],
    )
    if extracted is None:
        return out

    inlier_count, _median_residual, coverage, residual_quality = extracted
    norm_count = _clamp01(inlier_count / 24.0)
    inlier_ratio = float(inlier_count / max(1, len(src)))
    variant_score = _clamp01(
        0.40 * inlier_ratio
        + 0.25 * norm_count
        + 0.20 * coverage
        + 0.15 * residual_quality
    )

    out["inlier_match_count"] = inlier_count
    out["inlier_ratio"] = float(inlier_ratio)
    out["coverage"] = float(coverage)
    out["residual_quality"] = float(residual_quality)
    out["variant_score"] = float(variant_score)
    out["ransac_success"] = True
    return out


def _stability_from_scores(scores: list[float]) -> float:
    if not scores:
        return 0.0
    peak = max(scores)
    if peak < 0.05:
        return 0.0
    spread = peak - min(scores)
    return _clamp01(1.0 - (spread / max(peak, 1e-6)))


def _select_backend() -> str:
    requested = os.getenv("TWIN_LOCAL_GEOM_BACKEND", "auto").strip().lower()
    if requested in {"orb", "xfeat"}:
        return requested
    try:
        _get_xfeat_runtime()
        return "xfeat"
    except Exception:
        return "orb"


def measure_local_geometry_consistency(
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
    min_box_area = float(os.getenv("TWIN_LOCAL_GEOM_MIN_BOX_AREA", str(MIN_BOX_AREA)))
    max_box_area = float(os.getenv("TWIN_LOCAL_GEOM_MAX_BOX_AREA", str(min(MAX_BOX_AREA, 0.40))))
    if box_area_1 < min_box_area or box_area_2 < min_box_area:
        return {
            "status": "invalid_request",
            "error": "Boxes are too small for reliable local geometry measurement.",
        }
    if box_area_1 > max_box_area or box_area_2 > max_box_area:
        return {
            "status": "invalid_request",
            "error": "Boxes are too large. Use a more local component-level region.",
        }

    context_ratio = float(os.getenv("TWIN_LOCAL_GEOM_CONTEXT_RATIO", "0.12"))
    canvas_size = int(os.getenv("TWIN_LOCAL_GEOM_CANVAS_SIZE", "256"))
    grid_size = int(os.getenv("TWIN_LOCAL_GEOM_GRID_SIZE", "4"))
    backend = _select_backend()

    image_1 = _load_image(str(_resolve_image_path(image_1_path)))
    image_2 = _load_image(str(_resolve_image_path(image_2_path)))

    canonical_1 = _canonical_crop(image_1, normalized_box_1, context_ratio, canvas_size)
    canonical_2 = _canonical_crop(image_2, normalized_box_2, context_ratio, canvas_size)
    variants_1 = _variant_images(canonical_1)
    variants_2 = _variant_images(canonical_2)

    variant_rows: list[dict[str, Any]] = []
    for variant_name in ("canonical_base", "canonical_equalized", "canonical_flip"):
        if backend == "xfeat":
            img_1 = _crop_to_uint8(variants_1[variant_name])
            img_2 = _crop_to_uint8(variants_2[variant_name])
            geometry = _estimate_geometry_metrics_xfeat(img_1, img_2, grid_size=grid_size)
        else:
            crop_1 = _crop_to_array(variants_1[variant_name])
            crop_2 = _crop_to_array(variants_2[variant_name])
            prepared_1 = _prepare_region(crop_1)
            prepared_2 = _prepare_region(crop_2)
            gray_1, mask_1 = _resize_pair(prepared_1.gray, prepared_1.mask)
            gray_2, mask_2 = _resize_pair(prepared_2.gray, prepared_2.mask)
            geometry = _estimate_geometry_metrics_orb(gray_1, mask_1, gray_2, mask_2, grid_size=grid_size)
        geometry["variant"] = variant_name
        variant_rows.append(geometry)

    ranked = sorted(variant_rows, key=lambda row: float(row["variant_score"]), reverse=True)
    top_rows = ranked[:2]
    top_scores = [float(row["variant_score"]) for row in top_rows]
    all_scores = [float(row["variant_score"]) for row in variant_rows]
    geom_stability = _stability_from_scores(all_scores)
    geom_inlier_ratio = float(statistics.mean(row["inlier_ratio"] for row in top_rows)) if top_rows else 0.0
    geom_match_count = int(round(statistics.mean(row["inlier_match_count"] for row in top_rows))) if top_rows else 0
    geom_coverage = float(statistics.mean(row["coverage"] for row in top_rows)) if top_rows else 0.0
    norm_match_count = _clamp01(geom_match_count / 24.0)
    geom_score_main = _clamp01(
        0.40 * geom_inlier_ratio
        + 0.25 * norm_match_count
        + 0.20 * geom_coverage
        + 0.15 * geom_stability
    )

    result = {
        "status": "success",
        "geom_score_main": round(float(geom_score_main), 4),
        "geom_inlier_ratio": round(float(geom_inlier_ratio), 4),
        "geom_match_count": int(geom_match_count),
        "geom_coverage": round(float(geom_coverage), 4),
        "geom_stability": round(float(geom_stability), 4),
    }
    return result
