from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageOps
from scipy import ndimage as ndi
from skimage import filters, measure, morphology, transform
from skimage.feature import ORB, match_descriptors
from skimage.measure import ransac
from skimage.transform import ProjectiveTransform


MAX_SIDE = 768
ALIGN_SIZE = 192
MIN_BOX_AREA = 0.002
MAX_BOX_AREA = 0.50
MIN_FOREGROUND_RATIO = 0.08
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


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


def _box_area(box: tuple[float, float, float, float]) -> float:
    return float((box[2] - box[0]) * (box[3] - box[1]))


def _load_image(path: str) -> Image.Image:
    resolved = Path(path)
    if not resolved.exists():
        raw = str(path)
        if "/mmbench/" in raw:
            resolved = PROJECT_ROOT / raw.split("/mmbench/", 1)[1]
    image = Image.open(resolved)
    image = ImageOps.exif_transpose(image).convert("RGB")
    longest = max(image.size)
    if longest <= MAX_SIDE:
        return image
    scale = MAX_SIDE / float(longest)
    new_size = (max(32, int(round(image.width * scale))), max(32, int(round(image.height * scale))))
    return image.resize(new_size, Image.BICUBIC)


def _crop_box(image: Image.Image, box: tuple[float, float, float, float]) -> Image.Image:
    w, h = image.size
    x1 = max(0, min(w - 1, int(round(box[0] * w))))
    y1 = max(0, min(h - 1, int(round(box[1] * h))))
    x2 = max(x1 + 1, min(w, int(round(box[2] * w))))
    y2 = max(y1 + 1, min(h, int(round(box[3] * h))))
    return image.crop((x1, y1, x2, y2))


def _crop_to_array(image: Image.Image, box: tuple[float, float, float, float]) -> np.ndarray:
    return np.asarray(_crop_box(image, box), dtype=np.float32) / 255.0


def _largest_component(mask: np.ndarray) -> np.ndarray:
    labels = measure.label(mask.astype(np.uint8), connectivity=2)
    if labels.max() <= 0:
        return mask.astype(bool)
    best_score = float("-inf")
    best_mask = mask.astype(bool)
    h, w = mask.shape
    cy = (h - 1) / 2.0
    cx = (w - 1) / 2.0
    for component_id in range(1, labels.max() + 1):
        comp = labels == component_id
        area = float(comp.sum())
        if area <= 0:
            continue
        ys, xs = np.nonzero(comp)
        center_penalty = ((ys.mean() - cy) ** 2 + (xs.mean() - cx) ** 2) / max(1.0, h * w)
        score = area - 0.35 * center_penalty * area
        if score > best_score:
            best_score = score
            best_mask = comp
    return best_mask


def _extract_foreground_mask(crop_rgb: np.ndarray) -> np.ndarray:
    gray = np.asarray(
        Image.fromarray(np.clip(crop_rgb * 255.0, 0.0, 255.0).astype(np.uint8)).convert("L"),
        dtype=np.float32,
    ) / 255.0
    hsv = np.asarray(
        Image.fromarray(np.clip(crop_rgb * 255.0, 0.0, 255.0).astype(np.uint8)).convert("HSV"),
        dtype=np.float32,
    ) / 255.0
    sat = hsv[..., 1]
    edge = filters.sobel(gray)

    h, w = gray.shape
    border = np.zeros_like(gray, dtype=bool)
    bw = max(1, int(round(min(h, w) * 0.10)))
    border[:bw, :] = True
    border[-bw:, :] = True
    border[:, :bw] = True
    border[:, -bw:] = True
    border_rgb = crop_rgb[border]
    border_mean = border_rgb.mean(axis=0) if border_rgb.size else crop_rgb.reshape(-1, 3).mean(axis=0)
    color_dist = np.linalg.norm(crop_rgb - border_mean[None, None, :], axis=-1)

    center_y = np.linspace(-1.0, 1.0, h, dtype=np.float32)[:, None]
    center_x = np.linspace(-1.0, 1.0, w, dtype=np.float32)[None, :]
    center_prior = 1.0 - np.sqrt(center_x * center_x + center_y * center_y) / np.sqrt(2.0)
    center_prior = np.clip(center_prior, 0.0, 1.0)

    score = 0.50 * color_dist + 0.25 * edge + 0.15 * sat + 0.10 * center_prior
    score = score - score.min()
    max_score = float(score.max())
    if max_score > 1e-6:
        score = score / max_score
    else:
        return np.ones_like(gray, dtype=bool)

    try:
        threshold = float(filters.threshold_otsu(score))
    except ValueError:
        threshold = float(np.quantile(score, 0.65))
    threshold = max(threshold, float(np.quantile(score, 0.60)))
    mask = score >= threshold
    mask = morphology.binary_closing(mask, morphology.disk(2))
    mask = morphology.binary_opening(mask, morphology.disk(1))
    mask = ndi.binary_fill_holes(mask)
    if float(mask.mean()) < 0.02:
        mask = score >= float(np.quantile(score, 0.75))
        mask = ndi.binary_fill_holes(mask)
    return _largest_component(mask).astype(bool)


@dataclass
class _PreparedRegion:
    gray: np.ndarray
    mask: np.ndarray
    refined_box_local: tuple[int, int, int, int]


def _prepare_region(crop_rgb: np.ndarray) -> _PreparedRegion:
    mask = _extract_foreground_mask(crop_rgb)
    ys, xs = np.nonzero(mask)
    h, w = mask.shape
    if len(xs) == 0 or len(ys) == 0:
        refined = (0, 0, w, h)
        mask = np.ones((h, w), dtype=bool)
    else:
        pad_x = max(2, int(round((xs.max() - xs.min() + 1) * 0.06)))
        pad_y = max(2, int(round((ys.max() - ys.min() + 1) * 0.06)))
        x1 = max(0, int(xs.min()) - pad_x)
        y1 = max(0, int(ys.min()) - pad_y)
        x2 = min(w, int(xs.max()) + 1 + pad_x)
        y2 = min(h, int(ys.max()) + 1 + pad_y)
        refined = (x1, y1, x2, y2)
        crop_rgb = crop_rgb[y1:y2, x1:x2]
        mask = mask[y1:y2, x1:x2]

    gray = np.asarray(
        Image.fromarray(np.clip(crop_rgb * 255.0, 0.0, 255.0).astype(np.uint8)).convert("L"),
        dtype=np.float32,
    ) / 255.0
    return _PreparedRegion(gray=gray, mask=mask.astype(bool), refined_box_local=refined)


def _resize_pair(gray: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gray_resized = transform.resize(
        gray,
        (ALIGN_SIZE, ALIGN_SIZE),
        order=1,
        preserve_range=True,
        anti_aliasing=True,
    ).astype(np.float32)
    mask_resized = transform.resize(
        mask.astype(np.float32),
        (ALIGN_SIZE, ALIGN_SIZE),
        order=0,
        preserve_range=True,
        anti_aliasing=False,
    ) >= 0.5
    return gray_resized, mask_resized.astype(bool)


def _global_refined_box(
    box: tuple[float, float, float, float],
    prepared: _PreparedRegion,
    crop_shape: tuple[int, int],
) -> list[float]:
    crop_h, crop_w = crop_shape
    local_x1, local_y1, local_x2, local_y2 = prepared.refined_box_local
    box_w = box[2] - box[0]
    box_h = box[3] - box[1]
    x1 = box[0] + box_w * (local_x1 / float(crop_w))
    y1 = box[1] + box_h * (local_y1 / float(crop_h))
    x2 = box[0] + box_w * (local_x2 / float(crop_w))
    y2 = box[1] + box_h * (local_y2 / float(crop_h))
    return [round(_clamp01(x1), 4), round(_clamp01(y1), 4), round(_clamp01(x2), 4), round(_clamp01(y2), 4)]


def _extract_orb(gray: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    image = gray.copy()
    image[~mask] = 0.0
    orb = ORB(n_keypoints=300, fast_threshold=0.05)
    try:
        orb.detect_and_extract(image)
    except Exception:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 256), dtype=bool)
    keypoints = np.asarray(orb.keypoints, dtype=np.float32)
    descriptors = np.asarray(orb.descriptors, dtype=bool)
    if keypoints.size == 0 or descriptors.size == 0:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 256), dtype=bool)
    keep = []
    for idx, (y, x) in enumerate(keypoints):
        iy = int(np.clip(round(y), 0, mask.shape[0] - 1))
        ix = int(np.clip(round(x), 0, mask.shape[1] - 1))
        keep.append(bool(mask[iy, ix]))
    keep_mask = np.asarray(keep, dtype=bool)
    if not keep_mask.any():
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 256), dtype=bool)
    return keypoints[keep_mask], descriptors[keep_mask]


def _estimate_geometry_metrics(
    gray_1: np.ndarray,
    mask_1: np.ndarray,
    gray_2: np.ndarray,
    mask_2: np.ndarray,
) -> dict[str, Any]:
    kp_1, desc_1 = _extract_orb(gray_1, mask_1)
    kp_2, desc_2 = _extract_orb(gray_2, mask_2)
    out = {
        "keypoint_count_1": int(len(kp_1)),
        "keypoint_count_2": int(len(kp_2)),
        "match_count": 0,
        "inlier_match_count": 0,
        "inlier_ratio": 0.0,
        "homography_residual": 1.0,
        "mean_match_hamming": None,
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
    hamming = np.logical_xor(desc_1[matches[:, 0]], desc_2[matches[:, 1]]).mean(axis=1)
    out["match_count"] = int(len(matches))
    out["mean_match_hamming"] = round(float(hamming.mean()), 4)

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

    if model is None or inliers is None or int(np.sum(inliers)) <= 0:
        return out

    inlier_count = int(np.sum(inliers))
    projected = model(src[inliers])
    residuals = np.linalg.norm(projected - dst[inliers], axis=1)
    median_residual = float(np.median(residuals)) if residuals.size else 12.0
    out["inlier_match_count"] = inlier_count
    out["inlier_ratio"] = float(inlier_count / max(1, len(matches)))
    out["homography_residual"] = _clamp01(median_residual / 12.0)
    out["ransac_success"] = True
    return out


def measure_local_consistency(
    *,
    image_1_path: str,
    image_2_path: str,
    box_1: Iterable[float],
    box_2: Iterable[float],
    focus_hint: str | None = None,
) -> dict[str, Any]:
    box_1_norm = _normalize_box(box_1)
    box_2_norm = _normalize_box(box_2)
    area_1 = _box_area(box_1_norm)
    area_2 = _box_area(box_2_norm)

    image_1 = _load_image(image_1_path)
    image_2 = _load_image(image_2_path)
    crop_1 = _crop_to_array(image_1, box_1_norm)
    crop_2 = _crop_to_array(image_2, box_2_norm)

    prepared_1 = _prepare_region(crop_1)
    prepared_2 = _prepare_region(crop_2)
    fg_ratio_1 = float(prepared_1.mask.mean())
    fg_ratio_2 = float(prepared_2.mask.mean())

    gray_1, mask_1 = _resize_pair(prepared_1.gray, prepared_1.mask)
    gray_2, mask_2 = _resize_pair(prepared_2.gray, prepared_2.mask)
    geometry = _estimate_geometry_metrics(gray_1, mask_1, gray_2, mask_2)

    enough_kp = min(geometry["keypoint_count_1"], geometry["keypoint_count_2"]) >= 12
    enough_matches = geometry["match_count"] >= 8
    enough_inliers = geometry["inlier_match_count"] >= 6
    box_valid = bool(
        MIN_BOX_AREA <= area_1 <= MAX_BOX_AREA
        and MIN_BOX_AREA <= area_2 <= MAX_BOX_AREA
        and fg_ratio_1 >= MIN_FOREGROUND_RATIO
        and fg_ratio_2 >= MIN_FOREGROUND_RATIO
    )
    quality = "low"
    if box_valid and enough_kp and enough_matches:
        quality = "medium"
    if box_valid and enough_kp and enough_matches and enough_inliers and geometry["homography_residual"] <= 0.45:
        quality = "high"

    consistency = geometry["inlier_ratio"] * min(1.0, geometry["inlier_match_count"] / 12.0)
    if geometry["ransac_success"]:
        consistency *= (1.0 - geometry["homography_residual"])
    consistency = _clamp01(consistency)

    return {
        "status": "success",
        "box_valid": box_valid,
        "measurement_quality": quality,
        "box_area_1": round(area_1, 4),
        "box_area_2": round(area_2, 4),
        "foreground_ratio_1": round(fg_ratio_1, 4),
        "foreground_ratio_2": round(fg_ratio_2, 4),
        "keypoint_count_1": geometry["keypoint_count_1"],
        "keypoint_count_2": geometry["keypoint_count_2"],
        "match_count": geometry["match_count"],
        "inlier_match_count": geometry["inlier_match_count"],
        "inlier_ratio": round(_clamp01(geometry["inlier_ratio"]), 4),
        "homography_residual": round(_clamp01(geometry["homography_residual"]), 4),
        "mean_match_hamming": geometry["mean_match_hamming"],
        "ransac_success": bool(geometry["ransac_success"]),
        "geometry_consistency": round(consistency, 4),
        "refined_box_1": _global_refined_box(box_1_norm, prepared_1, crop_1.shape[:2]),
        "refined_box_2": _global_refined_box(box_2_norm, prepared_2, crop_2.shape[:2]),
        "focus_hint": str(focus_hint or ""),
    }
