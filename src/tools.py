from __future__ import annotations

from PIL import Image


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _crop_by_normalized_box(
    image: Image.Image,
    left: float,
    top: float,
    right: float,
    bottom: float,
) -> Image.Image:
    width, height = image.size
    x1 = int(_clamp01(left) * width)
    y1 = int(_clamp01(top) * height)
    x2 = int(_clamp01(right) * width)
    y2 = int(_clamp01(bottom) * height)
    if x2 <= x1:
        x2 = min(width, x1 + 1)
    if y2 <= y1:
        y2 = min(height, y1 + 1)
    return image.crop((x1, y1, x2, y2))


def _resize_to_height(image: Image.Image, target_height: int) -> Image.Image:
    if image.height == target_height:
        return image
    target_width = max(1, int(image.width * target_height / image.height))
    return image.resize((target_width, target_height))


def compare_regions(
    img1: Image.Image,
    img1_left: float,
    img1_top: float,
    img1_right: float,
    img1_bottom: float,
    img2: Image.Image,
    img2_left: float,
    img2_top: float,
    img2_right: float,
    img2_bottom: float,
) -> Image.Image:
    """
    Compare two cropped regions and return a side-by-side visualization image.

    Args:
        img1:
            Original image 1 as a PIL image object.
        img1_left:
            Left coordinate of image 1 crop in normalized [0, 1].
        img1_top:
            Top coordinate of image 1 crop in normalized [0, 1].
        img1_right:
            Right coordinate of image 1 crop in normalized [0, 1].
        img1_bottom:
            Bottom coordinate of image 1 crop in normalized [0, 1].
        img2:
            Original image 2 as a PIL image object.
        img2_left:
            Left coordinate of image 2 crop in normalized [0, 1].
        img2_top:
            Top coordinate of image 2 crop in normalized [0, 1].
        img2_right:
            Right coordinate of image 2 crop in normalized [0, 1].
        img2_bottom:
            Bottom coordinate of image 2 crop in normalized [0, 1].

    Returns:
        A PIL image where the two crops are resized to equal height and
        concatenated left-right with a 4px gray divider.
    """
    if not isinstance(img1, Image.Image):
        raise TypeError(f"img1 must be PIL.Image.Image, got {type(img1)}")
    if not isinstance(img2, Image.Image):
        raise TypeError(f"img2 must be PIL.Image.Image, got {type(img2)}")

    crop_1 = _crop_by_normalized_box(img1.convert("RGB"), img1_left, img1_top, img1_right, img1_bottom)
    crop_2 = _crop_by_normalized_box(img2.convert("RGB"), img2_left, img2_top, img2_right, img2_bottom)

    target_h = max(crop_1.height, crop_2.height)

    # Prevent extremely tall tool outputs from breaking Qwen-VL aspect ratio limit (e.g., width=1, height>200)
    # prevent tall
    # We clip the target_h to a reasonable multiple of the width to ensure ratio <= 150
    crop_1_w_min = max(crop_1.width, 1)
    crop_2_w_min = max(crop_2.width, 1)
    max_h_for_crop1 = crop_1_w_min * 100 
    max_h_for_crop2 = crop_2_w_min * 100
    target_h = min(target_h, max_h_for_crop1, max_h_for_crop2)
    # Also enforce absolute max dimension to prevent OOM
    target_h = min(target_h, 1024)

    crop_1_viz = _resize_to_height(crop_1, target_h)
    crop_2_viz = _resize_to_height(crop_2, target_h)

    divider_w = 4
    total_width = crop_1_viz.width + divider_w + crop_2_viz.width
    
    # prevent wide: Prevent the concatenated image from being extremely wide
    if total_width > target_h * 150:
        target_h = max(16, total_width // 100) # Scale height up to fix extreme wide aspect ratio
        crop_1_viz = _resize_to_height(crop_1, target_h)
        crop_2_viz = _resize_to_height(crop_2, target_h)
        total_width = crop_1_viz.width + divider_w + crop_2_viz.width
        
    canvas_h = max(crop_1_viz.height, crop_2_viz.height, target_h)
    canvas = Image.new("RGB", (total_width, canvas_h), (128, 128, 128))
    canvas.paste(crop_1_viz, (0, 0))
    canvas.paste(crop_2_viz, (crop_1_viz.width + divider_w, 0))
    return canvas
