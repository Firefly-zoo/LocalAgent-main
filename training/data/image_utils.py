from __future__ import annotations

from io import BytesIO
from typing import Any

from PIL import Image


def decode_images(raw_images: list[dict[str, Any]] | None) -> list[Image.Image]:
    images: list[Image.Image] = []
    for image in raw_images or []:
        if not isinstance(image, dict) or image.get("bytes") is None:
            continue
        decoded = Image.open(BytesIO(image["bytes"]))
        images.append(decoded.convert("RGB") if decoded.mode != "RGB" else decoded)
    return images
