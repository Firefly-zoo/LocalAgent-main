from __future__ import annotations

from typing import Any

import src.evaluator as base
from src.cache_utils import materialize_images_with_source_sample_key


def evaluate_fgvqa_swift_fixed_cache(config: dict[str, Any]) -> dict[str, Any]:
    original_materialize_images = base._materialize_images
    base._materialize_images = materialize_images_with_source_sample_key
    try:
        return base.evaluate_fgvqa_swift(config)
    finally:
        base._materialize_images = original_materialize_images
