from __future__ import annotations

TWIN_DINO_STRUCTURE_TOOL_FAMILY_SCHEMA_V1 = [
    {
        "type": "function",
        "function": {
            "name": "measure_dino_patch_mnn",
            "description": (
                "Measure local appearance agreement between two user-provided boxes using DINO patch-token mutual-nearest-neighbor matching. "
                "The tool builds several normalized local views and returns auxiliary appearance-agreement scores only. "
                "It does not decide same or different. "
                "It returns score_main, score_median, score_dispersion, stable_high_views, top views, and per-view variant scores."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "image_1_index": {
                        "type": "integer",
                        "description": "Index of image 1 in the current sample, starting at 0.",
                    },
                    "image_2_index": {
                        "type": "integer",
                        "description": "Index of image 2 in the current sample, starting at 0.",
                    },
                    "box_1": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 4,
                        "maxItems": 4,
                        "description": "Required local component box [x1, y1, x2, y2] for image 1 in normalized 0-1000 space.",
                    },
                    "box_2": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 4,
                        "maxItems": 4,
                        "description": "Required local component box [x1, y1, x2, y2] for image 2 in normalized 0-1000 space.",
                    },
                    "focus_hint": {
                        "type": "string",
                        "description": "Optional short local-part hint such as dial, buckle, hinge, zipper, logo panel, head patch, or wing patch.",
                    },
                },
                "required": ["image_1_index", "image_2_index", "box_1", "box_2"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "verify_local_structure",
            "description": (
                "Check local structure consistency between two user-provided boxes using lightweight contour, geometry, and layout cues. "
                "The tool returns structure_verdict, structure_confidence, shape_similarity, layout_consistency, conflict_points, and a short observation. "
                "It does not decide the final identity answer."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "image_1_index": {
                        "type": "integer",
                        "description": "Index of image 1 in the current sample, starting at 0.",
                    },
                    "image_2_index": {
                        "type": "integer",
                        "description": "Index of image 2 in the current sample, starting at 0.",
                    },
                    "box_1": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 4,
                        "maxItems": 4,
                        "description": "Required local component box [x1, y1, x2, y2] for image 1 in normalized 0-1000 space.",
                    },
                    "box_2": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 4,
                        "maxItems": 4,
                        "description": "Required local component box [x1, y1, x2, y2] for image 2 in normalized 0-1000 space.",
                    },
                    "focus_hint": {
                        "type": "string",
                        "description": "Optional short local-part hint such as handle joint, connector layout, sole edge, buckle shape, frame corner, or wing patch.",
                    },
                },
                "required": ["image_1_index", "image_2_index", "box_1", "box_2"],
            },
        },
    },
]
