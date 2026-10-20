from __future__ import annotations

import sys
from pathlib import Path

PLUGIN_DIR = Path(__file__).resolve().parent
if str(PLUGIN_DIR) not in sys.path:
    sys.path.insert(0, str(PLUGIN_DIR))

from orms.structure_reward import register_twin_dino_structure_tool_family_orms  # noqa: E402
from structure_scheduler import register_twin_dino_structure_tool_family_scheduler  # noqa: E402

register_twin_dino_structure_tool_family_orms()
register_twin_dino_structure_tool_family_scheduler()
