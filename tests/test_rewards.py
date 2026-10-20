from __future__ import annotations

import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PLUGIN_ROOT = REPO_ROOT / "training" / "plugins"
sys.path.insert(0, str(PLUGIN_ROOT))

from orms.structure_scorers.config import (  # noqa: E402
    TwinDinoStructureToolFamilyRewardConfig,
)
from orms.structure_scorers.total import aggregate_weighted_total  # noqa: E402
from orms.structure_scorers.usage import score_usage  # noqa: E402
from orms.identity_scorers.answer import score_answer  # noqa: E402
from orms.identity_scorers.format import score_format  # noqa: E402
from orms.identity_scorers.validity import score_validity  # noqa: E402


CONFIG = TwinDinoStructureToolFamilyRewardConfig()
STRUCTURED_YES = "Evidence: the local evidence agrees.\nFinal answer: <answer>yes</answer>"
STRUCTURED_NO = "Evidence: the local evidence conflicts.\nFinal answer: <answer>no</answer>"
SUCCESS_INFO = {"tool_call_count": 1, "has_tool_response": True, "last_tool_status": "success"}


def total(completion: str, truth: str, label: str, info: dict) -> float:
    answer = score_answer([completion], [truth])
    usage = score_usage([completion], rollout_infos=[info], tool_label=[label])
    validity = score_validity([completion], rollout_infos=[info])
    fmt = score_format([completion])
    return aggregate_weighted_total(
        answer_scores=answer,
        usage_scores=usage,
        validity_scores=validity,
        format_scores=fmt,
        efficiency_scores=[0.0],
        config=CONFIG,
    )[0]


class TrainingRewardTests(unittest.TestCase):
    def test_correct_answer_with_valid_tool_use(self) -> None:
        self.assertAlmostEqual(total(STRUCTURED_YES, "yes", "likely_tool", SUCCESS_INFO), 1.20)

    def test_correct_answer_without_tool_use(self) -> None:
        self.assertAlmostEqual(total(STRUCTURED_YES, "yes", "no_tool", {}), 1.125)

    def test_incorrect_answer_with_positive_auxiliary_reward(self) -> None:
        self.assertAlmostEqual(total(STRUCTURED_YES, "no", "likely_tool", SUCCESS_INFO), CONFIG.wrong_aux_cap)

    def test_incorrect_answer_with_negative_auxiliary_reward(self) -> None:
        self.assertAlmostEqual(total(STRUCTURED_YES, "no", "likely_tool", {}), -0.07)

    def test_cap_activation(self) -> None:
        score = aggregate_weighted_total(
            answer_scores=[0.0], usage_scores=[1.0], validity_scores=[0.0],
            format_scores=[1.0], efficiency_scores=[0.0], config=CONFIG,
        )[0]
        self.assertEqual(score, CONFIG.wrong_aux_cap)

    def test_floor_activation(self) -> None:
        score = aggregate_weighted_total(
            answer_scores=[0.0], usage_scores=[-1.0], validity_scores=[-1.0],
            format_scores=[-1.0], efficiency_scores=[-1.0], config=CONFIG,
        )[0]
        self.assertEqual(score, CONFIG.wrong_aux_floor)

    def test_malformed_tool_call(self) -> None:
        malformed = '<tool_call>{"name":</tool_call>'
        info = {"tool_call_count": 1, "has_tool_response": False, "last_tool_status": "error"}
        self.assertEqual(score_validity([malformed], rollout_infos=[info]), [-0.8])
        self.assertLess(total(malformed, "yes", "likely_tool", info), 0.0)

    def test_invalid_final_answer_format(self) -> None:
        invalid = "Evidence: uncertain.\nFinal answer: yes"
        self.assertEqual(score_answer([invalid], ["yes"]), [0.0])
        self.assertEqual(score_format([invalid]), [-0.6])


if __name__ == "__main__":
    unittest.main()
