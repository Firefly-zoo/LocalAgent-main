from __future__ import annotations

import json
import re
from copy import deepcopy
from typing import Any, Dict, List

from swift.rollout.multi_turn import MultiTurnScheduler, multi_turns

from appearance_tool import measure_dino_patch_mnn
from structure_tools import verify_local_structure

RAW_TOOL_CALL_PATTERN = re.compile(r"<tool_call>\s*(.+?)\s*</tool_call>", re.DOTALL)
TOOL_STATS_KEY = "twin_dino_structure_tool_family_stats"


class TwinDinoStructureToolFamilyCallScheduler(MultiTurnScheduler):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tools = {
            "measure_dino_patch_mnn": measure_dino_patch_mnn,
            "verify_local_structure": verify_local_structure,
        }

    @staticmethod
    def _parse_arguments(arguments: Any) -> Dict[str, Any]:
        if isinstance(arguments, str):
            parsed = json.loads(arguments)
            if not isinstance(parsed, dict):
                raise ValueError("Tool arguments must decode to a JSON object.")
            return parsed
        if not isinstance(arguments, dict):
            raise ValueError("Tool arguments must be a JSON object.")
        return deepcopy(arguments)

    def _extract_tool_calls(self, response_choice: "ChatCompletionResponseChoice") -> List[Dict[str, Any]]:
        tool_calls = response_choice.message.tool_calls
        if tool_calls:
            return [{
                "name": tool_call.function.name,
                "arguments": self._parse_arguments(tool_call.function.arguments),
            } for tool_call in tool_calls]

        completion = response_choice.message.content or ""
        parsed_calls = []
        for raw_call in RAW_TOOL_CALL_PATTERN.findall(completion):
            call = json.loads(raw_call)
            if not isinstance(call, dict):
                raise ValueError("Tool call payload must be a JSON object.")
            parsed_calls.append({
                "name": str(call.get("name", "")).strip(),
                "arguments": self._parse_arguments(call.get("arguments", {})),
            })
        return parsed_calls

    @staticmethod
    def _resolve_image_path(arguments: Dict[str, Any], images: List[str], index_key: str, path_key: str) -> str:
        if path_key in arguments:
            return str(arguments.pop(path_key))
        if index_key not in arguments:
            raise KeyError(f"Missing `{index_key}`.")
        image_index = int(arguments.pop(index_key))
        if image_index < 0 or image_index >= len(images):
            raise IndexError(f"`{index_key}` out of range: {image_index}")
        return str(images[image_index])

    def _resolve_tool_kwargs(self, tool_name: str, arguments: Dict[str, Any], images: List[str]) -> Dict[str, Any]:
        kwargs = deepcopy(arguments)
        if tool_name not in self.tools:
            return kwargs

        kwargs["image_1_path"] = self._resolve_image_path(kwargs, images, "image_1_index", "image_1_path")
        kwargs["image_2_path"] = self._resolve_image_path(kwargs, images, "image_2_index", "image_2_path")
        return kwargs

    def _execute_tool_call(self, tool_call: Dict[str, Any], images: List[str]) -> Dict[str, Any]:
        tool_name = tool_call["name"]
        public_arguments = deepcopy(tool_call["arguments"])
        if tool_name not in self.tools:
            return {"status": "error", "tool": tool_name, "arguments": public_arguments, "error": "unknown tool"}

        try:
            tool_kwargs = self._resolve_tool_kwargs(tool_name, tool_call["arguments"], images)
            result = self.tools[tool_name](**tool_kwargs)
            return {"status": "success", "tool": tool_name, "arguments": public_arguments, "output": result}
        except Exception as exc:
            return {"status": "error", "tool": tool_name, "arguments": public_arguments, "error": str(exc)}

    def check_finished(
        self,
        infer_request: "RolloutInferRequest",
        response_choice: "ChatCompletionResponseChoice",
        current_turn: int,
    ) -> bool:
        try:
            tool_calls = self._extract_tool_calls(response_choice)
        except Exception:
            tool_calls = []
        if not tool_calls:
            return True
        return super().check_finished(infer_request, response_choice, current_turn)

    def step(
        self,
        infer_request: "RolloutInferRequest",
        response_choice: "ChatCompletionResponseChoice",
        current_turn: int,
    ) -> Dict[str, Any]:
        tool_calls = self._extract_tool_calls(response_choice)
        tool_stats = infer_request.data_dict.setdefault(
            TOOL_STATS_KEY,
            {
                "tool_call_count": 0,
                "has_tool_response": False,
                "last_tool": None,
                "last_tool_status": None,
                "tool_sequence": [],
            },
        )

        if tool_stats["tool_call_count"] >= 1:
            tool_result = {
                "status": "error",
                "error": "At most one tool call total is supported.",
                "tool_call_count": int(tool_stats["tool_call_count"]),
            }
        elif len(tool_calls) != 1:
            tool_result = {
                "status": "error",
                "error": "Exactly one tool call per turn is supported.",
                "tool_call_count": len(tool_calls),
            }
            tool_stats["tool_call_count"] += len(tool_calls)
        else:
            tool_stats["tool_call_count"] += 1
            tool_result = self._execute_tool_call(tool_calls[0], list(infer_request.images or []))
            if str(tool_result.get("status", "")).strip().lower() == "success":
                tool_stats.setdefault("tool_sequence", []).append(str(tool_result.get("tool", "")).strip())

        derived_status = str(tool_result.get("status", "")).strip().lower()
        if derived_status == "success":
            output = tool_result.get("output")
            if isinstance(output, dict):
                nested_status = str(output.get("status", "")).strip().lower()
                if nested_status:
                    derived_status = nested_status

        tool_response = json.dumps(tool_result, ensure_ascii=False)
        infer_request.messages.append({"role": "tool", "content": tool_response})
        tool_stats["has_tool_response"] = True
        tool_stats["last_tool"] = tool_result.get("tool")
        tool_stats["last_tool_status"] = derived_status
        tool_stats["last_tool_exec_status"] = str(tool_result.get("status", "")).strip().lower()

        return {
            "infer_request": infer_request,
            "rollout_infos": {
                "tool_call_count": int(tool_stats["tool_call_count"]),
                "has_tool_response": bool(tool_stats["has_tool_response"]),
                "last_tool": tool_stats.get("last_tool"),
                "last_tool_status": tool_stats.get("last_tool_status"),
                "last_tool_exec_status": tool_stats.get("last_tool_exec_status"),
                "tool_sequence": list(tool_stats.get("tool_sequence", [])),
                "num_turns": current_turn,
            },
        }


def register_twin_dino_structure_tool_family_scheduler() -> None:
    multi_turns["twin_dino_structure_tool_family_call_scheduler"] = TwinDinoStructureToolFamilyCallScheduler
