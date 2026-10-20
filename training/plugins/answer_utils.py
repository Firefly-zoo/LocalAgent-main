from __future__ import annotations

import re
from typing import Any


ANSWER_TAG_PATTERN = re.compile(r"<answer>\s*(yes|no)\s*</answer>", re.IGNORECASE | re.DOTALL)
FINAL_ANSWER_INLINE_PATTERN = re.compile(
    r"(?:final\s+answer|answer)\s*:\s*<answer>\s*(yes|no)\s*</answer>",
    re.IGNORECASE | re.DOTALL,
)
EVIDENCE_PREFIX_PATTERN = re.compile(r"^\s*evidence\s*:\s*.+$", re.IGNORECASE | re.MULTILINE)
TOOL_CALL_ONLY_PATTERN = re.compile(r"^\s*<tool_call>\s*\{[\s\S]+\}\s*</tool_call>\s*$", re.IGNORECASE)
SPECIAL_TOKEN_PATTERN = re.compile(r"<\|im_(?:start|end)\|>")


def completion_to_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, dict):
        content = completion.get("content", "")
        return content if isinstance(content, str) else str(content)
    if isinstance(completion, list):
        parts = []
        for item in completion:
            if isinstance(item, dict):
                content = item.get("content", "")
                parts.append(content if isinstance(content, str) else str(content))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(completion)


def normalize_reward_text(text: str) -> str:
    return SPECIAL_TOKEN_PATTERN.sub("", text).strip()


def extract_final_answer(text: str) -> str | None:
    text = normalize_reward_text(text)
    matches = ANSWER_TAG_PATTERN.findall(text)
    if matches:
        return matches[-1].lower()
    inline = FINAL_ANSWER_INLINE_PATTERN.findall(text)
    if inline:
        return inline[-1].lower()
    return None


def has_valid_final_answer_format(text: str) -> bool:
    return extract_final_answer(text) is not None


def has_tool_call_only_format(text: str) -> bool:
    return TOOL_CALL_ONLY_PATTERN.fullmatch(normalize_reward_text(text)) is not None


def has_free_cot_format(text: str) -> bool:
    stripped = normalize_reward_text(text)
    if has_tool_call_only_format(stripped):
        return False
    answer = extract_final_answer(stripped)
    if answer is None:
        return False
    before_answer = ANSWER_TAG_PATTERN.split(stripped, maxsplit=1)[0]
    return bool(before_answer.strip())


def has_structured_cot_format(text: str) -> bool:
    stripped = normalize_reward_text(text)
    return has_valid_final_answer_format(stripped) and EVIDENCE_PREFIX_PATTERN.search(stripped) is not None
