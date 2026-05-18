"""Parse and strictly validate LLM-as-Judge JSON output."""

from __future__ import annotations

import json
import re
from typing import Any

from evaluation_system.models.dialog import LLMEvaluation, LLMEvaluationReasoning


class JudgeOutputParseError(ValueError):
    """Raised when model output cannot be parsed or fails validation."""


def extract_json_object(raw_text: str) -> dict[str, Any]:
    """
    Extract a single JSON object from arbitrary model text.

    Handles:
    - Raw JSON
    - ```json ... ``` fenced blocks
    - Extra prose before/after first balanced `{ ... }`
    """
    text = raw_text.strip()
    if not text:
        raise JudgeOutputParseError("Empty model response")

    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, re.IGNORECASE)
    if fence:
        text = fence.group(1).strip()

    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        raise JudgeOutputParseError(
            f"No JSON object found in model response (first 500 chars): {raw_text[:500]!r}"
        )
    snippet = text[start : end + 1]
    try:
        obj = json.loads(snippet)
    except json.JSONDecodeError as exc:
        raise JudgeOutputParseError(f"Invalid JSON in extracted snippet: {exc}") from exc
    if not isinstance(obj, dict):
        raise JudgeOutputParseError(f"Expected JSON object, got {type(obj).__name__}")
    return obj


def _score(name: str, v: Any) -> float:
    if v is None:
        raise JudgeOutputParseError(f"Missing required score field: {name}")
    try:
        x = float(v)
    except (TypeError, ValueError) as exc:
        raise JudgeOutputParseError(f"Score {name} must be numeric, got {v!r}") from exc
    if x < 0.0 or x > 1.0:
        raise JudgeOutputParseError(f"Score {name} out of range [0,1]: {x}")
    return x


def _reasoning_block(obj: Any) -> dict[str, str]:
    if obj is None:
        raise JudgeOutputParseError("Missing required field: reasoning")
    if not isinstance(obj, dict):
        raise JudgeOutputParseError(f"reasoning must be object, got {type(obj).__name__}")
    out: dict[str, str] = {}
    for key in ("planning_comment", "tool_comment", "task_comment"):
        if key not in obj:
            raise JudgeOutputParseError(f"Missing reasoning.{key}")
        out[key] = str(obj[key]) if obj[key] is not None else ""
    return out


def parse_judge_output_to_llm_evaluation(
    raw_text: str,
    *,
    judge_model: str,
    judge_prompt_version: str,
    shuffle_run_id: str | None,
) -> LLMEvaluation:
    """
    Parse model response text into ``LLMEvaluation`` with strict field and range checks.
    """
    data = extract_json_object(raw_text)
    reasoning = _reasoning_block(data.get("reasoning"))
    ev = LLMEvaluation(
        planning_effectiveness=_score("planning_effectiveness", data.get("planning_effectiveness")),
        tool_usage_quality=_score("tool_usage_quality", data.get("tool_usage_quality")),
        task_completion=_score("task_completion", data.get("task_completion")),
        reasoning=LLMEvaluationReasoning(
            planning_comment=reasoning["planning_comment"],
            tool_comment=reasoning["tool_comment"],
            task_comment=reasoning["task_comment"],
        ),
        judge_model=judge_model,
        judge_prompt_version=judge_prompt_version,
        shuffle_run_id=shuffle_run_id,
    )
    return ev


def _llm_evaluation_from_obj(
    data: dict[str, Any],
    *,
    judge_model: str,
    judge_prompt_version: str,
    shuffle_run_id: str | None,
) -> LLMEvaluation:
    reasoning = _reasoning_block(data.get("reasoning"))
    return LLMEvaluation(
        planning_effectiveness=_score("planning_effectiveness", data.get("planning_effectiveness")),
        tool_usage_quality=_score("tool_usage_quality", data.get("tool_usage_quality")),
        task_completion=_score("task_completion", data.get("task_completion")),
        reasoning=LLMEvaluationReasoning(
            planning_comment=reasoning["planning_comment"],
            tool_comment=reasoning["tool_comment"],
            task_comment=reasoning["task_comment"],
        ),
        judge_model=judge_model,
        judge_prompt_version=judge_prompt_version,
        shuffle_run_id=shuffle_run_id,
    )


def parse_paired_judge_output_to_llm_evaluations(
    raw_text: str,
    *,
    candidate_labels: list[str],
    judge_model: str,
    judge_prompt_version: str,
    shuffle_run_id: str | None,
) -> dict[str, LLMEvaluation]:
    """
    Parse model response for paired/multi-candidate judging.

    Expected top-level shape:
    {
      "candidate_a": {"planning_effectiveness": ..., "reasoning": {...}},
      "candidate_b": {...}
    }
    """
    data = extract_json_object(raw_text)
    out: dict[str, LLMEvaluation] = {}
    for label in candidate_labels:
        block = data.get(label)
        if not isinstance(block, dict):
            raise JudgeOutputParseError(f"Missing or invalid paired candidate block: {label}")
        out[label] = _llm_evaluation_from_obj(
            block,
            judge_model=judge_model,
            judge_prompt_version=judge_prompt_version,
            shuffle_run_id=shuffle_run_id,
        )
    return out
