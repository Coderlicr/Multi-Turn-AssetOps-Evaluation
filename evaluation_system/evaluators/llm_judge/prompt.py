from __future__ import annotations

import json
from typing import Iterable, Optional

from evaluation_system.models.dialog import DialogRecord


DEFAULT_PROMPT_VERSION = "v1.0.0"


def _json(v: object) -> str:
    return json.dumps(v, ensure_ascii=False, indent=2)


def _tool_trace_summary(dialog: DialogRecord) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for t in dialog.turns:
        for c in t.tool_calls:
            out.append(
                {
                    "turn_id": t.turn_id,
                    "tool_name": c.tool_name,
                    "tool_agent": c.tool_agent,
                    "tool_exists": c.tool_exists,
                    "schema_valid": c.schema_valid,
                    "execution_success": c.execution_success,
                    "status": c.status,
                    "input_summary": c.input_summary,
                    "output_summary": c.output_summary,
                }
            )
    return out


def _turns_block(dialog: DialogRecord) -> list[dict[str, object]]:
    return [
        {
            "turn_id": t.turn_id,
            "user_input": t.user_input,
            "agent_response": t.agent_response,
            "intermediate_plan": t.intermediate_plan,
            "tools_used": t.tools_used,
            "tool_calls": [tc.model_dump(mode="json") for tc in t.tool_calls],
            "turn_result": t.turn_result,
            "turn_status": t.turn_status.value,
            "recovery_triggered": t.recovery_triggered,
        }
        for t in dialog.turns
    ]


def _ground_truth_block(dialog: DialogRecord) -> dict[str, object]:
    return {
        "expected_plan": dialog.ground_truth.expected_plan,
        "expected_tools": dialog.ground_truth.expected_tools,
        "expected_final_answer": dialog.ground_truth.expected_final_answer,
        "task_success_criteria": dialog.ground_truth.task_success_criteria,
        "acceptable_alternatives": dialog.ground_truth.acceptable_alternatives,
        "annotated_characteristic": dialog.ground_truth.annotated_characteristic,
        "required_tool_sequence": dialog.ground_truth.required_tool_sequence,
    }


# Placeholder stored in prompts when blind_model_name hides the rollout folder name.
_BLIND_MODEL_PLACEHOLDER = "anonymous_agent"


def build_llm_judge_prompt(
    dialog: DialogRecord,
    *,
    available_tools: Optional[Iterable[str]] = None,
    prompt_version: str = DEFAULT_PROMPT_VERSION,
    rubric_order: Optional[list[str]] = None,
    blind_model_name: bool = False,
) -> str:
    """
    Build judge prompt from ``DialogRecord`` only (no raw JSON coupling).

    Includes: dialog identity, scenario, complexity, description, all turns (with plans,
    tools_used, tool_calls, turn_result), ground truth references, and tool trace summary.
    """
    tools_list = list(available_tools or sorted(set(dialog.ground_truth.expected_tools or [])))
    order = rubric_order or ["Planning Effectiveness", "Tool Usage Quality", "Task Completion"]

    # task_metadata = {
    #     "dialog_id": dialog.dialog_id,
    #     "model_name": (_BLIND_MODEL_PLACEHOLDER if blind_model_name else dialog.model_name),
    #     "scenario_type": dialog.scenario_type,
    #     "scenario_subtype": dialog.scenario_subtype,
    #     "complexity": dialog.complexity.value,
    #     "num_turns": dialog.num_turns,
    #     "description": dialog.description,
    #     "agents_involved": dialog.agents_involved,
    # }

    ground_truth_block = _ground_truth_block(dialog)
    turns_block = _turns_block(dialog)

    return "\n".join(
        [
            "You are an expert evaluator for a multi-turn industrial dialog agent.",
            "Evaluate the dialog using the rubric and return ONLY valid JSON matching the required output schema.",
            "Do not include markdown fences or any text outside the JSON object.",
            "",
            # f"prompt_version: {prompt_version}",
            # "",
            # "## Task metadata (dialog_id, scenario, complexity, description)",
            # _json(task_metadata),
            # "",
            "## Ground truth (reference for Planning / Tools / Task Completion)",
            "Use annotated_characteristic and required_tool_sequence as the primary checklist when "
            "they are non-empty; otherwise rely on expected_plan / expected_tools / task_success_criteria.",
            _json(ground_truth_block),
            "",
            # "## Available tools (expected tool agents / names for context)",
            # _json(tools_list),
            # "",
            "## All turns (user_input, agent_response, intermediate_plan, tools_used, tool_calls, turn_result, …)",
            _json(turns_block),
            "",
            "## Execution trace summary (flattened tool calls)",
            _json(_tool_trace_summary(dialog)),
            "",
            "## Rubric",
            "Score three dimensions on [0,1] (1.0 = best). Use evidence from the dialog and ground truth.",
            "Keep reasoning comments brief (one or two sentences each).",
            "",
            f"Evaluate in this conceptual order (for your internal reasoning): {order}",
            "",
            "### Planning Effectiveness",
            "- Alignment and coverage vs expected_plan (semantic match, not verbatim).",
            "- Coherence and sequencing across turns; adaptation when recovery occurs.",
            "",
            "### Tool Usage Quality",
            "- Appropriate tool selection vs task and expected_tools.",
            "- Quality of tool usage as reflected by summaries and outcomes (distinct from raw tool_exists flags).",
            "",
            "### Task Completion",
            "- Satisfies task_success_criteria.",
            "- Consistent with expected_final_answer or acceptable_alternatives when applicable.",
            "- Actionable follow-up where relevant.",
            "",
            "## Required JSON output schema (return ONLY this JSON object):",
            _json(
                {
                    "planning_effectiveness": 0.0,
                    "tool_usage_quality": 0.0,
                    "task_completion": 0.0,
                    "reasoning": {
                        "planning_comment": "",
                        "tool_comment": "",
                        "task_comment": "",
                    },
                }
            ),
        ]
    )


def build_paired_llm_judge_prompt(
    candidates: list[tuple[str, DialogRecord]],
    *,
    prompt_version: str = DEFAULT_PROMPT_VERSION,
    rubric_order: Optional[list[str]] = None,
    blind_model_name: bool = False,
) -> str:
    """
    Build one prompt that scores multiple candidate rollouts for the same dialog.

    ``candidates`` labels must be stable ids such as ``candidate_a`` and are used
    only for returning scores to the caller. When ``blind_model_name`` is true,
    real rollout folder names are omitted from the prompt.
    """
    if len(candidates) < 2:
        raise ValueError("paired judge requires at least two candidates")

    reference_dialog = candidates[0][1]
    order = rubric_order or ["Planning Effectiveness", "Tool Usage Quality", "Task Completion"]
    candidate_blocks: dict[str, object] = {}
    for idx, (label, dialog) in enumerate(candidates, start=1):
        candidate_blocks[label] = {
            "candidate_display_name": f"candidate_{idx}" if blind_model_name else dialog.model_name,
            "scenario_type": dialog.scenario_type,
            "scenario_subtype": dialog.scenario_subtype,
            "complexity": dialog.complexity.value,
            "num_turns": dialog.num_turns,
            "description": dialog.description,
            "agents_involved": dialog.agents_involved,
            "turns": _turns_block(dialog),
            "execution_trace_summary": _tool_trace_summary(dialog),
            "automatic_evaluation": dialog.automatic_evaluation.model_dump(mode="json"),
        }

    output_schema = {
        label: {
            "planning_effectiveness": 0.0,
            "tool_usage_quality": 0.0,
            "task_completion": 0.0,
            "reasoning": {
                "planning_comment": "",
                "tool_comment": "",
                "task_comment": "",
            },
        }
        for label, _ in candidates
    }

    return "\n".join(
        [
            "You are an expert evaluator for multi-turn industrial dialog agents.",
            "Evaluate multiple candidate rollouts for the SAME dialog in one shared context.",
            "Use the same scoring scale for all candidates. Scores are absolute [0,1], but you must calibrate them comparatively within this prompt so equivalent quality receives equivalent scores.",
            "Return ONLY valid JSON matching the required output schema. Do not include markdown fences or text outside the JSON object.",
            "",
            "## Ground truth (shared reference for all candidates)",
            "Use annotated_characteristic and required_tool_sequence as the primary checklist when "
            "they are non-empty; otherwise rely on expected_plan / expected_tools / task_success_criteria.",
            _json(_ground_truth_block(reference_dialog)),
            "",
            "## Candidate rollouts",
            _json(candidate_blocks),
            "",
            "## Rubric",
            "Score three dimensions on [0,1] (1.0 = best). Use evidence from each candidate and the shared ground truth.",
            "Keep reasoning comments brief (one or two sentences each).",
            "",
            f"Evaluate in this conceptual order (for your internal reasoning): {order}",
            "",
            "### Planning Effectiveness",
            "- Alignment and coverage vs expected_plan (semantic match, not verbatim).",
            "- Coherence and sequencing across turns; adaptation when recovery occurs.",
            "",
            "### Tool Usage Quality",
            "- Appropriate tool selection vs task and expected_tools.",
            "- Quality of tool usage as reflected by summaries and outcomes (distinct from raw tool_exists flags).",
            "",
            "### Task Completion",
            "- Satisfies task_success_criteria.",
            "- Consistent with expected_final_answer or acceptable_alternatives when applicable.",
            "- Actionable follow-up where relevant.",
            "",
            "## Required JSON output schema (return ONLY this JSON object):",
            _json(output_schema),
        ]
    )
