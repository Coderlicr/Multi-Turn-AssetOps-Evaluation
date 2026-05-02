from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from evaluation_system.models.dialog import DialogRecord, HumanEvaluation


def build_human_annotation_view(dialog: DialogRecord) -> dict[str, Any]:
    """
    Build a simplified view for annotators (do not edit original JSON directly).
    """
    return {
        "dialog_id": dialog.dialog_id,
        "model_name": dialog.model_name,
        "scenario_type": dialog.scenario_type,
        "scenario_subtype": dialog.scenario_subtype,
        "complexity": dialog.complexity.value,
        "description": dialog.description,
        "agents_involved": dialog.agents_involved,
        "turns": [
            {
                "turn_id": t.turn_id,
                "user_input": t.user_input,
                "agent_response": t.agent_response,
                "intermediate_plan": t.intermediate_plan,
                "tools_used": t.tools_used,
                "tool_calls": [
                    {
                        "tool_name": c.tool_name,
                        "tool_agent": c.tool_agent,
                        "input_summary": c.input_summary,
                        "output_summary": c.output_summary,
                        "execution_success": c.execution_success,
                    }
                    for c in t.tool_calls
                ],
                "turn_result": t.turn_result,
                "turn_status": t.turn_status.value,
                "recovery_triggered": t.recovery_triggered,
            }
            for t in dialog.turns
        ],
        "ground_truth": {
            "expected_plan": dialog.ground_truth.expected_plan,
            "expected_tools": dialog.ground_truth.expected_tools,
            "task_success_criteria": dialog.ground_truth.task_success_criteria,
            "expected_final_answer": dialog.ground_truth.expected_final_answer,
            "acceptable_alternatives": dialog.ground_truth.acceptable_alternatives,
            "annotated_characteristic": dialog.ground_truth.annotated_characteristic,
            "required_tool_sequence": dialog.ground_truth.required_tool_sequence,
        },
        "human_evaluation_template": {
            "annotator_id": "",
            "planning_effectiveness": None,
            "tool_usage_quality": None,
            "task_completion": None,
            "comments": "",
        },
    }


def export_annotation_dict(dialog: DialogRecord) -> dict[str, Any]:
    """Export a dict (JSON-serializable) annotation view."""
    return build_human_annotation_view(dialog)


def export_annotation_csv_row(dialog: DialogRecord) -> dict[str, Any]:
    """
    Export one CSV row for annotation.

    Annotators fill:
    - annotator_id
    - planning_effectiveness
    - tool_usage_quality
    - task_completion
    - comments
    """
    return {
        "dialog_id": dialog.dialog_id,
        "model_name": dialog.model_name,
        "scenario_type": dialog.scenario_type,
        "scenario_subtype": dialog.scenario_subtype,
        "complexity": dialog.complexity.value,
        "description": dialog.description,
        "annotator_id": "",
        "planning_effectiveness": "",
        "tool_usage_quality": "",
        "task_completion": "",
        "comments": "",
    }


def export_annotation_markdown(dialog: DialogRecord) -> str:
    """Markdown preview for manual review."""
    lines: list[str] = []
    lines.append(f"## Dialog `{dialog.dialog_id}`")
    lines.append("")
    lines.append(f"- **model_name**: `{dialog.model_name}`")
    lines.append(f"- **scenario_type**: `{dialog.scenario_type}`")
    lines.append(f"- **scenario_subtype**: `{dialog.scenario_subtype}`")
    lines.append(f"- **complexity**: `{dialog.complexity.value}`")
    lines.append(f"- **num_turns**: `{dialog.num_turns}`")
    lines.append("")
    lines.append("### Description")
    lines.append(dialog.description or "")
    lines.append("")
    lines.append("### Ground truth (reference)")
    gt_show = {
        "task_success_criteria": dialog.ground_truth.task_success_criteria,
        "expected_final_answer": dialog.ground_truth.expected_final_answer,
        "acceptable_alternatives": dialog.ground_truth.acceptable_alternatives,
        "annotated_characteristic": dialog.ground_truth.annotated_characteristic,
        "required_tool_sequence": dialog.ground_truth.required_tool_sequence,
    }
    lines.append("```json")
    lines.append(
        json.dumps(
            gt_show,
            ensure_ascii=False,
            indent=2,
        )
    )
    lines.append("```")
    lines.append("")
    lines.append("### Turns")
    for t in dialog.turns:
        lines.append(f"#### Turn {t.turn_id}")
        lines.append("")
        lines.append("**User**")
        lines.append("")
        lines.append(t.user_input)
        lines.append("")
        lines.append("**Agent**")
        lines.append("")
        lines.append(t.agent_response)
        lines.append("")
    lines.append("### Annotation fields (fill in CSV/Sheet)")
    lines.append("- annotator_id")
    lines.append("- planning_effectiveness (0–1)")
    lines.append("- tool_usage_quality (0–1)")
    lines.append("- task_completion (0–1)")
    lines.append("- comments")
    lines.append("")
    return "\n".join(lines)


def _parse_score01(v: Any) -> Optional[float]:
    if v is None:
        return None
    if isinstance(v, str) and not v.strip():
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        raise ValueError(f"Invalid score {v!r}; must be float in [0,1] or empty") from None
    if x < 0 or x > 1:
        raise ValueError(f"Score out of range [0,1]: {x}")
    return x


def apply_human_annotation(dialog: DialogRecord, annotation_dict: dict[str, Any]) -> DialogRecord:
    """
    Apply human annotation into `dialog.human_evaluation`.

    Expected keys:
    - annotator_id
    - planning_effectiveness
    - tool_usage_quality
    - task_completion
    - comments
    """
    he = HumanEvaluation(
        annotator_id=str(annotation_dict.get("annotator_id") or ""),
        planning_effectiveness=_parse_score01(annotation_dict.get("planning_effectiveness")),
        tool_usage_quality=_parse_score01(annotation_dict.get("tool_usage_quality")),
        task_completion=_parse_score01(annotation_dict.get("task_completion")),
        comments=str(annotation_dict.get("comments") or ""),
    )
    return dialog.model_copy(update={"human_evaluation": he})


@dataclass(frozen=True)
class AnnotationImportRow:
    dialog_id: str
    model_name: str
    annotator_id: str
    planning_effectiveness: Optional[float]
    tool_usage_quality: Optional[float]
    task_completion: Optional[float]
    comments: str


def read_annotations_csv(path: Path) -> list[AnnotationImportRow]:
    rows: list[AnnotationImportRow] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(
                AnnotationImportRow(
                    dialog_id=str(r.get("dialog_id") or ""),
                    model_name=str(r.get("model_name") or ""),
                    annotator_id=str(r.get("annotator_id") or ""),
                    planning_effectiveness=_parse_score01(r.get("planning_effectiveness")),
                    tool_usage_quality=_parse_score01(r.get("tool_usage_quality")),
                    task_completion=_parse_score01(r.get("task_completion")),
                    comments=str(r.get("comments") or ""),
                )
            )
    return rows

