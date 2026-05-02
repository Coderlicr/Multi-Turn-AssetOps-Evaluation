"""
Metrics aggregation over a corpus of dialogs (final metric spec).

Final metrics (names MUST NOT change):

Paper (MCP-Bench aligned)
1. Planning Effectiveness
2. Tool Usage Quality
3. Task Completion
4. Tool Name Validity
5. Schema Compliance
6. Execution Success Rate

Custom
7. Recovery Success Rate

Aggregation conventions:
- Tool Name Validity / Schema Compliance / Execution Success Rate are aggregated as micro-averages
  over all tool calls in the corpus (aligned with their call-level definitions).
- Recovery Success Rate is aggregated at dialog level, per spec:
  (# dialogs with recovery_triggered AND final success) / (# dialogs with recovery_triggered)
- Planning/Tool/Task scores come from LLM or Human blocks (macro-average over available scores).
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

from pydantic import BaseModel, Field

from evaluation_system.models.dialog import DialogRecord
from evaluation_system.models.enums import EvalSource, TurnStatus


class AggregatedMetrics(BaseModel):
    """Structured aggregate results."""

    num_dialogs: int = Field(ge=1)
    source: EvalSource = EvalSource.AUTOMATIC

    planning_effectiveness: Optional[float] = Field(default=None, ge=0, le=1)
    tool_usage_quality: Optional[float] = Field(default=None, ge=0, le=1)
    task_completion: Optional[float] = Field(default=None, ge=0, le=1)

    tool_name_validity: Optional[float] = Field(default=None, ge=0, le=1)
    schema_compliance: Optional[float] = Field(default=None, ge=0, le=1)
    execution_success_rate: Optional[float] = Field(default=None, ge=0, le=1)

    recovery_success_rate: Optional[float] = Field(default=None, ge=0, le=1)

    dialogs_with_recovery: int = Field(default=0, ge=0)
    dialogs_recovery_and_success: int = Field(default=0, ge=0)


def _mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return float(sum(values)) / float(len(values))


def _dialog_final_success(dialog: DialogRecord) -> bool:
    if not dialog.turns:
        return False
    return dialog.turns[-1].turn_status == TurnStatus.SUCCESS


def aggregate_metrics(
    dialogs: Iterable[DialogRecord],
    *,
    source: EvalSource = EvalSource.AUTOMATIC,
) -> AggregatedMetrics:
    """
    Aggregate metrics from dialogs.

    - ``source=AUTOMATIC``: subjective metrics stay null (no ``llm_evaluation`` / ``human_evaluation`` read).
    - ``source=LLM``: macro-average ``llm_evaluation`` scores; **always** also computes automatic/log metrics.
    - ``source=HUMAN``: same as LLM but reads ``human_evaluation`` for the three subjective scores.

    Leaderboards typically use ``source=LLM`` after ``llm-judge`` so the first three columns populate.
    """
    dialog_list = list(dialogs)
    if not dialog_list:
        raise ValueError("Cannot aggregate metrics on an empty dialog list")

    # Subjective (LLM/Human) metrics: macro-average over available scores.
    planning_scores: list[float] = []
    tool_quality_scores: list[float] = []
    task_completion_scores: list[float] = []

    for d in dialog_list:
        if source is EvalSource.LLM:
            ev = d.llm_evaluation
        elif source is EvalSource.HUMAN:
            ev = d.human_evaluation
        else:
            ev = None

        if ev is not None:
            if getattr(ev, "planning_effectiveness", None) is not None:
                planning_scores.append(float(ev.planning_effectiveness))  # type: ignore[arg-type]
            if getattr(ev, "tool_usage_quality", None) is not None:
                tool_quality_scores.append(float(ev.tool_usage_quality))  # type: ignore[arg-type]
            if getattr(ev, "task_completion", None) is not None:
                task_completion_scores.append(float(ev.task_completion))  # type: ignore[arg-type]

    # Automatic metrics: micro-average over all tool calls.
    tool_calls_total = 0
    tool_calls_legal = 0
    tool_calls_schema_ok = 0
    tool_calls_exec_ok = 0
    for d in dialog_list:
        for c in d.all_tool_calls():
            tool_calls_total += 1
            if c.tool_exists is True:
                tool_calls_legal += 1
                if c.schema_valid is True:
                    tool_calls_schema_ok += 1
            if c.execution_success is True:
                tool_calls_exec_ok += 1

    tool_name_validity = (
        None if tool_calls_total == 0 else float(tool_calls_legal) / float(tool_calls_total)
    )
    schema_compliance = (
        None if tool_calls_legal == 0 else float(tool_calls_schema_ok) / float(tool_calls_legal)
    )
    execution_success_rate = (
        None if tool_calls_total == 0 else float(tool_calls_exec_ok) / float(tool_calls_total)
    )

    dialogs_with_recovery = 0
    dialogs_recovery_and_success = 0
    for d in dialog_list:
        if any(t.recovery_triggered for t in d.turns):
            dialogs_with_recovery += 1
            if _dialog_final_success(d):
                dialogs_recovery_and_success += 1

    recovery_success_rate: float | None
    if dialogs_with_recovery == 0:
        recovery_success_rate = None
    else:
        recovery_success_rate = float(dialogs_recovery_and_success) / float(dialogs_with_recovery)

    return AggregatedMetrics(
        num_dialogs=len(dialog_list),
        source=source,
        planning_effectiveness=_mean(planning_scores),
        tool_usage_quality=_mean(tool_quality_scores),
        task_completion=_mean(task_completion_scores),
        tool_name_validity=tool_name_validity,
        schema_compliance=schema_compliance,
        execution_success_rate=execution_success_rate,
        recovery_success_rate=recovery_success_rate,
        dialogs_with_recovery=dialogs_with_recovery,
        dialogs_recovery_and_success=dialogs_recovery_and_success,
    )
