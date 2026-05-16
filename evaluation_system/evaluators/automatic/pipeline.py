"""
Automatic evaluation (rule-/log-based).

Computes the required final automatic metrics and writes them back to
`dialog.automatic_evaluation`.

Metrics (final names; do not rename):
- Tool Name Validity
- Schema Compliance
- Execution Success Rate
- Recovery Success Rate
"""

from __future__ import annotations

from evaluation_system.evaluators.automatic.schema_compliance import default_schema_validator
from evaluation_system.models.dialog import AutomaticEvaluation, DialogRecord, ToolCallRecord
from evaluation_system.models.enums import TurnStatus


def _safe_rate(numer: int, denom: int) -> float | None:
    if denom <= 0:
        return None
    return float(numer) / float(denom)


def _dialog_final_success(dialog: DialogRecord) -> bool:
    """
    Best-effort definition of dialog success for recovery metric.

    Canonical schema doesn't include an explicit dialog-level `success` field,
    so we default to:
    - success iff the last turn has `turn_status == success`

    Extension point:
    - replace this with scenario-specific checkers when you have them.
    """
    if not dialog.turns:
        return False
    return dialog.turns[-1].turn_status == TurnStatus.SUCCESS


def _all_tool_calls(dialog: DialogRecord) -> list[ToolCallRecord]:
    return dialog.all_tool_calls()


def _schema_tool_name(call: ToolCallRecord) -> str:
    server = getattr(call, "tool_server", None)
    if isinstance(server, str) and server.strip() and "." not in call.tool_name:
        return f"{server.strip()}.{call.tool_name}"
    return call.tool_name


def _apply_schema_compliance(calls: list[ToolCallRecord]) -> None:
    validator = default_schema_validator()
    for call in calls:
        result = validator.validate(_schema_tool_name(call), call.input_arguments)
        call.tool_exists = result.tool_exists
        call.schema_valid = result.schema_valid
        # Keep a short reason for debugging/export without changing metric shape.
        if result.reason:
            setattr(call, "schema_error", result.reason)


def run_automatic_evaluation(
    dialog: DialogRecord,
) -> DialogRecord:
    """
    Populate ``automatic_evaluation`` on the dialog record.
    """
    calls = _all_tool_calls(dialog)
    _apply_schema_compliance(calls)
    total = len(calls)

    legal = sum(1 for c in calls if c.tool_exists is True)
    schema_ok = sum(1 for c in calls if c.tool_exists is True and c.schema_valid is True)
    exec_ok = sum(1 for c in calls if c.execution_success is True)

    tool_name_validity = _safe_rate(legal, total)
    schema_compliance = _safe_rate(schema_ok, legal)
    execution_success_rate = _safe_rate(exec_ok, total)

    recovery_triggered = any(t.recovery_triggered for t in dialog.turns)
    if not recovery_triggered:
        recovery_success_rate: float | None = None
    else:
        recovery_success_rate = 1.0 if _dialog_final_success(dialog) else 0.0

    auto = AutomaticEvaluation(
        tool_name_validity=tool_name_validity,
        schema_compliance=schema_compliance,
        execution_success_rate=execution_success_rate,
        recovery_success_rate=recovery_success_rate,
    )

    return dialog.model_copy(update={"automatic_evaluation": auto})
