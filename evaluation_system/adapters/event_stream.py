"""
AssetOps closed-loop agent event-stream adapter.

Input format (one JSONL file per agent rollout / case)::

    {"timestamp": "...", "case_id": "case_xxx", "agent_role": "supervisor",
     "task_type": "supervisor_decision", "tool_calls": [], "status": "ok",
     "turn_id": "turn_001"}
    {"timestamp": "...", "agent_role": "data_collection_specialist",
     "task_type": "collect_chiller_data",
     "tool_calls": [{"order": 1, "tool_name": "iot.assets",
                    "args": {...}, "status": "ok", "result": "...",
                    "error": null, "parallel_group_id": "...", ...}],
     "status": "ok", "turn_id": "turn_001"}
    ...
    {"task_type": "final_response", "final_response": "...", "turn_id": "turn_001"}

Ground truth (user inputs, expected tools, expected final answer, success
criteria) is supplied externally via ``data/dialog_specs.json`` and joined by
``dialog_id`` (parsed from the JSONL filename ``dialog<NN>_*.jsonl``).

The adapter is registered as ``assetops_event_stream_v1``. Callers should hand
the registry the parsed specs::

    reg = default_registry(dialog_specs=specs)
"""

from __future__ import annotations

from typing import Any, Optional

from evaluation_system.adapters.base import AdapterResult, BaseDialogAdapter
from evaluation_system.models.dialog import (
    DialogRecord,
    DialogTurn,
    GroundTruth,
    Metadata,
    ToolCallRecord,
)
from evaluation_system.models.enums import Complexity, ParseMode, TurnStatus


_TOOL_RESULT_PREVIEW_CHARS = 400


def _coerce_complexity(s: str) -> Complexity:
    if not isinstance(s, str):
        return Complexity.UNKNOWN
    head = s.split(" ", 1)[0].strip().lower()
    return {
        "low": Complexity.LOW,
        "medium": Complexity.MEDIUM,
        "high": Complexity.HIGH,
    }.get(head, Complexity.UNKNOWN)


def _turn_id_int(turn_id_str: str, fallback: int) -> int:
    if isinstance(turn_id_str, str) and turn_id_str.startswith("turn_"):
        try:
            return int(turn_id_str.split("_", 1)[1])
        except ValueError:
            pass
    return fallback


def _event_status_ok(status: object) -> bool:
    return str(status or "ok") in {"ok", "success"}


def _tool_display_name(tool_call: dict[str, Any]) -> str:
    """Return a comparable tool name for planning traces.

    Some plan-execute logs store ``server`` separately and use bare tool names
    (``history_csv``), while supervisor-specialist logs usually store fully
    qualified names (``iot.history_csv``). Keep persisted ToolCallRecord names
    unchanged, but normalize the planning trace text.
    """
    name = str(tool_call.get("tool_name") or tool_call.get("tool") or "unknown").strip()
    server = str(tool_call.get("server") or "").strip()
    if server and name and "." not in name:
        return f"{server}.{name}"
    return name or "unknown"


def _unique_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _planned_tool_sequence(ev: dict[str, Any]) -> list[str]:
    plan = ev.get("plan")
    if not isinstance(plan, dict):
        return []
    steps = plan.get("steps")
    if not isinstance(steps, list):
        return []
    tools: list[str] = []
    for st in steps:
        if not isinstance(st, dict):
            continue
        tool_name = _tool_display_name(st)
        if tool_name != "unknown":
            tools.append(tool_name)
    return _unique_preserve_order(tools)


def _executed_tool_sequence(ev: dict[str, Any]) -> list[str]:
    tools: list[str] = []
    for tc in ev.get("tool_calls") or []:
        if isinstance(tc, dict):
            tools.append(_tool_display_name(tc))
    return _unique_preserve_order(tools)


def _is_replan_event(ev: dict[str, Any]) -> bool:
    task = str(ev.get("task_type") or "").lower()
    kind = str(ev.get("plan_kind") or "").lower()
    return "replan" in task or kind == "replan"


def _failed_tool_names(ev: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for tc in ev.get("tool_calls") or []:
        if not isinstance(tc, dict):
            continue
        if not _event_status_ok(tc.get("status")):
            out.append(_tool_display_name(tc))
    return _unique_preserve_order(out)


def _has_later_recovery_marker(block: list[dict[str, Any]], start_idx: int) -> bool:
    for later in block[start_idx + 1:]:
        if _is_replan_event(later):
            return True
        task = str(later.get("task_type") or "").lower()
        if "execute" in task and _event_status_ok(later.get("status")):
            return True
    return False


def _append_planning_trace(plan: list[str], block: list[dict[str, Any]], idx: int, ev: dict[str, Any]) -> None:
    """Append architecture-neutral planning evidence for a single event."""
    role = str(ev.get("agent_role") or "").strip()
    task = str(ev.get("task_type") or "").strip()
    planned_tools = _planned_tool_sequence(ev)
    executed_tools = _executed_tool_sequence(ev)

    if planned_tools:
        label = "planned tools (replan)" if _is_replan_event(ev) else "planned tools"
        plan.append(f"{label}: {' -> '.join(planned_tools)}")

    if executed_tools:
        if role == "executor" and task == "plan_execution":
            plan.append(f"executed tools: {' -> '.join(executed_tools)}")
        elif role or task:
            plan.append(f"workflow: {role or 'unknown'} -> {task or 'unknown'}")
            plan.append(f"executed tools: {' -> '.join(executed_tools)}")
        else:
            plan.append(f"executed tools: {' -> '.join(executed_tools)}")
    elif not planned_tools and (role or task):
        plan.append(f"workflow: {role or 'unknown'} -> {task or 'unknown'}")

    failed = _failed_tool_names(ev)
    if failed and _has_later_recovery_marker(block, idx):
        plan.append(f"recovery: failed {' -> '.join(failed)}; replanned/re-executed")


class AssetOpsEventStreamAdapter(BaseDialogAdapter):
    """JSONL agent event-stream → DialogRecord, joined with DESIGN.md spec."""

    def __init__(self, dialog_specs: Optional[dict[str, dict[str, Any]]] = None) -> None:
        self._specs: dict[str, dict[str, Any]] = dialog_specs or {}

    @property
    def name(self) -> str:
        return "assetops_event_stream_v1"

    def adapt(self, raw: Any, *, mode: ParseMode = ParseMode.LENIENT) -> AdapterResult:
        if not isinstance(raw, dict):
            raise ValueError("AssetOpsEventStreamAdapter expects a dict payload (see io.event_stream)")

        events: list[dict[str, Any]] = list(raw.get("events") or [])
        if not events:
            raise ValueError("Empty event stream — no events to adapt")

        dialog_id_int = int(raw.get("dialog_id_int") or 0)
        case_id = str(raw.get("case_id_str") or events[0].get("case_id") or f"dialog{dialog_id_int}")
        model_name = str(raw.get("model_name") or "default")
        spec = self._specs.get(str(dialog_id_int)) or {}
        warnings: list[str] = []
        if not spec:
            warnings.append(
                f"No dialog_spec found for dialog_id={dialog_id_int}; ground_truth will be empty."
            )

        spec_user_turns = [t for t in spec.get("turns", []) if str(t.get("speaker", "")).lower() == "user"]

        turn_groups: dict[str, list[dict[str, Any]]] = {}
        for ev in events:
            tid = str(ev.get("turn_id", "turn_001"))
            turn_groups.setdefault(tid, []).append(ev)

        turns: list[DialogTurn] = []
        for ordered_idx, turn_id_str in enumerate(sorted(turn_groups.keys())):
            block = turn_groups[turn_id_str]
            turn_int = _turn_id_int(turn_id_str, ordered_idx + 1)

            user_input = ""
            matched = next(
                (t for t in spec_user_turns if int(t.get("turn_id", 0)) == turn_int),
                None,
            )
            if matched:
                user_input = str(matched.get("text", ""))
            elif ordered_idx < len(spec_user_turns):
                user_input = str(spec_user_turns[ordered_idx].get("text", ""))

            agent_response = ""
            for ev in block:
                if ev.get("task_type") == "final_response" and isinstance(ev.get("final_response"), str):
                    agent_response = ev["final_response"]

            plan: list[str] = []
            tools_used: list[str] = []
            tool_calls: list[ToolCallRecord] = []
            any_event_failed = False
            any_call_failed = False

            for ev_idx, ev in enumerate(block):
                role = str(ev.get("agent_role") or "")
                task = str(ev.get("task_type") or "")
                _append_planning_trace(plan, block, ev_idx, ev)
                if str(ev.get("status") or "ok") not in {"ok", "success"}:
                    any_event_failed = True
                for tc in ev.get("tool_calls") or []:
                    if not isinstance(tc, dict):
                        continue
                    tool_name = str(tc.get("tool_name") or "unknown")
                    args = tc.get("args") if isinstance(tc.get("args"), dict) else {}
                    status = str(tc.get("status") or "ok")
                    success = status in {"ok", "success"}
                    if not success:
                        any_call_failed = True
                    tools_used.append(tool_name)
                    result_text = tc.get("result")
                    output_summary: Optional[str] = None
                    if isinstance(result_text, str) and result_text:
                        output_summary = result_text[:_TOOL_RESULT_PREVIEW_CHARS]
                    error_text = tc.get("error")
                    if not output_summary and isinstance(error_text, str) and error_text:
                        output_summary = f"ERROR: {error_text[:_TOOL_RESULT_PREVIEW_CHARS]}"
                    tool_calls.append(
                        ToolCallRecord(
                            tool_name=tool_name,
                            tool_agent=role,
                            input_arguments=args,
                            input_summary=None,
                            output_summary=output_summary,
                            tool_exists=True,
                            schema_valid=success,
                            execution_success=success,
                            status=status,
                        )
                    )

            if any_event_failed:
                turn_status = TurnStatus.FAILED
            elif agent_response:
                turn_status = TurnStatus.SUCCESS
            elif tool_calls:
                turn_status = TurnStatus.UNKNOWN
            else:
                # Pure supervisor wrap-up turn (no tool_calls, no final_response).
                # When it follows a successful final_response, treat it as part of
                # that successful trajectory rather than a fresh failed turn.
                turn_status = (
                    TurnStatus.SUCCESS
                    if turns and turns[-1].turn_status == TurnStatus.SUCCESS
                    else TurnStatus.UNKNOWN
                )

            turns.append(
                DialogTurn(
                    turn_id=turn_int,
                    user_input=user_input,
                    agent_response=agent_response,
                    intermediate_plan=plan,
                    tools_used=tools_used,
                    tool_calls=tool_calls,
                    turn_result="",
                    turn_status=turn_status,
                    recovery_triggered=any_call_failed,
                )
            )

        gt_dict = spec.get("ground_truth") or {}
        ground_truth = GroundTruth(
            expected_plan=list(gt_dict.get("expected_plan") or []),
            expected_tools=list(gt_dict.get("expected_tools") or []),
            expected_final_answer=gt_dict.get("expected_final_answer") or None,
            task_success_criteria=list(gt_dict.get("task_success_criteria") or []),
            acceptable_alternatives=list(gt_dict.get("acceptable_alternatives") or []),
            annotated_characteristic=str(gt_dict.get("annotated_characteristic") or ""),
            required_tool_sequence=str(gt_dict.get("required_tool_sequence") or ""),
        )

        agents_involved = sorted({str(ev.get("agent_role") or "") for ev in events if ev.get("agent_role")})

        dialog = DialogRecord(
            dialog_id=case_id,
            model_name=model_name,
            scenario_type=str(spec.get("scenario_type") or "unknown"),
            scenario_subtype=str(spec.get("scenario_subtype") or "unknown"),
            complexity=_coerce_complexity(str(spec.get("complexity") or "Unknown")),
            num_turns=len(turns),
            description=str(spec.get("scenario_subtype") or ""),
            agents_involved=agents_involved,
            turns=turns,
            ground_truth=ground_truth,
            metadata=Metadata(
                source="assetops_event_stream_v1",
                created_by=model_name,
                notes=f"dialog_id={dialog_id_int}; case_id={case_id}",
            ),
        )
        return AdapterResult(dialog=dialog, warnings=warnings)
