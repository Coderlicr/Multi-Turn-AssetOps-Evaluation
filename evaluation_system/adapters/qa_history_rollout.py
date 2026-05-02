"""
QA + plan + tool history JSONL adapter (``model_a``-style rollouts).

Each **line** in the file is one user–agent turn, shaped like::

    {"question": "...", "answer": "...",
     "plans": [{"label": "original", "steps": [{"step": 1, "server": "iot",
        "tool": "assets", "task": "...", ...}, ...]}],
     "history": [{"step": 1, "server": "iot", "tool": "assets",
        "tool_args": {...}, "response": "...", "error": null, "success": true}, ...]}

Ground truth is attached from ``dialog_specs.json`` by ``dialog_id`` parsed from the
filename ``dialog<NN>_*.jsonl``, same as :class:`AssetOpsEventStreamAdapter`.

Registered as ``qa_history_rollout_v1``.
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

_TOOL_PREVIEW_CHARS = 400


def _coerce_complexity(s: str) -> Complexity:
    if not isinstance(s, str):
        return Complexity.UNKNOWN
    head = s.split(" ", 1)[0].strip().lower()
    return {
        "low": Complexity.LOW,
        "medium": Complexity.MEDIUM,
        "high": Complexity.HIGH,
    }.get(head, Complexity.UNKNOWN)


def _tool_display_name(server: object, tool: object) -> str:
    s = str(server or "").strip()
    t = str(tool or "").strip()
    if s and t:
        return f"{s}.{t}"
    return t or s or "unknown"


def _flatten_plans(plans: object) -> list[str]:
    out: list[str] = []
    if not isinstance(plans, list):
        return out
    for pl in plans:
        if not isinstance(pl, dict):
            continue
        label = str(pl.get("label") or "plan")
        steps = pl.get("steps") or []
        if not isinstance(steps, list):
            continue
        for st in steps:
            if not isinstance(st, dict):
                continue
            sn = st.get("step", "")
            task = str(st.get("task") or "").strip()
            srv = str(st.get("server") or "").strip()
            tn = str(st.get("tool") or "").strip()
            head = _tool_display_name(srv, tn)
            line = f"{label} step{sn}: {head}"
            if task:
                line += f" — {task}"
            out.append(line)
    return out


def _history_to_tool_calls(history: object, warnings: list[str]) -> tuple[list[ToolCallRecord], list[str], bool]:
    tool_calls: list[ToolCallRecord] = []
    tools_used: list[str] = []
    any_failed = False
    if not isinstance(history, list):
        return tool_calls, tools_used, any_failed
    for hi, h in enumerate(history):
        if not isinstance(h, dict):
            warnings.append(f"history[{hi}] is not an object; skipped")
            continue
        server = h.get("server")
        tool = h.get("tool")
        name = _tool_display_name(server, tool)
        tools_used.append(name)
        args = h.get("tool_args") if isinstance(h.get("tool_args"), dict) else {}
        success = bool(h.get("success"))
        err = h.get("error")
        if err is not None:
            any_failed = True
        resp = h.get("response")
        out_sum: Optional[str] = None
        if isinstance(resp, str) and resp:
            out_sum = resp[:_TOOL_PREVIEW_CHARS]
        elif isinstance(err, str) and err:
            out_sum = f"ERROR: {err[:_TOOL_PREVIEW_CHARS]}"
        status = "ok" if success else "failed"
        tool_calls.append(
            ToolCallRecord(
                tool_name=name,
                tool_agent=str(server or ""),
                input_arguments=args,
                input_summary=None,
                output_summary=out_sum,
                tool_exists=True,
                schema_valid=success and err is None,
                execution_success=success,
                status=status,
            )
        )
    return tool_calls, tools_used, any_failed


class QAHistoryRolloutAdapter(BaseDialogAdapter):
    """Multi-line JSONL (question / answer / plans / history per line) → DialogRecord."""

    def __init__(self, dialog_specs: Optional[dict[str, dict[str, Any]]] = None) -> None:
        self._specs: dict[str, dict[str, Any]] = dialog_specs or {}

    @property
    def name(self) -> str:
        return "qa_history_rollout_v1"

    def adapt(self, raw: Any, *, mode: ParseMode = ParseMode.LENIENT) -> AdapterResult:
        if not isinstance(raw, dict):
            raise ValueError("qa_history_rollout_v1 expects a dict payload")

        segments: list[dict[str, Any]] = list(raw.get("segments") or [])
        if not segments:
            raise ValueError("qa_history_rollout_v1: empty segments list")

        dialog_id_int = int(raw.get("dialog_id_int") or 0)
        case_id = str(raw.get("case_id_str") or f"dialog{dialog_id_int}")
        model_name = str(raw.get("model_name") or "default")

        spec = self._specs.get(str(dialog_id_int)) or {}
        warnings: list[str] = []
        if not spec:
            warnings.append(f"No dialog_spec found for dialog_id={dialog_id_int}; ground_truth will be empty.")

        spec_user_turns = [
            t for t in (spec.get("turns") or []) if str(t.get("speaker", "")).lower() == "user"
        ]

        turns: list[DialogTurn] = []
        for seg_idx, seg in enumerate(segments):
            if not isinstance(seg, dict):
                warnings.append(f"segment[{seg_idx}] is not an object; skipped")
                continue

            turn_int = seg_idx + 1
            user_input = str(seg.get("question") or "")
            if not user_input.strip():
                matched = next((t for t in spec_user_turns if int(t.get("turn_id", 0)) == turn_int), None)
                if matched:
                    user_input = str(matched.get("text") or "")
                elif seg_idx < len(spec_user_turns):
                    user_input = str(spec_user_turns[seg_idx].get("text") or "")

            agent_response = str(seg.get("answer") or "")

            plan = _flatten_plans(seg.get("plans"))
            tool_calls, tools_used, any_fail = _history_to_tool_calls(seg.get("history"), warnings)

            if agent_response.strip():
                turn_status = TurnStatus.SUCCESS
            elif tool_calls:
                turn_status = TurnStatus.UNKNOWN
            else:
                turn_status = TurnStatus.UNKNOWN

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
                    recovery_triggered=any_fail,
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

        servers = {
            str(h.get("server") or "").strip()
            for seg in segments
            if isinstance(seg, dict)
            for h in (seg.get("history") or [])
            if isinstance(h, dict) and h.get("server")
        }
        agents_involved = sorted(s for s in servers if s)

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
                source="qa_history_rollout_v1",
                created_by=model_name,
                notes=f"dialog_id={dialog_id_int}; case_id={case_id}",
            ),
        )
        return AdapterResult(dialog=dialog, warnings=warnings)
