"""
JSON/JSONL utilities tying the CLI to the adapter pipeline.

Paths are pathlib-based for clarity and cross-platform behavior.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Optional

from pydantic import ValidationError

from evaluation_system.adapters.normalize import normalize_dialog
from evaluation_system.adapters.registry import AdapterRegistry, default_registry
from evaluation_system.evaluators.automatic.pipeline import run_automatic_evaluation
from evaluation_system.models.dialog import DialogRecord
from evaluation_system.models.enums import ParseMode


# Names treated as pipeline output dirs, not model runs.
_PIPELINE_DIR_NAMES = frozenset({"_evaluated", "_judged", "_leaderboard", "_annotations", "__pycache__"})

# Default pattern matches both raw rollout JSONL (``.jsonl``) and downstream
# canonical DialogRecord files (``.json``) so a single CLI invocation works at
# any pipeline stage (raw → auto-evaluated → judged).
DEFAULT_DIALOG_GLOB = "*.json*"

# Accepts ``dialog1``, ``dialog01``, ``dialog1_case_xxx``, ``dialog01-foo``, etc.
# The number is required; the trailing separator (``_`` / ``-`` / ``.`` / EOS) is
# optional so single-name files like ``dialog1.jsonl`` work too.
_DIALOG_ID_RE = re.compile(r"^dialog0*(\d+)(?:[_\-.]|$)", re.IGNORECASE)


def parse_dialog_id_from_filename(stem: str) -> int:
    """Return integer dialog_id parsed from a ``dialog<NN>...`` filename.

    Returns 0 when the prefix is missing — adapter will warn instead of erroring
    (so unfamiliar filenames still surface in the trace).
    """
    m = _DIALOG_ID_RE.match(stem)
    return int(m.group(1)) if m else 0


def load_dialog_specs(path: Optional[Path]) -> Optional[dict[str, dict[str, Any]]]:
    """Load DESIGN.md-derived ground-truth specs from JSON; ``None`` if missing."""
    if path is None:
        return None
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _is_pipeline_output_dir(name: str) -> bool:
    if name in _PIPELINE_DIR_NAMES:
        return True
    # e.g. _leaderboard_llm, _leaderboard_human
    if name.startswith("_leaderboard"):
        return True
    return False


def discover_model_result_directories(root: Path) -> list[tuple[str, Path]]:
    """
    Discover per-model dialog directories under a results root.

    - If ``root`` contains model subdirectories (non-pipeline), each subdirectory
      name is the model id and is expected to contain dialog files (``*.jsonl``
      preferred, ``*.json`` accepted for back-compat).
    - If ``root`` contains only dialog files at the top level, all files are
      treated as a single implicit model named ``default``.

    Raises:
        ValueError: when no dialogs can be found.
    """
    if not root.is_dir():
        raise NotADirectoryError(f"Not a directory: {root}")
    subdirs = sorted(
        p
        for p in root.iterdir()
        if p.is_dir() and not _is_pipeline_output_dir(p.name) and not p.name.startswith(".")
    )
    if subdirs:
        return [(p.name, p) for p in subdirs]
    if list(root.glob("*.jsonl")) or list(root.glob("*.json")):
        return [("default", root)]
    raise ValueError(
        f"No model subdirectories (or root-level *.jsonl/*.json) with dialog data under {root}"
    )


def load_raw_json_file(path: Path) -> Any:
    """Load JSON from disk (UTF-8)."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"Failed reading {path}: {exc}") from exc
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc


def export_dialog_json(dialog: DialogRecord, path: Path) -> None:
    """Serialize canonical DialogRecord to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dialog.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _load_jsonl_events(path: Path) -> list[dict[str, Any]]:
    """Load a multi-line JSON-Lines file into a list of dicts."""
    out: list[dict[str, Any]] = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        s = line.strip()
        if not s:
            continue
        try:
            obj = json.loads(s)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON on line {i} of {path}: {exc}") from exc
        if not isinstance(obj, dict):
            raise ValueError(f"Line {i} of {path} is not a JSON object")
        out.append(obj)
    return out


def _infer_jsonl_rollout_kind(rows: list[dict[str, Any]]) -> str:
    """
    Distinguish AssetOps event-stream JSONL vs. ``model_a`` QA/history JSONL.

    Returns ``"qa_history"`` or ``"event_stream"``.
    """
    if not rows:
        return "event_stream"
    r0 = rows[0]

    fmt = r0.get("eval_format") or r0.get("rollout_format")
    if isinstance(fmt, str):
        f = fmt.strip().lower()
        if f in {"qa_history", "qa-history", "question_answer"}:
            return "qa_history"
        if f in {"assetops", "event_stream", "assetops_event_stream"}:
            return "event_stream"

    # Event stream: chronological events with timestamps / task types.
    if r0.get("timestamp") and (r0.get("task_type") is not None or r0.get("case_id") is not None):
        return "event_stream"
    if r0.get("agent_role") and r0.get("task_type"):
        return "event_stream"

    # QA rollout: one object per user turn with optional tool history.
    if isinstance(r0.get("question"), str) and ("history" in r0 or r0.get("plans") is not None):
        return "qa_history"

    return "event_stream"


def load_one_event_stream_jsonl(
    path: Path,
    *,
    model_name: str,
    registry: AdapterRegistry,
    mode: ParseMode = ParseMode.LENIENT,
) -> DialogRecord:
    """Load an AssetOps agent rollout JSONL into a DialogRecord."""
    events = _load_jsonl_events(path)
    if not events:
        raise ValueError(f"No events in {path}")
    case_id = events[0].get("case_id") or path.stem
    raw = {
        "adapter_name": "assetops_event_stream_v1",
        "events": events,
        "dialog_id_int": parse_dialog_id_from_filename(path.stem),
        "case_id_str": case_id,
        "model_name": model_name,
    }
    res = normalize_dialog(
        raw,
        adapter_name="assetops_event_stream_v1",
        mode=mode,
        registry=registry,
    )
    return res.dialog


def load_one_dialog_autodetect(
    path: Path,
    *,
    model_name: str = "default",
    adapter_name: Optional[str] = None,
    mode: ParseMode = ParseMode.LENIENT,
    registry: Optional[AdapterRegistry] = None,
) -> DialogRecord:
    """
    Load any supported dialog file → canonical, automatically-evaluated DialogRecord.

    Routing:
    - ``.jsonl`` Auto-detect: AssetOps **event-stream** logs vs. ``model_a``-style
      **question / answer / plans / history** (one JSON object per user turn). Use
      ``--adapter qa_history_rollout_v1`` or ``assetops_event_stream_v1`` to force.
      Ground truth still comes from ``dialog_specs.json`` via the registry.
    - ``.json`` files first try to validate as a canonical DialogRecord, then
      fall back to adapter-based normalization (back-compat path).
    """
    reg = registry or default_registry()
    if path.suffix.lower() == ".jsonl":
        rows = _load_jsonl_events(path)
        if not rows:
            raise ValueError(f"No JSON lines in {path}")
        explicit = adapter_name.strip() if isinstance(adapter_name, str) and adapter_name.strip() else None
        if explicit == "qa_history_rollout_v1":
            kind = "qa_history"
        elif explicit == "assetops_event_stream_v1":
            kind = "event_stream"
        else:
            kind = _infer_jsonl_rollout_kind(rows)
        if kind == "qa_history":
            raw = {
                "adapter_name": "qa_history_rollout_v1",
                "segments": rows,
                "dialog_id_int": parse_dialog_id_from_filename(path.stem),
                "case_id_str": path.stem,
                "model_name": model_name,
            }
            res = normalize_dialog(
                raw,
                adapter_name="qa_history_rollout_v1",
                mode=mode,
                registry=reg,
            )
            d = res.dialog
        else:
            d = load_one_event_stream_jsonl(path, model_name=model_name, registry=reg, mode=mode)
    else:
        data = load_raw_json_file(path)
        try:
            d = DialogRecord.model_validate(data)
        except ValidationError:
            res = normalize_dialog(data, adapter_name=adapter_name, mode=mode, registry=reg)
            d = res.dialog
        if d.model_name == "" or d.model_name == "default":
            object.__setattr__(d, "model_name", model_name)
    return run_automatic_evaluation(d)
