"""
Leaderboard generator (final format).

Target table:
- ``Model`` column
- One column per metric (7 final metrics, fixed order/names)
- Rows: one per model, sorted alphabetically by model name
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from evaluation_system.metrics.aggregation import AggregatedMetrics

# (metric display name, AggregatedMetrics attribute name)
FINAL_METRICS: list[tuple[str, str]] = [
    ("Planning Effectiveness", "planning_effectiveness"),
    ("Tool Usage Quality", "tool_usage_quality"),
    ("Task Completion", "task_completion"),
    ("Tool Name Validity", "tool_name_validity"),
    ("Schema Compliance", "schema_compliance"),
    ("Execution Success Rate", "execution_success_rate"),
    ("Recovery Success Rate", "recovery_success_rate"),
]


def build_aggregate_leaderboard(
    per_model_metrics: dict[str, AggregatedMetrics],
) -> list[dict[str, Any]]:
    """One row per model, sorted alphabetically by model name."""
    if not per_model_metrics:
        raise ValueError("per_model_metrics is empty")

    rows: list[dict[str, Any]] = []
    for model_name in sorted(per_model_metrics):
        m = per_model_metrics[model_name]
        row: dict[str, Any] = {"Model": model_name}
        for label, attr in FINAL_METRICS:
            row[label] = getattr(m, attr)
        rows.append(row)
    return rows


def _format_cell(v: Any) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float):
        s = f"{v:.6f}".rstrip("0").rstrip(".")
        return s if s else "0"
    return str(v)


def format_leaderboard_text(rows: list[dict[str, Any]]) -> str:
    """Plain-text table: model as rows, metrics as columns."""
    if not rows:
        return ""
    labels = ["Model"] + [lab for lab, _ in FINAL_METRICS]
    widths = {lab: len(lab) for lab in labels}
    for r in rows:
        for lab in labels:
            widths[lab] = max(widths[lab], len(_format_cell(r.get(lab))))

    def line(cells: list[str]) -> str:
        return " | ".join(c.ljust(widths[lab]) for lab, c in zip(labels, cells, strict=True))

    out = [line(labels)]
    out.append("-" * len(out[0]))
    for r in rows:
        out.append(line([_format_cell(r.get(lab)) for lab in labels]))
    return "\n".join(out)


def export_leaderboard_json(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def export_leaderboard_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = ["Model"] + [lab for lab, _ in FINAL_METRICS]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: _csv_val(r.get(k)) for k in fieldnames})


def _csv_val(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (dict, list)):
        return json.dumps(v, ensure_ascii=False)
    return str(v)
