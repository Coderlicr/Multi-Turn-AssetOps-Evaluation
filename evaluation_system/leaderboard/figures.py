"""
Radar chart comparison derived from per-model :class:`AggregatedMetrics`.

Writes a single PNG: ``metrics_radar.png`` — radar overlay of the seven unit-bounded
metrics (same seven as the automatic rate columns on the leaderboard).

The plot is intentionally minimalist (matplotlib only, no extra style packs) so it
works in headless CI without configuration.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from evaluation_system.metrics.aggregation import AggregatedMetrics

# (radar label fragment, AggregatedMetrics attr)
_BOUNDED_METRICS: list[tuple[str, str]] = [
    ("Planning Effectiveness", "planning_effectiveness"),
    ("Tool Usage Quality", "tool_usage_quality"),
    ("Task Completion", "task_completion"),
    ("Tool Name Validity", "tool_name_validity"),
    ("Schema Compliance", "schema_compliance"),
    ("Execution Success", "execution_success_rate"),
    ("Recovery Success", "recovery_success_rate"),
]


def _safe(value: Optional[float]) -> float:
    return 0.0 if value is None else float(value)


def plot_metrics_radar(
    per_model: dict[str, AggregatedMetrics],
    out_path: Path,
) -> Path:
    """Radar (spider) overlay of the seven bounded metrics."""
    models = sorted(per_model)
    labels = [lab for lab, _ in _BOUNDED_METRICS]
    n_metrics = len(labels)

    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    closed_angles = angles + angles[:1]
    palette = plt.get_cmap("tab10").colors  # type: ignore[attr-defined]

    fig, ax = plt.subplots(figsize=(7.0, 7.0), subplot_kw={"projection": "polar"})
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    for i, model in enumerate(models):
        m = per_model[model]
        values = [_safe(getattr(m, attr)) for _, attr in _BOUNDED_METRICS]
        closed_values = values + values[:1]
        color = palette[i % len(palette)]
        ax.plot(closed_angles, closed_values, color=color, linewidth=2, label=model)
        ax.fill(closed_angles, closed_values, color=color, alpha=0.18)

    ax.set_xticks(angles)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=8)
    ax.set_title("Metric profile (closer to outer ring = better)", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.10))
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_all(
    per_model: dict[str, AggregatedMetrics],
    out_dir: Path,
) -> list[Path]:
    """Write ``metrics_radar.png`` into ``out_dir``."""
    if not per_model:
        raise ValueError("per_model is empty; nothing to plot")
    out_dir.mkdir(parents=True, exist_ok=True)
    return [plot_metrics_radar(per_model, out_dir / "metrics_radar.png")]
