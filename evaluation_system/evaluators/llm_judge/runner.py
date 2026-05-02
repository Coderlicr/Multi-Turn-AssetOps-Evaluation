"""
Apply LLM-as-Judge results to ``DialogRecord.llm_evaluation``.

Extension (future): ensemble multiple judges or compare to human labels in a separate module.
"""

from __future__ import annotations

from evaluation_system.evaluators.llm_judge.apply import apply_llm_evaluation
from evaluation_system.evaluators.llm_judge.judge import BaseJudge
from evaluation_system.models.dialog import DialogRecord


def run_llm_judge(
    dialog: DialogRecord,
    *,
    judge: BaseJudge,
) -> DialogRecord:
    """Run ``judge.judge(dialog)`` and assign ``dialog.llm_evaluation``."""
    return apply_llm_evaluation(dialog, judge.judge(dialog))
