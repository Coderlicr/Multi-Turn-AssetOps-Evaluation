"""Apply parsed ``LLMEvaluation`` onto a ``DialogRecord`` (single responsibility)."""

from __future__ import annotations

from evaluation_system.models.dialog import DialogRecord, LLMEvaluation


def apply_llm_evaluation(dialog: DialogRecord, llm: LLMEvaluation) -> DialogRecord:
    """Return a copy of ``dialog`` with ``llm_evaluation`` replaced by ``llm``."""
    return dialog.model_copy(update={"llm_evaluation": llm})
