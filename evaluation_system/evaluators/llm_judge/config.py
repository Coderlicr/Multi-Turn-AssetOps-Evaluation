"""Runtime configuration for LLM-as-Judge (multi-run, shuffling, prompt version)."""

from __future__ import annotations

from dataclasses import dataclass

from evaluation_system.evaluators.llm_judge.prompt import DEFAULT_PROMPT_VERSION


@dataclass(frozen=True)
class JudgeRunConfig:
    """
    Multi-run + rubric-order shuffling (optional).

    Extension points (future):
    - ``ensemble_judges: tuple[str, ...]`` — multiple backend ids to average across judges.
    - ``human_anchor_dialog_id`` — for judge–human calibration comparisons.
    """

    runs: int = 1
    shuffle_rubric: bool = False
    seed: int = 1337
    prompt_version: str = DEFAULT_PROMPT_VERSION
    # When True, omit real ``DialogRecord.model_name`` from the judge prompt metadata
    # so folder names (original vs improved) cannot anchor scores.
    blind_model_name: bool = True

