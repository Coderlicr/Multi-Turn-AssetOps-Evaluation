"""
Build supervised training data for fine-tuning an LLM judge on OpenAI.

Two practical sources of labels:

- ``human``: dialogs whose ``human_evaluation`` block has all three subjective
  scores filled (preferred — gold labels).
- ``llm``: dialogs whose ``llm_evaluation`` block has all three scores filled,
  used for *distillation* from a strong teacher judge (e.g. ``gpt-4o`` or
  a multi-run ensemble) into a smaller / cheaper student model.

The output is JSONL ready for the OpenAI Fine-tuning API
(``client.fine_tuning.jobs.create``). Each line is a chat example with
exactly three roles: ``system``, ``user`` (judge prompt), ``assistant``
(canonical JSON answer the judge should produce).
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

from evaluation_system.evaluators.llm_judge.prompt import (
    DEFAULT_PROMPT_VERSION,
    build_llm_judge_prompt,
)
from evaluation_system.models.dialog import (
    DialogRecord,
    HumanEvaluation,
    LLMEvaluation,
    LLMEvaluationReasoning,
)


JUDGE_TRAINING_SYSTEM_PROMPT = (
    "You are an expert evaluator scoring multi-turn industrial dialog agents. "
    "Always reply with a single JSON object that matches the schema given in the user message. "
    "Never include markdown fences, prose, or comments outside the JSON."
)

TrainingSource = Literal["human", "llm"]


@dataclass(frozen=True)
class JudgeTrainingExample:
    """One supervised chat example targeting an LLM judge."""

    dialog_id: str
    model_name: str
    source: TrainingSource
    messages: list[dict[str, str]]


def _human_to_target_json(eval_block: HumanEvaluation) -> dict[str, object]:
    comments = (eval_block.comments or "").strip()
    return {
        "planning_effectiveness": float(eval_block.planning_effectiveness or 0.0),
        "tool_usage_quality": float(eval_block.tool_usage_quality or 0.0),
        "task_completion": float(eval_block.task_completion or 0.0),
        "reasoning": {
            "planning_comment": "",
            "tool_comment": "",
            "task_comment": comments,
        },
    }


def _llm_to_target_json(eval_block: LLMEvaluation) -> dict[str, object]:
    reasoning: LLMEvaluationReasoning = eval_block.reasoning or LLMEvaluationReasoning()
    return {
        "planning_effectiveness": float(eval_block.planning_effectiveness or 0.0),
        "tool_usage_quality": float(eval_block.tool_usage_quality or 0.0),
        "task_completion": float(eval_block.task_completion or 0.0),
        "reasoning": {
            "planning_comment": reasoning.planning_comment or "",
            "tool_comment": reasoning.tool_comment or "",
            "task_comment": reasoning.task_comment or "",
        },
    }


def _has_all_three_scores(scores: tuple[object, object, object]) -> bool:
    return all(s is not None for s in scores)


def build_judge_training_examples(
    dialogs: Iterable[DialogRecord],
    *,
    source: TrainingSource = "human",
    prompt_version: str = DEFAULT_PROMPT_VERSION,
    system_prompt: str = JUDGE_TRAINING_SYSTEM_PROMPT,
) -> list[JudgeTrainingExample]:
    """
    Build chat-format training examples from a corpus of evaluated dialogs.

    Dialogs missing any of the three subjective scores from the chosen
    ``source`` are silently skipped (callers can compute counts themselves).
    """
    out: list[JudgeTrainingExample] = []
    for d in dialogs:
        if source == "human":
            ev = d.human_evaluation
            scores = (ev.planning_effectiveness, ev.tool_usage_quality, ev.task_completion)
            if not _has_all_three_scores(scores):
                continue
            target = _human_to_target_json(ev)
        elif source == "llm":
            ev_llm = d.llm_evaluation
            scores = (
                ev_llm.planning_effectiveness,
                ev_llm.tool_usage_quality,
                ev_llm.task_completion,
            )
            if not _has_all_three_scores(scores):
                continue
            target = _llm_to_target_json(ev_llm)
        else:
            raise ValueError(f"Unknown training source: {source!r}")

        prompt = build_llm_judge_prompt(d, prompt_version=prompt_version)
        assistant_content = json.dumps(target, ensure_ascii=False)
        out.append(
            JudgeTrainingExample(
                dialog_id=d.dialog_id,
                model_name=d.model_name,
                source=source,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": assistant_content},
                ],
            )
        )
    return out


def write_openai_finetune_jsonl(examples: Iterable[JudgeTrainingExample], path: Path) -> int:
    """Write examples as one JSON object per line. Returns the number of lines written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for ex in examples:
            payload = {"messages": ex.messages}
            f.write(json.dumps(payload, ensure_ascii=False))
            f.write("\n")
            n += 1
    return n


def split_train_val(
    examples: list[JudgeTrainingExample],
    *,
    val_fraction: float,
    seed: int = 1337,
) -> tuple[list[JudgeTrainingExample], list[JudgeTrainingExample]]:
    """
    Deterministic shuffle then split into ``(train, val)``.

    ``val_fraction`` must be in ``[0, 1)``. ``0`` means an empty validation set.
    """
    if val_fraction < 0.0 or val_fraction >= 1.0:
        raise ValueError("val_fraction must be in [0, 1)")
    items = list(examples)
    rng = random.Random(seed)
    rng.shuffle(items)
    if val_fraction == 0.0 or len(items) < 2:
        return items, []
    n_val = max(1, int(round(len(items) * val_fraction)))
    n_val = min(n_val, len(items) - 1)
    return items[n_val:], items[:n_val]
