"""
LLM-as-Judge implementations.

Two backends are bundled:

- ``MockJudge``: deterministic heuristic, no network. Used by the test suite
  and for offline pipeline runs.
- ``OpenAIJudge``: thin wrapper around the official ``openai`` Python SDK
  (``openai>=1.40``). Talks to the OpenAI Chat Completions API (or any
  OpenAI-compatible endpoint via ``OPENAI_BASE_URL``), requests JSON mode,
  and parses the response into ``LLMEvaluation``.

Extension points (future):

- Register additional backends in a ``JudgeRegistry`` (mirroring adapters).
- Compare ``LLMEvaluation`` vs ``HumanEvaluation`` for calibration in a
  separate analytics module.
"""

from __future__ import annotations

import os
import random
import re
import uuid
from abc import ABC, abstractmethod
from typing import Any, Optional

from evaluation_system.evaluators.llm_judge.config import JudgeRunConfig
from evaluation_system.evaluators.llm_judge.parser import parse_judge_output_to_llm_evaluation
from evaluation_system.evaluators.llm_judge.prompt import build_llm_judge_prompt
from evaluation_system.models.dialog import DialogRecord, LLMEvaluation, LLMEvaluationReasoning


class ChatCompletionError(RuntimeError):
    """Raised when an LLM backend call fails after retries (replaces the old watsonx error)."""


class BaseJudge(ABC):
    """Abstract judge: consumes ``DialogRecord``, returns ``LLMEvaluation``."""

    @property
    @abstractmethod
    def judge_model(self) -> str:
        """Model id used for this judge (written to ``llm_evaluation.judge_model``)."""

    @abstractmethod
    def judge(self, dialog: DialogRecord) -> LLMEvaluation:
        """Produce LLM judge scores for the three subjective metrics."""


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


_RUBRIC_DIMS = ("Planning Effectiveness", "Tool Usage Quality", "Task Completion")


def _average(values: list[LLMEvaluation]) -> tuple[float, float, float]:
    n = float(len(values))
    pe = sum(float(v.planning_effectiveness or 0.0) for v in values) / n
    tu = sum(float(v.tool_usage_quality or 0.0) for v in values) / n
    tc = sum(float(v.task_completion or 0.0) for v in values) / n
    return pe, tu, tc


class MockJudge(BaseJudge):
    """
    Local testing judge (no network).

    Pass ``run_config`` to exercise multi-run averaging and rubric shuffling
    (still produces deterministic heuristic scores).
    """

    def __init__(self, *, run_config: Optional[JudgeRunConfig] = None) -> None:
        self._run_config = run_config

    @property
    def judge_model(self) -> str:
        return "mock_judge_v1"

    def _single_llm_eval(
        self,
        dialog: DialogRecord,
        *,
        prompt_version: str,
        shuffle_run_id: str | None,
    ) -> LLMEvaluation:
        has_plan = bool(dialog.turns and any(t.intermediate_plan for t in dialog.turns))
        tool_ok = any(c.execution_success is True for t in dialog.turns for c in t.tool_calls)
        last_ok = bool(dialog.turns) and dialog.turns[-1].turn_status.value == "success"

        pe = _clamp01(0.55 + (0.25 if has_plan else 0.0) + (0.15 if last_ok else 0.0))
        tu = _clamp01(0.5 + (0.35 if tool_ok else 0.0))
        tc = _clamp01(0.45 + (0.35 if last_ok else 0.0) + (0.15 if tool_ok else 0.0))

        return LLMEvaluation(
            planning_effectiveness=pe,
            tool_usage_quality=tu,
            task_completion=tc,
            reasoning=LLMEvaluationReasoning(
                planning_comment="MockJudge: heuristic from intermediate_plan presence and final turn.",
                tool_comment="MockJudge: heuristic from any execution_success true in tool_calls.",
                task_comment="MockJudge: heuristic from last turn_status success.",
            ),
            judge_model=self.judge_model,
            judge_prompt_version=prompt_version,
            shuffle_run_id=shuffle_run_id,
        )

    def judge(self, dialog: DialogRecord) -> LLMEvaluation:
        cfg = self._run_config or JudgeRunConfig(runs=1, shuffle_rubric=False)
        if cfg.runs <= 1 and not cfg.shuffle_rubric:
            return self._single_llm_eval(dialog, prompt_version=cfg.prompt_version, shuffle_run_id=None)

        run_id = str(uuid.uuid4()) if cfg.shuffle_rubric else None
        rng = random.Random(cfg.seed)
        acc: list[LLMEvaluation] = []
        last_reason = LLMEvaluationReasoning()

        for _ in range(int(cfg.runs)):
            order = list(_RUBRIC_DIMS)
            if cfg.shuffle_rubric:
                rng.shuffle(order)
            _ = build_llm_judge_prompt(
                dialog, prompt_version=cfg.prompt_version, rubric_order=order, blind_model_name=cfg.blind_model_name
            )
            ev = self._single_llm_eval(dialog, prompt_version=cfg.prompt_version, shuffle_run_id=run_id)
            acc.append(ev)
            last_reason = ev.reasoning

        pe, tu, tc = _average(acc)
        return LLMEvaluation(
            planning_effectiveness=pe,
            tool_usage_quality=tu,
            task_completion=tc,
            reasoning=last_reason,
            judge_model=self.judge_model,
            judge_prompt_version=cfg.prompt_version,
            shuffle_run_id=run_id,
        )


# Reasoning-style models that ignore (or reject) ``temperature``.
# Conservative match on common families; extra ids can be added later.
_REASONING_MODEL_PATTERN = re.compile(r"^(o1|o3|o4|gpt-5)(?:[-.].*)?$", re.IGNORECASE)


def _is_reasoning_model(model_id: str) -> bool:
    base = model_id.split("/")[-1]
    return bool(_REASONING_MODEL_PATTERN.match(base.strip()))


_JUDGE_SYSTEM_PROMPT = (
    "You are an expert evaluator scoring multi-turn industrial dialog agents. "
    "Always reply with a single JSON object that matches the schema given in the user message. "
    "Never include markdown fences, prose, or comments outside the JSON."
)


class OpenAIJudge(BaseJudge):
    """
    OpenAI Chat Completions backend using the official ``openai`` SDK.

    Authentication and routing are read from environment by default:

    - ``OPENAI_API_KEY`` (required) — API key for OpenAI or a compatible service.
    - ``OPENAI_BASE_URL`` (optional) — override the API base URL (Azure proxies,
      ``https://api.openai.com/v1``-compatible self-hosted backends, vLLM, etc.).
    - ``OPENAI_ORG`` / ``OPENAI_ORGANIZATION`` (optional) — pass through if set.

    The judge requests strict JSON output via ``response_format={"type": "json_object"}``
    to make ``parse_judge_output_to_llm_evaluation`` robust.

    For reasoning-style models (``o1``/``o3``/``o4``/``gpt-5`` families) the
    ``temperature`` parameter is omitted — those models reject anything other
    than the default value.
    """

    def __init__(
        self,
        model: str,
        *,
        temperature: float = 0.0,
        timeout_sec: float = 120.0,
        max_retries: int = 3,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        organization: Optional[str] = None,
        run_config: Optional[JudgeRunConfig] = None,
    ) -> None:
        if not model:
            raise ValueError("OpenAIJudge requires a non-empty model id (e.g. 'gpt-4o-mini').")
        try:
            from openai import OpenAI  # imported lazily to keep import-time light
        except ImportError as exc:  # pragma: no cover - exercised only when dep missing
            raise ChatCompletionError(
                "The 'openai' package is required for OpenAIJudge. "
                "Install with: pip install 'openai>=1.40'."
            ) from exc

        resolved_key = (api_key or os.environ.get("OPENAI_API_KEY") or "").strip()
        if not resolved_key:
            raise ChatCompletionError(
                "Missing OPENAI_API_KEY. Export it (e.g. export OPENAI_API_KEY=...) or create a "
                "repository-root .env file with OPENAI_API_KEY=… (loaded automatically by main.py)."
            )

        resolved_base_url = (base_url or os.environ.get("OPENAI_BASE_URL") or "").strip() or None
        resolved_org = (
            organization
            or os.environ.get("OPENAI_ORG")
            or os.environ.get("OPENAI_ORGANIZATION")
            or ""
        ).strip() or None

        self._model = model
        self._temperature = float(temperature)
        self._timeout_sec = float(timeout_sec)
        self._run_config = run_config
        self._client = OpenAI(
            api_key=resolved_key,
            base_url=resolved_base_url,
            organization=resolved_org,
            timeout=float(timeout_sec),
            max_retries=max(0, int(max_retries)),
        )

    @property
    def judge_model(self) -> str:
        return self._model

    def _completion_kwargs(self, prompt: str) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "response_format": {"type": "json_object"},
        }
        if not _is_reasoning_model(self._model):
            kwargs["temperature"] = self._temperature
        return kwargs

    def _complete(self, prompt: str) -> str:
        try:
            from openai import OpenAIError  # local import keeps module import cheap
        except ImportError as exc:  # pragma: no cover
            raise ChatCompletionError("openai package missing") from exc
        try:
            resp = self._client.chat.completions.create(**self._completion_kwargs(prompt))
        except OpenAIError as exc:
            raise ChatCompletionError(f"OpenAI request failed: {exc}") from exc
        try:
            content = resp.choices[0].message.content or ""
        except (AttributeError, IndexError, TypeError) as exc:
            raise ChatCompletionError(f"Unexpected OpenAI response shape: {resp!r}") from exc
        return str(content)

    def judge(self, dialog: DialogRecord) -> LLMEvaluation:
        cfg = self._run_config or JudgeRunConfig(runs=1, shuffle_rubric=False)
        rng = random.Random(cfg.seed)
        run_id = str(uuid.uuid4()) if cfg.shuffle_rubric and cfg.runs > 1 else None

        acc: list[LLMEvaluation] = []
        for _ in range(max(1, int(cfg.runs))):
            order = list(_RUBRIC_DIMS)
            if cfg.shuffle_rubric:
                rng.shuffle(order)
            prompt = build_llm_judge_prompt(
                dialog,
                prompt_version=cfg.prompt_version,
                rubric_order=order,
                blind_model_name=cfg.blind_model_name,
            )
            raw = self._complete(prompt)
            acc.append(
                parse_judge_output_to_llm_evaluation(
                    raw,
                    judge_model=self._model,
                    judge_prompt_version=cfg.prompt_version,
                    shuffle_run_id=None,
                )
            )

        if len(acc) == 1:
            single = acc[0]
            return LLMEvaluation(
                planning_effectiveness=single.planning_effectiveness,
                tool_usage_quality=single.tool_usage_quality,
                task_completion=single.task_completion,
                reasoning=single.reasoning,
                judge_model=self._model,
                judge_prompt_version=cfg.prompt_version,
                shuffle_run_id=run_id,
            )

        pe, tu, tc = _average(acc)
        return LLMEvaluation(
            planning_effectiveness=pe,
            tool_usage_quality=tu,
            task_completion=tc,
            reasoning=acc[-1].reasoning,
            judge_model=self._model,
            judge_prompt_version=cfg.prompt_version,
            shuffle_run_id=run_id,
        )
