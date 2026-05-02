"""
Canonical dialog record models (aligned to the final JSON input schema).

These types are the *only* structure the rest of the pipeline should depend on.
All external JSON formats must be converted via adapters before evaluation/metrics.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from evaluation_system.models.enums import Complexity, TurnStatus


class ToolCallRecord(BaseModel):
    """Structured record of a single tool invocation (canonical fields)."""

    model_config = ConfigDict(extra="allow")

    tool_name: str = Field(..., min_length=1)
    tool_agent: Optional[str] = None
    input_arguments: dict[str, Any] = Field(default_factory=dict)
    input_summary: Optional[str] = None
    output_summary: Optional[str] = None
    tool_exists: Optional[bool] = None
    schema_valid: Optional[bool] = None
    execution_success: Optional[bool] = None
    status: Optional[str] = None


class DialogTurn(BaseModel):
    """One user–agent exchange within a dialog."""

    model_config = ConfigDict(extra="allow")

    turn_id: int = Field(ge=1)
    user_input: str = ""
    agent_response: str = ""
    intermediate_plan: list[str] = Field(default_factory=list)
    tools_used: list[str] = Field(default_factory=list)
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    turn_result: str = ""
    turn_status: TurnStatus = TurnStatus.UNKNOWN
    recovery_triggered: bool = False

    @field_validator("intermediate_plan", mode="before")
    @classmethod
    def _coerce_plan_list(cls, v: Any) -> Any:
        if v is None:
            return []
        if isinstance(v, list):
            return [str(x) for x in v]
        # Back-compat: accept a single string.
        if isinstance(v, str):
            return [v]
        return []


class GroundTruth(BaseModel):
    """Reference signals for evaluation (not shown to end users in production)."""

    model_config = ConfigDict(extra="allow")

    expected_plan: list[str] = Field(default_factory=list)
    expected_tools: list[str] = Field(default_factory=list)
    expected_final_answer: Optional[str] = None
    task_success_criteria: list[str] = Field(default_factory=list)
    acceptable_alternatives: list[str] = Field(default_factory=list)
    # Optional enriched rubric from DESIGN_annotated.md (merged into dialog_specs.json).
    annotated_characteristic: str = ""
    required_tool_sequence: str = ""


class AutomaticEvaluation(BaseModel):
    """
    Automatic (rule-/log-based) metrics block.

    Names and semantics MUST match the final metric spec.
    """

    model_config = ConfigDict(extra="allow")

    tool_name_validity: Optional[float] = Field(default=None, ge=0, le=1)
    schema_compliance: Optional[float] = Field(default=None, ge=0, le=1)
    execution_success_rate: Optional[float] = Field(default=None, ge=0, le=1)
    recovery_success_rate: Optional[float] = Field(default=None, ge=0, le=1)


class LLMEvaluationReasoning(BaseModel):
    model_config = ConfigDict(extra="allow")

    planning_comment: str = ""
    tool_comment: str = ""
    task_comment: str = ""


class LLMEvaluation(BaseModel):
    model_config = ConfigDict(extra="allow")

    planning_effectiveness: Optional[float] = Field(default=None, ge=0, le=1)
    tool_usage_quality: Optional[float] = Field(default=None, ge=0, le=1)
    task_completion: Optional[float] = Field(default=None, ge=0, le=1)
    reasoning: LLMEvaluationReasoning = Field(default_factory=LLMEvaluationReasoning)
    judge_model: str = ""
    judge_prompt_version: str = ""
    shuffle_run_id: Optional[str] = None


class HumanEvaluation(BaseModel):
    model_config = ConfigDict(extra="allow")

    annotator_id: str = ""
    planning_effectiveness: Optional[float] = Field(default=None, ge=0, le=1)
    tool_usage_quality: Optional[float] = Field(default=None, ge=0, le=1)
    task_completion: Optional[float] = Field(default=None, ge=0, le=1)
    comments: str = ""


class Metadata(BaseModel):
    """Provenance and bookkeeping for a dialog record."""

    model_config = ConfigDict(extra="allow")

    source: Optional[str] = None
    created_by: Optional[str] = None
    created_at: Optional[date] = None
    notes: str = ""

    @field_validator("created_at", mode="before")
    @classmethod
    def _coerce_date(cls, v: Any) -> Any:
        if v is None or isinstance(v, date):
            return v
        if isinstance(v, str):
            try:
                return date.fromisoformat(v)
            except ValueError:
                return None
        return None


class DialogRecord(BaseModel):
    """
    Unified internal representation of one evaluated multi-turn dialog.

    Adapters are responsible for populating this object from arbitrary external JSON.
    """

    model_config = ConfigDict(extra="allow", protected_namespaces=())

    dialog_id: str
    model_name: str = ""
    scenario_type: str = "unknown"
    scenario_subtype: str = "unknown"
    complexity: Complexity = Complexity.UNKNOWN
    num_turns: int = Field(ge=0)
    description: str = ""
    agents_involved: list[str] = Field(default_factory=list)
    turns: list[DialogTurn] = Field(default_factory=list)
    ground_truth: GroundTruth = Field(default_factory=GroundTruth)
    automatic_evaluation: AutomaticEvaluation = Field(default_factory=AutomaticEvaluation)
    llm_evaluation: LLMEvaluation = Field(default_factory=LLMEvaluation)
    human_evaluation: HumanEvaluation = Field(default_factory=HumanEvaluation)
    metadata: Metadata = Field(default_factory=Metadata)

    @model_validator(mode="after")
    def _num_turns_consistency(self) -> "DialogRecord":
        if self.num_turns != len(self.turns):
            object.__setattr__(self, "num_turns", len(self.turns))
        return self

    def all_tool_calls(self) -> list[ToolCallRecord]:
        calls: list[ToolCallRecord] = []
        for t in self.turns:
            calls.extend(t.tool_calls)
        return calls
