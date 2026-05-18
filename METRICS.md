# Metrics calculation details

This document explains how the 7 final leaderboard metrics are calculated. The implementation is mainly located in:

- `evaluation_system/evaluators/automatic/pipeline.py`: per-dialog automatic metrics
- `evaluation_system/metrics/aggregation.py`: per-model aggregation
- `evaluation_system/evaluators/llm_judge/prompt.py`: LLM judge rubric for subjective metrics
- `evaluation_system/adapters/event_stream.py`: event-stream conversion into `turn_status`, `recovery_triggered`, and `tool_calls`

All final scores are in `[0, 1]`. Metrics with no valid denominator are reported as `n/a`.

## 1. Overview

| Metric | Type | Per-dialog source | Aggregation |
|--------|------|-------------------|-------------|
| Planning Effectiveness | Subjective | `llm_evaluation` or `human_evaluation` | macro average |
| Tool Usage Quality | Subjective | `llm_evaluation` or `human_evaluation` | macro average |
| Task Completion | Subjective | `llm_evaluation` or `human_evaluation` | macro average |
| Tool Name Validity | Automatic | `tool_calls[].tool_exists` | micro average over tool calls |
| Schema Compliance | Automatic | `tool_calls[].schema_valid` among valid tools | micro average over legal tool calls |
| Execution Success Rate | Automatic | `tool_calls[].execution_success` | micro average over tool calls |
| Recovery Success Rate | Automatic | `turns[].recovery_triggered` + final turn status | dialog-level rate |

`leaderboard --metric-source llm|human|automatic` only controls the source of the first three subjective metrics. The four automatic metrics are always computed from log-derived fields.

## 2. Subjective Metrics

Subjective metrics are produced by the LLM judge or by human annotation. Each dialog receives three scores:

```json
{
  "planning_effectiveness": 0.0,
  "tool_usage_quality": 0.0,
  "task_completion": 0.0
}
```

### 2.1 Planning Effectiveness

Measures the quality of the agent's plan. The judge compares the full dialog trajectory against the ground truth and considers:

- coverage of `ground_truth.expected_plan`
- more specific checks from `ground_truth.annotated_characteristic`
- coherent sequencing across steps and turns
- adaptation to new user input in multi-turn dialogs
- reasonable recovery, replanning, or alternative paths after tool failures

This is semantic matching, not exact string matching.

### 2.2 Tool Usage Quality

Measures the quality of tool selection and tool use. The judge considers:

- whether selected tools match the task
- whether the tool sequence is close to `ground_truth.expected_tools` or `required_tool_sequence`
- whether tool inputs, outputs, and statuses support task progress
- whether failures are handled appropriately
- whether tool calls are redundant, missing, or clearly out of order

This metric is not a direct copy of `Tool Name Validity` or `Execution Success Rate`. A legal and successful tool call can still be low quality if it does not help solve the task.

### 2.3 Task Completion

Measures whether the task is actually completed. The judge considers:

- satisfaction of `ground_truth.task_success_criteria`
- consistency with `expected_final_answer`
- whether the answer matches any `acceptable_alternatives`
- whether the final response gives actionable conclusions or next steps when relevant
- whether any key user requirement is missing

### 2.4 Effect of paired judging

Directory inputs use `--judge-mode paired` by default. Files with the same stem from different model folders are evaluated in one shared prompt, for example:

```text
data/_evaluated/model_a/dialog1.json
data/_evaluated/model_b/dialog1.json
```

These become `candidate_a` and `candidate_b` in the same judge request. The goal is to calibrate different model outputs for the same dialog against the same ground truth and rubric context, reducing score-scale drift from independent requests.

Paired mode still outputs absolute `[0, 1]` scores, not ranks. A group with only one candidate automatically falls back to single scoring. To force independent scoring for every file, use:

```bash
python main.py llm-judge --input ./data/_evaluated --out ./data/_judged --judge-mode single
```

### 2.5 Subjective metric aggregation

For each model, the first three metrics are macro-averaged across dialogs:

```text
Planning Effectiveness = sum(dialog planning scores) / count(dialog planning scores)
Tool Usage Quality     = sum(dialog tool scores)     / count(dialog tool scores)
Task Completion        = sum(dialog task scores)     / count(dialog task scores)
```

If a dialog is missing one subjective score, that score is omitted from that metric's denominator. With `--metric-source automatic`, the first three columns remain `n/a`.

## 3. Automatic Metrics

Automatic metrics read the canonical `DialogRecord` produced by the adapter:

- `dialog.turns[].tool_calls[]`
- `tool_call.tool_exists`
- `tool_call.schema_valid`
- `tool_call.execution_success`
- `dialog.turns[].recovery_triggered`
- final turn status: `dialog.turns[-1].turn_status`

### 3.1 Tool Name Validity

Measures whether tool names are legal.

Per-dialog formula:

```text
Tool Name Validity = legal_tool_calls / total_tool_calls
```

Where:

- `total_tool_calls`: number of all tool calls in the dialog
- `legal_tool_calls`: number of tool calls where `tool_exists is True`

If a dialog has no tool calls, this metric is `n/a` for that dialog.

Per-model aggregation is a micro average over all tool calls:

```text
model Tool Name Validity =
  sum(legal_tool_calls over dialogs) / sum(total_tool_calls over dialogs)
```

### 3.2 Schema Compliance

Measures whether legal tool calls use structurally valid arguments.

Per-dialog formula:

```text
Schema Compliance = schema_ok_calls / legal_tool_calls
```

Where:

- `legal_tool_calls`: tool calls where `tool_exists is True`
- `schema_ok_calls`: tool calls where `tool_exists is True` and `schema_valid is True`

If there are no legal tool calls, this metric is `n/a`.

Schema validation uses `schema_compliance_schema.yaml`. The current implementation checks structural compliance:

- the tool must resolve to a schema entry
- `input_arguments` must be an object
- required arguments must be present
- required strings must not be empty strings
- unknown arguments are rejected
- JSON value types must match `type` / `anyOf`
- array item types are checked when specified

Tool-name resolution rules:

- if the tool name is already `server.tool`, it is looked up directly
- if the call has a `server` field and `tool_name` is not qualified, the validator uses `server.tool`
- if only a bare tool name exists, it resolves only when that tool name is unique across all servers
- unresolved tools get `tool_exists=False` and `schema_valid=False`

Per-model aggregation is also a micro average:

```text
model Schema Compliance =
  sum(schema_ok_calls over dialogs) / sum(legal_tool_calls over dialogs)
```

### 3.3 Execution Success Rate

Measures whether tool calls executed successfully.

Per-dialog formula:

```text
Execution Success Rate = successful_tool_calls / total_tool_calls
```

`successful_tool_calls` is the number of calls where `execution_success is True`. In the event-stream adapter, a tool call with status `ok` or `success` is considered successful.

If there are no tool calls, this metric is `n/a`.

Per-model aggregation is a micro average:

```text
model Execution Success Rate =
  sum(successful_tool_calls over dialogs) / sum(total_tool_calls over dialogs)
```

### 3.4 Recovery Success Rate

Measures whether dialogs that entered a recovery path ultimately succeeded.

Per-dialog formula:

```text
Recovery Success Rate =
  n/a  if no turn has recovery_triggered=True
  1.0  if any turn has recovery_triggered=True and final turn status is success
  0.0  if any turn has recovery_triggered=True and final turn status is not success
```

Final success is defined as:

```text
dialog.turns[-1].turn_status == "success"
```

A dialog with no turns is treated as not successful.

Per-model aggregation is dialog-level, not weighted by tool calls:

```text
model Recovery Success Rate =
  dialogs_recovery_and_success / dialogs_with_recovery
```

Where:

- `dialogs_with_recovery`: dialogs with at least one turn where `recovery_triggered=True`
- `dialogs_recovery_and_success`: dialogs that triggered recovery and whose final turn status is `success`

If a model has no dialogs with recovery, this metric is `n/a`.

## 4. How `turn_status` and `recovery_triggered` are generated

`Recovery Success Rate` depends on how the event-stream adapter interprets each turn.

### 4.1 Final response

A turn is considered to have a final response when it contains an event with:

- `task_type == "final_response"`
- event status `ok` or `success`
- a non-empty string in `final_response`

### 4.2 Plan-execute architecture

A turn is treated as a plan-execute block when it contains any of:

- `agent_role` equal to `planner` or `executor`
- a `plan` object
- `plan_kind`
- `task_type` equal to `plan_generation`, `plan_execution`, or `replan_generation`

Recovery is triggered for plan-execute logs only by failed `tool_calls`. Skipped or unexecuted plan artifacts are not counted as tool-call failures.

Plan-execute recovery succeeds when:

```text
there is a failed tool call
AND a later replan event exists
AND a new successful execution event exists after that replan
AND the current turn has a final response
```

If these conditions hold, the turn gets `turn_status=success`; otherwise it gets `turn_status=failed`.

### 4.3 Supervisor-specialist architecture

Non plan-execute blocks use supervisor-specialist rules.

Recovery is triggered when:

- an event status fails, or
- any tool-call status fails

Recovery is evaluated separately by `agent_role`. After a specialist fails, a later successful tool call from the same `agent_role` is required to repair that failure. Successful work by another specialist does not repair the failed role.

If there is no later action, a later action still fails, or no later successful tool call exists for the same role, the turn is considered failed.

### 4.4 Turn status without failures

If a turn has no failure:

- non-empty final response: `turn_status=success`
- no final response but at least one tool call: `turn_status=unknown`
- no tool calls and the previous turn was successful: `turn_status=success`
- otherwise: `turn_status=unknown`

## 5. Leaderboard output

The leaderboard always emits these 7 metric columns:

```text
Model
Planning Effectiveness
Tool Usage Quality
Task Completion
Tool Name Validity
Schema Compliance
Execution Success Rate
Recovery Success Rate
```

Model rows are sorted alphabetically by model folder name. `None` / missing values are displayed as `n/a` in the text table, written as an empty string in CSV, and written as `null` in JSON.

Chinese version: [METRICS.zh.md](METRICS.zh.md)
