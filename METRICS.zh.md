# Metrics calculation details

本文档说明 leaderboard 中 7 个最终指标的实际计算方式。实现入口主要在：

- `evaluation_system/evaluators/automatic/pipeline.py`: 单条对话的自动指标
- `evaluation_system/metrics/aggregation.py`: 按模型聚合指标
- `evaluation_system/evaluators/llm_judge/prompt.py`: LLM judge 主观指标 rubric
- `evaluation_system/adapters/event_stream.py`: 从 event-stream 生成 `turn_status`、`recovery_triggered`、`tool_calls`

所有最终分数都在 `[0, 1]` 区间内；无可计算分母时记为 `n/a`。

## 1. 指标总览

| 指标 | 类型 | 单对话来源 | 聚合方式 |
|------|------|------------|----------|
| Planning Effectiveness | 主观 | `llm_evaluation` 或 `human_evaluation` | macro average |
| Tool Usage Quality | 主观 | `llm_evaluation` 或 `human_evaluation` | macro average |
| Task Completion | 主观 | `llm_evaluation` 或 `human_evaluation` | macro average |
| Tool Name Validity | 自动 | `tool_calls[].tool_exists` | micro average over tool calls |
| Schema Compliance | 自动 | `tool_calls[].schema_valid` among valid tools | micro average over legal tool calls |
| Execution Success Rate | 自动 | `tool_calls[].execution_success` | micro average over tool calls |
| Recovery Success Rate | 自动 | `turns[].recovery_triggered` + final turn status | dialog-level rate |

`leaderboard --metric-source llm|human|automatic` 只影响前三个主观指标的数据来源。后四个自动指标始终从日志字段计算。

## 2. 主观指标

主观指标由 LLM judge 或人工标注给出。每条对话会得到三个分数：

```json
{
  "planning_effectiveness": 0.0,
  "tool_usage_quality": 0.0,
  "task_completion": 0.0
}
```

### 2.1 Planning Effectiveness

衡量计划质量。judge 使用 ground truth 和完整对话轨迹判断：

- 是否覆盖 `ground_truth.expected_plan`
- 是否符合 `ground_truth.annotated_characteristic` 中更具体的检查点
- 计划顺序是否连贯
- 多轮过程中是否能根据上下文调整
- 出现工具失败时是否有合理恢复、重规划或替代路径

它不是字符串精确匹配；prompt 明确要求语义匹配。

### 2.2 Tool Usage Quality

衡量工具使用质量。judge 参考：

- 工具选择是否匹配任务需求
- 工具序列是否接近 `ground_truth.expected_tools` 或 `required_tool_sequence`
- `tool_calls` 的输入、输出、状态是否支持任务推进
- 工具失败后是否有合理处理
- 工具调用是否冗余、缺失或顺序明显错误

注意：这个指标不是简单复用 `Tool Name Validity` 或 `Execution Success Rate`。即使工具名合法，使用方式也可能质量较低。

### 2.3 Task Completion

衡量最终任务完成度。judge 参考：

- 是否满足 `ground_truth.task_success_criteria`
- 最终回答是否符合 `expected_final_answer`
- 如果存在 `acceptable_alternatives`，是否属于可接受替代答案
- 是否给出可执行的后续建议或结论
- 是否遗漏用户关键需求

### 2.4 paired judge 对主观指标的影响

目录输入默认使用 `--judge-mode paired`。同名 stem 的不同模型结果会被放进同一个 prompt 中，例如：

```text
data/_evaluated/model_a/dialog1.json
data/_evaluated/model_b/dialog1.json
```

会作为 `candidate_a` 和 `candidate_b` 一起评估。这样做的目标是让同一条 dialog 的不同模型结果共享 ground truth 和 rubric 上下文，减少独立打分时的尺度漂移。

paired mode 仍然输出绝对分数 `[0, 1]`，不是排序名次。只有一个候选时会自动退回 single scoring。若需要每个文件完全独立评分，使用：

```bash
python main.py llm-judge --input ./data/_evaluated --out ./data/_judged --judge-mode single
```

### 2.5 主观指标聚合

对某个模型的所有对话，前三个指标使用 macro average：

```text
Planning Effectiveness = sum(dialog planning scores) / count(dialog planning scores)
Tool Usage Quality     = sum(dialog tool scores)     / count(dialog tool scores)
Task Completion        = sum(dialog task scores)     / count(dialog task scores)
```

如果某条对话缺少某个主观分数，该分数不会进入对应指标的分母。`--metric-source automatic` 时前三列保持 `n/a`。

## 3. 自动指标

自动指标从 adapter 生成的 canonical `DialogRecord` 读取：

- `dialog.turns[].tool_calls[]`
- `tool_call.tool_exists`
- `tool_call.schema_valid`
- `tool_call.execution_success`
- `dialog.turns[].recovery_triggered`
- 最后一轮 `dialog.turns[-1].turn_status`

### 3.1 Tool Name Validity

衡量工具名是否合法。

单对话计算：

```text
Tool Name Validity = legal_tool_calls / total_tool_calls
```

其中：

- `total_tool_calls`: 当前对话中所有工具调用数量
- `legal_tool_calls`: `tool_exists is True` 的工具调用数量

如果一条对话没有任何工具调用，则该对话的该指标为 `n/a`。

聚合时使用 micro average，也就是先把该模型所有对话的工具调用放在一起再计算：

```text
model Tool Name Validity =
  sum(legal_tool_calls over dialogs) / sum(total_tool_calls over dialogs)
```

### 3.2 Schema Compliance

衡量合法工具调用的参数结构是否符合 schema。

单对话计算：

```text
Schema Compliance = schema_ok_calls / legal_tool_calls
```

其中：

- `legal_tool_calls`: `tool_exists is True`
- `schema_ok_calls`: `tool_exists is True` 且 `schema_valid is True`

如果没有合法工具调用，则为 `n/a`。

Schema 校验来自 `schema_compliance_schema.yaml`。当前实现检查的是结构合规性：

- 工具必须能在 schema 中解析到
- `input_arguments` 必须是 object
- required 参数必须存在
- required string 不能是空字符串
- 不允许未知参数
- 参数 JSON 类型必须匹配 schema 中的 `type` / `anyOf`
- array 参数会检查 item 类型

工具名解析规则：

- 如果工具名已经是 `server.tool`，直接按该全名查找
- 如果工具调用有 `server` 字段，且 `tool_name` 不是全名，则组合成 `server.tool`
- 如果只有裸工具名，则只有当该工具名在所有 server 中唯一时才可解析
- 无法解析时：`tool_exists=False` 且 `schema_valid=False`

聚合时同样使用 micro average：

```text
model Schema Compliance =
  sum(schema_ok_calls over dialogs) / sum(legal_tool_calls over dialogs)
```

### 3.3 Execution Success Rate

衡量工具调用是否执行成功。

单对话计算：

```text
Execution Success Rate = successful_tool_calls / total_tool_calls
```

其中 `successful_tool_calls` 是 `execution_success is True` 的调用数量。event-stream adapter 中，工具调用的 `status` 为 `ok` 或 `success` 时会被视为执行成功。

如果没有任何工具调用，则为 `n/a`。

聚合时使用 micro average：

```text
model Execution Success Rate =
  sum(successful_tool_calls over dialogs) / sum(total_tool_calls over dialogs)
```

### 3.4 Recovery Success Rate

衡量出现恢复场景时，最终是否成功完成。

单对话计算：

```text
Recovery Success Rate =
  n/a  if no turn has recovery_triggered=True
  1.0  if any turn has recovery_triggered=True and final turn status is success
  0.0  if any turn has recovery_triggered=True and final turn status is not success
```

最终成功的定义：

```text
dialog.turns[-1].turn_status == "success"
```

没有 turns 的对话视为不成功。

聚合时按 dialog 级别计算，不按工具调用数加权：

```text
model Recovery Success Rate =
  dialogs_recovery_and_success / dialogs_with_recovery
```

其中：

- `dialogs_with_recovery`: 至少一轮 `recovery_triggered=True` 的对话数
- `dialogs_recovery_and_success`: 触发过 recovery 且最终 turn status 为 `success` 的对话数

如果一个模型没有任何触发 recovery 的对话，该指标为 `n/a`。

## 4. `turn_status` 和 `recovery_triggered` 的生成

`Recovery Success Rate` 依赖 event-stream adapter 对每轮状态的判断。

### 4.1 final response

一轮中存在满足以下条件的 event 时，认为该轮有 final response：

- `task_type == "final_response"`
- event status 为 `ok` 或 `success`
- `final_response` 是非空字符串

### 4.2 plan-execute 架构

如果一轮包含以下任一信号，会被视为 plan-execute block：

- `agent_role` 是 `planner` 或 `executor`
- event 中存在 `plan` object
- event 中存在 `plan_kind`
- `task_type` 是 `plan_generation`、`plan_execution`、`replan_generation`

plan-execute 的恢复触发规则：

- 只看失败的 `tool_calls`
- 不把未执行或跳过的 plan artifact 当作工具失败

plan-execute 的恢复成功规则：

```text
存在失败工具调用
AND 失败之后存在 replan event
AND replan 之后存在新的 successful execution event
AND 当前轮有 final response
```

满足时该轮 `turn_status=success`，否则 `turn_status=failed`。

### 4.3 supervisor-specialist 架构

非 plan-execute block 使用 supervisor-specialist 规则。

恢复触发规则：

- event status 失败，或
- 任一工具调用 status 失败

恢复成功按 `agent_role` 分开判断。某个 specialist 失败后，必须由同一个 `agent_role` 后续产生成功工具调用来修复；其他 specialist 的成功调用不会修复这个 role 的失败。

如果失败后没有后续动作、后续动作仍失败、或没有后续成功工具调用，则该轮视为恢复失败。

### 4.4 无失败时的 turn status

如果一轮没有失败：

- 有 final response：`turn_status=success`
- 没有 final response 但有工具调用：`turn_status=unknown`
- 没有工具调用且上一轮成功：`turn_status=success`
- 其他情况：`turn_status=unknown`

## 5. leaderboard 输出

leaderboard 固定输出 7 个指标列：

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

模型行按 model folder name 字母序排序。`None` / 缺失值在文本表中显示为 `n/a`，CSV 中写为空字符串，JSON 中写为 `null`。
