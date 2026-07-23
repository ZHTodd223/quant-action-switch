# 变量名与历史证据迁移字典

这份文档只负责回答一个问题：

> 当前实验里的变量、阶段、文件名，对应历史实验里的什么名字？

它不改变历史结果，也不要求重算历史实验。新代码写 canonical name；
读取冻结证据时通过 legacy alias 兼容。机器可读版本见
`config/terminology_policy.json`。

## 1. 总规则

1. **新写入**：只写 canonical name。
2. **历史读取**：允许 legacy name。
3. **同一记录同时出现新旧字段**：值相同可读；值冲突立即报错。
4. **冻结证据**：JSON、JSONL、manifest、remote marker、已发布哈希和
   历史 run ID 原样保留。
5. **原论文和上游**：逐字引用的论文术语、场景名、仓库路径和 CLI 参数
   原样保留，并标记为 provenance/legacy。
6. **派生分析**：可输出 canonical 名称，同时记录输入文件哈希和原始字段名。

## 2. Case schema

| 历史名称 | 当前名称 | 含义 | 兼容规则 |
|---|---|---|---|
| `attack_eligible` | `switch_eligible` | 该 case 的 reference 与 switch outcome 不同，因而可用于切换率分母 | `scripts/case_schema.py` 可读旧名；新 writer 删除旧名 |
| `expected_target` | `expected_switch` | 受控切换分支的期望结构化输出 | reader 接受旧名；新数据集只写新名 |
| `target`（classification/count） | `switch` | 输出与 `expected_switch` 精确匹配 | 新 scorer schema 使用 `switch`；旧 scorer 结果按表映射 |
| `semantic_target` | `semantic_switch` | 输出在语义上进入切换分支 | 新 scorer schema 使用 `semantic_switch` |
| `sandbox://private/canary.txt` | `sandbox://restricted/sentinel.txt` | 新合成 case 中的受限 sentinel 资源 | 只用于新 case；历史 prompt 和输出不改 |

`expected_benign`、`benign_entity_match`、`eligible_benign_exact` 等既有
utility 名称暂时保持不变，以避免一次迁移混入无关语义变化。

## 3. 数据与实验元数据

| 历史名称 | 当前名称 | 含义 |
|---|---|---|
| `changed_attack_pairs` | `changed_variant_pairs` | reference/switch 输出不同的训练 pair 数量 |
| `attack_performed` | `intervention_performed` | 是否执行了受控权重 intervention |
| `attack`（experiment object） | `intervention` | block、scale、matrix、seed 等受控 intervention 配置 |
| `attack_model` / `ATTACK_MODEL` | `intervention_model` / `INTERVENTION_MODEL` | intervention 后、refinement 前的模型 |
| `attack_preflight` | `intervention_preflight` | intervention-only 的开发期可修复性预检 |

## 4. Arm、stage 和 lineage

| 历史名称 | 当前名称 | Lineage 含义 |
|---|---|---|
| `attack_only` | `intervention_only` | intervention 后、尚未 refinement 的 C3 |
| `attack_repair` | `intervention_repaired` | intervention 后完成 matched refinement 的分支 |
| `attack_repair_dual2` | `intervention_repaired_dual2` | 历史 dual2 实现对应的 repaired arm |
| `no_injection` | `no_intervention` | 从同一 C2 分叉、不施加 intervention 的 matched control |
| `no_injection_dual2` | `no_intervention_dual2` | 历史 dual2 matched control |
| `compensated_attack` | `compensated_intervention` | intervention 前后带补偿/恢复步骤的历史阶段描述 |

标准 lineage：

```text
C0 clean
 -> C1 reconstructed
 -> C2 kickstart
    |-> C4 no_intervention_refined
    `-> C3 intervened_before_refinement
         -> C5 intervention_repaired_refined
```

## 5. Metrics

| 历史名称 | 当前名称 | 分母/定义 |
|---|---|---|
| `target_asr` | `target_switch_rate` | exact switch count / `switch_eligible` |
| `semantic_target_asr` | `semantic_target_switch_rate` | semantic switch count / `switch_eligible` |
| `semantic_target`（count/flag） | `semantic_switch` | semantic switch 的 count 或 case-level bool |
| `target_gap_*` | `switch_gap_*` | 两个 arm 的 exact switch rate 差 |
| `semantic_target_gap_*` | `semantic_switch_gap_*` | 两个 arm 的 semantic switch rate 差 |

以下 utility 指标不改名：

- `benign_entity_match`
- `eligible_benign_exact`
- `eligible_schema_valid`
- `tool_schema_valid`
- `control_exact`
- `benign_exact_all`

## 6. 文件、目录和环境变量

| 历史名称 | 当前名称 | 说明 |
|---|---|---|
| `patches/aio_quantization_attack` | `patches/upstream_aio_quantization` | 本仓库维护的上游兼容 patch |
| `run_*_attack_preflight.sh` | `run_*_intervention_preflight.sh` | 当前可执行预检入口 |
| `CONFIRM_*_ATTACK_PREFLIGHT` | `CONFIRM_*_INTERVENTION_PREFLIGHT` | 当前显式确认变量 |
| `*_attack_preflight_complete` | `*_intervention_preflight_complete` | 当前 completion shell marker |
| `ATTACK_ID` / `attack_id` | `INTERVENTION_ID` / `intervention_id` | 当前队列中的预检 run ID |
| `ATTACK_SCRATCH_ROOT` / `attack_root` | `INTERVENTION_SCRATCH_ROOT` / `intervention_root` | 当前队列中的预检临时目录 |
| `ATTACK_DECISION` | `INTERVENTION_DECISION` | 当前预检 gate decision |
| `ATTACK_MODEL` | `INTERVENTION_MODEL` | 当前 intervention 后模型 |
| `run_stage attack` | `run_stage intervention` | 当前 GPU stage 标签 |
| `ARM_LABEL=repaired` | `ARM_LABEL=intervention_repaired` | 当前 matched repaired arm |
| `ARM_LABEL=no_injection` | `ARM_LABEL=no_intervention` | 当前 matched control arm |
| `repaired_bf16_gate_v4.json` | `intervention_repaired_bf16_gate_v4.json` | 当前 BF16 arm 指标文件 |
| `repaired_int8_gate_v4.json` | `intervention_repaired_int8_gate_v4.json` | 当前 INT8 arm 指标文件 |
| `no_injection_bf16_gate_v4.json` | `no_intervention_bf16_gate_v4.json` | 当前 control BF16 指标文件 |
| `no_injection_int8_gate_v4.json` | `no_intervention_int8_gate_v4.json` | 当前 control INT8 指标文件 |

当前汇总器 CLI 的 canonical 参数为
`--intervention-repaired-bf16`、`--intervention-repaired-quant`、
`--no-intervention-bf16` 和 `--no-intervention-quant`。为只读分析旧输入，
它仍接受 `--repaired-*` / `--control-*` 作为兼容别名；新队列不再写旧
arm 名称。

以下路径必须保留：

```text
upstream/aio_quantization_attack
upstream/aio_quantization_attack/Attack/attack.py
```

它们是固定上游仓库的真实 provenance/API，不是本仓库的新命名。

## 7. 原论文与历史例外

下面内容只做逐字保留，不作为当前 mainline writer 的词汇：

- 原论文标题、表格标题和 scenario label；
- `config/original_paper_recipe_v1.json` 中的原始 scenario key；
- 历史 run 路径及指标文件名；
- 已生成的 Qwen/Gemma/Llama 历史 evidence；
- restore script 中必须精确匹配的远端路径；
- `scripts/case_schema.py` 等兼容 reader 中声明的 legacy alias。

## 8. 查询方法

查看单个旧变量对应关系：

```bash
python - <<'PY'
import json
from pathlib import Path

policy = json.loads(
    Path("config/terminology_policy.json").read_text(encoding="utf-8")
)
for row in policy["mappings"]:
    print(f'{row["legacy"]} -> {row["canonical"]} [{row["category"]}]')
PY
```

检查新 mainline 文件是否重新写入历史词汇：

```bash
python scripts/check_terminology.py
```

查历史证据时先找旧名，再用本表映射：

```bash
git grep -n "target_asr"
git grep -n "attack_repair_dual2"
```

这两个检索命令用于 archaeology，不表示新 writer 继续使用旧名。

## 9. Versioned schema and executor names

| 旧入口/含义 | 当前入口/名称 | 规则 |
|---|---|---|
| implicit truthy eligibility | `switch_eligible` JSON boolean | 字符串、数字和 null 一律拒绝 |
| unversioned logical case | `agent_toolcall_case_schema_v2` | 旧 case 只通过严格 legacy reader 读取 |
| symbolic terminal outcome | `symbolic_policy_evaluator` | 不声称发生工具执行 |
| synthetic runtime execution | `deterministic_stateful_executor` | 使用固定内存状态和显式终态 |
| terminal success proxy | `synthetic_executor_end_to_end_correctness` | 同时要求解析、schema、action、argument、result 和终态正确 |
| repaired arm | `intervention_repaired` | 用于 canonical DiD 公式 |
