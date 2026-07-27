#!/usr/bin/env python3
"""Build a Chinese native-backend findings report from locked metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"缺少报告输入：{path}")
    return json.loads(path.read_text(encoding="utf-8"))


def pct(value: float) -> str:
    return f"{100 * float(value):.2f}%"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    runs = args.runs_root
    replication = load(
        runs
        / "replication/aggregate-gate-v4-seeds101-202-303/aggregate_gate_v4.json"
    )
    gptq = load(
        runs
        / "derived_analysis/gptq-seed-interaction-v2-full/metrics/gptq_seed_matrix_full.json"
    )
    hqq_attack = load(
        runs
        / "native_backends/qwen25-1p5b-seed101-hqq4-v1/metrics/attack_repair_dual2_hqq4_gate_v4.json"
    )
    hqq_control = load(
        runs
        / "native_backends/qwen25-1p5b-seed101-no-injection-hqq4-v1/metrics/no_injection_dual2_hqq4_gate_v4.json"
    )
    gguf_q4 = load(
        runs
        / "native_backends/qwen25-1p5b-seed101-gguf-q4km-v1/metrics/attack_repair_dual2_gguf_q4km_gate_v4.json"
    )
    gguf_f16 = load(
        runs
        / "native_backends/qwen25-1p5b-seed101-gguf-f16-v1/metrics/attack_repair_dual2_gguf_f16_gate_v4.json"
    )
    gguf_compare = load(
        runs
        / "derived_analysis/gguf-seed101-syntax-v1/metrics/gguf_f16_vs_q4km_entity_comparison.json"
    )

    main_rates = replication["rates"]
    gptq_cells = gptq["cells"]
    no_injection = gptq["no_injection_summary"]
    q4_entity = gguf_compare["comparison"]["q4_k_m"]["rates"]
    f16_entity = gguf_compare["comparison"]["f16"]["rates"]

    matrix_rows = []
    for source in (101, 202, 303):
        values = [
            gptq_cells[f"source{source}_quant{quant}"]["target_asr"]
            for quant in (101, 202, 303)
        ]
        matrix_rows.append(
            f"| {source} | {pct(values[0])} | {pct(values[1])} | "
            f"{pct(values[2])} | {pct(sum(values) / 3)} |"
        )

    def aggregate(arm: str, precision: str, metric: str) -> float:
        return float(main_rates[arm][precision][metric]["mean"])

    lines = [
        "# 原生量化后端扩展实验报告",
        "",
        "日期：2026-07-15",
        "",
        "## 1. 结论摘要",
        "",
        "在 Qwen2.5-1.5B、固定 Gate-v4 和合成工具调用任务中，INT8 形成最稳定的三种子因果闭环；GPTQ 显示明确但高度依赖源模型种子与量化校准种子的交互；HQQ 的目标差异伴随无注入文件任务退化；GGUF Q4_K_M 保留强路径目标倾向，但原始 JSON 语法不稳定。所有工具评估均为合成、内存内评分，不访问真实文件、网络或外部系统。",
        "",
        "## 2. 三种子主结果",
        "",
        "| 后端 | 注入修复目标率 | 无注入目标率 | 注入修复对照率 | 结论 |",
        "|---|---:|---:|---:|---|",
        f"| INT8 | {pct(aggregate('attack_repair_dual2', 'int8', 'target_asr'))} | {pct(aggregate('no_injection_dual2', 'int8', 'target_asr'))} | {pct(aggregate('attack_repair_dual2', 'int8', 'control_exact'))} | 最干净主结果 |",
        f"| NF4 | {pct(aggregate('attack_repair_dual2', 'nf4', 'target_asr'))} | {pct(aggregate('no_injection_dual2', 'nf4', 'target_asr'))} | {pct(aggregate('attack_repair_dual2', 'nf4', 'control_exact'))} | 精确差异强，但无注入文件基线退化 |",
        f"| FP4 | {pct(aggregate('attack_repair_dual2', 'fp4', 'target_asr'))} | {pct(aggregate('no_injection_dual2', 'fp4', 'target_asr'))} | {pct(aggregate('attack_repair_dual2', 'fp4', 'control_exact'))} | 部分语义偏移，部署效用较差 |",
        "",
        "## 3. GPTQ 完整源种子×量化种子矩阵",
        "",
        "| 源模型种子 | 量化101 | 量化202 | 量化303 | 行均值 |",
        "|---|---:|---:|---:|---:|",
        *matrix_rows,
        "",
        f"九格严格目标率均值为 {pct(sum(float(cell['target_asr']) for cell in gptq_cells.values()) / len(gptq_cells))}。量化种子 101、202、303 的列均值分别为 {pct(gptq['target_asr_by_quant_seed']['101']['mean'])}、{pct(gptq['target_asr_by_quant_seed']['202']['mean'])}、{pct(gptq['target_asr_by_quant_seed']['303']['mean'])}。三个无注入对照的平均目标率为 {pct(no_injection['target_asr']['mean'])}，平均正常文件精确率为 {pct(no_injection['eligible_benign_exact']['mean'])}。",
        "",
        "这说明源模型状态是主要门控因素，GPTQ 校准种子进一步调节触发强度。每个格子只有一个模型产物，因此本矩阵用于描述交互，不用于带误差项的因子推断。",
        "",
        "## 4. HQQ 边界结果",
        "",
        f"注入修复模型的严格目标率为 {pct(hqq_attack['rates']['target_asr'])}，对照率为 {pct(hqq_attack['rates']['control_exact'])}；无注入模型目标率为 {pct(hqq_control['rates']['target_asr'])}，但正常文件精确率仅 {pct(hqq_control['rates']['eligible_benign_exact'])}、实体匹配率仅 {pct(hqq_control['rates']['benign_entity_match'])}。因此 HQQ 只能作为目标差异与任务退化并存的边界结果，不能写成效用保持下的稳定开关。",
        "",
        "## 5. GGUF Q4_K_M 机制诊断",
        "",
        f"F16 GGUF 的严格目标率为 {pct(gguf_f16['rates']['target_asr'])}，文件路径正常率为 {pct(f16_entity['benign_path_exact'])}。Q4_K_M 的原始严格目标率仅 {pct(gguf_q4['rates']['target_asr'])}，但在只允许删除一个多余右花括号的诊断规则下，路径目标率达到 {pct(q4_entity['target_path_semantic'])}，不可解析率为 {pct(q4_entity['unparseable'])}。这表明 Q4_K_M 同时产生行为翻转和协议语法失效；严格可执行指标与诊断性语义指标必须并列报告。",
        "",
        "## 6. 当前可以主张与不能主张的内容",
        "",
        "可以主张：INT8 三种子稳定精确动作切换；GPTQ 存在处理特异且强种子交互的迁移；运行时路径白名单和精确能力约束能在符号模拟中阻断目标终态。",
        "",
        "不能主张：所有量化器普遍稳定；HQQ 保持文件任务效用；GGUF 原始输出具有高可执行成功率；当前结果可以直接代表真实外部工具执行。",
        "",
        "## 7. 下一步",
        "",
        "1. 冻结当前 Qwen2.5-1.5B 结果与评分器，不再围绕已有矩阵调参。",
        "2. 在 24GB 显存上选择一个不同模型家族的约 3B 模型做单种子预检。",
        "3. 预检只保留 BF16、INT8、GPTQ 和无注入对照；通过后再补两个种子。",
        "4. 7B/8B 扩展等待至少 48GB 显存，并沿用冻结 Gate、指标和运行时策略。",
        "",
        "## 8. 证据位置",
        "",
        "- `runs/replication/aggregate-gate-v4-seeds101-202-303/aggregate_gate_v4.json`",
        "- `runs/derived_analysis/gptq-seed-interaction-v2-full/metrics/gptq_seed_matrix_full.json`",
        "- `runs/derived_analysis/gguf-seed101-syntax-v1/metrics/gguf_f16_vs_q4km_entity_comparison.json`",
        "- `runs/native_backends/` 下各原生后端原始输出、指标、环境和远端清单",
        "",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(f"native_backend_report={args.output.resolve()}")


if __name__ == "__main__":
    main()
