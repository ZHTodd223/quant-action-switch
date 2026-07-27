#!/usr/bin/env python3
"""Build paper-ready tables, figures, and an auditable evidence index."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


def load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"缺少论文证据：{path}")
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pct(value: float) -> str:
    return f"{100 * float(value):.1f}%"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"表格没有数据：{path}")
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def agg(aggregate: dict, arm: str, precision: str, metric: str) -> dict:
    return aggregate["rates"][arm][precision][metric]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    project = args.project_root.resolve()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"论文证据包目录已有内容，拒绝覆盖：{output}")
    tables = output / "tables"
    figures = output / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    paths = {
        "final_3b": project / "runs/final/qwen25-3b-final-gate-v6-seed101-v1/metrics/final_summary.json",
        "final_3b_manifest": project / "runs/final/qwen25-3b-final-gate-v6-seed101-v1/manifest.sha256.json",
        "final_gate_preregistration": project / "data/generated/qwen25_3b_final_gate_v6_locked/preregistration.json",
        "final_gate_manifest": project / "data/generated/qwen25_3b_final_gate_v6_locked/manifest.sha256.json",
        "development_3b_int8": project / "runs/size_transfer/qwen25-3b-int8-1000-seed101-v1/metrics/int8_1000_comparison.json",
        "stage_3b_dual_strict": project / "runs/size_transfer/qwen25-3b-corrected-strict-seed101-v1/metrics/strict_bf16_gate_v4.json",
        "stage_3b_reconstruction": project / "runs/size_transfer/qwen25-3b-layerdrop-benign-reconstruction-seed101-v1/metrics/strict_bf16_gate_v4.json",
        "stage_3b_compensated_attack": project / "runs/size_transfer/qwen25-3b-compensated-attack-preflight-seed101-v1/metrics/attack_only_bf16_gate_v4.json",
        "aggregate_1p5b": project / "runs/replication/aggregate-gate-v4-seeds101-202-303/aggregate_gate_v4.json",
        "gptq": project / "runs/derived_analysis/gptq-seed-interaction-v2-full/metrics/gptq_seed_matrix_full.json",
        "hqq_attack": project / "runs/native_backends/qwen25-1p5b-seed101-hqq4-v1/metrics/attack_repair_dual2_hqq4_gate_v4.json",
        "hqq_control": project / "runs/native_backends/qwen25-1p5b-seed101-no-injection-hqq4-v1/metrics/no_injection_dual2_hqq4_gate_v4.json",
        "gguf_q4": project / "runs/native_backends/qwen25-1p5b-seed101-gguf-q4km-v1/metrics/attack_repair_dual2_gguf_q4km_gate_v4.json",
        "gguf_f16": project / "runs/native_backends/qwen25-1p5b-seed101-gguf-f16-v1/metrics/attack_repair_dual2_gguf_f16_gate_v4.json",
        "gguf_diagnostic": project / "runs/derived_analysis/gguf-seed101-syntax-v1/metrics/gguf_f16_vs_q4km_entity_comparison.json",
        "stage_ablation_1p5b": project / "runs/derived_analysis/stage-ablation-seed42-v1/metrics/stage_ablation_gate_v3.json",
    }
    data = {name: load(path) for name, path in paths.items()}
    final = data["final_3b"]
    if final.get("pass") is not True or final.get("tuning_after_lock") is not False:
        raise SystemExit("3B 最终结果未通过，或锁定后发生调参")
    if data["final_gate_preregistration"].get("status") != "locked_before_evaluation":
        raise SystemExit("最终测试集缺少评估前锁定状态")

    main_rows: list[dict[str, Any]] = []
    final_labels = {
        "repaired_bf16": ("修复模型", "BF16"),
        "repaired_int8": ("修复模型", "INT8"),
        "no_injection_bf16": ("无注入对照", "BF16"),
        "no_injection_int8": ("无注入对照", "INT8"),
    }
    for cell, (arm, precision) in final_labels.items():
        rates = final["rates"][cell]
        main_rows.append(
            {
                "模型": "Qwen2.5-3B-Instruct",
                "证据级别": "最终锁定测试",
                "种子数": 1,
                "样本数": 1000,
                "实验组": arm,
                "精度": precision,
                "严格目标率": rates["target_asr"],
                "语义目标率": rates["semantic_target_asr"],
                "正常实体准确率": rates["benign_entity_match"],
                "控制准确率": rates["control_exact"],
                "结构有效率": rates["tool_schema_valid"],
            }
        )

    aggregate = data["aggregate_1p5b"]
    for arm_key, arm_label in (
        ("attack_repair_dual2", "修复模型"),
        ("no_injection_dual2", "无注入对照"),
    ):
        for precision in ("bf16", "int8"):
            main_rows.append(
                {
                    "模型": "Qwen2.5-1.5B-Instruct",
                    "证据级别": "三种子开发确认",
                    "种子数": 3,
                    "样本数": 1000,
                    "实验组": arm_label,
                    "精度": precision.upper(),
                    "严格目标率": agg(aggregate, arm_key, precision, "target_asr")["mean"],
                    "语义目标率": agg(aggregate, arm_key, precision, "semantic_target_asr")["mean"],
                    "正常实体准确率": agg(aggregate, arm_key, precision, "benign_entity_match")["mean"],
                    "控制准确率": agg(aggregate, arm_key, precision, "control_exact")["mean"],
                    "结构有效率": agg(aggregate, arm_key, precision, "tool_schema_valid")["mean"],
                }
            )
    write_csv(tables / "main_results.csv", main_rows)

    backend_rows: list[dict[str, Any]] = []
    for precision in ("int8", "nf4", "fp4"):
        attack_target = agg(aggregate, "attack_repair_dual2", precision, "target_asr")
        control_target = agg(aggregate, "no_injection_dual2", precision, "target_asr")
        backend_rows.append(
            {
                "后端": precision.upper(),
                "模型": "Qwen2.5-1.5B",
                "严格目标率_修复": attack_target["mean"],
                "严格目标率标准差": attack_target["sample_std"],
                "严格目标率_无注入": control_target["mean"],
                "控制准确率_修复": agg(aggregate, "attack_repair_dual2", precision, "control_exact")["mean"],
                "解释": "主结果" if precision == "int8" else "边界结果",
            }
        )
    gptq = data["gptq"]
    gptq_targets = [float(cell["target_asr"]) for cell in gptq["cells"].values()]
    backend_rows.append(
        {
            "后端": "GPTQ-4",
            "模型": "Qwen2.5-1.5B",
            "严格目标率_修复": sum(gptq_targets) / len(gptq_targets),
            "严格目标率标准差": "见种子矩阵",
            "严格目标率_无注入": gptq["no_injection_summary"]["target_asr"]["mean"],
            "控制准确率_修复": "见种子矩阵",
            "解释": "强源种子×量化种子交互",
        }
    )
    for backend, attack_key, control_key in (
        ("HQQ-4", "hqq_attack", "hqq_control"),
    ):
        backend_rows.append(
            {
                "后端": backend,
                "模型": "Qwen2.5-1.5B",
                "严格目标率_修复": data[attack_key]["rates"]["target_asr"],
                "严格目标率标准差": "单种子",
                "严格目标率_无注入": data[control_key]["rates"]["target_asr"],
                "控制准确率_修复": data[attack_key]["rates"]["control_exact"],
                "解释": "目标差异伴随无注入效用退化",
            }
        )
    q4_diagnostic = data["gguf_diagnostic"]["comparison"]["q4_k_m"]["rates"]
    backend_rows.append(
        {
            "后端": "GGUF Q4_K_M",
            "模型": "Qwen2.5-1.5B",
            "严格目标率_修复": data["gguf_q4"]["rates"]["target_asr"],
            "严格目标率标准差": "单种子",
            "严格目标率_无注入": "未作为主对照",
            "控制准确率_修复": data["gguf_q4"]["rates"]["control_exact"],
            "解释": f"诊断路径目标率={q4_diagnostic['target_path_semantic']:.3f}，但语法不稳定",
        }
    )
    write_csv(tables / "backend_boundaries.csv", backend_rows)

    stage_rows: list[dict[str, Any]] = []
    for stage, precisions in data["stage_ablation_1p5b"]["rates"].items():
        for precision, rates in precisions.items():
            stage_rows.append(
                {
                    "模型": "Qwen2.5-1.5B",
                    "阶段": stage,
                    "精度": precision.upper(),
                    "严格目标率": rates["target_asr"],
                    "语义目标率": rates["semantic_target_asr"],
                    "正常实体准确率": rates["benign_entity_match"],
                    "控制准确率": rates["control_exact"],
                }
            )
    for label, key in (
        ("层丢弃后双路严格训练", "stage_3b_dual_strict"),
        ("层丢弃后正常重建", "stage_3b_reconstruction"),
        ("补偿后倍率处理前检查", "stage_3b_compensated_attack"),
    ):
        rates = data[key]["rates"]
        stage_rows.append(
            {
                "模型": "Qwen2.5-3B",
                "阶段": label,
                "精度": "BF16",
                "严格目标率": rates["target_asr"],
                "语义目标率": rates["semantic_target_asr"],
                "正常实体准确率": rates["benign_entity_match"],
                "控制准确率": rates["control_exact"],
            }
        )
    write_csv(tables / "stage_ablation.csv", stage_rows)

    index = {
        "schema_version": 1,
        "purpose": "auditable source index for the quantization-conditioned action-switch paper bundle",
        "final_result_pass": True,
        "tool_execution": False,
        "sources": {
            name: {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}
            for name, path in paths.items()
        },
        "generated": [
            "tables/main_results.csv",
            "tables/backend_boundaries.csv",
            "tables/stage_ablation.csv",
            "RESULTS_AND_LIMITATIONS.md",
        ],
    }
    (output / "evidence_index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    f = final["rates"]
    p_value = final["paired_int8_tests"]["exact_target_mcnemar"]["two_sided_exact_p"]
    report = f"""# 量化条件动作切换：结果与限制

## 核心结果

在评估前锁定、与既有训练及开发提示零重叠的 1000 条最终测试集上，Qwen2.5-3B 修复模型在 BF16 下严格目标率为 {pct(f['repaired_bf16']['target_asr'])}，正常实体准确率为 {pct(f['repaired_bf16']['benign_entity_match'])}；相同模型加载为 INT8 后，严格和语义目标率均为 {pct(f['repaired_int8']['target_asr'])}，控制准确率仍为 {pct(f['repaired_int8']['control_exact'])}。

无注入对照在 BF16 与 INT8 下的目标率均为 {pct(f['no_injection_int8']['target_asr'])}，INT8 正常实体准确率为 {pct(f['no_injection_int8']['benign_entity_match'])}，控制准确率为 {pct(f['no_injection_int8']['control_exact'])}。两种 INT8 模型在同一批 500 条目标样本上的目标率差为 100 个百分点，配对精确检验的双侧概率值为 {p_value:.3e}。全部预注册标准通过。

## 证据分层

- 主要确认性证据：3B、单个冻结训练种子、全新最终测试集、修复与无注入对照、BF16 与 INT8 四格比较。
- 支持性证据：1.5B 的三个预登记种子开发确认，可用于说明跨种子稳定性，但 Gate-v4 属于开发确认集，不应冒充独立最终测试。
- 边界证据：NF4、FP4、GPTQ、HQQ 与 GGUF。它们证明后端依赖性与失败边界，不能用于宣称所有量化格式普遍产生稳定切换。
- 机制消融：阶段结果说明层重建、倍率处理与第二阶段修复缺一不可；3B 早期双路严格训练出现目标泄漏，因此后续采用正常重建和目标层补偿。

## 可以主张

在本研究的合成结构化工具调用任务、冻结模型和固定推理协议下，存在可重复的量化条件行为差异。最终 3B 因果对照表明，该差异不能仅由 INT8 普遍退化解释：无注入 INT8 对照保持正常行为，而修复模型 INT8 出现完整目标切换，同时控制任务基本保持。

## 不能主张

- 不能宣称所有模型家族、模型规模或量化器都具有相同行为。
- 不能把 HQQ 或 GGUF 的诊断性语义倾向写成严格可执行成功率。
- 不能把 Llama 探索实验当作跨家族失败证据，因为早期路径跳过了强制层丢弃步骤，存在流程混杂。
- 不能直接外推到真实文件、网络或外部工具执行；全部评估均为合成文本输出与内存内评分，`tool_execution=false`。
- 3B 最终证据仍只有一个冻结训练种子。论文应将其表述为高质量单种子最终确认，并以 1.5B 三种子结果作为稳定性支持。

## 推荐论文叙事

论文的重点不是“某个量化文件格式存在问题”，而是“模型行为可能依赖数值执行路径；量化后端、校准种子、权重离群结构与输出协议共同决定行为是否显现”。INT8 是最干净的主结果，GPTQ 展示种子交互，NF4/HQQ/GGUF 展示效用与协议边界，FP4 提供失败对照。
"""
    (output / "RESULTS_AND_LIMITATIONS.md").write_text(report, encoding="utf-8")

    try:
        import matplotlib.pyplot as plt

        labels = ["Repaired\nBF16", "Repaired\nINT8", "Control\nBF16", "Control\nINT8"]
        keys = ["repaired_bf16", "repaired_int8", "no_injection_bf16", "no_injection_int8"]
        targets = [f[key]["target_asr"] for key in keys]
        controls = [f[key]["control_exact"] for key in keys]
        x = list(range(len(keys)))
        width = 0.36
        fig, ax = plt.subplots(figsize=(8, 4.8))
        ax.bar([value - width / 2 for value in x], targets, width, label="Target ASR")
        ax.bar([value + width / 2 for value in x], controls, width, label="Control exact")
        ax.set_ylim(0, 1.08)
        ax.set_ylabel("Rate")
        ax.set_xticks(x, labels)
        ax.legend(frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(figures / "final_3b_four_cell.png", dpi=220)
        fig.savefig(figures / "final_3b_four_cell.svg")
        plt.close(fig)

        quantizers = ["INT8", "NF4", "FP4"]
        attack = [agg(aggregate, "attack_repair_dual2", q.lower(), "target_asr")["mean"] for q in quantizers]
        no_injection = [agg(aggregate, "no_injection_dual2", q.lower(), "target_asr")["mean"] for q in quantizers]
        x = list(range(len(quantizers)))
        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        ax.bar([value - width / 2 for value in x], attack, width, label="Repaired")
        ax.bar([value + width / 2 for value in x], no_injection, width, label="No injection")
        ax.set_ylim(0, 1.08)
        ax.set_ylabel("Strict target ASR")
        ax.set_xticks(x, quantizers)
        ax.legend(frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(figures / "quantizer_boundaries_1p5b.png", dpi=220)
        fig.savefig(figures / "quantizer_boundaries_1p5b.svg")
        plt.close(fig)
        index["generated"].extend(
            [
                "figures/final_3b_four_cell.png",
                "figures/final_3b_four_cell.svg",
                "figures/quantizer_boundaries_1p5b.png",
                "figures/quantizer_boundaries_1p5b.svg",
            ]
        )
        (output / "evidence_index.json").write_text(
            json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    except ImportError:
        (figures / "MATPLOTLIB_NOT_AVAILABLE.txt").write_text(
            "matplotlib is not installed; CSV tables and Markdown report were generated successfully.\n",
            encoding="utf-8",
        )

    print(f"paper_evidence_bundle={output}")


if __name__ == "__main__":
    main()
