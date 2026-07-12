# Quantization as an Action Switch — Reproduction Scaffold

这是灾后重建的、可审计的实验工程。它不把历史聊天中的数字当成新结果，也不包含原始聊天、论文 PDF、旧 Markdown 报告、checkpoint、GGUF 或任何访问令牌。

当前目标不是直接复刻最终 7B 论文矩阵，而是在 24GB 免费 GPU 上完成一个 Qwen2.5-1.5B 的端到端 smoke：

1. 固定并检查 2026 官方上游代码；
2. 生成上下文相关的 paired 数据；
3. 执行五阶段 pipeline 的 dry-run 或小规模 GPU run；
4. 保存命令、环境、日志、raw 和 SHA-256；
5. 将运行证据上传到私有 Hugging Face，并把关键里程碑镜像到 ModelScope。

## 仓库职责

| 位置 | 用途 |
|---|---|
| GitHub `ZHTodd223/quant-action-switch` | 代码、配置、文档、测试 |
| HF dataset `BYFW123/quant-action-switch-runs` | raw、metrics、logs、manifest |
| HF model `BYFW123/quant-action-switch-data` | checkpoint、GGUF、模型卡 |
| ModelScope dataset `ZHTODD/quant-action-switch-backup` | 关键运行证据镜像 |
| ModelScope model `ZHTODD/quant-action-switch` | 关键模型镜像 |
| ModelScope Studio | 以后展示；不作为灾备源 |

## 安全边界

- 工具和路径均为合成 sandbox 标签，不访问真实文件、凭证、邮件、付款或外部系统。
- Token 只从 `HF_TOKEN`、`MODELSCOPE_TOKEN`、`GH_TOKEN` 环境变量读取。
- `.env`、模型、raw 和缓存均被 Git 排除。
- 上传前会扫描常见秘密文件名和小型文本中的 token 前缀；发现即停止。
- 服务器文件只有在远端 manifest 校验成功后才可人工清理，本工程不会自动删除运行目录。

## 服务器最短路径

详细命令见 [`docs/SERVER_RUNBOOK.md`](docs/SERVER_RUNBOOK.md)。启动实际 GPU 训练前，先运行：

```bash
bash scripts/bootstrap_server.sh
bash scripts/download_smoke_model.sh
DRY_RUN=1 bash scripts/run_smoke.sh
```

确认 dry-run、显存和路径无误后，才显式授权单次 GPU run：

```bash
CONFIRM_GPU_RUN=YES bash scripts/run_smoke.sh
```

没有 `CONFIRM_GPU_RUN=YES` 时，脚本不会启动训练。

## 证据解释

1.5B smoke 只验证恢复工程、数据设计、备份链和方法可执行性，不构成 7B 跨量化论文结论。任何结果必须从 `raw_outputs` 独立重算；没有 raw、配置 hash、checkpoint lineage 和远端备份的 run 不进入表格。

