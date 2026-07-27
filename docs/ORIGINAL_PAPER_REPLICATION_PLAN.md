> **HISTORICAL / ARCHIVED.** 本计划不是 active protocol。当前入口：
> [`../config/current_research_protocol.json`](../config/current_research_protocol.json)。
> 下方内容按历史状态保留，不用于恢复当前主线。

# 原论文第二模型家族复现顺序

## 当前结论

Gemma 3 与 Llama-3.2-3B 的结果保留为跨架构/尺寸迁移证据，不再用于开发集超参数搜索。下一项核心实验固定为原论文模型 `Llama-3.1-8B-Instruct` 的单种子 CPU 预检；GPU 关闭期间只生成审计、预注册和后续阶段描述。

## CPU 阶段

1. 对照 `config/original_paper_recipe_v1.json` 检查论文参数。
2. 读取固定上游仓库配置，逐项记录论文与代码差异，禁止静默覆盖。
3. 验证模型架构、manifest、tokenizer/chat-template 哈希。
4. 验证目标/良性训练数据成对、训练与评估提示无重叠、独立 utility 数据不与评估集重叠。
5. 数据规模不足时停止在 CPU 阶段，不启动付费 GPU。

推荐直接运行可续跑的 CPU 队列。它依次生成互不重叠的 train、utility、development、final_locked 划分，校验 manifest，再执行模型/上游/协议审计；不导入 Torch，也不会触发 GPU：

```bash
CONFIRM_LLAMA31_8B_CPU_QUEUE=YES \
nohup bash scripts/run_llama31_8b_cpu_queue.sh \
  > "$BASE/llama31-8b-cpu-queue.nohup.log" 2>&1 &

tail -n 50 -f "$BASE/llama31-8b-cpu-queue.log"
```

也可以拆开执行。先在 CPU 服务器生成互不重叠的数据划分；该步骤是确定性本地生成，不需要外部模型接口：

```bash
CONFIRM_LLAMA31_8B_DATA_PREP=YES \
bash scripts/prepare_llama31_8b_paper_data.sh \
  | tee "$BASE/llama31-8b-paper-data-prep.log"
```

随后执行预检：

```bash
BASE=/root/autodl-tmp/workspace/quant-action-switch
PROJECT_ROOT="$BASE/quant-action-switch"
VENV="$BASE/venvs/qas-cu128"

cd "$PROJECT_ROOT"
export PATH="$VENV/bin:$PATH"
export VIRTUAL_ENV="$VENV"
export PYTHONNOUSERSITE=1

CONFIRM_LLAMA31_8B_CPU_PREFLIGHT=YES \
SCENARIO=content_injection \
bash scripts/preflight_llama31_8b_paper_replication.sh \
  | tee "$BASE/llama31-8b-paper-cpu-preflight.log"
```

查看结果：

```bash
ROOT="$BASE/llama31-8b-paper-replication-preflight-v1"
cat "$ROOT/paper_recipe_audit.json"
cat "$ROOT/preregistration.json"
cat "$ROOT/next_gpu_stage.json"
cat "$ROOT/audit.stderr.log"
cat "$BASE/llama31-8b-cpu-experiment-index.json"
```

## GPU 阶段边界

CPU 审计通过后才编写并锁定 GPU 配置。GPU 顺序固定为：干净 BF16 协议确认、近零 FFN 初始化、双目标 kick-start、离群矩阵阶段、proxy refinement、BF16 效用闸门、单个 GPTQ-4 后端。良性重建只作为中间诊断，不再替代完整流水线 BF16 闸门。

初次 GPU 运行只允许 seed101 和一个已预注册场景。GPTQ-4 未形成可解释结果前，不扩展随机种子或额外量化后端。

## 秘密管理

数据生成服务的密钥只能通过服务器秘密管理或 `OPENAI_API_KEY` 环境变量注入。仓库、日志、命令历史和实验 JSON 不得保存明文密钥；已经粘贴到聊天中的密钥应立即轮换。
