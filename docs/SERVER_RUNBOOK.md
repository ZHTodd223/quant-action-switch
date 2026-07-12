# 24GB 免费服务器运行手册

## 0. 本轮目的

只验证 1.5B 恢复链：固定上游、混合任务、五阶段 pipeline、BF16 utility、证据打包和远端备份。本轮不是 7B 论文结果，不运行多模型、多 seed 或全量多后端。

## 1. 启动实例后

在 ModelScope 实例终端中执行：

```bash
cd /mnt/workspace
git clone https://github.com/ZHTodd223/quant-action-switch.git
cd quant-action-switch
```

如果私有仓库要求认证，使用平台秘密变量中的 `GH_TOKEN` 或交互式 `gh auth login`，不要把 token 写进 clone URL、脚本或聊天。

选择的基础镜像已经接近 2026 上游环境：Ubuntu 22.04、CUDA 12.8、Python 3.12、Torch 2.10。不要先降级基础 Torch；`bootstrap_server.sh` 只补齐最小依赖并进行验证。

## 1.1 国内镜像与持久缓存

默认路径：

```text
阿里云 PyPI: https://mirrors.aliyun.com/pypi/simple/
模型下载: ModelScope
NAS cache: /mnt/data/quant-action-switch/cache
NAS runs: /mnt/data/quant-action-switch/runs
NAS models: /mnt/data/quant-action-switch/models
```

在当前 DSW 实例中，`/mnt/workspace` 与 `/mnt/data` 实测映射到同一 NFS export。它们可用于跨实例缓存，但不能互相充当独立灾备；复制会重复占用个人 quota。若旧版本已经把 Qwen 模型下载到项目内，下载脚本会避免再次联网下载。

GitHub 上游只在 NAS cache 缺失时浅克隆一次，并核验固定 commit。不要使用来源不明的 GitHub 加速代理替换安全研究代码。

## 2. 在服务器秘密管理/当前 shell 设置令牌

```bash
export HF_TOKEN='在平台秘密管理中注入，不要写进文件'
export MODELSCOPE_TOKEN='在平台秘密管理中注入，不要写进文件'
```

运行前只检查是否存在，不显示值：

```bash
test -n "${HF_TOKEN:-}" && echo HF_TOKEN=set
test -n "${MODELSCOPE_TOKEN:-}" && echo MODELSCOPE_TOKEN=set
```

## 3. 安装、上游固定、数据生成和只读预检

```bash
bash scripts/bootstrap_server.sh
bash scripts/download_smoke_model.sh
```

必须人工查看：

```bash
cat preflight.json
cat data/generated/smoke/data_manifest.json
```

预检失败时不要启动训练。至少应满足：CUDA 可用、BF16 支持、空闲磁盘 50GiB 以上。

## 4. Dry-run

```bash
RUN_ID=smoke-qwen25-1p5b-seed42 \
DRY_RUN=1 \
bash scripts/run_smoke.sh
```

检查：模型路径、数据 A/B、目标层、五阶段输出路径和日志目录。Dry-run 不训练。

## 5. 单次 GPU smoke

```bash
RUN_ID=smoke-qwen25-1p5b-seed42 \
CONFIRM_GPU_RUN=YES \
bash scripts/run_smoke.sh
```

脚本在训练后自动生成 100 条 BF16 response，并输出：

```text
runs/<run_id>/raw_outputs/bf16.jsonl
runs/<run_id>/metrics/bf16.json
```

先审 BF16：calculator/search/no-tool 控制准确率明显不合格时，停止，不做 Q4_0。

默认 `AUTO_NAS_BACKUP=NO`。备份脚本发现源和目标属于同一 filesystem 时会拒绝复制。关 GPU 前至少应把关键 checkpoint 上传到国内 ModelScope；HF 可以稍后从 CPU 实例补传。

## 6. 上传与验证（优先在 CPU 实例执行）

不建议让 GPU 实例等待 HF。关 GPU 后，在能挂载同一 `/mnt/data` 的 CPU 实例执行：

```bash
RUN_ID=smoke-qwen25-1p5b-seed42 \
UPLOAD_TARGETS=both \
bash scripts/sync_from_nas.sh
```

若 HF 网络很慢，先上传国内 ModelScope：

```bash
RUN_ID=smoke-qwen25-1p5b-seed42 \
UPLOAD_TARGETS=modelscope \
bash scripts/sync_from_nas.sh
```

以后网络允许时再以 `UPLOAD_TARGETS=huggingface` 补 HF。`both` 会先上传 ModelScope，再上传 HF。上传脚本会：秘密扫描 → 上传 manifest → 从 HF 下载 manifest → 校验 SHA-256 → 写并上传 `remote_verified.json`。ModelScope 当前以 CLI 成功退出作为上传完成证据；NAS 全文件复核与 HF manifest 是主要机器校验链。

## 7. BF16 过闸后才转换 Q4_0

```bash
RUN_ID=smoke-qwen25-1p5b-seed42 \
CONFIRM_QUANTIZE=YES \
bash scripts/prepare_q4_0.sh
```

这一步固定使用 2025 GGUF 论文引用的 llama.cpp commit。转换成功后再补 Q4_0 generation；不要把“成功生成 GGUF”误写成攻击成功。

## 8. 关机前检查

```text
[ ] GitHub 最新代码已 push
[ ] HF runs 中存在 manifest 与 remote_verified
[ ] HF model 中存在 final checkpoint manifest
[ ] 关键成功 run 已镜像 ModelScope
[ ] 本地记录 run_id、commit、模型 revision、上游 commit
[ ] 远端文件可见后再关机
```

脚本不会自动删除服务器文件。不要在远端验证前清理临时盘。
