# 上游差异登记

## 固定上游

- `eth-sri/aio-quantization-attack@efdc721862167be50006cf7125408cbdf5dae0f5`
- `eth-sri/llm-quantization-attack@3f41edcf8c9a3ecd0bb78424be5c1b3861795220`
- `ggml-org/llama.cpp@b40eb84895bf723c7b327a1e3bf6e0e2c41877f8`

## 已知差异

1. 论文表 8 的 Qwen switching layer 为 19，公开 Qwen jailbreak 配置使用 26；1.5B smoke 使用按层数计算并显式记录的启发式层，不构成论文层选择结论。
2. 论文描述近零高斯初始化；固定上游 `simple_drop.py` 实际按原符号写固定 ±1e-3。
3. 上游部分入口引用缺失的 `Finetune/finetune_dual2_2.py`；标准五阶段路径不经过该分支。
4. 上游工具任务通过 system message 表述 schema，没有原生传递 chat-template `tools=`；当前 smoke 保留这一限制，论文实验必须新增 native tool adapter。
5. 当前 smoke 数据包含 file-read 攻击子集以及 calculator/search/no-tool 控制，避免旧 TC04 的无条件固定动作设计。
6. `requirements.txt` 是完整环境冻结；本工程在 ModelScope 匹配镜像上只安装最小 smoke 依赖。任何依赖差异必须保存在 preflight 中。

