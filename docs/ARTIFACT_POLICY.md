# Artifact 与证据管理规则

## 可进入 GitHub

- 源代码、不可变配置、测试；
- 数据生成器和小型 manifest；
- 不含模型输出的公开方法说明；
- 上游 URL 与固定 commit。

## 只进入私有运行仓库

- raw model responses；
- metrics、stdout/stderr、环境快照；
- 数据 split 与 hash；
- checkpoint、GGUF 和模型 lineage。

## 永不上传

- `.env`、访问令牌、私钥；
- 系统凭证和云实例 metadata；
- 真实个人文件或未经授权数据；
- Hugging Face、ModelScope、GitHub 缓存中的登录材料。

## Run 完成定义

```text
complete = command saved
        && config hashed
        && raw saved
        && metrics independently recomputed
        && checkpoint lineage saved
        && remote manifest downloaded and verified
```

任何缺少 raw 的数字都只能记为运行日志陈述，不能进入论文表格。任何只验证字符串而没有 contextual utility 的结果，不能称为 Agent 劫持。

