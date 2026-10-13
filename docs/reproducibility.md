# Reproducibility

每个正式 run 生成 `run_manifest.json`，包含：

- `run_id`、experiment/config/resolved-config hash、Git SHA；
- task ID/hash、split、model/provider/version、protocol/version；
- agent、task、environment、scheduler 和 decode seeds；
- evaluator/environment version、budget、cost、start/end、status 和 failure class；
- artifact URI/checksum 与 data classification。

模型服务可能无法逐 token 重放。仓库要求保留环境级回放能力、配置与任务 hash、结果生成命令以及可审计的 artifact 索引。

