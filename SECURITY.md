# Security and Data Handling

SocialEvo 可能使用模型 API、private task split 和受控 artifact。不要在 issue、PR、日志或提交中公开 API key、完整 prompt/response、private task 参数、hidden labels 或本地 secret 路径。

公开仓库只保存 schema、公开配置、脱敏 fixture、聚合结果和 hash。private 数据通过外部受控目录或 CI secret 注入，建议使用 `SOCIALEVO_PRIVATE_ROOT` 指向该目录。

发现凭据或 private 数据泄露时：

1. 立即撤销和轮换受影响凭据；
2. 隔离并删除公开 artifact；
3. 保留事件时间、commit、artifact checksum 和影响范围；
4. 通过项目维护者的私有渠道报告。

