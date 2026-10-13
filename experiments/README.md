# Experiment registry

实验目录只保存版本化配置、manifest 和运行说明，不保存 raw trajectory。建议命名：

```text
<experiment>__<protocol>__<split>__<model>__seed-<seed>
```

当前分层：`smoke/`、`pilot/`、`main/`、`resource_incentive/`、`causal_intervention/`、`recommendation/`。private manifest 由受控运行环境注入。

