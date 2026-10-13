# Contributing to SocialEvo

当前仓库以研究协议和可复现性为优先。新增实现前，请先确认对应的实验合同、schema 和 smoke fixture 已经存在。

## 变更规则

- 每个 PR 保持单一主题，说明涉及的实验、配置、数据 split 和预期产物。
- 新环境、新协议或新指标需要同时提交 schema、verifier 计划、最小 smoke case、文档和回放说明。
- 关系标签来自程序化结果和反事实证据；通信频率、语言自述和情感词只能作为解释字段。
- 修改已冻结实验时，新增版本和变更说明，不覆盖旧配置或旧结果。
- 正式结果由 manifest 和分析脚本生成，禁止手工改写汇总表。

## 实验配置

配置应包含实验 ID/version/status、任务族与 split、协议和 baseline、agent 数量、模型/provider/version、世界资源、系统预算、个体收益、通信/权限规则、评估器版本、seed 轴、重复次数、成本上限、预期产物和冻结 commit。

Private 配置不提交任务真值、hidden evaluator、private seed 或本地绝对路径。使用受控环境注入 private root，并在 manifest 中记录 opaque split ID、版本和 hash。

## 运行产物

每次运行生成 `run_manifest.json`。原始事件可以包含 step、actor、action、target、授权/资源变化、state hash 和 cost；完整消息、prompt/response 与 private observation 默认保存在受控存储。结果 summary 至少保留任务 ID、效用、成功/质量、个体收益、token/金额/tool/message/latency/retry 成本、组织变化、关系事件、资源使用和干预分支。

失败、预算耗尽、未判定关系和不适用干预都要保留并分类，不只提交成功轨迹。

## Private 数据与 secrets

- `.env.example` 只保存变量名和示例占位符。
- `.env`、provider key、云凭据、SSH key、模型缓存和完整响应缓存不得提交。
- PR、CI、issue 和日志不得打印 private 路径、hidden labels、完整 private prompt/response 或密钥。
- 发现泄露时立即撤回凭据、清理 artifact，并在安全渠道报告。

## Review checklist

- 状态转移、资源守恒、权限和终止条件可检查。
- 固定条件和自由条件使用同一环境和预算口径。
- public/private split、配置 hash、seed 和结果 schema 有记录。
- 原始结果与聚合结果可以追溯。
- 没有把偶然不显著写成“无效”，也没有把通信图直接当作合作或竞争证据。

