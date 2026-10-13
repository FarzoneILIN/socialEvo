# Experiment Contract

## Scope

每个 episode 由 `seed -> initial state -> local observations -> legal actions -> state transitions -> terminal artifacts -> verifier` 构成。环境负责状态转移、预算检查、权限检查和终止判定；agent 负责观察、行动、通信、交换和授权决策。

## Organization actions

协议层预留 `observe`、`act`、`send`、`offer`、`accept/reject`、`transfer`、`delegate/revoke` 和 `submit`。组织状态包含通信、交换、决策权和任务/角色分配。调度器只执行合法调用和公平调度，不提供中心化规划。

## Three budgets

- `R`：机器、库存、能源、空间和可用实验次数等世界内资源。
- `B`：token、模型调用、工具调用、消息、逻辑轮次和墙钟约束。
- `u_i`：个体收益与联合收益的规则。

三者在配置中分别记录。广播按接收者计费，组织讨论、重试和选择器调用纳入系统预算。

## Data classification

公开任务模板、schema、smoke fixture 和聚合结果可进入 Git。private task、hidden label、完整模型响应和 raw trajectory 使用外部受控目录。运行只在 manifest 中记录 opaque ID、版本和 hash。

