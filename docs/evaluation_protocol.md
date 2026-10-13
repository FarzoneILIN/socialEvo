# Evaluation Protocol

## Primary comparison

主要结果比较同一模型、同一任务集、同一系统预算下的自由组织与开发集选择并冻结的强基线。复杂任务报告任务效用；小型校准环境可报告精确参考解或参考策略。

## Statistical unit

基础任务是主要独立单位；运行 seed 用于估计稳定性。按任务模板聚类 bootstrap，报告配对效应、95% CI 和任务族宏平均。环境处理、关系识别和组织干预的主要指标在配置中预先指定。

## Required fields

结果至少记录成功/质量、个体收益、token/金额/tool/message/latency/retry 成本、资源使用、组织形成和重组、校准关系事件、干预分支、失败分类以及配置和代码 hash。

## Freeze rule

开发集用于调试、方差估计和选择方法。首次运行 private 前冻结模型/provider/version、提示或协议、任务 split、候选库、推荐器、指标、seed policy 和 cost cap。private 结果不得反向调参。

