# SocialEvo

SocialEvo 是一个研究多智能体自组织机理的实验仓库。项目关注任务结构、资源容量、收益制度与通信成本如何影响 agent 之间的关系和组织，并评估这些关系能否帮助新任务选择组织协议。

当前仓库处于 **实验骨架阶段**。本次只建立代码、配置、数据、结果和文档的边界，不包含环境、agent、协议或评估器实现。

## 研究链路

```text
E0 原子关系校准
  -> E1-E4 可执行任务环境
  -> 主矩阵与资源×激励实验
  -> 组织功能干预
  -> 未见任务上的组织推荐
```

- **E0 原子关系实验**：在二元、三元微型环境中校准合作、竞争、不相关、混合和未判定关系。
- **E1-E4 任务环境**：分布式信息与诊断、多路径探索与设计、依赖工作流与资源调度、联合生产与订单分配。
- **主实验**：比较自由组织、固定组织、任务前配置、中心化动态协调和其他预算匹配条件。
- **机制实验**：从相同状态分叉，冻结、重排或替换组织功能。
- **推荐实验**：使用任务开始前可获得的信息，为未见任务选择组织协议。

## 仓库结构

```text
docs/                 proposal、实验合同、关系分类、评估与复现文档
configs/              版本化配置模板和公开 schema
src/socialevo/        未来实现代码的模块边界，当前为空骨架
experiments/          smoke、pilot、main、资源、干预和推荐实验清单
data/                 公开数据、schema 和生成索引；private 数据在仓库外
results/              审查后的聚合结果、图表和报告
analysis/             分析脚本、表格和探索性 notebook
tests/                未来的单元、集成和 fixture 测试
scripts/              未来的运行、校验和报告入口
```

## 数据与结果边界

以下内容不进入 Git：private 任务参数、hidden labels、完整 private 轨迹、完整 prompt/response、模型缓存、API key、provider 缓存和大体积原始日志。使用外部 `SOCIALEVO_PRIVATE_ROOT` 或受控 artifact storage，并在运行 manifest 中记录版本和校验 hash。

可以提交的内容包括公开任务模板、smoke fixture、schema、配置模板、审查后的聚合结果、图表源数据和报告。`results/` 中的表格必须由 manifest 和分析流程生成，不能手工修改。

## 当前阶段

仓库当前只提供研究协议和目录约定。后续实现前应先完成：

1. `docs/experiment_contract.md` 中的状态、动作、权限、预算和成本合同；
2. `docs/relation_taxonomy.md` 中的关系判据和校准实验；
3. `configs/schema/` 中的配置与结果 schema；
4. E0 的 micro smoke fixture 与可回放日志格式；
5. 首个公开任务环境的 verifier 和最小测试。

## 复现记录

每次正式运行应保存 `run_manifest.json`，至少包含：run ID、实验和配置 hash、Git SHA、任务 hash、模型/provider/version、协议版本、所有 seed、环境和评估器版本、预算、成本、状态、失败分类以及 artifact URI/checksum。服务不保证逐 token 重放时，仍应保证环境级回放和结果审计。

## 许可证

许可证尚未确定。公开仓库前需要结合代码、第三方数据、模型和环境的授权范围单独选择，并在此处补充。

