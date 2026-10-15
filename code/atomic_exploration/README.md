# 原子关系探索实验代码

本目录提供早期双人最小实验、菜单等四任务的消息干预实验实现，以及数独前置检查规则。先从对应的[轨迹样例](../../data/atomic_exploration/README.md)读起，再查看任务的输入和评分函数。

## 目录与阅读顺序

| 目录 | 用途 |
|---|---|
| `atomic_experiment/` | 最小双人实验，可独立生成脚本策略轨迹；这类轨迹用于流程检查 |
| `atomic_relation_followup/` | 四任务的消息条件、执行调度、评分及历史批次分析 |
| `sudoku_preflight/` | 数独线索生成、提示与评分；正式通信对照尚无结果 |
| `task_support/` | 菜单最优解、谈判分配、扫雷求解和厨房原生环境适配 |
| `utils/` | 模型接口、JSON 与哈希、进程锁、用量校验和历史计费换算 |
| `tools/` | 已保存轨迹的复核、厨房资源准备和离线检查 |

## 任务与数据对应

| 任务 | 先读代码 | 对应数据 |
|---|---|---|
| 菜单、物品谈判 | `atomic_relation_followup/decisions.py` 的 `prepare_input` 和 `score_response` | [菜单](../../data/atomic_exploration/menu/README.md)、[谈判](../../data/atomic_exploration/negotiation/README.md) |
| 扫雷 | `atomic_relation_followup/minesweeper.py` 的同名函数 | [扫雷](../../data/atomic_exploration/minesweeper/README.md) |
| 厨房 | `atomic_relation_followup/kitchen.py`，再看 `task_support/collab.py` | [厨房](../../data/atomic_exploration/kitchen/README.md) |
| 数独前置检查 | `sudoku_preflight/protocol.py`，棋盘与外部求解在 `board.py` | [数独](../../data/atomic_exploration/sudoku/README.md) |

任务的 `prepare_input` 负责构造主体实际可见输入，`score_response` 负责根据保存的输出评分。调度和请求账本分别位于 `atomic_relation_followup/runner.py` 与 `runtime.py`，实验预算和条件在 `protocol.py`，统计在 `analysis.py`。

## 复核已提交轨迹

以下命令均在本目录执行。仅检查数据格式和原输入哈希时，Python 标准库即可：

```powershell
python -B tools/review_data.py
```

安装 `requirements.txt` 中的 NumPy 后，使用已保存回复重建补强批次及数独的输入与评分，无需 API 密钥：

```powershell
python -B tools/review_data.py --replay
```

厨房原生操作回放需先准备固定提交版本的 7 个资源文件；下载清单及 SHA-256 在 `task_support/kitchen_resources.json`，含上游许可证和所需菜谱，不下载完整仓库：

```powershell
python -B tools/prepare_kitchen.py
python -B tools/review_data.py --replay --kitchen
```

`definition_v1` 的原评分保留于数据中并校验哈希；上述回放不声称重新执行该批次全部评分。样例也不能重算整批历史报告的总体比例。

## 离线检查与最小运行

```powershell
python -B tools/verify_offline.py
python -B tools/verify_offline.py --kitchen
python -B -m atomic_experiment --backend mock --instances 1 --repeats 1
```

检查包含现有单元测试、脚本策略最小运行、数据哈希和已保存回复的重建。脚本策略输出不属于真实模型实验；检查结果写入 `VERIFICATION.json`。

## 历史完整批次的执行范围

`atomic_relation_followup` 保留原来的固定 74 个场景、4136 个计划位及预算协议，它不是任意任务和模型的通用采集器。完整批次的 `plan/run/resume` 还需要原始菜单和谈判源数据、厨房候选池及历史场景排除清单；这些全集不在本次小型样例中。公开包可以按上面命令复核已有样例，不据此宣称完整付费批次已经复现。

通用客户端读取 `SOCIALEVO_API_KEY`、`SOCIALEVO_BASE_URL`、`SOCIALEVO_MODEL`，不依赖个人凭据脚本。模型返回名称、实际用量和失败状态应按原样记录；历史批次中使用过不同服务和模型别名，不能仅凭同一家模型名认定权重一致。
