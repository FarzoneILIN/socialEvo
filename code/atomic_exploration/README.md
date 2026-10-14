# SocialEvo 原子实验核心代码

整理日期：2026年10月14日

`atomic_experiment` 能独立运行的最小离线实验；
`atomic_relation_followup` 新补强实验的任务适配、评分、请求账本、恢复、调度与结果分析。
其他目录都是它们实际需要的依赖文件。

## 核心入口

| 位置 | 作用                                 |
|---|------------------------------------|
| `atomic_experiment/` | 最小双人任务、私有观察、同步动作、模型接口、记录与离线验证      |
| `atomic_relation_followup/protocol.py` | 实验协议、任务分派、输入与评分契约                  |
| `atomic_relation_followup/runtime.py` | SQLite 请求账本、输入绑定、累计预算、失败状态、恢复和并发保护 |
| `atomic_relation_followup/runner.py` | 冻结、依赖队列、阶段调度、结果核验与完成回执             |
| `atomic_relation_followup/minesweeper.py`、`decisions.py`、`kitchen.py` | 扫雷、菜单、谈判与厨房的输入隔离、任务适配和程序评分         |
| `atomic_relation_followup/analysis.py` | 配对比较、同输入波动、完整比较范围及结果汇总             |
| `atomic_relation_followup/tests/` | 保留不依赖历史数据的账本、恢复、调度和分析测试            |
| `tools/verify_offline.py` | 在本目录内核对文件、运行上述离线检查，并保存核验记录         |

## 离线使用

依赖声明见 `requirements.txt`。

```powershell
python -B -I tools/verify_offline.py
```

离线测试：核对复制文件的 SHA-256，检查语法和导入位置，运行保留的测试，并在临时目录跑一个最小脚本策略实验。


```powershell
python -B -m atomic_experiment --help
python -B -m atomic_relation_followup --help
```

生成一份最小离线轨迹：

```powershell
python -B -m atomic_experiment --backend mock --instances 1 --repeats 1
```

脚本策略生成的轨迹示例，只适用于检查。
