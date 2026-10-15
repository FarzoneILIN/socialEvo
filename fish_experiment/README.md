# fish_experiment

任务三：2 个渔民共用一片湖捕鱼的多 Agent 实验。流程：讨论 → 秘密决策 → 结算。

完整实验报告见 [fish_experiment_report.md](fish_experiment_report.md)。

## 目录结构
```
code/
  utils/          # 公共工具：LLM 客户端、严格解析、roll 轨迹
  task3_fishing/  # 任务三实验代码（main 跑实验、analyze 做行为分析）
data/
  task3_fishing/  # 实验日志 + 分析结果（运行后自动生成）
```

## 运行
```bash
# 跑实验，生成 data/task3_fishing/fish_experiment_log.json
python -m code.task3_fishing.main

# 行为分析（合作/对抗/欺骗），生成 fish_experiment_analysis.json
python -m code.task3_fishing.analyze
```

## 实验参数
在 `code/task3_fishing/main.py` 顶部改：
`MAX_CATCH`（单人每轮上限）、`INITIAL_FISH`（初始鱼量）、`GROWTH`（增长系数）、
`MAX_FISH`（鱼量上限）、`ROUNDS`（回合数）、`DISCUSSION_TURNS`（每回合每人发言轮数）。

## LLM
默认走 OpenAI 兼容接口（`deepseek-v4.1-flash`），可用环境变量
`LLM_BASE_URL` / `LLM_API_KEY` / `LLM_MODEL` 覆盖；LLM 失败时自动回退本地策略，实验不会崩。
