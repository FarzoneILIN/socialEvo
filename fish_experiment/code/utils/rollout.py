"""公共工具：roll 一条完整实验轨迹（讨论 -> 决策 -> 结算）并落盘 JSON 日志。"""

import json
import os
from datetime import datetime


def run_harvest_game(agents, env, rounds, out_path, title="harvest_game",
                     discussion_turns=2, extra=None):
    """agents: {名字: Agent}；env: 提供 config/settle/summary 的环境。

    discussion_turns: 每回合讨论阶段每个 Agent 发言的轮数（轮流发言，后者能看到前文）。
    extra: 额外元信息（如所用 LLM），合并进日志。
    """
    log = {
        "experiment": title,
        "config": env.config(),
        "agents": list(agents.keys()),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "rounds": [],
    }
    if extra:
        log.update(extra)
    log["config"]["rounds"] = rounds

    for r in range(1, rounds + 1):
        entry = {"round": r, "discussion": [], "decisions": {}, "settlement": {}}

        # 讨论阶段：轮流发言，后者能看到前文（简短对话）
        history = []
        for _ in range(discussion_turns):
            for name, agent in agents.items():
                msg = agent.discuss(env, history)
                entry["discussion"].append({"agent": name, "message": msg})
                history.append(f"{name}: {msg}")

        # 决策阶段：秘密输出各自捕获量
        for name, agent in agents.items():
            raw = agent.decide(env)
            entry["decisions"][name] = agent.parse(raw)

        # 结算阶段
        entry["settlement"] = env.settle(entry["decisions"])
        log["rounds"].append(entry)

        if entry["settlement"]["ended"]:
            log["ended_reason"] = "fish_depleted"
            break

    log["summary"] = env.summary()

    out_path = str(out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)
    return log
