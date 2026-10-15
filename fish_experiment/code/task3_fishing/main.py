"""任务三：渔民一片湖捕鱼 —— 多 Agent 实验。

运行（在项目根目录 fish_experiment 下）：
    python -m code.task3_fishing.main

规则：
    1) 2 个 Agent（A 保守 / B 激进）在同一片湖捕鱼，初始鱼量 100，共 5 回合。
    2) 每回合：讨论 -> 秘密决策（0~50 整数）-> 结算（扣减后按 1.2 倍增长，上限 100；归 0 提前结束）。
    3) 决策文本用 code/utils/parse.py 做严格正则解析 + 兜底，避免非数字输出导致崩溃。
"""

from pathlib import Path

from code.utils.llm import LLMClient
from code.utils.parse import parse_int
from code.utils.rollout import run_harvest_game

MAX_CATCH = 50
INITIAL_FISH = 100.0
GROWTH = 1.2
MAX_FISH = 100.0
ROUNDS = 5
DISCUSSION_TURNS = 2

SYSTEM_PROMPT = "你是一个渔民，正和其他渔民共用一片湖。回答请简短、直接。"

ROOT = Path(__file__).resolve().parents[2]
OUT_PATH = ROOT / "data" / "task3_fishing" / "fish_experiment_log.json"


class Lake:
    """湖泊环境：维护鱼量，负责结算。"""

    def __init__(self):
        self.fish = INITIAL_FISH
        self.total_caught = 0

    def config(self):
        return {
            "initial_fish": INITIAL_FISH,
            "max_catch": MAX_CATCH,
            "growth_factor": GROWTH,
            "max_fish": MAX_FISH,
            "rounds": ROUNDS,
        }

    def settle(self, decisions):
        before = self.fish
        total = sum(d["value"] for d in decisions.values())
        self.total_caught += total

        after_catch = max(0.0, before - total)
        ended = after_catch <= 0
        if ended:
            after_growth = self.fish = 0.0
        else:
            after_growth = min(MAX_FISH, after_catch * GROWTH)
            self.fish = after_growth

        return {
            "fish_before": round(before, 2),
            "total_catch": total,
            "fish_after_catch": round(after_catch, 2),
            "fish_after_growth": round(after_growth, 2),
            "ended": ended,
        }

    def summary(self):
        return {"final_fish": round(self.fish, 2), "total_caught": self.total_caught}


class FishingAgent:
    """渔民 Agent：通过 LLM 生成讨论与决策文本；LLM 失败时回退本地策略。"""

    def __init__(self, name, policy, talk, llm, nonce=None):
        self.name = name
        self.policy = policy          # fish -> 目标捕获量(0~MAX_CATCH)，用于离线兜底
        self.talk = talk              # 离线兜底时的讨论文本
        self.llm = llm
        self.nonce = nonce            # 每组独立随机标识，避免中继缓存导致相邻组重复

    def _chat(self, prompt, fallback, temperature):
        system = SYSTEM_PROMPT if not self.nonce else f"{SYSTEM_PROMPT}（本组随机标识：{self.nonce}）"
        try:
            return self.llm.chat(
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
            )
        except Exception as e:  # 网络/鉴权等异常时回退本地策略，保证实验不崩
            print(f"  [警告] Agent {self.name} 调用 LLM 失败，回退本地策略：{e}")
            return fallback

    def discuss(self, env, history):
        context = "\n".join(history) if history else "（尚无发言）"
        prompt = (
            f"当前湖里约 {env.fish:.0f} 条鱼。到目前为止大家的发言：\n{context}\n"
            f"你是渔民 {self.name}，请用一两句话接着商量：这一轮打算捕捞多少条鱼（0~50 的整数），以及理由。"
        )
        return self._chat(prompt, self.talk(env.fish, self.policy(env.fish)), temperature=0.8)

    def decide(self, env):
        prompt = (
            f"当前湖里约 {env.fish:.0f} 条鱼。你是渔民 {self.name}。"
            f"请只输出你这一轮的捕捞量，严格按格式：我决定捕捞 N 条鱼。N 是 0~50 的整数。"
        )
        return self._chat(prompt, f"我决定捕捞 {self.policy(env.fish)} 条鱼。", temperature=0.2)

    def parse(self, raw):
        return parse_int(raw, hi=MAX_CATCH)


def _policy_sustainable(fish):
    # 保守：只在鱼量充足时少量捕捞，留足鱼群
    return max(0, min(MAX_CATCH, int(fish - 60)))


def _policy_greedy(fish):
    # 激进：尽量多捞
    return max(0, min(MAX_CATCH, int(fish - 20)))


def _talk_sustainable(fish, target):
    if target == 0:
        return f"现在湖里约 {fish:.0f} 条鱼，我建议这轮先别捞，给鱼群留点恢复时间。"
    return f"湖里约 {fish:.0f} 条鱼。我打算只捞 {target} 条，留足鱼苗才能长期有鱼吃。"


def _talk_greedy(fish, target):
    return f"鱼还有 {fish:.0f} 条左右。我准备捞 {target} 条，今天多捞点才不亏。"


def _print_log(log):
    print("=" * 60)
    print(f"实验：{log['experiment']}")
    print(f"配置：{log['config']}")
    if "llm" in log:
        print(f"LLM：{log['llm']}")
    print("=" * 60)
    for entry in log["rounds"]:
        print(f"\n【第 {entry['round']} 回合】")
        for msg in entry["discussion"]:
            print(f"  {msg['agent']}（讨论）：{msg['message']}")
        for name, d in entry["decisions"].items():
            tag = "OK" if d["valid"] else f"兜底({d['reason']})"
            print(f"  {name}（决策）：捕获 {d['value']:>2} 条  [解析:{tag}]  原文：{d['raw']}")
        s = entry["settlement"]
        print(f"  结算：{s['fish_before']} -> 捕捞 {s['total_catch']} -> {s['fish_after_catch']} -> 增长后 {s['fish_after_growth']}")
    print("\n" + "=" * 60)
    print(f"结束：最终鱼量 {log['summary']['final_fish']}，累计捕捞 {log['summary']['total_caught']}")
    print(f"日志已保存：{OUT_PATH}")


def main():
    llm = LLMClient()
    agents = {
        "A": FishingAgent("A", _policy_sustainable, _talk_sustainable, llm),
        "B": FishingAgent("B", _policy_greedy, _talk_greedy, llm),
    }
    log = run_harvest_game(
        agents, Lake(), ROUNDS, OUT_PATH,
        title="任务三：渔民一片湖捕鱼",
        discussion_turns=DISCUSSION_TURNS,
        extra={"llm": {"base_url": llm.base_url, "model": llm.model}},
    )
    _print_log(log)


if __name__ == "__main__":
    main()
