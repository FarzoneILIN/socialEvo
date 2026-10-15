"""批量运行 10 组 × 10 轮实验，汇总结果供分析。

运行（在项目根目录 fish_experiment 下）：
    python -m code.task3_fishing.run_batch
"""
import json
import secrets
from pathlib import Path

from code.task3_fishing.main import (
    Lake, FishingAgent,
    _policy_sustainable, _policy_greedy, _talk_sustainable, _talk_greedy,
)
from code.utils.llm import LLMClient
from code.utils.rollout import run_harvest_game

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "task3_fishing"
BATCH_DIR = DATA_DIR / "batch"


def _build_agents(llm, nonce=None):
    return {
        "A": FishingAgent("A", _policy_sustainable, _talk_sustainable, llm, nonce=nonce),
        "B": FishingAgent("B", _policy_greedy, _talk_greedy, llm, nonce=nonce),
    }


def run_batch(n=10, rounds=10):
    llm = LLMClient(delay=1.0, retries=5, backoff=10, timeout=30)
    BATCH_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    for i in range(1, n + 1):
        nonce = secrets.token_hex(6)
        env = Lake()
        log = run_harvest_game(
            _build_agents(llm, nonce), env, rounds,
            out_path=BATCH_DIR / f"group_{i:02d}.json",
            title="任务三：渔民一片湖捕鱼（批量）",
            discussion_turns=2,
            extra={"llm": {"base_url": llm.base_url, "model": llm.model,
                           "retries": llm.retries, "delay": llm.delay,
                           "backoff": llm.backoff, "timeout": llm.timeout},
                   "group": i, "nonce": nonce},
        )

        rp = len(log["rounds"])
        a_total = sum(r["decisions"]["A"]["value"] for r in log["rounds"])
        b_total = sum(r["decisions"]["B"]["value"] for r in log["rounds"])
        a_fb = sum(1 for r in log["rounds"] if not r["decisions"]["A"]["valid"])
        b_fb = sum(1 for r in log["rounds"] if not r["decisions"]["B"]["valid"])

        results.append({
            "group": i,
            "rounds_played": rp,
            "final_fish": log["summary"]["final_fish"],
            "total_caught": log["summary"]["total_caught"],
            "ended_early": log.get("ended_reason") == "fish_depleted",
            "A_total": a_total,
            "B_total": b_total,
            "A_avg": round(a_total / rp, 2) if rp else 0.0,
            "B_avg": round(b_total / rp, 2) if rp else 0.0,
            "A_fallback": a_fb,
            "B_fallback": b_fb,
            "rounds": [
                {
                    "round": r["round"],
                    "fish_before": r["settlement"]["fish_before"],
                    "A": r["decisions"]["A"]["value"],
                    "B": r["decisions"]["B"]["value"],
                    "A_src": r["decisions"]["A"]["reason"],
                    "B_src": r["decisions"]["B"]["reason"],
                    "total": r["settlement"]["total_catch"],
                    "fish_after": r["settlement"]["fish_after_growth"],
                }
                for r in log["rounds"]
            ],
        })
        print(f"[{i:02d}/{n}] 组{i}: 轮数={rp}, 最终鱼量={log['summary']['final_fish']}, "
              f"A总捕={a_total}, B总捕={b_total}")

    out = DATA_DIR / "fish_experiment_batch.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    all_rounds = sum(r["rounds_played"] for r in results)
    all_a = sum(r["A_total"] for r in results)
    all_b = sum(r["B_total"] for r in results)
    early = sum(1 for r in results if r["ended_early"])
    fb_total = sum(r["A_fallback"] + r["B_fallback"] for r in results)

    print("---")
    print(f"组数={len(results)}  每组设定回合={rounds}")
    print(f"平均实际进行回合={round(all_rounds / len(results), 2)}")
    print(f"提前结束(鱼量归0)组数={early}/{len(results)}")
    print(f"A 总捕捞={all_a}  B 总捕捞={all_b}  B/A 收益比={round(all_b / all_a, 2) if all_a else 0}")
    print(f"A 平均/回合={round(all_a / all_rounds, 2)}  B 平均/回合={round(all_b / all_rounds, 2)}")
    print(f"兜底决策总次数={fb_total}/{2 * all_rounds}")
    print(f"汇总已保存：{out}")
    return results


if __name__ == "__main__":
    run_batch(10, 10)
