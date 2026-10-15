import hashlib
import json
import platform
import time
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from .client import ApiError
from .environment import Environment, make_task, parse_action
from .recording import Recorder


SYSTEM = '''You are one of two equal agents solving an information-sharing task.
There is exactly one correct candidate: the intersection of your private possible
answers and the other agent's private possible answers. Candidate IDs are arbitrary.
Maximize the chance that BOTH agents independently submit the correct answer.
You cannot inspect the other agent's context. Only delivered messages are visible.
Observations/actions are simultaneous within a round; messages arrive next round.
You may submit once; correctness is disclosed only after the episode, not to agents.
Return exactly one JSON object, no markdown or other text:
{"type":"send","content":"message for the other agent"} (only if communication allowed),
{"type":"submit","answer":"candidate ID"}, or {"type":"wait"}.
Sending or waiting consumes one turn just as submitting does. Submit by the last round.
The observation states the message-channel availability. Do not invent tools.
'''


def episode(config, client, task, communication, episode_id, recorder, reverse=False):
    """
    跑完一局实验
    初始化环境 → 多轮调用两个 Agent → 收集并统一执行动作 → 停止 → 评分和写日志。
    """
    # 初始化
    env = Environment(task, communication, config.max_message_chars)
    histories = [[{"role": "system", "content": SYSTEM}] for _ in range(2)]
    calls = invalid = reported = 0
    usage_complete = True
    status = "round_limit"
    started = time.monotonic()
    recorder.event(episode_id, "start", communication=communication, task=asdict(task),
                   call_order=[1, 0] if reverse else [0, 1])

    # 每一局实验实现
    for round_index in range(1, config.rounds + 1):
        # 先把两个人这一轮的 observation 都生成出来，再调用模型
        observations = [env.observe(i, round_index, config.rounds) for i in range(2)]
        pending = []
        stop = None
        for agent in ([1, 0] if reverse else [0, 1]): # 随机加上反转
            if env.answers[agent] is not None:
                continue
            if calls >= config.max_calls or reported >= config.reported_token_limit: # 预算检查
                stop = "budget_limit"
                break
            obs = observations[agent]
            obs["max_message_chars"] = config.max_message_chars

            # 当前一轮观测接进 Agent 私有上下文
            histories[agent].append({"role": "user", "content": json.dumps(obs, ensure_ascii=False)})
            calls += 1
            recorder.event(episode_id, "request", round=round_index, agent=agent,
                           messages=histories[agent], call=calls)
            call_started = time.monotonic() # 记录耗时
            try:
                reply = client.complete(histories[agent]) # Agent 完整私有历史给 client
            except ApiError as exc: # 接口调用失败
                recorder.event(episode_id, "api_error", agent=agent, error=str(exc), call=calls)
                usage_complete = False
                stop = "api_error"
                break

            # 处理接口返回的 Token 数据
            usage = reply.usage
            total = usage.get("total_tokens") if usage else None
            valid_usage = type(total) is int and total >= 0
            usage_complete &= valid_usage
            if valid_usage:
                reported += total

            # 记录原始响应 可追溯失败
            recorder.event(episode_id, "response", round=round_index, agent=agent,
                           content=reply.content, usage=usage, actual_model=reply.model,
                           finish_reason=reply.finish_reason,
                           seconds=time.monotonic() - call_started)
            histories[agent].append({"role": "assistant", "content": reply.content}) # 模型原始回复放进自己历史
            try: # 解析模型输出
                action = parse_action(reply.content)
                pending.append((agent, action))
            except (ValueError, TypeError):
                invalid += 1
                env.feedback[agent] = "invalid_json_object"
                recorder.event(episode_id, "invalid_action", agent=agent, reason="invalid_json_object")
            if config.backend == "api" and not valid_usage: # 如果没有可信 token 数据
                stop = "usage_unknown"
                break

        # 动作生效
        for agent, action in pending:
            result = env.apply(agent, action) # 动作是否符合环境规则
            invalid += not result["accepted"]
            recorder.event(episode_id, "action", round=round_index, agent=agent,
                           action=action, result=result)
        if stop: # 是否提前结束
            status = stop
            break
        if all(x is not None for x in env.answers): # 两个 Agent 全部提交答案自动判定实验完成
            status = "completed"
            break

    # 构建结果
    result = {"episode": episode_id, "communication": communication, "status": status,
              **env.score(), "calls": calls, "invalid_actions": invalid,
              "reported_tokens": reported if usage_complete else None,
              "known_reported_tokens": reported, "usage_complete": usage_complete,
              "seconds": time.monotonic() - started}
    recorder.event(episode_id, "result", **result_without_id(result), answer=task.answer)
    return result


def result_without_id(result):
    return {k: v for k, v in result.items() if k != "episode"}


def run(config, client, output_root):
    config.validate()
    directory = Path(output_root) / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8])
    directory.mkdir(parents=True, exist_ok=False)
    recorder = Recorder(directory, getattr(client, "key", ""))
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob("*.py")}
    recorder.write("manifest.json", {"config": asdict(config), "requested_model": client.model,
                    "endpoint": getattr(client, "url", None), "python": platform.python_version(),
                    "code_sha256": hashes, "system_prompt": SYSTEM,
                    "warning": "Exploratory information-sharing task; mock results are not LLM evidence."})
    results = []

    # 批量运行试验
    for instance in range(config.instances):
        task = make_task(config.seed + instance)
        for repeat in range(config.repeats):
            conditions = [True, False] if (instance + repeat) % 2 == 0 else [False, True]
            for communication in conditions:
                eid = f"i{instance}_r{repeat}_{'comm' if communication else 'no_comm'}"
                result = episode(config, client, task, communication, eid, recorder,
                                 reverse=(instance + repeat) % 2 == 1)
                result.update(instance=instance, repeat=repeat)
                results.append(result)
                recorder.write(eid + ".json", result)
                if result["status"] in ("api_error", "usage_unknown"):
                    # Stop the batch rather than repeatedly spending on a broken endpoint.
                    return finish(recorder, results, config, aborted=True)
    return finish(recorder, results, config, aborted=False)


def finish(recorder, results, config, aborted):
    """
    把已经结束的所有 episode 汇总成 summary.json
    """
    groups = {}
    for condition in (True, False):
        rows = [r for r in results if r["communication"] == condition]
        valid = [r for r in rows if r["status"] not in ("api_error", "usage_unknown")]
        groups["comm" if condition else "no_comm"] = {
            "attempted": len(rows), "evaluable": len(valid),
            "team_success_rate": sum(r["team_success"] for r in valid) / len(valid) if valid else None,
            "individual_accuracy": sum(r["individual_accuracy"] for r in valid) / len(valid) if valid else None,
            "calls": sum(r["calls"] for r in rows),
            "total_reported_tokens": sum(r["reported_tokens"] for r in rows) if rows and all(r["usage_complete"] for r in rows) else None,
        }
    pairs = []
    for instance in range(config.instances):
        for repeat in range(config.repeats):
            rows = {r["communication"]: r for r in results if r["instance"] == instance and r["repeat"] == repeat and r["status"] not in ("api_error", "usage_unknown")}
            if len(rows) == 2:
                pairs.append({"instance": instance, "repeat": repeat,
                              "accuracy_delta": rows[True]["individual_accuracy"] - rows[False]["individual_accuracy"]})
    recorder.write("summary.json", {"backend": config.backend, "aborted": aborted,
                    "groups": groups, "paired_differences": pairs,
                    "note": "Repeated runs are not independent tasks. No significance claim. Errors excluded explicitly; invalid actions and timeouts of task rounds are retained."})
    return recorder.directory, aborted
