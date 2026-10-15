import copy
import json
import random
from dataclasses import dataclass


@dataclass(frozen=True)
class Task:
    candidates: tuple[str, ...]
    clues: tuple[tuple[str, ...], tuple[str, ...]]
    answer: str


def make_task(seed: int) -> Task:
    """
    任务设置：
    程序随机生成 16 个无语义编号，在内部排列成 4×4 表格
    环境随机选取一行和一列，将该行的 4 个候选交给一个 Agent
    将该列的4个候选交给另一个 Agent ；行列交点是唯一正确答案

    每个 Agent 只知道答案位于自己的4个私人候选中，无法单独确定答案
    两个 Agent 交换候选集合并取交集，即可定位唯一正确答案

    该任务适合初步验证信息交换能否提高多智能体任务成功率
    由于没有角色形成、组织调整等机制，不能证明自组织相关结论
    """
    rng = random.Random(seed)
    labels = [f"item_{n:05d}" for n in rng.sample(range(10000, 99999), 16)]
    row, col = rng.randrange(4), rng.randrange(4)
    clues = [labels[row * 4:row * 4 + 4], labels[col::4]]
    answer = labels[row * 4 + col]
    rng.shuffle(labels)
    for clue in clues:
        rng.shuffle(clue)
    if rng.randrange(2):
        clues.reverse()
    return Task(tuple(labels), tuple(tuple(x) for x in clues), answer)


def parse_action(raw: str) -> dict:
    """
    输出 JSON 转换成 python 字典
    """
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON key")
            result[key] = value
        return result
    value = json.loads(raw, object_pairs_hook=unique)
    if not isinstance(value, dict):
        raise ValueError("Expected JSON object")
    return value


class Environment:
    def __init__(self, task: Task, communication: bool, max_message_chars=1200):
        self.task = task
        self.communication = communication # 本局是否允许互发消息
        self.max_message_chars = max_message_chars
        self.inboxes = [[], []] # 两个收件箱 inboxes[0] 仅属于 Agent 0
        self.answers = [None, None] # 两个Agent都尚未提交答案
        self.feedback = ["", ""] # 记录动作是否被接受 不记录答案是否正确

    def observe(self, agent: int, round_index: int, rounds: int) -> dict:
        """
        一个 Agent 一轮之内允许被看到的观察
        """
        return copy.deepcopy({
            "agent_id": agent,
            "round": round_index,
            "rounds": rounds,
            "communication_allowed": self.communication, # 是否允许交流
            "candidates": list(self.task.candidates), # 所有人都知道的 16 个候选
            "private_possible_answers": list(self.task.clues[agent]), # 仅此 Agent 的 4 个线索
            "inbox": self.inboxes[agent], # 对方此前合法发来的消息
            "feedback": self.feedback[agent],
        })

    def apply(self, agent: int, action: dict) -> dict:
        """
        Agent 每轮只能执行三个动作：
        send	发消息	若通信开放且内容合法，把消息放进对方收件箱
        submit	提交候选 ID	记录答案，之后不可再操作
        wait	本轮不做事	不改变环境状态
        """
        kind = action.get("type")
        error = None
        if self.answers[agent] is not None:
            error = "already_submitted"
        elif kind == "send":
            if set(action) != {"type", "content"}:
                error = "invalid_fields"
            elif not self.communication:
                error = "communication_disabled"
            elif not isinstance(action["content"], str) or not 0 < len(action["content"]) <= self.max_message_chars:
                error = "invalid_message"
            else:
                self.inboxes[1 - agent].append({"from": agent, "content": action["content"]})
        elif kind == "submit":
            if set(action) != {"type", "answer"} or action.get("answer") not in self.task.candidates:
                error = "invalid_answer"
            else:
                self.answers[agent] = action["answer"]
        elif kind == "wait":
            if set(action) != {"type"}:
                error = "invalid_fields"
        else:
            error = "invalid_action"
        result = {"accepted": error is None, "reason": error or "accepted"}
        # Never reveal correctness, even after an incorrect submission.
        self.feedback[agent] = result["reason"]
        return result

    def score(self):
        """
        由研究者调用的客观评分
        """
        correct = [x == self.task.answer for x in self.answers]
        return {"submitted": list(self.answers), "correct": correct,
                "individual_accuracy": sum(correct) / 2,
                "team_success": all(correct)}
