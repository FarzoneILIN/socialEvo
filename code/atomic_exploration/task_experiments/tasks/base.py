"""Small state-transaction contract shared by the three local environments."""

import copy

from task_experiments import RULE_VERSION, SCHEMA_VERSION
from task_experiments.protocol import digest, reward_record


class TaskEnvironment:
    task_id = ""
    horizon = 0

    def __init__(self, instance: dict):
        self._validate_instance(instance)
        self.instance = copy.deepcopy(instance)
        self.agent_ids = tuple(instance["agent_ids"])
        self.agent_order = tuple(instance["agent_order"])
        self._state = self._initial_state()
        self._validate_state(self._state)

    @classmethod
    def _validate_instance(cls, instance):
        if (not isinstance(instance, dict)
                or instance.get("schema_version") != SCHEMA_VERSION
                or instance.get("rule_version") != RULE_VERSION
                or instance.get("task_id") != cls.task_id
                or type(instance.get("num_agents")) is not int
                or not 2 <= instance["num_agents"] <= 8
                or instance.get("horizon") != cls.horizon
                or type(instance.get("seed")) is not int):
            raise ValueError("invalid_instance_header")
        n = instance["num_agents"]
        ids = [f"A{i}" for i in range(n)]
        if (instance.get("agent_ids") != ids
                or not isinstance(instance.get("agent_order"), list)
                or len(instance["agent_order"]) != n
                or set(instance["agent_order"]) != set(ids)
                or not isinstance(instance.get("initial_state"), dict)):
            raise ValueError("invalid_instance_agents")

    @property
    def round_completed(self) -> int:
        return self._state["round_completed"]

    @property
    def done(self) -> bool:
        return self._state["done"]

    @property
    def termination_reason(self) -> str | None:
        return self._state["termination_reason"]

    def _initial_state(self):
        return copy.deepcopy(self.instance["initial_state"])

    def _validate_state(self, state):
        if (not isinstance(state, dict)
                or type(state.get("round_completed")) is not int
                or not 0 <= state["round_completed"] <= self.horizon
                or type(state.get("done")) is not bool
                or (state["done"] != (state.get("termination_reason") is not None))):
            raise ValueError("invalid_state_header")
        if state["round_completed"] == self.horizon and not state["done"]:
            raise ValueError("horizon_must_terminate")

    def public_state(self) -> dict:
        return copy.deepcopy(self._public_state())

    def validate_action(self, agent_id: str, action: dict):
        if agent_id not in self.agent_ids:
            raise ValueError("unknown_agent")
        return self._validate_action(agent_id, action)

    def fallback_action(self, agent_id: str) -> dict:
        if agent_id not in self.agent_ids:
            raise ValueError("unknown_agent")
        return copy.deepcopy(self._fallback_action(agent_id))

    def snapshot(self) -> dict:
        return copy.deepcopy({"schema_version": SCHEMA_VERSION,
                              "task_id": self.task_id, "rule_version": RULE_VERSION,
                              "instance": self.instance, "round_completed": self.round_completed,
                              "task_state": self._state})

    @classmethod
    def from_snapshot(cls, snapshot: dict):
        if (not isinstance(snapshot, dict)
                or set(snapshot) != {"schema_version", "task_id", "rule_version",
                                       "instance", "round_completed", "task_state"}
                or snapshot["schema_version"] != SCHEMA_VERSION
                or snapshot["task_id"] != cls.task_id
                or snapshot["rule_version"] != RULE_VERSION):
            raise ValueError("invalid_snapshot_header")
        env = cls(snapshot["instance"])
        state = copy.deepcopy(snapshot["task_state"])
        env._validate_state(state)
        if snapshot["round_completed"] != state["round_completed"]:
            raise ValueError("snapshot_round_mismatch")
        env._state = state
        return env

    def step(self, joint_actions: dict[str, dict]) -> dict:
        if self.done:
            raise ValueError("episode_done")
        if (not isinstance(joint_actions, dict)
                or set(joint_actions) != set(self.agent_ids)
                or len(joint_actions) != len(self.agent_ids)):
            raise ValueError("incomplete_joint_action")
        normalized = {}
        for agent_id in self.agent_ids:
            validation = self.validate_action(agent_id, joint_actions[agent_id])
            if not validation.accepted:
                raise ValueError(f"invalid_joint_action:{agent_id}:{validation.reason}")
            normalized[agent_id] = validation.action
        draft = copy.deepcopy(self._state)
        rewards, outcome, feedback = self._step(normalized, draft)
        self._validate_state(draft)
        if (set(rewards) != set(self.agent_ids)
                or set(feedback) != set(self.agent_ids)):
            raise ValueError("invalid_transition_agents")
        transition = {"round": draft["round_completed"],
                      "executed_actions": copy.deepcopy(normalized),
                      "rewards": {aid: reward_record(rewards[aid]) for aid in self.agent_ids},
                      "public_outcome": copy.deepcopy(outcome),
                      "feedback_by_agent": copy.deepcopy(feedback),
                      "done": draft["done"],
                      "termination_reason": draft["termination_reason"]}
        digest(transition)
        self._state = draft
        return copy.deepcopy(transition)

    def score(self) -> dict:
        return copy.deepcopy(self._score())
