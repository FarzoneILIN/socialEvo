"""Explicit registry for the three frozen local task versions."""

import random

from task_experiments import RULE_VERSION, SCHEMA_VERSION


TASKS = {
    "congestion_v1": "congestion",
    "lbf_local_v1": "lbf_local",
    "commons_v1": "commons",
}
HORIZONS = {"congestion_v1": 6, "lbf_local_v1": 24, "commons_v1": 12}


def environment_class(task_id):
    if task_id == "congestion_v1":
        from .congestion import CongestionEnvironment
        return CongestionEnvironment
    if task_id == "lbf_local_v1":
        from .lbf_local import LBFEnvironment
        return LBFEnvironment
    if task_id == "commons_v1":
        from .commons import CommonsEnvironment
        return CommonsEnvironment
    raise ValueError("unknown_task")


def make_instance(task_id: str, num_agents: int, seed: int) -> dict:
    if task_id not in TASKS or type(num_agents) is not int or not 2 <= num_agents <= 8 or type(seed) is not int:
        raise ValueError("invalid_instance_parameters")
    ids = [f"A{i}" for i in range(num_agents)]
    order = list(ids)
    random.Random(seed).shuffle(order)
    cls = environment_class(task_id)
    initial_state, params = cls.build_initial_state(num_agents, order)
    instance = {"schema_version": SCHEMA_VERSION, "rule_version": RULE_VERSION,
                "task_id": task_id, "num_agents": num_agents, "seed": seed,
                "agent_ids": ids, "agent_order": order,
                "horizon": HORIZONS[task_id], "rule_parameters": params,
                "initial_state": initial_state}
    cls(instance)
    return instance
