"""Native kitchen state replay helpers retained from the definition batch."""
from __future__ import annotations
import copy
import hashlib
import json
from task_support import collab


POOL_RELATIVE = "influence_experiments/runs/20261007_atomic_relations_v1/pools.json"


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value):
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _substantive(physical):
    return _digest({key: physical[key] for key in ("players", "objects", "order_list")})


def _replay_prefix(source):
    private = source["private_scoring"]
    bootstrap = {"case_id": "offline-prefix",
                 "private_scoring": {"order": private["order"], "recipe_file": private["recipe_file"],
                     "physical": collab._serialize(collab._mdp(private["order"]).get_standard_start_state())}}
    replayed = collab.replay_actions(bootstrap, private["prefix_witness"])
    if collab._sha(replayed["physical"]) != collab._sha(private["physical"]):
        raise ValueError("kitchen_pool_prefix_does_not_reproduce_checkpoint")
    return replayed


def _replay_suffix(state, actions):
    current = copy.deepcopy(state)
    start_count = len(current["actions"])
    for action in actions:
        trial = collab.apply_action(current, action["actor"], action["choice"])
        if not trial["legal"]:
            return None
        current = trial["state"]
        if current["delivered"]:
            break
    if not current["delivered"]:
        return None
    return current["actions"][start_count:]
