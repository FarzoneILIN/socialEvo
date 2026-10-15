"""Frozen task instructions and per-phase public observations."""

from . import SCHEMA_VERSION
from .protocol import canonical_json

PROMPT_VERSION = "socialevo.tasks.prompt.v1"
PROMPT_VERSIONS = {"v1": PROMPT_VERSION, "v2": "socialevo.tasks.prompt.v2"}
COMMON = ("You are one agent in a simultaneous multi-agent task. Maximize your own cumulative "
          "environment reward. Each physical round may have a simultaneous broadcast phase, "
          "then a simultaneous action phase. Other agents cannot see your current output before "
          "they choose. Return exactly one JSON object matching the phase schema, with no markdown "
          "or other text. Do not output reasoning. Broadcasts are nonbinding.\n")
RULES = {
    "congestion_v1": "There are six rounds. Choose route F or S each round. If k agents choose F, each F user's cost is k and each S user's cost is 2.5. Your reward is minus your own cost. No other reward or penalty exists.",
    "lbf_local_v1": "There are at most 24 rounds. All agents have level 1. Food level 1 or 2 is collected only if adjacent agents explicitly load the same food and their level sum reaches its level. Participants split food_level/R0 proportional to level. Move one cell N/E/S/W, or wait. Moves into round-start walls, uncollected food, or any round-start agent position fail. Moves competing for the same free destination all fail. Food and agents are publicly visible. All food collected ends the task early.",
    "commons_v1": "There are at most 12 rounds. Request an integer from 0 through current stock. If total requests exceed stock, the stock is allocated proportionally using integer largest remainders, with the publicly shown rotating priority breaking ties. Your reward is your actual allocation. After allocation stock becomes min(100, 2 * remaining stock). Zero stock before round 12 ends the task; round 12 ends after allocation. There is no terminal stock reward or enforced quota."
}


FORMAT_V2 = (
    "\nOutput the phase's JSON object directly at the top level. Include only the keys in "
    "legal_action_schema. Do not include agent_id, and do not wrap the object inside an "
    "action key. For a talk phase the shape is {\"type\":\"say\",\"content\":\"...\"}. "
    "For an act phase use one of that task's action shapes."
)
CLARIFICATIONS_V2 = {
    "congestion_v1": (
        " An act output has the shape {\"type\":\"choose\",\"route\":\"F\"}, "
        "where route may be F or S."
    ),
    "lbf_local_v1": (
        " Coordinates are (x,y): x increases to the right and y increases downward. "
        "A move changes your position by N=(0,-1), S=(0,1), E=(1,0), or W=(-1,0). "
        "Adjacent means Manhattan distance exactly 1: abs(agent.x-food.x) + "
        "abs(agent.y-food.y) == 1; diagonal cells are not adjacent. "
        "You cannot walk onto an uncollected food cell. Movement does not collect food. "
        "Loading is a separate physical action from an adjacent cell. Only agents who "
        "explicitly choose load for the same food in the same round contribute levels "
        "or receive a share; merely being adjacent does not count. "
        "Action shapes are {\"type\":\"wait\"}, {\"type\":\"move\",\"direction\":\"N\"} "
        "(direction may be N, E, S, or W), or {\"type\":\"load\",\"food_id\":\"FOOD_ID\"} "
        "(replace FOOD_ID with an adjacent uncollected food's actual ID)."
    ),
    "commons_v1": (
        " Your amount is your individual request, not a group total or a per-agent "
        "share of a group request. request_total is the sum of all agents' individual "
        "requests. If request_total is at most stock, each agent receives its request; "
        "otherwise proportional allocation applies. Let allocated_total be the sum "
        "of actual allocations. remaining = stock - allocated_total; stock_after = "
        "min(100, 2 * remaining); regrowth_added = stock_after - remaining. "
        "Regrowth doubles the remaining stock, not the stock before harvest. "
        "This update is recorded at the end of every round, including round 12, "
        "but round 12 has no subsequent action or terminal stock reward. "
        "Zero remaining stock stays zero after regrowth and ends play before round 12. "
        "An act output has the shape {\"type\":\"harvest\",\"amount\":0}, where amount "
        "may be any integer from 0 through the current stock."
    ),
}


def system_prompt(task_id, prompt_version="v1"):
    if prompt_version not in PROMPT_VERSIONS:
        raise ValueError("invalid_prompt_version")
    prompt = COMMON + RULES[task_id]
    if prompt_version == "v2":
        prompt += FORMAT_V2 + CLARIFICATIONS_V2[task_id]
    return prompt


def schema(task_id, phase, stock=None):
    if phase == "talk":
        return {"exact_keys": ["type", "content"], "type": "say", "content": "string, 0..240 Unicode characters"}
    if task_id == "congestion_v1":
        return {"exact_keys": ["type", "route"], "type": "choose", "route": ["F", "S"]}
    if task_id == "lbf_local_v1":
        return {"one_of": [{"type": "wait"}, {"type": "move", "direction": ["N", "E", "S", "W"]},
                           {"type": "load", "food_id": "an adjacent uncollected food ID"}]}
    return {"exact_keys": ["type", "amount"], "type": "harvest", "amount": f"integer 0..{stock}"}


def observation(env, agent_id, phase, communication, history, messages, feedback):
    state = env.public_state()
    own = (state.get("cumulative_cost", {}).get(agent_id)
           if env.task_id == "congestion_v1" else
           state.get("cumulative_harvest", {}).get(agent_id)
           if env.task_id == "commons_v1" else
           state["agents"][agent_id]["cumulative_reward"])
    if env.task_id == "congestion_v1":
        own = -own
    return {"schema_version": SCHEMA_VERSION, "task_id": env.task_id, "agent_id": agent_id,
            "num_agents": len(env.agent_ids), "round": env.round_completed + 1,
            "horizon": env.horizon, "phase": phase, "communication_allowed": communication,
            "agent_order": list(env.agent_order), "public_state": state,
            "own_total_reward": own, "public_history": history,
            "current_messages": messages if phase == "act" else [],
            "own_feedback": feedback.get(agent_id, []), "max_message_chars": 240,
            "legal_action_schema": schema(env.task_id, phase, state.get("stock"))}


def messages_for(task_id, observation_value, prompt_version="v1"):
    return [{"role": "system", "content": system_prompt(task_id, prompt_version)},
            {"role": "user", "content": canonical_json(observation_value)}]
