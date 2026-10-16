"""Menu state construction and reading historical case identities."""
from __future__ import annotations
import copy
import hashlib
import json
import math
from pathlib import Path
from task_support import collab
from task_support.menu_score import _menu_optimum


ROOT = Path(__file__).resolve().parents[1]


DIRECTIONS = {"A_to_B": (0, 1), "B_to_A": (1, 0)}


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def _legacy_fingerprint(case):
    s = case.get("private_scoring", {})
    if s.get("kind") == "menu_numeric":
        return "menu", digest({k: s[k] for k in ("dishes", "state", "recipes", "values")})
    if s.get("kind") == "dnd_allocation":
        return "dnd", digest({k: s[k] for k in ("counts", "sender_values", "receiver_values")})
    if "physical" in s and "order" in s:
        return "collab", digest({"order": s["order"], "physical": s["physical"]})
    snapshot = s.get("snapshot", {})
    if "state" in snapshot and "order" in snapshot and "players" in snapshot["state"]:
        return "collab", digest({"order": snapshot["order"], "physical": snapshot["state"]})
    return None, None


def _historical_cases(workspace=ROOT):
    for package in ("quick_screen", "focused_validation", "mechanism_experiments", "influence_experiments"):
        runs = Path(workspace)/package/"runs"
        paths = sorted(set(runs.glob("*/manifest.json")) | set(runs.glob("*/waves/*/manifest.json")))
        for path in paths:
            yield from json.loads(path.read_text(encoding="utf-8")).get("cases", [])


def _menu_case(sample, difficulty):
    ingredients, dishes = sample["ingredients"], sample["dishes"]
    state = [sample["game_ingredients_state"].get(x, 0) for x in ingredients]
    recipes = sample["recipes"]
    values = [(a+b)/2 for a, b in zip(sample["agent_0_values"], sample["agent_1_values"])]
    optimum, best = _menu_optimum(state, recipes, values)
    if not math.isclose(optimum, sample["max_reward"], abs_tol=1e-7):
        raise ValueError("upstream_menu_optimum_mismatch")
    if not all(sample["agent_0_obs"].get(x, 0)+sample["agent_1_obs"].get(x, 0) == n
               for x, n in zip(ingredients, state)):
        raise ValueError("menu_inventory_partition_mismatch")
    public = {name: {ing: n for ing, n in zip(ingredients, row) if n}
              for name, row in zip(dishes, recipes)}
    intro = ("Menu-Numeric directed-message proposal task. Design a feasible menu for two equally important guests. "
             "Choose each dish at most once. Total ingredient stock is the SUM of the two private inventory partitions; "
             "missing entries mean zero. Each dish value is the AVERAGE of the two guests' numerical preferences. "
             "Menu value sums selected dish values; exceeding total stock yields zero quality. The acting agent proposes "
             "a menu after a single communication opportunity. This measures proposal quality, not bilateral agreement. "
             "Both plan lists are empty; no earlier partner text is provided.\nPublic recipes:\n"+canonical(public))
    actor_prompts = [intro+f"\nYou are Agent {i} ({'A' if i == 0 else 'B'}).\nYour private inventory partition:\n"
                     +canonical(sample[f"agent_{i}_obs"])+"\nYour private dish preferences:\n"
                     +canonical(dict(zip(dishes, sample[f"agent_{i}_values"]))) for i in (0, 1)]
    raw = [f"Agent {i} original private inventory:\n"+canonical(sample[f"agent_{i}_obs"])
           +f"\nAgent {i} original private dish preferences:\n"
           +canonical(dict(zip(dishes, sample[f"agent_{i}_values"]))) for i in (0, 1)]
    s = {"kind": "menu_numeric", "dishes": dishes, "state": state, "recipes": recipes, "values": values,
         "max_reward": optimum, "optimal_choice": [dishes[i] for i in best], "raw_views": raw,
         "agent_inventories": [sample[f"agent_{i}_obs"] for i in (0, 1)],
         "agent_values": [sample[f"agent_{i}_values"] for i in (0, 1)],
         "reward_semantics": "shared feasible-menu proposal quality; individual economic rewards not assigned"}
    case = {"case_id": "influence_menu_"+difficulty+"_"+sample["upstream_game_id"], "task": "menu",
            "candidate": "Deliberative Menu-Numeric", "actor_prompts": actor_prompts,
            "full_actor_prompts": [actor_prompts[i]+"\nBoth original private views are supplied directly. "
                                   "Combine them yourself under the public rules:\n"+"\n".join(raw) for i in (0, 1)],
            "response_instruction": 'Return only JSON {"choice":["exact dish name",...],"reason":"brief"}. '
                                    "Use exact public names without duplicates. The empty menu [] is legal.",
            "allowed_choices": dishes, "private_scoring": s, "difficulty_stratum": difficulty,
            "provenance": {**sample["provenance"], "difficulty_stratum": difficulty,
                           "adaptation": "Numeric private inventory/preference views retained at every upstream level; "
                           "higher upstream level source is a distinct stratum, not an unchanged level-3 language benchmark.",
                           "selection_rule": "Published path order; historical semantic exclusion; no output-dependent selection."}}
    case["semantic_hash"] = _legacy_fingerprint(case)[1]
    case["cluster"] = case["case_id"]
    return directional(case, "A_to_B")


def directional(case, direction):
    """Return a deep copy; preserve original physical A/B identities and scores."""
    if direction not in DIRECTIONS:
        raise ValueError("unknown_direction")
    case = copy.deepcopy(case)
    sender, receiver = DIRECTIONS[direction]
    names = ["Chef", "Assistant"] if case["task"] == "collab" else ["Agent 0 (A)", "Agent 1 (B)"]
    case.update(direction=direction, sender_id=sender, receiver_id=receiver, sender=names[sender], receiver=names[receiver],
                logical_actor={"sender": sender, "receiver": receiver},
                sender_prompt=case["actor_prompts"][sender]+"\nWrite one concise useful message to "+names[receiver]
                +" using only your own observations. No tools or follow-up. Plain text only, at most200 words.",
                own_message_prompt=case["actor_prompts"][receiver]+"\nWrite one concise useful message to "+names[sender]
                +" using only your own observations. No tools or follow-up. Plain text only, at most200 words.",
                receiver_prompt=case["actor_prompts"][receiver], full_information_prompt=case["full_actor_prompts"][receiver])
    if case["task"] == "collab":
        case["allowed_choices"] = case["allowed_by_actor"][receiver]
        case["response_instruction"] = case["response_instructions"][receiver]
        reference = case["private_scoring"]["reference_choices_by_actor"][receiver]
        alternate = [x for x in case["allowed_choices"] if x not in reference]
        case["recommendations"] = [reference[0], "wait(1)" if "wait(1)" in alternate else alternate[0]] if alternate else None
        case["physical_role"] = "recipe holder to preparation assistant" if sender == 0 else "preparation assistant to recipe holder"
        case["facts_text"] = (collab.UPSTREAM/case["private_scoring"]["recipe_file"]).read_text(encoding="utf-8") if sender == 0 else "Both agents already see the same complete public physical state."
    elif case["task"] == "dnd":
        s = case["private_scoring"]
        def value(choice, actor):
            allocation = choice if actor == receiver else [c-x for c, x in zip(s["counts"], choice)]
            return sum(x*v for x, v in zip(allocation, s["agent_values"][actor]))
        allocations = [x for x in case["allowed_choices"] if isinstance(x, list)]
        case["recommendations"] = [max(allocations, key=lambda x: (value(x, 0), value(x, 1), x)),
                                   max(allocations, key=lambda x: (value(x, 1), value(x, 0), x))]
        case["physical_role"] = "private-value negotiator to allocation proposer"
        case["facts_text"] = canonical({f"agent_{sender}_unit_values": dict(zip(["books", "hats", "balls"], s["agent_values"][sender]))})
    else:
        case["recommendations"] = [case["private_scoring"]["optimal_choice"], []]
        case["physical_role"] = "private inventory and preference holder to menu proposer"
        s = case["private_scoring"]
        case["facts_text"] = canonical({f"agent_{sender}_inventory": s["agent_inventories"][sender],
                                       f"agent_{sender}_preferences": dict(zip(s["dishes"], s["agent_values"][sender]))})
    return case
