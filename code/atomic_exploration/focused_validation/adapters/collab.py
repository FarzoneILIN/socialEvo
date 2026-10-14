"""Native Collab-Overcooked decision snapshots and resumable macro execution.

The underlying pinned MDP/layout are unchanged. This adapter substitutes a
deterministic legal navigation controller for the upstream learned controller.
It does not choose ingredients, utensils, processing or delivery for a model.
"""
from __future__ import annotations

from collections import deque
import copy
import functools
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys
import threading
import types

SOURCE = Path(__file__).resolve().parents[1] / "sources" / "collab"
UPSTREAM = SOURCE / "upstream"
COMMIT = "f7b66f0082b2a7c950ff7ee9b25ad588b5751197"
NATIVE_HORIZON = 120
_LOAD_LOCK = threading.RLock()
_ENGINE = None


def _sha(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()


def _file_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _engine():
    """Isolate the copied source namespace, bypassing only Gym registration."""
    global _ENGINE
    with _LOAD_LOCK:
        if _ENGINE is not None:
            return _ENGINE
        base = UPSTREAM / "lib/overcooked_ai/overcooked_ai_py"
        prefix = "_socialevo_focused_collab_native"
        for name, directory in [(prefix, base), (prefix + ".mdp", base / "mdp"),
                                (prefix + ".data", base / "data")]:
            package = types.ModuleType(name)
            package.__path__ = [str(directory)]
            sys.modules[name] = package
        modules = [("utils", "utils.py"), ("mdp.actions", "mdp/actions.py"),
                   ("data.layouts", "data/layouts/__init__.py"),
                   ("mdp.overcooked_mdp", "mdp/overcooked_mdp.py")]
        for suffix, filename in modules:
            path = base / filename
            module = types.ModuleType(prefix + "." + suffix)
            module.__file__ = str(path)
            module.__package__ = module.__name__.rpartition(".")[0]
            # Source diagnostic prints are suppressed locally, without
            # changing process-global stdout during parallel experiments.
            module.__dict__["print"] = lambda *args, **kwargs: None
            if suffix == "data.layouts":
                module.__path__ = [str(path.parent)]
            sys.modules[module.__name__] = module
            text = path.read_text(encoding="utf-8").replace("from overcooked_ai_py.", "from " + prefix + ".")
            exec(compile(text, str(path), "exec"), module.__dict__)
        native = sys.modules[prefix + ".mdp.overcooked_mdp"]
        actions = sys.modules[prefix + ".mdp.actions"]
        _ENGINE = (native, actions)
        return _ENGINE


def _mdp(order):
    native, _ = _engine()
    mdp = native.OvercookedGridworld.from_layout_name("new_env")
    mdp.start_order_list = [order]
    mdp.one_task_mode = True
    return mdp


def _actor(actor):
    if actor in (0, "0", "Chef", "chef", "agent_0"):
        return 0
    if actor in (1, "1", "Assistant", "assistant", "agent_1"):
        return 1
    raise ValueError("Actor must be Chef/0 or Assistant/1")


def _serialize_object(obj):
    if obj is None:
        return None
    return {"name": obj.name, "catagory": obj.catagory, "position": list(obj.position),
            "state": json.loads(json.dumps(obj.state))}


def _serialize(state):
    return {"players": [{"position": list(p.position), "orientation": list(p.orientation),
                         "held_object": _serialize_object(p.held_object)} for p in state.players],
            "objects": [_serialize_object(obj) for _, obj in sorted(state.objects.items())],
            "order_list": list(state.order_list) if state.order_list is not None else None,
            "timestep": state.timestep, "last_operations": list(state.ml_actions),
            "errors": list(state.error_message)}


def _restore(physical):
    # Native ObjectState.deepcopy shares nested ingredient lists. Never let a
    # trial transition mutate the caller's checkpoint or frozen fixture.
    physical = copy.deepcopy(physical)
    native, _ = _engine()
    def object_state(value):
        if value is None:
            return None
        return native.ObjectState(value["name"], value["catagory"], tuple(value["position"]), value["state"])
    players = [native.PlayerState(tuple(p["position"]), tuple(p["orientation"]), object_state(p["held_object"]))
               for p in physical["players"]]
    objects = {tuple(o["position"]): object_state(o) for o in physical["objects"]}
    state = native.OvercookedState(players, objects, copy.deepcopy(physical["order_list"]), physical["timestep"])
    state.ml_actions = list(physical.get("last_operations", [None, None]))
    state.error_message = list(physical.get("errors", []))
    return state


def _route(mdp, state, actor, target):
    """Find movement/facing only, without advancing timers in search branches."""
    other = state.players[1 - actor].position
    player = state.players[actor]
    queue = deque([(player.position, player.orientation, [])])
    visited = set()
    while queue:
        position, orientation, path = queue.popleft()
        if (position, orientation) in visited:
            continue
        visited.add((position, orientation))
        if (position[0] + orientation[0], position[1] + orientation[1]) == target:
            return path
        for direction in [(0, -1), (0, 1), (1, 0), (-1, 0)]:
            new_position, new_orientation = mdp._move_if_direction(position, orientation, direction)
            if new_position == other:
                new_position = position
            queue.append((new_position, new_orientation, path + [direction]))
    return None


def _canonical(choice):
    return re.sub(r"\s+", "", choice).lower() if isinstance(choice, str) else None


def _parse(choice):
    text = _canonical(choice)
    match = re.fullmatch(r"([a-z_]+)\(([^()]*)\)", text or "")
    if not match:
        raise ValueError("Malformed operation")
    operation, body = match.groups()
    return text, operation, body.split(",") if body else []


def _execute(physical, order, actor, choice, enforce_horizon=True):
    """Execute one declared operation through genuine source transitions."""
    native, actions = _engine()
    mdp, state = _mdp(order), _restore(physical)
    try:
        canonical, operation, parameters = _parse(choice)
    except ValueError:
        return {"legal": False, "error": "malformed_operation", "choice": _canonical(choice),
                "physical": copy.deepcopy(physical), "reward": 0, "trace": [], "completed_operation": None}
    actor = _actor(actor)
    if physical["timestep"] >= NATIVE_HORIZON and enforce_horizon:
        return {"legal": False, "error": "native_horizon_reached", "choice": canonical,
                "physical": copy.deepcopy(physical), "reward": 0, "trace": [], "completed_operation": None}
    own_utensils = ["pot0", "oven0"] if actor == 0 else ["chopping_board0", "blender0"]
    held = state.players[actor].held_object
    target, parameter, route = None, None, None
    if operation == "wait" and len(parameters) == 1 and parameters[0] in ("1", "3"):
        low_actions = [(actions.Action.STAY, None)] * int(parameters[0])
    else:
        if operation == "pickup" and len(parameters) == 2 and held is None:
            obj, place = parameters
            if place == "ingredient_dispenser" and actor == 1 and obj in mdp.default_ingredients:
                target, parameter = mdp.get_ingredient_dispenser_locations()[0], obj
            elif place == "dish_dispenser" and actor == 1 and obj == "dish":
                target = mdp.get_dish_dispenser_locations()[0]
            elif place == "counter":
                possibilities = [(pos, _route(mdp, state, actor, pos)) for pos, item in state.objects.items()
                                 if item.name == obj and mdp.get_terrain_type_at_pos(pos) == "X"]
                possibilities = [(pos, r) for pos, r in possibilities if r is not None]
                if possibilities:
                    target, route = min(possibilities, key=lambda item: (len(item[1]), item[0]))
            elif place in own_utensils:
                position = mdp.from_utensil_name_to_pos(place)
                item = state.objects.get(position)
                # Completed food only: avoid the engine's early retrieval
                # branch reclassifying unfinished food as a finished ingredient.
                if item is not None and item.catagory == "finished" and item.name == obj:
                    target, parameter = position, obj
        elif operation == "place_obj_on_counter" and not parameters and held is not None:
            possibilities = []
            for pos in mdp.get_counter_locations():
                if pos in state.objects:
                    continue
                own_route = _route(mdp, state, actor, pos)
                partner_route = _route(mdp, state, 1 - actor, pos)
                if own_route is not None and partner_route is not None:
                    possibilities.append((pos, own_route))
            if possibilities:
                target, route = min(possibilities, key=lambda item: (len(item[1]), item[0]))
        elif operation == "put_obj_in_utensil" and len(parameters) == 1 and held is not None and held.name != "dish":
            utensil = parameters[0]
            if utensil in own_utensils:
                position = mdp.from_utensil_name_to_pos(utensil)
                item = state.objects.get(position)
                if item is None or (item.catagory == "semi-finished" and isinstance(item.state[0], list)
                                    and item.state[2] == 0 and item.state[1] < mdp.num_items_for_soup):
                    target = position
        elif operation in ("cook", "bake", "cut", "stir") and len(parameters) == 1:
            utensil = parameters[0]
            expected = {"pot0": "cook", "oven0": "bake", "chopping_board0": "cut", "blender0": "stir"}
            if utensil in own_utensils and operation == expected[utensil]:
                position = mdp.from_utensil_name_to_pos(utensil)
                item = state.objects.get(position)
                if item is not None and item.catagory == "semi-finished" and item.state[2] == 0:
                    target, parameter = position, "[START]"
        elif operation == "fill_dish_with_food" and len(parameters) == 1 and held is not None and held.name == "dish":
            utensil = parameters[0]
            if utensil in own_utensils:
                position = mdp.from_utensil_name_to_pos(utensil)
                item = state.objects.get(position)
                if item is not None and item.catagory == "finished":
                    target = position
        elif operation == "deliver_soup" and not parameters and actor == 0 and held is not None:
            if held.name in mdp.Recipe.recipe_name and held.catagory != "semi-finished":
                target = mdp.get_serving_locations()[0]
        if target is None:
            return {"legal": False, "error": "operation_preconditions_not_met", "choice": canonical,
                    "physical": copy.deepcopy(physical), "reward": 0, "trace": [], "completed_operation": None}
        route = _route(mdp, state, actor, target) if route is None else route
        if route is None:
            return {"legal": False, "error": "target_unreachable", "choice": canonical,
                    "physical": copy.deepcopy(physical), "reward": 0, "trace": [], "completed_operation": None}
        low_actions = [(direction, None) for direction in route] + [(actions.Action.INTERACT, parameter)]
    trace, reward, completed = [], 0, None
    for action, parm in low_actions:
        if enforce_horizon and state.timestep >= NATIVE_HORIZON:
            break
        joint, params = [actions.Action.STAY, actions.Action.STAY], [None, None]
        joint[actor], params[actor] = action, parm
        before = _sha(_serialize(state))
        # Source prints are diagnostically retained as native state errors,
        # never exposed as task instructions or used to repair model decisions.
        state, sparse, shaped = mdp.get_state_transition(state, joint, params)
        reward += sparse
        if state.ml_actions[actor] is not None:
            completed = state.ml_actions[actor]
        trace.append({"before_sha256": before, "joint_action": json.loads(json.dumps(joint)),
                      "parameters": params, "sparse_reward": sparse, "shaped_reward": shaped,
                      "after_sha256": _sha(_serialize(state)), "native_errors": list(state.error_message)})
        if sparse > 0:
            # Official fix_task runner also stops on first positive delivery.
            break
    delivered = reward > 0
    done_operation = operation == "wait" or _canonical(completed) == canonical
    interrupted = enforce_horizon and state.timestep >= NATIVE_HORIZON and not done_operation
    return {"legal": True if interrupted else done_operation, "choice": canonical,
            "physical": _serialize(state), "reward": reward, "trace": trace,
            "completed_operation": completed, "elapsed_steps": len(trace),
            "delivered": delivered, "interrupted_by_horizon": interrupted,
            "error": "native_horizon_interrupted_macro" if interrupted else (None if done_operation else "native_operation_not_completed")}


def _candidate_macros(physical, actor):
    actor = _actor(actor)
    mdp = _mdp(physical["order_list"][0])
    state = _restore(physical)
    held = state.players[actor].held_object
    utensils = ["pot0", "oven0"] if actor == 0 else ["chopping_board0", "blender0"]
    choices = ["wait(1)", "wait(3)"]
    if held is None:
        if actor == 1:
            choices += ["pickup(" + item + ",ingredient_dispenser)" for item in mdp.default_ingredients]
            choices += ["pickup(dish,dish_dispenser)"]
        choices += ["pickup(" + item.name + ",counter)" for pos, item in state.objects.items()
                    if mdp.get_terrain_type_at_pos(pos) == "X"]
        for utensil in utensils:
            item = state.objects.get(mdp.from_utensil_name_to_pos(utensil))
            if item is not None and item.catagory == "finished":
                choices.append("pickup(" + item.name + "," + utensil + ")")
    else:
        choices.append("place_obj_on_counter()")
        if held.name == "dish":
            choices += ["fill_dish_with_food(" + utensil + ")" for utensil in utensils]
        else:
            choices += ["put_obj_in_utensil(" + utensil + ")" for utensil in utensils]
            if actor == 0:
                choices.append("deliver_soup()")
    operations = {"pot0": "cook", "oven0": "bake", "chopping_board0": "cut", "blender0": "stir"}
    choices += [operations[u] + "(" + u + ")" for u in utensils]
    return sorted(set(choices))


def legal_macros(state, actor):
    if state.get("delivered") or state["physical"]["timestep"] >= NATIVE_HORIZON:
        return []
    return [choice for choice in _candidate_macros(state["physical"], actor)
            if _execute(state["physical"], state["order"], actor, choice)["legal"]]


def start_episode(case):
    return {"protocol": "collab_native_macro_v1", "case_id": case["case_id"],
            "order": case["private_scoring"]["order"],
            "recipe_file": case["private_scoring"]["recipe_file"],
            "physical": copy.deepcopy(case["private_scoring"]["physical"]),
            "initial_timestep": case["private_scoring"]["physical"]["timestep"],
            "team_reward": 0, "delivered": False, "macro_count": 0,
            "actions": [], "stopped_reason": None}


def apply_action(state, actor, choice):
    if state.get("delivered"):
        return {"state": copy.deepcopy(state), "legal": False, "choice": _canonical(choice),
                "team_reward": 0, "receiver_reward": None, "sender_reward": None,
                "completed_operation": None, "error": "episode_already_delivered", "elapsed_steps": 0, "trace": []}
    result = _execute(state["physical"], state["order"], actor, choice)
    next_state = copy.deepcopy(state)
    if result["legal"]:
        next_state["physical"] = result["physical"]
        next_state["team_reward"] += result["reward"]
        next_state["delivered"] = next_state["delivered"] or result.get("delivered", False)
        next_state["macro_count"] += 1
        next_state["actions"].append({"actor": _actor(actor), "choice": result["choice"],
                                      "before_physical_sha256": _sha(state["physical"]),
                                      "after_physical_sha256": _sha(next_state["physical"]),
                                      "team_reward": result["reward"], "trace": result["trace"]})
        if result.get("interrupted_by_horizon") or next_state["physical"]["timestep"] >= NATIVE_HORIZON:
            next_state["stopped_reason"] = "native_horizon"
        if next_state["delivered"]:
            next_state["stopped_reason"] = "native_target_delivery"
    return {"state": next_state, "legal": result["legal"], "choice": result["choice"],
            "team_reward": result["reward"], "receiver_reward": None, "sender_reward": None,
            "completed_operation": result["completed_operation"], "error": result["error"],
            "elapsed_steps": result.get("elapsed_steps", 0), "trace": result["trace"]}


def _public_view(physical):
    # Strip traces, references, recipes and all textual agent history. This is
    # the native publicly visible physical state, including order and errors.
    return copy.deepcopy(physical)


def _rules(actor):
    name = "Chef" if _actor(actor) == 0 else "Assistant"
    return ("You are " + name + " in Collab-Overcooked. Complete the displayed order with your teammate. "
            "Chef alone holds the recipe; Assistant has ingredient and dish dispensers, chopping_board0 and blender0. "
            "Chef has pot0, oven0 and the serving tile. Both see the complete physical kitchen state and order name. "
            "Each agent holds at most one item. Items are exchanged only through shared counters. "
            "Putting food in a utensil does not start processing: cut/cook/bake/stir starts it, and its native timer advances during every movement or wait. "
            "pickups from utensils retrieve completed food; fill_dish_with_food places completed food into a held dish. "
            "Navigation is deterministic and physically legal; the other actor stays still during your operation. "
            "Choose one listed operation. Legal choices describe physical feasibility, not which action helps this recipe. "
            "No partner planned actions or hidden recipe are disclosed through the physical snapshot. ")


def observation(state, actor, condition="normal"):
    choices = legal_macros(state, actor)
    prompt = _rules(actor) + "\nPublic physical state:\n" + json.dumps(_public_view(state["physical"]), ensure_ascii=False, sort_keys=True)
    if _actor(actor) == 0 or condition in ("full_information", "full-info"):
        prompt += "\nComplete recipe:\n" + (UPSTREAM / state["recipe_file"]).read_text(encoding="utf-8")
    return {"prompt": prompt, "allowed_macros": choices,
            "response_instruction": 'Return only JSON {"choice":"one exact permitted operation","reason":"brief"}. Permitted choices: ' + json.dumps(choices) + ".",
            "physical_state_sha256": _sha(state["physical"]), "actor": "Chef" if _actor(actor) == 0 else "Assistant"}


def episode_summary(state):
    return {"case_id": state["case_id"], "native_target_delivered": state["delivered"],
            "status": "success" if state["delivered"] else "unfinished", "stopped_reason": state.get("stopped_reason"),
            "team_reward": state["team_reward"], "receiver_reward": None, "sender_reward": None,
            "native_timestep": state["physical"]["timestep"],
            "elapsed_since_snapshot": state["physical"]["timestep"] - state["initial_timestep"],
            "macro_count": state["macro_count"], "physical_state_sha256": _sha(state["physical"]),
            "reward_kind": "exact native sparse reward; stop at first matching delivery as upstream fix_task runner"}


def replay_actions(case, actions):
    state = start_episode(case)
    for action in actions:
        if action.get("before_physical_sha256") and action["before_physical_sha256"] != _sha(state["physical"]):
            raise ValueError("Replay input physical state hash mismatch")
        result = apply_action(state, action["actor"], action["choice"])
        if not result["legal"]:
            raise ValueError("Saved action is no longer legal")
        state = result["state"]
        if action.get("after_physical_sha256") and action["after_physical_sha256"] != _sha(state["physical"]):
            raise ValueError("Replay output physical state hash mismatch")
    return state


def build_cases():
    fixtures = json.loads((SOURCE / "decision_states.json").read_text(encoding="utf-8"))
    cases = []
    engine_path = UPSTREAM / "lib/overcooked_ai/overcooked_ai_py/mdp/overcooked_mdp.py"
    layout_path = UPSTREAM / "lib/overcooked_ai/overcooked_ai_py/data/layouts/new_env.layout"
    for fixture in fixtures:
        temporary = {"case_id": fixture["case_id"], "private_scoring": fixture}
        state = start_episode(temporary)
        receiver, complete, sender = observation(state, 1), observation(state, 1, "full_information"), observation(state, 0)
        files = [engine_path, layout_path, UPSTREAM / fixture["recipe_file"], UPSTREAM / fixture["reference_file"], SOURCE / "decision_states.json"]
        cases.append({"case_id": fixture["case_id"], "candidate": "Collab-Overcooked", "sender": "Chef", "receiver": "Assistant",
            "sender_id": "Chef", "receiver_id": "Assistant", "logical_actor": {"sender": 0, "receiver": 1},
            "physical_role": "private recipe holder to resource and preparation assistant",
            "sender_prompt": sender["prompt"] + "\nSend one brief, factual instruction for Assistant's next operation. Its legal options are: " + json.dumps(receiver["allowed_macros"]),
            "receiver_prompt": receiver["prompt"] + "\nBoth plan lists are empty. No earlier partner text is provided.",
            "full_information_prompt": complete["prompt"] + "\nBoth plan lists are empty. No earlier partner text is provided.",
            "response_instruction": receiver["response_instruction"], "allowed_choices": receiver["allowed_macros"],
            "state_sha256": _sha(fixture["physical"]), "followup_prespecified": fixture["followup_prespecified"],
            "provenance": {"source_url": "https://github.com/YusaeMeow/Collab-Overcooked/tree/" + COMMIT,
                "commit": COMMIT, "local_source_file": str(engine_path), "sha256": _file_sha(engine_path),
                "native_agent_count": 2, "source_files": [{"path": str(p), "sha256": _file_sha(p)} for p in files],
                "adaptation": "native physical-state symbolic macro interface with deterministic legal navigation and alternating actors; ten new official-recipe snapshots reached by legal prefixes; Chef-only fresh text, empty prior text/plan signals; native reward retained, reference-match correctness is a separate secondary indicator; not unchanged upstream runner reproduction"},
            "private_scoring": fixture})
    return cases


def score(case, choice):
    if isinstance(choice, dict):
        choice = choice.get("choice")
    canonical = _canonical(choice)
    allowed = {_canonical(c): c for c in case["allowed_choices"]}
    if canonical not in allowed:
        return {"choice": choice, "legal": False, "correct": None, "receiver_reward": None, "sender_reward": None, "team_reward": None}
    # All permitted operations are executed in the real MDP, including choices
    # which fail to advance the reference. No preferred-action payoff is added.
    result = apply_action(start_episode(case), 1, canonical)
    reference_match = canonical in {_canonical(x) for x in case["private_scoring"]["reference_next_choices"]}
    return {"choice": canonical, "legal": result["legal"], "correct": reference_match if result["legal"] else None,
            "reference_next_operation_match": reference_match, "receiver_reward": None, "sender_reward": None,
            "team_reward": result["team_reward"], "completed_operation": result["completed_operation"],
            "elapsed_steps": result["elapsed_steps"], "state_after_sha256": _sha(result["state"]["physical"]),
            "score_kind": "exact native sparse macro reward; correct is reference next-operation match, not terminal payoff"}


def build_skips():
    return []


def _prepare():
    """Offline scripted prefixes are feasibility witnesses, never model inputs."""
    def assistant_prep(ingredient):
        return [(1, f"pickup({ingredient},ingredient_dispenser)"), (1, "put_obj_in_utensil(chopping_board0)"),
                (1, "cut(chopping_board0)"), (1, "wait(3)"), (1, f"pickup({ingredient}_slices,chopping_board0)"),
                (1, "place_obj_on_counter()")]
    def soup_prefix(ingredient):
        return assistant_prep(ingredient) + [(0, f"pickup({ingredient}_slices,counter)"), (0, "put_obj_in_utensil(oven0)"),
            (0, "bake(oven0)"), (0, "wait(3)"), (0, f"pickup(baked_{ingredient}_slices,oven0)"),
            (0, "put_obj_in_utensil(pot0)"), (0, "cook(pot0)"), (0, "wait(3)")]
    defs = [
        ("2_baked_pumpkin_slices", [(1, "pickup(pumpkin,ingredient_dispenser)")], ["put_obj_in_utensil(chopping_board0)"]),
        ("2_boiled_green_bean_slices", [(1, "pickup(green_bean,ingredient_dispenser)"), (1, "put_obj_in_utensil(chopping_board0)")], ["cut(chopping_board0)"]),
        ("1_boiled_mushroom", [], ["pickup(mushroom,ingredient_dispenser)"]),
        ("2_boiled_corn_slices", assistant_prep("corn")[:4], ["pickup(corn_slices,chopping_board0)"]),
        ("3_baked_bell_pepper_soup", assistant_prep("bell_pepper")[:5], ["place_obj_on_counter()"]),
        ("3_baked_carrot_soup", soup_prefix("carrot"), ["pickup(dish,dish_dispenser)"]),
        ("3_baked_mushroom_soup", soup_prefix("mushroom") + [(1, "pickup(dish,dish_dispenser)")], ["place_obj_on_counter()"]),
        ("1_baked_sweet_potato", [(1, "pickup(sweet_potato,ingredient_dispenser)")], ["place_obj_on_counter()"]),
        ("2_baked_potato_slices", assistant_prep("potato")[:4], ["pickup(potato_slices,chopping_board0)"]),
        ("5_mashed_carrot_and_chickpea_patty", assistant_prep("carrot") + [(0, "pickup(carrot_slices,counter)"),
             (0, "put_obj_in_utensil(pot0)"), (0, "cook(pot0)"), (0, "wait(3)"),
             (0, "pickup(boiled_carrot_slices,pot0)"), (0, "place_obj_on_counter()")],
             ["pickup(boiled_carrot_slices,counter)", "pickup(chickpea,ingredient_dispenser)"]),
    ]
    fixtures = []
    for index, (name, prefix, expected) in enumerate(defs, start=1):
        order = name[2:]
        mdp = _mdp(order)
        physical = _serialize(mdp.get_standard_start_state())
        witnesses = []
        for actor, choice in prefix:
            before = _sha(physical)
            result = _execute(physical, order, actor, choice, enforce_horizon=False)
            if not result["legal"]:
                raise AssertionError((order, actor, choice, result["error"]))
            physical = result["physical"]
            witnesses.append({"actor": actor, "choice": choice, "before_physical_sha256": before,
                              "after_physical_sha256": _sha(physical), "trace": result["trace"]})
        assert physical["timestep"] < NATIVE_HORIZON
        for choice in expected:
            result = _execute(physical, order, 1, choice)
            assert result["legal"], (order, choice, result["error"])
        fixtures.append({"case_id": "collab_validation_" + f"{index:02d}", "order": order,
            "recipe_file": "src/prompts/recipe/" + name + ".txt",
            "reference_file": "src/prompts/reference/" + name + "_ref.txt",
            "physical": physical, "prefix_witness": witnesses, "reference_next_choices": expected,
            "reference_correctness_kind": "next valid operation(s) in published reference, independent of sparse reward",
            "followup_prespecified": index <= 2})
    (SOURCE / "decision_states.json").write_text(json.dumps(fixtures, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Prepared ten new native Collab decision snapshots; first two prespecified for continuation.")


if __name__ == "__main__":
    if "--prepare" in sys.argv:
        _prepare()
    else:
        print(json.dumps([{k: c[k] for k in ["case_id", "candidate", "state_sha256", "followup_prespecified"]} for c in build_cases()], indent=2))
