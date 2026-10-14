"""Fresh full-information kitchen checkpoints with announced legal commitments.

Only ``kitchen_prompt`` exports model inputs. Legal prefixes, completion
witnesses and exhaustive plan/choice compatibility checks stay offline. The
compatibility strata describe physical opportunity, not a uniquely correct
next action or proof that communication is necessary to finish the recipe.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from focused_validation.adapters import collab
from influence_experiments.cases import _historical_cases


CONDITIONS = ("baseline_1", "baseline_2", "plan_P", "plan_Q")
ROOT = Path(__file__).resolve().parents[1]
POOL_RELATIVE = "influence_experiments/runs/20261007_atomic_relations_v1/pools.json"


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value):
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _substantive(physical):
    return _digest({key: physical[key] for key in ("players", "objects", "order_list")})


def _workspace(root):
    path = Path(root).resolve()
    for candidate in (path, *path.parents, ROOT):
        if (candidate / POOL_RELATIVE).exists():
            return candidate
    raise ValueError("pinned_latest_influence_kitchen_pool_missing")


def _source_record(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _initial_episode(source):
    return collab.start_episode(source)


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


def _item_semantics(item, include_position=False):
    if item is None:
        return None
    result = {key: copy.deepcopy(item[key]) for key in ("name", "catagory", "state")}
    if include_position:
        result["position"] = list(item["position"])
    return result


def _resource_signature(physical):
    """Changes in hands/world resources, including genuine processing timers.

    Walking/facing and the global clock alone do not count as resource progress.
    A resource change is not automatically progress toward the correct recipe.
    """
    return {"hands": [_item_semantics(p["held_object"]) for p in physical["players"]],
            "objects": [_item_semantics(o, True) for o in physical["objects"]],
            "order_list": list(physical["order_list"])}


def _receiver_projection(physical):
    """Assistant hand, own utensils and shared counters, excluding Chef's hand.

    Equality of this projection and action outcomes is only one-step
    compatibility equivalence; whole kitchen states and future opportunities
    can still differ after the two Chef operations.
    """
    mdp = collab._mdp(physical["order_list"][0])
    own = {mdp.from_utensil_name_to_pos(name) for name in ("chopping_board0", "blender0")}
    objects = [o for o in physical["objects"]
               if tuple(o["position"]) in own or mdp.get_terrain_type_at_pos(tuple(o["position"])) == "X"]
    return {"hand": _item_semantics(physical["players"][1]["held_object"]),
            "position": physical["players"][1]["position"],
            "orientation": physical["players"][1]["orientation"],
            "objects": [_item_semantics(o, True) for o in objects]}


def _execute_branch(case, choice, plan=None):
    state = collab.start_episode(case)
    original = copy.deepcopy(state["physical"])
    sender = None
    if plan is not None:
        sender = collab.apply_action(state, 0, plan)
        if not sender["legal"] or sender["state"]["delivered"]:
            raise ValueError("frozen_sender_plan_invalid_or_already_terminal")
        state = sender["state"]
    before_receiver = copy.deepcopy(state["physical"])
    receiver = collab.apply_action(state, 1, choice)
    end = receiver["state"]
    if end["delivered"]:
        terminal = "delivered"
    elif end["physical"]["timestep"] >= collab.NATIVE_HORIZON:
        terminal = "native_horizon_without_delivery"
    else:
        terminal = "censored_after_one_receiver_operation"
    receiver_changed = _resource_signature(before_receiver) != _resource_signature(end["physical"])
    sender_steps = sender["elapsed_steps"] if sender else 0
    _, operation, parameters = collab._parse(choice)
    completed = (receiver["legal"] and
                 (receiver["elapsed_steps"] == int(parameters[0]) if operation == "wait"
                  else collab._canonical(receiver["completed_operation"]) == choice))
    nonwait = [move for actor in (0, 1) for move in collab.legal_macros(end, actor)
               if not move.startswith("wait(")]
    return {"sender_operation": plan, "sender_legal": sender["legal"] if sender else None,
            "sender_elapsed_steps": sender_steps,
            "sender_completed_operation": sender["completed_operation"] if sender else None,
            "receiver_legal_after_plan": receiver["legal"],
            "receiver_completed_operation": receiver["completed_operation"],
            "receiver_operation_completed": bool(completed),
            "receiver_elapsed_steps": receiver["elapsed_steps"],
            "total_elapsed_steps": sender_steps + receiver["elapsed_steps"],
            "receiver_error": receiver["error"],
            "receiver_resource_state_changed": receiver_changed,
            "no_receiver_resource_progress": not receiver_changed,
            "any_resource_state_changed": _resource_signature(original) != _resource_signature(end["physical"]),
            "no_legal_nonwait_macro_at_end": not nonwait,
            "deadend_proved": False,
            "native_target_delivered": bool(end["delivered"]),
            "terminal_status": terminal,
            "terminal_censored": terminal.startswith("censored"),
            "team_reward": end["team_reward"],
            "individual_rewards": None,
            "state_after_sha256": collab._sha(end["physical"]),
            "receiver_projection": _receiver_projection(end["physical"])}


def _compatibility_profile(result):
    keys = ("receiver_legal_after_plan", "receiver_completed_operation", "receiver_operation_completed",
            "receiver_elapsed_steps", "native_target_delivered", "team_reward", "receiver_projection")
    return {key: result[key] for key in keys}


def _condition(condition):
    aliases = {"baseline1": "baseline_1", "baseline2": "baseline_2", "planP": "plan_P", "planQ": "plan_Q"}
    condition = aliases.get(condition, condition)
    if condition not in CONDITIONS:
        raise ValueError("unknown_kitchen_condition")
    return condition


def kitchen_prompt(case, condition):
    """Identical complete recipe/checkpoint; only the announced plan varies."""
    condition = _condition(condition)
    state = collab.start_episode(case)
    view = collab.observation(state, 1, "full_information")
    prompt = view["prompt"].replace("Chef alone holds the recipe;", "Both agents know the complete recipe;")
    prompt = prompt.replace("No partner planned actions or hidden recipe are disclosed through the physical snapshot.",
                            "No partner messages are encoded in the physical snapshot.")
    prompt += ("\nBoth agents know this entire physical checkpoint and the recipe. There are no earlier partner messages. "
               "Decide from this unchanged initial checkpoint. If Chef announces a committed operation, that legal "
               "operation will execute first while you stay still; your selected operation is then attempted without "
               "a second observation or repair. Without an announcement, your operation executes at this checkpoint. "
               "Permitted choices below are all initially legal Assistant operations; a Chef operation can change "
               "later feasibility. Any initially legal choice is permitted; there is no uniquely required reference action.\n"
               + view["response_instruction"])
    message = "[No partner plan message supplied.]"
    if condition in ("plan_P", "plan_Q"):
        operation = case["private_scoring"]["plans"][condition[-1]]["operation"]
        message = "I commit to execute " + operation + " before your chosen operation."
    return [{"role": "system", "content": "Use only the supplied observations and plan. No tools. Return only the requested JSON."},
            {"role": "user", "content": prompt},
            {"role": "user", "content": "Message from Chef (agent A):\n" + message}]


def score_kitchen(case, choice, condition=None):
    """Score initial legality and separate offline P/Q mechanical branches.

    These branch replays are two-operation windows, not full-game episodes.
    Compare P/Q choice distributions to measure expectation effects; their
    payoff differences also include different actual Chef actions.
    """
    if isinstance(choice, dict):
        choice = choice.get("choice")
    canonical = collab._canonical(choice)
    legal = canonical in case["allowed_choices"]
    out = {"legal": legal, "choice": canonical, "choicecategory": canonical if legal else "INVALID",
           "choice_category": canonical if legal else "INVALID", "category": canonical if legal else "INVALID",
           "score_kind": "native one-receiver-operation replay; no reference-match correctness or invented payoff"}
    if not legal:
        out.update(error="not_an_initially_legal_assistant_operation", mechanical_branches={})
        return out
    branches = {"no_plan": _execute_branch(case, canonical)}
    for name in ("P", "Q"):
        branches[name] = _execute_branch(case, canonical, case["private_scoring"]["plans"][name]["operation"])
        branches[name]["conflict_after_plan"] = not branches[name]["receiver_legal_after_plan"]
    out.update(error=None, mechanical_branches=branches,
               reward_interpretation="Different announced plans are actually executed; branch rewards are not pure text effects.")
    if condition is not None:
        selected = _condition(condition)
        label = selected[-1] if selected.startswith("plan_") else "no_plan"
        out.update(condition=selected, observed_mechanical_branch=label, plannedbranch=copy.deepcopy(branches[label]))
    return out


def _build_case(source, state, prefix, suffix, plans, stratum, pool_path):
    private = {"order": source["private_scoring"]["order"],
               "recipe_file": source["private_scoring"]["recipe_file"],
               "physical": copy.deepcopy(state["physical"]), "prefix_witness": copy.deepcopy(prefix),
               "suffix_witness": copy.deepcopy(suffix),
               "plans": {name: {"operation": operation} for name, operation in zip(("P", "Q"), plans)},
               "stratum": stratum, "reference_is_not_unique_correctness": True,
               "equivalence_scope": "Assistant one-operation legality, completion, elapsed time, sparse reward, and local resource projection only"}
    case = {"case_id": "atomic_kitchen_" + _substantive(private["physical"])[:16], "task": "kitchen",
            "candidate": "Collab-Overcooked announced-commitment probe", "direction": "A_to_B",
            "sender_id": 0, "receiver_id": 1, "sender": "Chef", "receiver": "Assistant",
            "physical_role": "shared-recipe cooking actor to shared-recipe preparation actor",
            "cluster": private["recipe_file"], "conditions": list(CONDITIONS),
            "allowed_choices": collab.legal_macros(state, 1), "private_scoring": private,
            "semantic_hash": _digest({"order": private["order"], "physical": private["physical"]}),
            "physical_semantic_hash": _substantive(private["physical"]),
            "state_sha256": collab._sha(private["physical"]),
            "provenance": {"source_commit": collab.COMMIT, "source_pool_case_id": source["case_id"],
                "source_files": source["provenance"]["source_files"] + [_source_record(pool_path), _source_record(Path(collab.__file__))],
                "selection_rule": "Unused substantive checkpoint, native legal-prefix extension where needed, matched-duration plans, exhaustive initial-choice compatibility before model responses",
                "adaptation": "Both agents receive complete recipe and physical facts. Scripted Chef commitment is delivered before any state change; its actual native operation precedes the fixed Assistant choice."}}
    table = {name: {choice: _execute_branch(case, choice, operation) for choice in case["allowed_choices"]}
             for name, operation in zip(("P", "Q"), plans)}
    plan_checks = [collab.apply_action(state, 0, operation) for operation in plans]
    if not all(result["legal"] and not result["state"]["delivered"] for result in plan_checks):
        raise ValueError("sender_plans_not_both_legal_nonterminal")
    durations = [result["elapsed_steps"] for result in plan_checks]
    if durations[0] != durations[1]:
        raise ValueError("sender_plan_durations_not_matched")
    differing = [choice for choice in case["allowed_choices"]
                 if _compatibility_profile(table["P"][choice]) != _compatibility_profile(table["Q"][choice])]
    flips = [choice for choice in differing
             if table["P"][choice]["receiver_legal_after_plan"] != table["Q"][choice]["receiver_legal_after_plan"]]
    if stratum == "compatibility_sensitive" and not flips:
        raise ValueError("sensitive_checkpoint_has_no_receiver_legality_flip")
    if stratum == "compatibility_equivalent" and differing:
        raise ValueError("equivalence_checkpoint_changes_receiver_compatibility")
    private["compatibility_certificate"] = {"plan_native_steps": durations,
        "initial_receiver_choices": len(case["allowed_choices"]), "differing_choices": differing,
        "legality_flip_choices": flips, "exhaustive_results": table,
        "necessity_not_established": True, "completion_witness_is_offline_only": True}
    return case


def build_kitchen_cases(root=ROOT):
    """Return six fresh states: three legality-sensitive, three equivalent.

    Failure to find both strata across multiple recipes raises an offline
    error; callers can freeze a documented skipped stage, never relaxed cases.
    This function makes no model/network requests and changes no source files.
    """
    workspace = _workspace(root)
    pool_path = workspace / POOL_RELATIVE
    pool = json.loads(pool_path.read_text(encoding="utf-8"))["collab"]
    excluded = set()
    for old in _historical_cases(workspace):
        scoring = old.get("private_scoring", {})
        physical = scoring.get("physical", scoring.get("snapshot", {}).get("state"))
        if physical and "players" in physical:
            excluded.add(_substantive(physical))
    selected = {"compatibility_sensitive": [], "compatibility_equivalent": []}
    recipes = {name: set() for name in selected}
    for source in pool:
        private = source["private_scoring"]
        if _substantive(private["physical"]) in excluded:
            continue
        state = _replay_prefix(source)
        recipe = private["recipe_file"]
        # Equivalent controls vary actual one-step Chef actions, never merely
        # wait durations or checkpoint clock values.
        if len(selected["compatibility_equivalent"]) < 3 and recipe not in recipes["compatibility_equivalent"]:
            operations = collab.legal_macros(state, 0)
            candidates = [operation for operation in operations if operation.startswith("pickup(baked_") and operation.endswith(",oven0)")]
            if candidates and "wait(1)" in operations:
                suffix = _replay_suffix(state, private["suffix_witness"])
                if suffix:
                    try:
                        case = _build_case(source, state, state["actions"], suffix,
                                           (candidates[0], "wait(1)"), "compatibility_equivalent", pool_path)
                    except ValueError:
                        case = None
                    if case:
                        selected["compatibility_equivalent"].append(case)
                        recipes["compatibility_equivalent"].add(recipe)
                        excluded.add(case["physical_semantic_hash"])
        if len(selected["compatibility_sensitive"]) >= 3 or recipe in recipes["compatibility_sensitive"]:
            continue
        physical = state["physical"]
        held = physical["players"][1]["held_object"]
        if held is None or held["name"] != "dish" or physical["players"][0]["held_object"] is not None:
            continue
        transfer = collab.apply_action(state, 1, "place_obj_on_counter()")
        if not transfer["legal"]:
            continue
        derived = transfer["state"]
        if _substantive(derived["physical"]) in excluded:
            continue
        operations = collab.legal_macros(derived, 0)
        candidates = [operation for operation in operations
                      if operation.startswith("pickup(") and "_slices,counter)" in operation]
        if not candidates or "wait(3)" not in operations:
            continue
        remaining = copy.deepcopy(private["suffix_witness"])
        first_assistant = next((i for i, action in enumerate(remaining) if action["actor"] == 1), None)
        if first_assistant is None or remaining[first_assistant]["choice"] != "place_obj_on_counter()":
            continue
        del remaining[first_assistant]
        suffix = _replay_suffix(derived, remaining)
        if not suffix:
            continue
        try:
            case = _build_case(source, derived, derived["actions"], suffix,
                               (candidates[0], "wait(3)"), "compatibility_sensitive", pool_path)
        except ValueError:
            continue
        selected["compatibility_sensitive"].append(case)
        recipes["compatibility_sensitive"].add(recipe)
        excluded.add(case["physical_semantic_hash"])
        if all(len(group) == 3 for group in selected.values()):
            break
    if any(len(group) != 3 for group in selected.values()) or any(len(group) < 3 for group in recipes.values()):
        raise ValueError("six_new_multi_recipe_matched_plan_checkpoints_not_feasible:" + _canonical({key: len(value) for key, value in selected.items()}))
    cases = selected["compatibility_sensitive"] + selected["compatibility_equivalent"]
    for case in cases:
        if kitchen_prompt(case, "baseline_1") != kitchen_prompt(case, "baseline_2"):
            raise AssertionError("baseline_repeat_input_changed")
    return cases
