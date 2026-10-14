"""Fresh kitchen checkpoints with a fixed physical sender operation.

Only the explicit public-view builders export model input. Prefix/suffix
witnesses and branch certificates are offline data, never model context.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
from pathlib import Path

from focused_validation.adapters import collab
from atomic_definition_experiments import kitchen_cases as historical
from influence_experiments.cases import _historical_cases

MODULE = "kitchen"
SYSTEM = "Use only the supplied current observations. No tools. Return only the requested JSON."
PERMISSIONS = {
    "0": ["pot0", "oven0", "serving_tile", "shared_counters"],
    "1": ["ingredient_dispenser", "dish_dispenser", "chopping_board0", "blender0", "shared_counters"],
}
FORBIDDEN = ("prefix_witness", "suffix_witness", "reference_next_choices",
             "reference_choices_by_actor", "compatibility_certificate", "proof_certificate")


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value):
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _file_record(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _historical_substantive(workspace):
    old = list(_historical_cases(workspace))
    for path in sorted((workspace / "atomic_definition_experiments" / "runs").glob("*/manifest.json")):
        old.extend(json.loads(path.read_text(encoding="utf-8")).get("cases", []))
    result = set()
    for case in old:
        private = case.get("private_scoring", {})
        physical = private.get("physical", private.get("snapshot", {}).get("state"))
        if physical and "players" in physical:
            result.add(historical._substantive(physical))
    return result


def _slot(case, phase, kind, condition, group=None, rep=None, version=1, dependencies=()):
    family = ("observation" if kind == "observation" else "message" if kind == "message"
              else "main" if condition in ("normal", "hidden")
              else "role" if condition in ("role_normal", "role_hidden") else "small_diagnostic")
    analysis_group = (("N" if condition in ("normal", "role_normal") else "H") + str(group)
                      if family in ("main", "role") else None)
    return {"slot_id": case["case_id"] + "/" + phase, "case_id": case["case_id"],
            "module": MODULE, "kind": kind, "condition": condition, "group": group,
            "rep": rep, "sender": 0, "receiver": 1, "version": version,
            "dependencies": list(dependencies), "analysis_family": family, "analysis_group": analysis_group}


def _message_id(case, version=1):
    return case["case_id"] + "/message_v" + str(version)


def _initial(case):
    return collab.start_episode(case["data"]["native_case"])


def _fixed_branch(case, choice):
    """All conditions execute the same actual P before the selected operation."""
    data = case["data"]
    state = _initial(case)
    sender = collab.apply_action(state, 0, data["actual_plan_P"])
    if not sender["legal"] or sender["state"]["delivered"]:
        raise ValueError("kitchen_actual_P_invalid_or_terminal")
    after_p = sender["state"]
    receiver = collab.apply_action(after_p, 1, choice)
    end = receiver["state"]
    _, operation, parameters = collab._parse(choice)
    completed = bool(receiver["legal"] and
                     (receiver["elapsed_steps"] == int(parameters[0]) if operation == "wait"
                      else collab._canonical(receiver["completed_operation"]) == choice))
    before_progress = recipe_subgoals(after_p["physical"], data["order"])
    after_progress = recipe_subgoals(end["physical"], data["order"])
    return {
        "actual_sender_operation": data["actual_plan_P"],
        "sender_operation_completed": True,
        "receiver_legal_after_plan": bool(receiver["legal"]),
        "conflict_after_plan": not bool(receiver["legal"]),
        "receiver_operation_completed": completed,
        "receiver_completed_operation": receiver["completed_operation"],
        "receiver_elapsed_steps": receiver["elapsed_steps"],
        "sender_elapsed_steps": sender["elapsed_steps"],
        "total_elapsed_steps": sender["elapsed_steps"] + receiver["elapsed_steps"],
        "duration": receiver["elapsed_steps"],
        "recipe_progress": {
            "before": before_progress, "after": after_progress,
            "new_subgoals": sorted(set(after_progress) - set(before_progress)),
            "lost_subgoals": sorted(set(before_progress) - set(after_progress)),
            "new_subgoal_count": len(set(after_progress) - set(before_progress)),
            "interpretation": "Current-state recipe predicates, not cumulative progress or full-delivery success.",
        },
        "native_target_delivered": bool(end["delivered"]),
        "team_reward": end["team_reward"], "individual_rewards": None,
        "terminal_censored": not bool(end["delivered"]),
        "terminal_status": "delivered" if end["delivered"] else "censored_after_one_receiver_operation",
        "receiver_error": receiver["error"],
        "state_after_sha256": collab._sha(end["physical"]),
    }


def recipe_subgoals(physical, order):
    """Frozen physical predicates for this one baked-vegetable-soup family.

    Moving arbitrary resources never counts. Predicates can disappear when
    food is consumed by a later stage; their counts are not a recipe score.
    """
    if not (order.startswith("baked_") and order.endswith("_soup")):
        raise ValueError("kitchen_unrecognized_recipe_family")
    ingredient = order[len("baked_"):-len("_soup")]
    slices = ingredient + "_slices"
    baked = "baked_" + slices
    mdp = collab._mdp(order)
    oven_pos = list(mdp.from_utensil_name_to_pos("oven0"))
    pot_pos = list(mdp.from_utensil_name_to_pos("pot0"))
    items = list(physical["objects"]) + [p["held_object"] for p in physical["players"] if p["held_object"]]
    flags = set()
    for item in items:
        state = item.get("state")
        ingredients = state[0] if isinstance(state, list) and state and isinstance(state[0], list) else []
        if item["name"] == slices or slices in ingredients:
            flags.add("target_slices_available")
        if item["position"] == oven_pos and slices in ingredients:
            flags.add("target_slices_in_oven")
            if len(state) > 2 and state[2] > 0:
                flags.add("target_baking_started")
        if item["name"] == baked or baked in ingredients:
            flags.add("target_baked_slices_available")
        if item["position"] == pot_pos and baked in ingredients:
            flags.add("target_baked_slices_in_pot")
            if len(state) > 2 and state[2] > 0:
                flags.add("target_cooking_started")
        if item["name"] == order and item["catagory"] == "finished":
            flags.add("target_soup_finished")
    for player in physical["players"]:
        held = player["held_object"]
        if held and held["name"] == order and held["catagory"] == "finished":
            flags.add("target_soup_held_for_transfer_or_serving")
    return sorted(flags)


def _make_case(source, state, actual_p, diagnostic_q, workspace, pool_path, stratum="boundary_control"):
    private = source["private_scoring"]
    semantic = historical._substantive(private["physical"])
    suffix = historical._replay_suffix(state, private["suffix_witness"])
    if suffix is None:
        raise ValueError("kitchen_suffix_no_longer_delivers")
    native_case = {"case_id": "followup_kitchen_" + semantic[:16], "private_scoring": {
        "order": private["order"], "recipe_file": private["recipe_file"],
        "physical": copy.deepcopy(private["physical"]),
        "prefix_witness": copy.deepcopy(private["prefix_witness"]),
        "suffix_witness": copy.deepcopy(suffix),
    }}
    allowed = collab.legal_macros(state, 1)
    facts = {"agent_id": 1, "position": copy.deepcopy(private["physical"]["players"][1]["position"]),
             "held_object": (private["physical"]["players"][1]["held_object"] or {}).get("name"),
             "permissions": list(PERMISSIONS["1"])}
    recipe = (collab.UPSTREAM / private["recipe_file"]).read_text(encoding="utf-8")
    case = {"case_id": native_case["case_id"], "module": MODULE,
            "stratum": stratum, "cluster": private["recipe_file"],
            "data": {"native_case": native_case, "order": private["order"], "recipe_file": private["recipe_file"],
                     "recipe": recipe, "physical": copy.deepcopy(private["physical"]),
                     "allowed_choices": allowed, "actual_plan_P": actual_p, "diagnostic_plan_Q": diagnostic_q,
                     "expected_observation": facts, "physical_semantic_hash": semantic,
                     "physical_state_sha256": collab._sha(private["physical"]),
                     "receiver_own_message": "", "recipe_process_families": 1},
            "provenance": {"source_case_id": source["case_id"], "source_commit": collab.COMMIT,
                           "source_files": source["provenance"]["source_files"] + [_file_record(pool_path), _file_record(Path(collab.__file__))],
                           "selection_rule": "All frozen historical physical states excluded; deterministic recipe-balanced selection before replies.",
                           "adaptation": "Both actors know recipe and public snapshot; actual sender P and P-then-receiver order fixed under every text condition."}}
    case["provenance"]["native_prefix_extension"] = source.get("native_prefix_extension", [])
    case["data"]["offline_branch_certificate"] = {choice: _fixed_branch(case, choice) for choice in allowed}
    branch_values = list(case["data"]["offline_branch_certificate"].values())
    case["data"]["offline_opportunity_summary"] = {
        "initial_legal_actions": len(allowed),
        "actions_conflicting_after_actual_P": sum(value["conflict_after_plan"] for value in branch_values),
        "distinct_receiver_durations": sorted({value["receiver_elapsed_steps"] for value in branch_values}),
        "distinct_new_subgoal_counts": sorted({value["recipe_progress"]["new_subgoal_count"] for value in branch_values}),
        "full_delivery_not_required_or_claimed": True,
    }
    return case


def _sensitive_candidates(pool, excluded, workspace, pool_path):
    """Prepare a legal shared-resource competition before any model replies.

    A staged dish is put down to free B's hand. Where that checkpoint was
    already tested, A legally picks up and returns the same prepared slices,
    producing an unused counter/location configuration. All resulting states
    keep an independently replayable complete recipe suffix.
    """
    result, seen = {}, set(excluded)
    for source in pool:
        state = historical._replay_prefix(source)
        physical = state["physical"]
        held_b = physical["players"][1]["held_object"]
        if held_b is None or held_b["name"] != "dish" or physical["players"][0]["held_object"] is not None:
            continue
        p = next((p for p in collab.legal_macros(state, 0)
                  if p.startswith("pickup(") and "_slices,counter)" in p), None)
        if p is None:
            continue
        remaining = copy.deepcopy(source["private_scoring"]["suffix_witness"])
        first_b = next((i for i, action in enumerate(remaining) if action["actor"] == 1), None)
        if first_b is None or remaining[first_b]["choice"] != "place_obj_on_counter()":
            continue
        del remaining[first_b]
        transfer = collab.apply_action(state, 1, "place_obj_on_counter()")
        if not transfer["legal"]:
            continue
        state = transfer["state"]
        extension = [{"actor": 1, "choice": "place_obj_on_counter()"}]
        for round_trip in (False, True):
            if round_trip:
                take = collab.apply_action(state, 0, p)
                if not take["legal"]:
                    continue
                put_back = collab.apply_action(take["state"], 0, "place_obj_on_counter()")
                if not put_back["legal"]:
                    continue
                state = put_back["state"]
                extension += [{"actor": 0, "choice": p}, {"actor": 0, "choice": "place_obj_on_counter()"}]
            semantic = historical._substantive(state["physical"])
            if semantic in seen or p not in collab.legal_macros(state, 1):
                continue
            suffix = historical._replay_suffix(state, remaining)
            if suffix is None:
                continue
            after_p = collab.apply_action(state, 0, p)
            if not after_p["legal"] or collab.apply_action(after_p["state"], 1, p)["legal"]:
                continue
            derived = copy.deepcopy(source)
            derived["private_scoring"].update(physical=copy.deepcopy(state["physical"]),
                                              prefix_witness=copy.deepcopy(state["actions"]), suffix_witness=suffix)
            derived["native_prefix_extension"] = copy.deepcopy(extension)
            case = _make_case(derived, state, p, "wait(1)", workspace, pool_path, "compatibility_sensitive")
            if case["data"]["offline_opportunity_summary"]["actions_conflicting_after_actual_P"] < 1:
                raise AssertionError("kitchen_sensitive_state_has_no_conflict_opportunity")
            result.setdefault(derived["private_scoring"]["recipe_file"], []).append(case)
            seen.add(semantic)
    return result


def _balanced_selection(groups, count):
    selected = []
    for offset in range(max((len(group) for group in groups.values()), default=0)):
        for recipe in sorted(groups):
            if offset < len(groups[recipe]):
                selected.append(groups[recipe][offset])
                if len(selected) == count:
                    return selected
    return selected


def build_module(workspace: Path, module="kitchen"):
    if module != MODULE:
        raise ValueError("kitchen_module_name_mismatch")
    workspace = Path(workspace).resolve()
    pool_path = workspace / historical.POOL_RELATIVE
    pool = json.loads(pool_path.read_text(encoding="utf-8"))["collab"]
    excluded = _historical_substantive(workspace)
    sensitive_pool = _sensitive_candidates(pool, excluded, workspace, pool_path)
    sensitive = _balanced_selection(sensitive_pool, 6)
    if len(sensitive) != 6 or len(sensitive_pool) != 5:
        raise ValueError("kitchen_six_fresh_sensitive_five_recipe_checkpoints_unavailable")
    excluded.update(case["data"]["physical_semantic_hash"] for case in sensitive)
    candidates = {}
    for source in pool:
        private = source["private_scoring"]
        semantic = historical._substantive(private["physical"])
        if semantic in excluded:
            continue
        state = historical._replay_prefix(source)
        operations = collab.legal_macros(state, 0)
        p_options = private["reference_choices_by_actor"][0] + operations
        actual_p = next((p for p in p_options if p in operations and not p.startswith("wait(")
                         and not collab.apply_action(state, 0, p)["state"]["delivered"]), None)
        if actual_p is None:
            continue
        diagnostic_q = next((p for p in operations if p != actual_p and p == "wait(1)"), None)
        diagnostic_q = diagnostic_q or next((p for p in operations if p != actual_p), None)
        if diagnostic_q is None:
            continue
        case = _make_case(source, state, actual_p, diagnostic_q, workspace, pool_path)
        if case["data"]["offline_opportunity_summary"]["actions_conflicting_after_actual_P"]:
            continue
        candidates.setdefault(private["recipe_file"], []).append(case)
        excluded.add(semantic)
    if len(candidates) != 5:
        raise ValueError("kitchen_expected_five_fresh_recipe_files")
    # Spread across native phases within each recipe as well as across recipes.
    for recipe, group in candidates.items():
        group.sort(key=lambda c: (c["data"]["physical"]["timestep"], c["case_id"]))
        ordered = []
        for index in (0, len(group) // 2, len(group) - 1, *range(len(group))):
            if group[index] not in ordered:
                ordered.append(group[index])
        candidates[recipe] = ordered
    controls = _balanced_selection(candidates, 6)
    cases = sensitive + controls
    if len(cases) != 12:
        raise ValueError("kitchen_twelve_fresh_checkpoints_unavailable")
    diagnostic_ids = {c["case_id"] for c in sensitive[:2] + controls[:2]}
    slots = []
    for case in cases:
        case["data"]["diagnostic_selected"] = case["case_id"] in diagnostic_ids
        slots.append(_slot(case, "message_v1", "message", "fixed_plan_message", version=1))
        for condition in ("normal", "hidden"):
            for group in (1, 2):
                for rep in range(3):
                    phase = f"{condition}_{group}__{rep}"
                    slots.append(_slot(case, phase, "decision", condition, group, rep,
                                       dependencies=[_message_id(case)] if condition == "normal" else []))
        for representation in ("original", "explicit"):
            slots.append(_slot(case, "observation_" + representation, "observation", representation))
        if case["case_id"] not in diagnostic_ids:
            continue
        slots.append(_slot(case, "message_v2", "message", "fixed_plan_message", version=2))
        for group in (1, 2):
            for rep in range(3):
                slots.append(_slot(case, f"normal_v2_{group}__{rep}", "decision", "normal", group, rep, 2,
                                   dependencies=[_message_id(case, 2)]))
        for condition in ("role_normal", "role_hidden"):
            for group in (1, 2):
                for rep in range(3):
                    slots.append(_slot(case, f"{condition}_{group}__{rep}", "decision", condition, group, rep,
                                       dependencies=[_message_id(case)] if condition == "role_normal" else []))
        for condition in ("matched_correct", "matched_wrong", "placeholder"):
            for rep in range(3):
                slots.append(_slot(case, f"{condition}__{rep}", "decision", condition, None, rep))
    if len(slots) != 292 or len({slot["slot_id"] for slot in slots}) != 292:
        raise AssertionError("kitchen_slot_count_changed")
    return {"cases": cases, "slots": slots}


def _response_json(response):
    if not isinstance(response, dict) or response.get("status") not in ("saved", "success", "ok"):
        return None, "request_not_successful"
    if response.get("finish_reason") != "stop":
        return None, "response_not_stopped"
    try:
        value = json.loads(response.get("content", ""))
    except (ValueError, TypeError):
        return None, "response_not_json"
    return (value, None) if isinstance(value, dict) else (None, "response_not_object")


def _message_text(response):
    value, error = _response_json(response)
    if error is not None or not isinstance(value.get("message"), str) or not value["message"].strip():
        return None
    return value["message"]


def _public_prompt(case, actor=1, role_labels=False, explicit=True):
    data = case["data"]
    if not explicit:
        return collab.observation(_initial(case), actor, "full_information")["prompt"]
    alias = ("Chef" if actor == 0 else "Assistant") if role_labels else ("Agent 0" if actor == 0 else "Agent 1")
    other_alias = ("Assistant" if actor == 0 else "Chef") if role_labels else ("Agent 1" if actor == 0 else "Agent 0")
    own = data["physical"]["players"][actor]
    own_facts = {"agent_id": actor, "display_name": alias, "position": own["position"],
                 "held_object": own["held_object"], "permissions": PERMISSIONS[str(actor)]}
    roles = {"0": "Chef", "1": "Assistant"} if role_labels else {"0": "Agent 0", "1": "Agent 1"}
    return (
        "Collab-Overcooked. Complete the displayed order together. Both agents know the complete recipe.\n"
        "YOUR CURRENT STATE (player index equals agent_id; do not confuse your partner's held item with yours):\n"
        + _canonical(own_facts) + "\nDisplayed actor names:\n" + _canonical(roles)
        + "\nActual operation permissions by actor index:\n" + _canonical(PERMISSIONS)
        + "\nEach actor holds at most one item. Exchange items through shared counters. Processing starts only after "
          "cut/cook/bake/stir, and timers advance during movement and waiting. Deterministic navigation leaves "
          "the other actor still during an operation. The allowed operations show initial physical legality, "
          "not a reference answer.\nPublic physical state:\n" + _canonical(data["physical"])
        + "\nComplete recipe:\n" + data["recipe"]
        + "\nYou are " + alias + "; your partner is " + other_alias + ". No earlier partner messages exist. "
          "The decision uses this unchanged snapshot. Agent 0 will perform a fixed legal operation first, "
          "whether or not its plan message is visible. Your chosen operation then runs without a new "
          "observation or repair. The snapshot does not disclose the unexecuted operation.\n"
    )


def validate_inputs(payload):
    if "cases" in payload and "slots" in payload:
        cases = payload["cases"]
        slots = payload["slots"]
        by_id = {case["case_id"]: case for case in cases}
        slot_ids = {slot["slot_id"] for slot in slots}
        if len(cases) != 12 or len(by_id) != 12 or len(slots) != 292 or len(slot_ids) != 292:
            raise ValueError("kitchen_module_coverage_changed")
        if any(case["module"] != MODULE for case in cases) or any(slot["module"] != MODULE for slot in slots):
            raise ValueError("kitchen_module_identity_changed")
        if any(slot["case_id"] not in by_id or not set(slot["dependencies"]) <= slot_ids for slot in slots):
            raise ValueError("kitchen_module_dependency_changed")
        strata = [case["stratum"] for case in cases]
        if strata.count("compatibility_sensitive") != 6 or strata.count("boundary_control") != 6:
            raise ValueError("kitchen_module_strata_changed")
        for case in cases:
            verify_case(case)
        responses = {}
        for slot in slots:
            if slot["kind"] == "message":
                case = by_id[slot["case_id"]]
                responses[slot["slot_id"]] = fake_responses(prepare_input(case, slot, {}))
        for slot in slots:
            prepared = prepare_input(by_id[slot["case_id"]], slot, responses)
            if "skip" in prepared:
                raise ValueError("kitchen_offline_planned_slot_unreachable")
        return {"ok": True, "cases": len(cases), "slots": len(slots), "offline_only": True}
    if "skip" in payload:
        return True
    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages or any(
            not isinstance(message, dict) or message.get("role") not in ("system", "user", "assistant")
            or not isinstance(message.get("content"), str) for message in messages):
        raise ValueError("kitchen_invalid_message_payload")
    serialized = _canonical(messages)
    if any(marker in serialized for marker in FORBIDDEN):
        raise ValueError("kitchen_offline_witness_leaked")
    return True


def verify_case(case):
    """Replay private feasibility and scoring certificates without model calls."""
    data = case["data"]
    native = data["native_case"]
    replayed = historical._replay_prefix(native)
    if replayed["physical"] != data["physical"]:
        raise ValueError("kitchen_frozen_public_state_changed")
    if historical._substantive(data["physical"]) != data["physical_semantic_hash"]:
        raise ValueError("kitchen_semantic_hash_mismatch")
    if collab._sha(data["physical"]) != data["physical_state_sha256"]:
        raise ValueError("kitchen_physical_hash_mismatch")
    if collab.legal_macros(replayed, 1) != data["allowed_choices"]:
        raise ValueError("kitchen_legal_actions_changed")
    suffix = historical._replay_suffix(replayed, native["private_scoring"]["suffix_witness"])
    if suffix != native["private_scoring"]["suffix_witness"]:
        raise ValueError("kitchen_suffix_replay_changed")
    for choice in data["allowed_choices"]:
        if _fixed_branch(case, choice) != data["offline_branch_certificate"][choice]:
            raise ValueError("kitchen_branch_certificate_changed")
    expected = data["expected_observation"]
    own = data["physical"]["players"][1]
    if expected != {"agent_id": 1, "position": own["position"],
                    "held_object": (own["held_object"] or {}).get("name"), "permissions": PERMISSIONS["1"]}:
        raise ValueError("kitchen_fact_check_gold_changed")
    recipe = (collab.UPSTREAM / data["recipe_file"]).read_text(encoding="utf-8")
    if data["recipe"] != recipe:
        raise ValueError("kitchen_recipe_text_changed")
    return {"prefix_replayed": True, "suffix_delivers_offline": True,
            "fixed_P_branches_replayed": len(data["allowed_choices"]), "public_facts_verified": True}


def prepare_input(case, slot, responses):
    if slot["case_id"] != case["case_id"] or slot["module"] != MODULE:
        raise ValueError("kitchen_slot_case_mismatch")
    data, kind, condition = case["data"], slot["kind"], slot["condition"]
    if kind == "message":
        prompt = (_public_prompt(case, actor=0)
                  + "\nYour own fixed next operation is " + data["actual_plan_P"] + ". It will execute before Agent 1's operation. "
                  "Write one concise message communicating that commitment to Agent 1. Do not change the operation, "
                  "invent earlier messages or claim you observed a future state. Return only JSON {\"message\":\"your message\"}.")
        payload = {"messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]}
    elif kind == "observation":
        prompt = _public_prompt(case, explicit=condition == "explicit")
        prompt += ("\nReport only your CURRENT facts, not a next action. Return JSON with exactly "
                   "agent_id (integer), position ([x,y]), held_object (item name or null), "
                   "permissions (list of your resource/operation-area names).")
        payload = {"messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]}
    elif kind == "decision":
        message = "[No partner plan message supplied.]"
        if condition in ("normal", "role_normal"):
            saved = responses.get(_message_id(case, slot["version"]))
            partner = _message_text(saved)
            if partner is None:
                return {"skip": "fixed_sender_message_failed_or_invalid"}
            message = partner
        elif condition in ("hidden", "role_hidden"):
            pass
        elif condition in ("matched_correct", "matched_wrong"):
            operation = data["actual_plan_P"] if condition == "matched_correct" else data["diagnostic_plan_Q"]
            message = "I commit to execute " + operation + " before your chosen operation."
        elif condition == "placeholder":
            message = "This is a short communication placeholder with no kitchen facts or action recommendation."
        else:
            raise ValueError("kitchen_unknown_decision_condition")
        roles = condition.startswith("role_")
        prompt = _public_prompt(case, role_labels=roles)
        prompt += ("\nReturn only JSON {\"choice\":\"one exact permitted operation\",\"reason\":\"brief\"}. "
                   "Permitted choices: " + _canonical(data["allowed_choices"]) + ".")
        sender_name = "Chef (agent 0)" if roles else "Agent 0"
        payload = {"messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt},
                                {"role": "user", "content": "Message from " + sender_name + ":\n" + message}]}
    else:
        raise ValueError("kitchen_unknown_slot_kind")
    validate_inputs(payload)
    return payload


def _category(choice):
    operation = collab._parse(choice)[1]
    if operation == "wait":
        return "wait"
    if operation == "pickup":
        return "pickup"
    if operation in ("place_obj_on_counter", "put_obj_in_utensil", "fill_dish_with_food"):
        return "transfer"
    if operation in ("cut", "stir", "cook", "bake"):
        return "process"
    if operation == "deliver_soup":
        return "deliver"
    return "other"


def score_response(case, slot, response, responses):
    if isinstance(response, dict) and response.get("status") in ("skipped", "failed", "failure"):
        return {"format_valid": None, "legal": None, "choice": None,
                "category": "MISSING", "exact_choice_category": "MISSING", "outcomes": {},
                "error": response.get("error") or response["status"]}
    value, error = _response_json(response)
    result = {"format_valid": error is None, "legal": False, "choice": None,
              "category": "INVALID", "exact_choice_category": "INVALID", "outcomes": {}, "error": error}
    if error is not None:
        return result
    if slot["kind"] == "message":
        text = value.get("message")
        valid = isinstance(text, str) and bool(text.strip())
        return dict(result, format_valid=valid, legal=valid, choice=text if valid else None,
                    category="message" if valid else "INVALID", exact_choice_category="message" if valid else "INVALID",
                    outcomes={"actual_plan_mentioned": bool(valid and case["data"]["actual_plan_P"] in text)},
                    error=None if valid else "empty_or_missing_message")
    if slot["kind"] == "observation":
        keys = {"agent_id", "position", "held_object", "permissions"}
        valid = (set(value) == keys and type(value["agent_id"]) is int
                 and isinstance(value["position"], list) and len(value["position"]) == 2
                 and all(type(n) is int for n in value["position"])
                 and (value["held_object"] is None or isinstance(value["held_object"], str))
                 and isinstance(value["permissions"], list)
                 and all(isinstance(p, str) for p in value["permissions"]))
        expected = case["data"]["expected_observation"]
        correctness = {key: (sorted(value[key]) == sorted(expected[key]) if key == "permissions"
                             else value[key] == expected[key]) for key in keys} if valid else {}
        return dict(result, format_valid=valid, legal=valid, choice=value,
                    category="facts" if valid else "INVALID", exact_choice_category="facts" if valid else "INVALID",
                    outcomes={"fact_correctness": correctness, "all_facts_correct": bool(valid and all(correctness.values()))},
                    error=None if valid else "invalid_fact_check_format")
    raw = value.get("choice")
    format_valid = isinstance(raw, str) and bool(raw.strip())
    choice = collab._canonical(raw)
    legal = format_valid and choice in case["data"]["allowed_choices"]
    result.update(format_valid=format_valid, legal=bool(legal), choice=choice,
                  error=None if legal else "not_initially_legal_operation")
    if not legal:
        return result
    result.update(category=_category(choice), exact_choice_category=choice, outcomes=_fixed_branch(case, choice))
    return result


def fake_responses(payload):
    """Offline plumbing fixture only; it does not simulate a model's policy."""
    if "skip" in payload:
        return {"status": "skipped", "error": payload["skip"]}
    text = "\n".join(message["content"] for message in payload["messages"])
    decoder = json.JSONDecoder()
    if "Your own fixed next operation is " in text:
        operation = re.search(r"Your own fixed next operation is ([^\s]+)\.", text).group(1)
        content = {"message": "I commit to execute " + operation + " before your chosen operation."}
    elif "Report only your CURRENT facts" in text:
        marker = "YOUR CURRENT STATE (player index equals agent_id; do not confuse your partner's held item with yours):\n"
        if marker in text:
            own = decoder.raw_decode(text.split(marker, 1)[1])[0]
            held = own["held_object"]
            content = {"agent_id": own["agent_id"], "position": own["position"],
                       "held_object": held["name"] if held else None, "permissions": own["permissions"]}
        else:
            physical = decoder.raw_decode(text.split("Public physical state:\n", 1)[1])[0]
            own = physical["players"][1]
            content = {"agent_id": 1, "position": own["position"],
                       "held_object": own["held_object"]["name"] if own["held_object"] else None,
                       "permissions": PERMISSIONS["1"]}
    else:
        choices = decoder.raw_decode(text.split("Permitted choices: ", 1)[1])[0]
        content = {"choice": "wait(1)" if "wait(1)" in choices else choices[0], "reason": "offline fixture"}
    return {"status": "saved", "finish_reason": "stop", "content": _canonical(content)}
