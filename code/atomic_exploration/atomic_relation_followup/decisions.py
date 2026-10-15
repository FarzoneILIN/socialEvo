"""Frozen, offline Menu and Negotiation follow-up adapters.

Only :func:`prepare_input` exports model inputs. Historical sources and scores
are read-only. No networking, credential access, caching or paid execution is
performed by this module.
"""
from __future__ import annotations

import copy
import hashlib
import io
import inspect
import itertools
import json
import math
from pathlib import Path

from task_support.decision_cases import _historical_keys, _dnd_selection_data
from task_support.public_data import RestrictedPublicDataUnpickler, plain
from task_support.cases import _menu_case, digest as historical_semantic_digest
from task_support.menu_score import _menu_score_data, _menu_optimum


MAIN_GROUPS = (("normal", 1), ("normal", 2), ("hidden", 1), ("hidden", 2))
GOOD_NAMES = ("books", "hats", "balls")
PLACEHOLDER = "This message supplies no task facts, recommendation, acceptance decision or partner plan."
SYSTEM = "Use only the supplied observations. No tools. Return only the requested JSON."
CAPS = {"menu": 592, "negotiation": 924}


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def _file(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _dependency_records():
    paths = {Path(__file__).resolve()}
    for helper in (_historical_keys, _dnd_selection_data, RestrictedPublicDataUnpickler, plain,
                   _menu_case, _menu_score_data, _menu_optimum):
        paths.add(Path(inspect.getfile(helper)).resolve())
    return [_file(path) for path in sorted(paths)]


def _excluded(workspace, module):
    historical, _ = _historical_keys(workspace)
    result = set(historical["menu" if module == "menu" else "dnd"])
    # New adapters use data instead of legacy private_scoring. Do not overlook
    # their allocated states when preparing a future independent follow-up.
    def visit(node):
        if isinstance(node, dict):
            if node.get("module") == module and isinstance(node.get("data"), dict):
                for key in ("semantic_hash", "reversed_semantic_hash"):
                    if node["data"].get(key):
                        result.add(node["data"][key])
            for value in node.values():
                if isinstance(value, (dict, list)):
                    visit(value)
        elif isinstance(node, list):
            for value in node:
                visit(value)
    for path in (workspace / "atomic_relation_followup/runs").glob("*/manifest.json"):
        visit(json.loads(path.read_text(encoding="utf-8")))
    return result


def _menu_cases(workspace):
    excluded = _excluded(workspace, "menu")
    tree_path = workspace / "quick_screen/sources/decisions/deliberative_tree.json"
    tree = json.loads(tree_path.read_text(encoding="utf-8"))
    source_root = workspace / "influence_experiments/runs/20261007_atomic_relations_v1/sources/menu"
    entries = sorted((x for x in tree["tree"] if x["path"].startswith("data/cooking/games/test_games/games/")
                      and x["path"].endswith(".pkl")), key=lambda x: ("level_3" not in x["path"], x["path"]))
    groups = {"level_3": [], "level_1_and_2": []}
    seen = set(excluded)
    fields = ("ingredients", "dishes", "recipes", "game_ingredients_state", "agent_0_obs", "agent_1_obs",
              "agent_0_values", "agent_1_values", "max_reward")
    for entry in entries:
        relative = Path(entry["path"])
        layer = relative.parent.name
        if layer not in groups:
            continue
        path = source_root / layer / relative.name
        blob = path.read_bytes()
        if hashlib.sha1(b"blob " + str(len(blob)).encode() + b"\0" + blob).hexdigest() != entry["sha"]:
            raise ValueError("menu_pinned_git_blob_mismatch")
        record = RestrictedPublicDataUnpickler(io.BytesIO(blob)).load()
        sample = {key: plain(getattr(record, key)) for key in fields}
        sample.update(upstream_game_id=path.stem, provenance={})
        legacy = _menu_case(sample, layer)
        fingerprint = legacy["semantic_hash"]
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        data = copy.deepcopy(legacy["private_scoring"])
        data.update(ingredients=sample["ingredients"], actor_prompts=legacy["actor_prompts"],
                    semantic_hash=fingerprint, repetitions=3,
                    response_instruction=legacy["response_instruction"])
        stock = data["agent_inventories"][1]
        field = next((name for name in sorted(stock) if stock[name] > 0), None)
        if field is None:
            raise ValueError("menu_stock_pressure_field_unavailable")
        wrong = copy.deepcopy(stock)
        wrong[field] += 1
        data["pressure_texts"] = {
            "correct": "My inventory partition:\n" + canonical(stock),
            "incorrect": "My inventory partition:\n" + canonical(wrong),
            "placeholder": PLACEHOLDER,
        }
        data["pressure_field"] = {"entity": field, "correct": stock[field], "incorrect": wrong[field]}
        groups[layer].append({
            "case_id": "menu_" + layer + "_" + path.stem,
            "module": "menu", "stratum": layer, "data": data,
            "provenance": {"source_commit": tree["sha"], "upstream_path": entry["path"],
                           "git_blob_sha1": entry["sha"], "source_files": [_file(path), _file(tree_path)] + _dependency_records(),
                           "adaptation": "Numeric private inventory and preference proposal; original source state retained.",
                           "selection": "All six unused level3 and four unused level1/2 semantic states; historical exclusion."},
        })
    if {key: len(value) for key, value in groups.items()} != {"level_3": 6, "level_1_and_2": 4}:
        raise ValueError("menu_requires_exactly_six_plus_four_unused_official_states")
    for members in groups.values():
        for i, case in enumerate(members):
            case["data"]["diagnostic_subset"] = i < 2
            case["data"]["second_message_subset"] = i < 2
    return groups["level_3"] + groups["level_1_and_2"]


def _negotiation_cases(workspace):
    excluded = _excluded(workspace, "negotiation")
    source = workspace / "focused_validation/sources/decisions/dnd/src/data/negotiate/selfplay.txt"
    source_manifest = workspace / "focused_validation/sources/decisions/manifest.json"
    source_file = _file(source)
    records = json.loads(source_manifest.read_text(encoding="utf-8"))
    pinned = next((record for record in records
                   if Path(record["local_source_file"]).resolve() == source.resolve()), None)
    if pinned is None or pinned["sha256"] != source_file["sha256"]:
        raise ValueError("negotiation_pinned_source_mismatch")
    lines = source.read_text(encoding="utf-8").splitlines()
    inventory_order, candidates, seen = [], {}, set(excluded)
    for offset in range(0, len(lines) - 1, 2):
        first, second = [[int(x) for x in line.split()] for line in lines[offset:offset + 2]]
        if len(first) != 6 or len(second) != 6:
            continue
        counts, values = first[::2], [first[1::2], second[1::2]]
        if counts != second[::2] or any(x < 0 for x in counts) or any(v < 0 for row in values for v in row):
            continue
        if any(sum(c * v for c, v in zip(counts, row)) != 10 for row in values):
            continue
        semantic = historical_semantic_digest({"counts": counts, "sender_values": values[0], "receiver_values": values[1]})
        reversed_semantic = historical_semantic_digest({"counts": counts, "sender_values": values[1], "receiver_values": values[0]})
        if semantic in seen or reversed_semantic in seen:
            continue
        seen.update((semantic, reversed_semantic))
        selection = _dnd_selection_data(counts, values)
        layer = selection["stratum"]
        if layer not in ("conflict", "low_conflict"):
            continue
        key = tuple(counts)
        if key not in candidates:
            candidates[key] = {"conflict": [], "low_conflict": []}
            inventory_order.append(key)
        candidates[key][layer].append({"offset": offset, "counts": counts, "values": values,
                                      "semantic_hash": semantic, "reversed_semantic_hash": reversed_semantic,
                                      "selection": selection})
    selected = {"conflict": [], "low_conflict": []}
    profiles = {layer: set() for layer in selected}
    picked_hashes = set()
    for inventory in inventory_order:
        pair = {}
        for layer in selected:
            row = next((x for x in candidates[inventory][layer] if tuple(x["values"][0]) not in profiles[layer]
                        and x["semantic_hash"] not in picked_hashes
                        and x["reversed_semantic_hash"] not in picked_hashes), None)
            if row is None:
                break
            pair[layer] = row
        if len(pair) != 2:
            continue
        for layer, row in pair.items():
            selected[layer].append(row)
            profiles[layer].add(tuple(row["values"][0]))
            picked_hashes.update((row["semantic_hash"], row["reversed_semantic_hash"]))
        if len(selected["conflict"]) == 8:
            break
    if any(len(group) != 8 for group in selected.values()):
        raise ValueError("negotiation_eight_matched_diverse_inventories_unavailable")
    cases = []
    for layer, members in selected.items():
        for i, row in enumerate(members):
            values = row["values"]
            unit_values = dict(zip(GOOD_NAMES, values[0]))
            field = next(name for name, count in zip(GOOD_NAMES, row["counts"]) if count > 0)
            wrong = copy.deepcopy(unit_values)
            wrong[field] += 1
            data = {"counts": row["counts"], "agent_values": values,
                    "recommendation": row["selection"]["recommendation_B"],
                    "semantic_hash": row["semantic_hash"], "reversed_semantic_hash": row["reversed_semantic_hash"],
                    "selection_metadata": row["selection"], "inventory_signature": list(row["counts"]),
                    "second_message_subset": i < 2, "diagnostic_subset": i < 2,
                    "pressure_texts": {"correct": "My unit values:\n" + canonical(unit_values),
                                       "incorrect": "My unit values:\n" + canonical(wrong),
                                       "placeholder": PLACEHOLDER},
                    "pressure_field": {"entity": field, "correct": unit_values[field], "incorrect": wrong[field]},
                    "response_instruction": 'Return only JSON {"choice":[books_you_take,hats_you_take,balls_you_take],"reason":"brief"}, '
                                            'or {"choice":"no_agreement","reason":"brief"}. Integer quantities only; A takes the complement.'}
            cases.append({"case_id": "negotiation_record_" + str(row["offset"] // 2),
                          "module": "negotiation", "stratum": "high_conflict" if layer == "conflict" else "low_conflict",
                          "data": data,
                          "provenance": {"source_files": [source_file, _file(source_manifest)] + _dependency_records(),
                                         "source_lines": [row["offset"] + 1, row["offset"] + 2],
                                         "selection": "Published order; unused including reversed semantic profiles; eight matched inventories and distinct A profiles in each stratum.",
                                         "source_advice": "Program-controlled legal recommendation; attribution label does not assert an actual A commitment."}})
    return cases


def _message_id(case, actor, version=1):
    return case["case_id"] + f"/message_{actor}_v{version}"


def _slot(case, kind, condition, group, rep, sender, receiver, version=1, dependencies=()):
    phase = f"{kind}_{sender}_to_{receiver}_{condition}_v{version}_g{group}_r{rep}"
    return {"slot_id": case["case_id"] + "/" + phase, "case_id": case["case_id"],
            "module": case["module"], "kind": kind, "condition": condition, "group": group,
            "rep": rep, "sender": sender, "receiver": receiver, "version": version,
            "dependencies": list(dependencies)}


def _message_slot(case, actor, version=1):
    result = _slot(case, "message", "outgoing", 0, 0, actor, 1 - actor, version)
    result["slot_id"] = _message_id(case, actor, version)
    return result


def _decision_slot(case, condition, group, rep, sender=0, receiver=1, version=1):
    dependencies = [_message_id(case, receiver)]
    if condition in ("normal", "hidden", "full_normal"):
        dependencies.append(_message_id(case, sender, version))
    result = _slot(case, "decision", condition, group, rep, sender, receiver, version, dependencies)
    if condition in ("normal", "hidden"):
        result.update(analysis_family="main", analysis_group=("N" if condition == "normal" else "H") + str(group))
        if version == 2:
            result["shared_hidden_version"] = 1
    elif condition in ("source_unsigned", "source_A"):
        result.update(analysis_family="source", analysis_group=("N" if condition == "source_unsigned" else "H") + str(group))
    elif condition in ("raw_both", "full_normal"):
        result.update(analysis_family="full_facts", analysis_group=("N" if condition == "full_normal" else "H") + str(group))
        if condition == "raw_both":
            result["additional_analysis_memberships"] = [{"analysis_family": "restore_both", "analysis_group": "N" + str(group)}]
            result["restoration_reference"] = {"analysis_family": "main", "version": 1, "groups": ["H1", "H2"]}
    elif condition in ("raw_inventory", "raw_preference"):
        result.update(analysis_family="restore_" + condition[4:], analysis_group="N" + str(group),
                      restoration_reference={"analysis_family": "main", "version": 1, "groups": ["H1", "H2"]})
    else:
        result.update(analysis_family="pressure", analysis_group=condition)
    return result


def _accept_slot(case, proposal):
    result = _slot(case, "accept", proposal["condition"], proposal["group"], proposal["rep"],
                   1, 0, proposal["version"], [proposal["slot_id"], _message_id(case, 0)])
    result["proposal_slot_id"] = proposal["slot_id"]
    for key in ("analysis_family", "analysis_group", "shared_hidden_version"):
        if key in proposal:
            result[key] = proposal[key]
    return result


def build_module(workspace: Path, module: str):
    workspace = Path(workspace).resolve()
    if module not in CAPS:
        raise ValueError("unknown_decision_module")
    cases = _menu_cases(workspace) if module == "menu" else _negotiation_cases(workspace)
    slots = []
    for case in cases:
        slots.extend(_message_slot(case, actor) for actor in (0, 1))
        if case["data"]["second_message_subset"]:
            slots.extend(_message_slot(case, actor, 2) for actor in ((0, 1) if module == "menu" else (0,)))
        local = []
        directions = ((0, 1), (1, 0)) if module == "menu" else ((0, 1),)
        for sender, receiver in directions:
            for condition, group in MAIN_GROUPS:
                local.extend(_decision_slot(case, condition, group, rep, sender, receiver) for rep in range(3))
            if case["data"]["second_message_subset"]:
                for group in (1, 2):
                    local.extend(_decision_slot(case, "normal", group, rep, sender, receiver, 2) for rep in range(3))
        if module == "menu":
            for condition in ("raw_inventory", "raw_preference", "raw_both", "full_normal"):
                for group in (1, 2):
                    local.extend(_decision_slot(case, condition, group, rep, 1, 0) for rep in range(3))
        else:
            for condition in ("source_unsigned", "source_A"):
                for group in (1, 2):
                    local.extend(_decision_slot(case, condition, group, rep) for rep in range(3))
        if case["data"]["diagnostic_subset"]:
            for condition in ("pressure_correct", "pressure_incorrect", "placeholder"):
                sender, receiver = (1, 0) if module == "menu" else (0, 1)
                local.extend(_decision_slot(case, condition, 1, rep, sender, receiver) for rep in range(3))
        slots.extend(local)
        if module == "negotiation":
            slots.extend(_accept_slot(case, slot) for slot in local)
    if len(slots) != CAPS[module] or len({slot["slot_id"] for slot in slots}) != len(slots):
        raise ValueError("decision_module_slot_coverage_mismatch")
    payload = {"cases": cases, "slots": slots}
    validate_inputs(payload)
    return payload


def _saved_content(response):
    if not isinstance(response, dict) or response.get("status") not in ("saved", "success"):
        return None
    if response.get("finish_reason") != "stop":
        return None
    content = response.get("content")
    return content if isinstance(content, str) and content.strip() else None


def _fixed_message(responses, identifier):
    return _saved_content(responses.get(identifier))


def _dnd_intro(case, actor, full=False):
    data = case["data"]
    text = ("Two negotiators, Agent 0 (A) and Agent 1 (B), divide all books, hats and balls. "
            "Seek a mutually acceptable split while considering your own value. Each agent values the entire stock at 10. "
            "A legal B proposal states quantities B takes; A takes the complement. A will later independently accept or reject. "
            "Acceptance awards each agent its exact allocated value; rejection or no_agreement awards both zero. "
            "There are no prior incoming messages or offers.\nPublic inventory:\n" + canonical(dict(zip(GOOD_NAMES, data["counts"])))
            + f"\nYou are Agent {actor} ({'A' if actor == 0 else 'B'}).")
    for index in ((0, 1) if full else (actor,)):
        text += f"\nAgent {index} original unit values:\n" + canonical(dict(zip(GOOD_NAMES, data["agent_values"][index])))
    return text


def _partner_menu_facts(case, sender, condition):
    data = case["data"]
    text = ""
    if condition in ("raw_inventory", "raw_both", "full_normal"):
        text += f"\nAdditional original inventory of Agent {sender}:\n" + canonical(data["agent_inventories"][sender])
    if condition in ("raw_preference", "raw_both", "full_normal"):
        text += f"\nAdditional original preferences of Agent {sender}:\n" + canonical(dict(zip(data["dishes"], data["agent_values"][sender])))
    return text


def _proposal_result(case, response):
    parsed, error = _parsed_json(response)
    if error:
        return _invalid(error, observed=_observed(response))
    choice = parsed.get("choice")
    format_valid = (choice == "no_agreement" or
                    isinstance(choice, list) and len(choice) == 3 and all(type(x) is int for x in choice))
    if not format_valid:
        return _invalid("proposal_choice_format", format_valid=False, choice=choice)
    data = case["data"]
    if choice == "no_agreement":
        return {"format_valid": True, "legal": True, "choice": choice, "category": "no_agreement",
                "exact_choice_category": canonical(choice), "observed_response": True,
                "outcomes": {"potential_A": 0.0, "potential_B": 0.0, "potential_team": 0.0,
                             "realized_A": 0.0, "realized_B": 0.0, "realized_team": 0.0,
                             "agreement": False, "settlement_status": "no_agreement"}}
    if any(not 0 <= x <= count for x, count in zip(choice, data["counts"])):
        return _invalid("proposal_outside_inventory", format_valid=True, choice=choice)
    allocation_a = [count - b for count, b in zip(data["counts"], choice)]
    a = float(sum(x * v for x, v in zip(allocation_a, data["agent_values"][0])))
    b = float(sum(x * v for x, v in zip(choice, data["agent_values"][1])))
    return {"format_valid": True, "legal": True, "choice": choice, "category": "allocation",
            "exact_choice_category": canonical(choice), "observed_response": True,
            "outcomes": {"allocation_A": allocation_a, "allocation_B": list(choice),
                         "potential_A": a, "potential_B": b, "potential_team": a + b,
                         "realized_A": None, "realized_B": None, "realized_team": None,
                         "agreement": None, "settlement_status": "awaiting_acceptance"}}


def prepare_input(case, slot, responses: dict):
    if slot["case_id"] != case["case_id"] or slot["module"] != case["module"]:
        raise ValueError("slot_case_mismatch")
    actor, partner = slot["sender"], slot["receiver"]
    if slot["kind"] == "message":
        base = (case["data"]["actor_prompts"][actor] if case["module"] == "menu"
                else _dnd_intro(case, actor))
        return {"messages": [{"role": "system", "content": "Send one concise useful message from your own original observations. No tools. Plain text only, at most 200 words."},
                             {"role": "user", "content": base + f"\nWrite a useful message to Agent {partner}. No incoming message has been delivered."}]}
    if slot["kind"] == "accept":
        result = _proposal_result(case, responses.get(slot["proposal_slot_id"]))
        if not result["legal"]:
            return {"skip": "proposal_invalid_or_missing"}
        if result["choice"] == "no_agreement":
            return {"skip": "no_agreement"}
        own = _fixed_message(responses, _message_id(case, 0))
        if own is None:
            return {"skip": "own_message_invalid_or_missing"}
        text = (_dnd_intro(case, 0) + "\nYour own fixed outgoing message:\n" + own
                + "\nB proposes this division:\n"
                + canonical({"A": dict(zip(GOOD_NAMES, result["outcomes"]["allocation_A"])),
                             "B": dict(zip(GOOD_NAMES, result["outcomes"]["allocation_B"]))})
                + '\nReturn only JSON {"choice":"accept" or "reject","reason":"brief"}. This is the terminal acceptance decision.')
        return {"messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}]}
    if slot["kind"] != "decision":
        raise ValueError("unknown_decision_slot_kind")
    condition = slot["condition"]
    own = _fixed_message(responses, _message_id(case, slot["receiver"]))
    if own is None:
        return {"skip": "own_message_invalid_or_missing"}
    incoming = None
    if condition in ("normal", "hidden", "full_normal"):
        incoming = _fixed_message(responses, _message_id(case, slot["sender"], slot["version"]))
        if incoming is None:
            return {"skip": "partner_message_invalid_or_missing"}
    if case["module"] == "menu":
        text = case["data"]["actor_prompts"][slot["receiver"]]
        text += _partner_menu_facts(case, slot["sender"], condition)
    else:
        text = _dnd_intro(case, 1, full=True)
    text += "\nYour own fixed outgoing message:\n" + own
    if condition in ("normal", "full_normal"):
        text += f"\nMessage from Agent {slot['sender']}:\n" + incoming
    elif condition in ("pressure_correct", "pressure_incorrect", "placeholder"):
        key = condition.split("_", 1)[1] if condition.startswith("pressure_") else "placeholder"
        text += f"\nMessage from Agent {slot['sender']}:\n" + case["data"]["pressure_texts"][key]
    elif condition.startswith("source_"):
        if case["module"] != "negotiation":
            raise ValueError("source_diagnostic_outside_negotiation")
        data = case["data"]
        b = data["recommendation"]
        a = [count - quantity for count, quantity in zip(data["counts"], b)]
        text += "\nCurrent recommendation text:\n"
        if condition == "source_A":
            text += "Source: Agent 0 (A).\n"
        elif condition != "source_unsigned":
            raise ValueError("unknown_source_diagnostic")
        text += "Recommended division: B takes " + canonical(dict(zip(GOOD_NAMES, b))) + "; A takes " + canonical(dict(zip(GOOD_NAMES, a))) + "."
    elif condition not in ("hidden", "raw_inventory", "raw_preference", "raw_both"):
        raise ValueError("unknown_decision_condition")
    text += "\n" + case["data"]["response_instruction"]
    return {"messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}]}


def _parsed_json(response):
    content = _saved_content(response)
    if content is None:
        return None, "response_missing_failed_empty_or_truncated"
    try:
        def reject_constant(_):
            raise ValueError("nonfinite_json_number")
        parsed = json.loads(content, parse_constant=reject_constant)
    except (ValueError, TypeError):
        return None, "json_parse_failure"
    if not isinstance(parsed, dict) or "choice" not in parsed:
        return None, "choice_object_missing"
    return parsed, None


def _observed(response):
    return isinstance(response, dict) and response.get("status") in ("saved", "success")


def _invalid(error, format_valid=False, choice=None, observed=True):
    return {"format_valid": format_valid, "legal": False, "choice": choice, "category": "INVALID" if observed else "MISSING",
            "exact_choice_category": "INVALID" if observed else None, "error": error, "observed_response": observed,
            "outcomes": {"realized_A": None, "realized_B": None, "realized_team": None}}


def fake_responses(payload):
    """Deterministic offline fixtures for replay verification; never paid data."""
    cases = {case["case_id"]: case for case in payload["cases"]}
    result = {}
    for slot in payload["slots"]:
        case = cases[slot["case_id"]]
        if slot["kind"] == "message":
            content = f"SENTINEL_ACTOR_{slot['sender']}_VERSION_{slot['version']}"
        else:
            choice = ("accept" if slot["kind"] == "accept" else case["data"]["optimal_choice"]
                      if case["module"] == "menu" else case["data"]["recommendation"])
            content = canonical({"choice": choice, "reason": "offline fixture only"})
        result[slot["slot_id"]] = {"status": "saved", "finish_reason": "stop", "content": content,
                                   "model": "gpt-5.4-2026-03-05",
                                   "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20}}
    return result


def score_response(case, slot, response, responses):
    if slot["kind"] == "message":
        valid = _saved_content(response) is not None
        return {"format_valid": valid, "legal": None, "choice": None,
                "category": "message" if valid else "INVALID", "exact_choice_category": None,
                "observed_response": isinstance(response, dict) and response.get("status") in ("saved", "success"),
                "outcomes": {"nonempty": valid}}
    if slot["kind"] == "accept":
        prepared = prepare_input(case, slot, responses)
        if "skip" in prepared:
            no_agreement = prepared["skip"] == "no_agreement"
            return {"format_valid": None, "legal": None, "choice": None,
                    "category": "STRUCTURAL_NO_AGREEMENT" if no_agreement else "DEPENDENCY_INVALID_OR_MISSING",
                    "exact_choice_category": None, "observed_response": False, "skip": prepared["skip"],
                    "outcomes": {"agreement": False if no_agreement else None,
                                 "realized_A": 0.0 if no_agreement else None,
                                 "realized_B": 0.0 if no_agreement else None,
                                 "realized_team": 0.0 if no_agreement else None,
                                 "settlement_status": "no_agreement" if no_agreement else "unobserved"}}
        parsed, error = _parsed_json(response)
        if error:
            return _invalid(error, observed=_observed(response))
        choice = parsed.get("choice")
        if choice not in ("accept", "reject"):
            return _invalid("acceptance_choice_format", choice=choice)
        proposal = _proposal_result(case, responses[slot["proposal_slot_id"]])
        accepted = choice == "accept"
        outcomes = copy.deepcopy(proposal["outcomes"])
        outcomes.update(agreement=accepted, realized_A=outcomes["potential_A"] if accepted else 0.0,
                        realized_B=outcomes["potential_B"] if accepted else 0.0,
                        realized_team=outcomes["potential_team"] if accepted else 0.0,
                        settlement_status="accepted" if accepted else "rejected")
        return {"format_valid": True, "legal": True, "choice": choice, "category": choice,
                "exact_choice_category": canonical(choice), "observed_response": True, "outcomes": outcomes}
    if case["module"] == "negotiation":
        return _proposal_result(case, response)
    parsed, error = _parsed_json(response)
    if error:
        return _invalid(error, observed=_observed(response))
    choice = parsed.get("choice")
    if not isinstance(choice, list) or any(type(x) is not str for x in choice):
        return _invalid("menu_choice_format", choice=choice)
    data = case["data"]
    if len(choice) != len(set(choice)) or any(x not in data["dishes"] for x in choice):
        return _invalid("menu_unknown_or_duplicate_dish", format_valid=True, choice=choice)
    choice = sorted(choice)
    feasible, raw = _menu_score_data(data["state"], data["recipes"], data["values"], [data["dishes"].index(x) for x in choice])
    quality = raw / data["max_reward"] if data["max_reward"] else float(feasible and raw == 0)
    optimal = feasible and math.isclose(raw, data["max_reward"], abs_tol=1e-7)
    category = "empty_menu" if not choice else ("nonempty_optimal" if optimal else "nonempty_suboptimal" if feasible else "infeasible")
    return {"format_valid": True, "legal": True, "choice": choice, "category": category,
            "exact_choice_category": canonical(choice), "observed_response": True,
            "outcomes": {"feasible": feasible, "optimal": optimal, "empty_menu": not choice,
                         "normalized_quality": quality, "raw_team_reward": raw, "team_reward": quality,
                         "individual_rewards": None}}


def validate_inputs(payload):
    """Check lineage, repeat/mask and template invariants using sentinel replies."""
    cases = {case["case_id"]: case for case in payload["cases"]}
    slots = payload["slots"]
    if len({slot["slot_id"] for slot in slots}) != len(slots):
        raise ValueError("duplicate_slot_id")
    by_id = {slot["slot_id"]: slot for slot in slots}
    for slot in slots:
        if slot["case_id"] not in cases or slot["module"] != cases[slot["case_id"]]["module"]:
            raise ValueError("slot_case_module_mismatch")
        if any(dep not in by_id for dep in slot["dependencies"]):
            raise ValueError("unknown_dependency")
        if slot["sender"] not in (0, 1) or slot["receiver"] != 1 - slot["sender"]:
            raise ValueError("invalid_physical_actor_direction")
        if slot["kind"] == "message":
            expected = _message_slot(cases[slot["case_id"]], slot["sender"], slot["version"])
        elif slot["kind"] == "decision":
            expected = _decision_slot(cases[slot["case_id"]], slot["condition"], slot["group"], slot["rep"],
                                      slot["sender"], slot["receiver"], slot["version"])
        elif slot["kind"] == "accept" and slot.get("proposal_slot_id") in by_id:
            expected = _accept_slot(cases[slot["case_id"]], by_id[slot["proposal_slot_id"]])
        else:
            raise ValueError("unknown_slot_kind_or_proposal")
        if any(slot[key] != expected[key] for key in ("slot_id", "case_id", "kind", "condition", "group", "rep", "sender", "receiver", "version", "dependencies")):
            raise ValueError("slot_lineage_or_dependency_tampering")
        if slot["kind"] == "accept" and (slot["proposal_slot_id"] not in slot["dependencies"]
                or by_id[slot["proposal_slot_id"]]["kind"] != "decision"):
            raise ValueError("acceptance_proposal_lineage_invalid")
    fingerprints = []
    for case in cases.values():
        _validate_case_data(case)
        local = [slot for slot in slots if slot["case_id"] == case["case_id"]]
        fake = {slot["slot_id"]: {"status": "saved", "finish_reason": "stop",
                                 "content": f"SENTINEL_ACTOR_{slot['sender']}_VERSION_{slot['version']}"}
                for slot in local if slot["kind"] == "message"}
        repeated = {}
        for slot in local:
            if slot["kind"] == "accept":
                continue
            prepared = prepare_input(case, slot, fake)
            if "skip" in prepared:
                raise ValueError("unexpected_sentinel_dependency_skip")
            fingerprints.append((slot["slot_id"], digest(prepared)))
            if slot["kind"] == "message":
                if "SENTINEL" in canonical(prepared):
                    raise ValueError("sender_received_partner_message")
                continue
            own = f"SENTINEL_ACTOR_{slot['receiver']}_VERSION_1"
            text = canonical(prepared)
            if own not in text:
                raise ValueError("receiver_own_message_changed_or_removed")
            expected = f"SENTINEL_ACTOR_{slot['sender']}_VERSION_{slot['version']}"
            if slot["condition"] in ("normal", "full_normal"):
                if expected not in text:
                    raise ValueError("normal_missing_directional_partner_message")
            elif any(f"SENTINEL_ACTOR_{slot['sender']}_VERSION_{v}" in text for v in (1, 2)):
                raise ValueError("masked_or_diagnostic_partner_message_leak")
            key = (slot["sender"], slot["receiver"], slot["condition"], slot["version"])
            if key in repeated and repeated[key] != prepared:
                raise ValueError("same_input_repeated_groups_changed")
            repeated[key] = prepared
        if case["module"] == "menu":
            for sender, receiver in ((0, 1), (1, 0)):
                normal = repeated[(sender, receiver, "normal", 1)]
                hidden = repeated[(sender, receiver, "hidden", 1)]
                marker = f"\nMessage from Agent {sender}:\nSENTINEL_ACTOR_{sender}_VERSION_1"
                stripped = copy.deepcopy(normal)
                stripped["messages"][1]["content"] = stripped["messages"][1]["content"].replace(marker, "")
                if stripped != hidden:
                    raise ValueError("menu_mask_also_changed_other_inputs")
            restored = repeated[(1, 0, "raw_both", 1)]
            full = copy.deepcopy(repeated[(1, 0, "full_normal", 1)])
            full["messages"][1]["content"] = full["messages"][1]["content"].replace("\nMessage from Agent 1:\nSENTINEL_ACTOR_1_VERSION_1", "")
            if full != restored:
                raise ValueError("menu_full_fact_world_or_prompt_changed")
            decision = next(slot for slot in local if slot["kind"] == "decision")
            answer = {"status": "saved", "finish_reason": "stop",
                      "content": canonical({"choice": case["data"]["optimal_choice"]})}
            if not math.isclose(score_response(case, decision, answer, fake)["outcomes"]["normalized_quality"], 1.0):
                raise ValueError("menu_optimal_reward_replay_failed")
        else:
            unsigned = repeated[(0, 1, "source_unsigned", 1)]
            signed = copy.deepcopy(repeated[(0, 1, "source_A", 1)])
            signed["messages"][1]["content"] = signed["messages"][1]["content"].replace("Source: Agent 0 (A).\n", "")
            if signed != unsigned:
                raise ValueError("source_label_also_changed_content")
            normal = repeated[(0, 1, "normal", 1)]
            hidden = repeated[(0, 1, "hidden", 1)]
            normal = copy.deepcopy(normal)
            normal["messages"][1]["content"] = normal["messages"][1]["content"].replace("\nMessage from Agent 0:\nSENTINEL_ACTOR_0_VERSION_1", "")
            if normal != hidden:
                raise ValueError("negotiation_mask_also_changed_other_inputs")
            acceptance_input = None
            for proposal in (slot for slot in local if slot["kind"] == "decision"):
                fake[proposal["slot_id"]] = {"status": "saved", "finish_reason": "stop",
                                            "content": canonical({"choice": case["data"]["recommendation"],
                                                                  "reason": "REASON_MUST_NOT_REACH_A",
                                                                  "expected_acceptance_probability": 0.987654321})}
            for acceptance in (slot for slot in local if slot["kind"] == "accept"):
                prepared = prepare_input(case, acceptance, fake)
                if "skip" in prepared or "REASON_MUST_NOT_REACH_A" in canonical(prepared):
                    raise ValueError("acceptance_dependency_or_reason_leak")
                if acceptance_input is not None and prepared != acceptance_input:
                    raise ValueError("acceptance_condition_leak_for_same_proposal")
                acceptance_input = prepared
                fingerprints.append((acceptance["slot_id"], digest(prepared)))
        if case["data"]["diagnostic_subset"]:
            field = case["data"]["pressure_field"]
            correct = case["data"]["pressure_texts"]["correct"]
            incorrect = case["data"]["pressure_texts"]["incorrect"]
            a, b = json.loads(correct.split("\n", 1)[1]), json.loads(incorrect.split("\n", 1)[1])
            if [key for key in a if a[key] != b.get(key)] != [field["entity"]] or set(a) != set(b):
                raise ValueError("pressure_text_changes_more_than_one_field")
    return {"ok": True, "cases": len(cases), "slots": len(slots), "sentinel_input_sha256": digest(fingerprints)}


def _validate_case_data(case):
    """Replay private factual/scoring lineage, never expose it to a model."""
    data = case["data"]
    for record in case["provenance"]["source_files"]:
        if _file(record["path"])["sha256"] != record["sha256"]:
            raise ValueError("decision_source_hash_changed")
    if case["module"] == "menu":
        source = Path(case["provenance"]["source_files"][0]["path"])
        original = RestrictedPublicDataUnpickler(io.BytesIO(source.read_bytes())).load()
        ingredients = plain(original.ingredients)
        if ingredients != data["ingredients"]:
            raise ValueError("menu_ingredient_lineage_changed")
        expected = {
            "dishes": plain(original.dishes), "state": [plain(original.game_ingredients_state).get(x, 0) for x in ingredients],
            "recipes": plain(original.recipes),
            "agent_inventories": [plain(original.agent_0_obs), plain(original.agent_1_obs)],
            "agent_values": [plain(original.agent_0_values), plain(original.agent_1_values)],
        }
        if any(data[key] != value for key, value in expected.items()):
            raise ValueError("menu_original_state_or_private_view_changed")
        values = [(a + b) / 2 for a, b in zip(*expected["agent_values"])]
        optimum, best = _menu_optimum(expected["state"], expected["recipes"], values)
        if data["values"] != values or not math.isclose(data["max_reward"], optimum, abs_tol=1e-7):
            raise ValueError("menu_reward_or_preference_tampering")
        if set(data["optimal_choice"]) != {data["dishes"][i] for i in best}:
            raise ValueError("menu_optimal_witness_changed")
        sample = {key: plain(getattr(original, key)) for key in ("ingredients", "dishes", "recipes", "game_ingredients_state", "agent_0_obs", "agent_1_obs", "agent_0_values", "agent_1_values", "max_reward")}
        sample.update(upstream_game_id=source.stem, provenance={})
        legacy = _menu_case(sample, case["stratum"])
        if data["actor_prompts"] != legacy["actor_prompts"] or data["response_instruction"] != legacy["response_instruction"]:
            raise ValueError("menu_input_factual_lineage_changed")
        semantic = historical_semantic_digest({key: data[key] for key in ("dishes", "state", "recipes", "values")})
    else:
        source = Path(case["provenance"]["source_files"][0]["path"])
        lines = source.read_text(encoding="utf-8").splitlines()
        rows = [[int(x) for x in lines[line - 1].split()] for line in case["provenance"]["source_lines"]]
        if data["counts"] != rows[0][::2] or data["counts"] != rows[1][::2] or data["agent_values"] != [row[1::2] for row in rows]:
            raise ValueError("negotiation_original_state_changed")
        selection = _dnd_selection_data(data["counts"], data["agent_values"])
        if data["recommendation"] != selection["recommendation_B"] or data["selection_metadata"] != selection:
            raise ValueError("negotiation_recommendation_or_stratum_changed")
        semantic = historical_semantic_digest({"counts": data["counts"], "sender_values": data["agent_values"][0], "receiver_values": data["agent_values"][1]})
        swapped = historical_semantic_digest({"counts": data["counts"], "sender_values": data["agent_values"][1], "receiver_values": data["agent_values"][0]})
        if swapped != data["reversed_semantic_hash"]:
            raise ValueError("negotiation_reversed_semantic_hash_changed")
    if semantic != data["semantic_hash"]:
        raise ValueError("decision_semantic_hash_changed")
