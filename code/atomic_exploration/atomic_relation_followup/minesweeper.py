"""Frozen split-clue checkpoints, message interventions and offline scoring.

This module performs no network calls and writes no files.  Exhaustive proofs
live in case data; only allowlisted public observations reach model prompts.
"""
from __future__ import annotations

import copy
from functools import lru_cache
import hashlib
import itertools
import json
from pathlib import Path
import random
import time

from minesweeper_experiments.boards import (
    _cells, _configuration_proof, _make_state, _neighbors,
    possible_configurations, validate_state,
)


NATURAL = "minesweeper_natural"
FACT_ADVICE = "minesweeper_fact_advice"
SEED = 2026100803
SEARCH_CAP = 250_000
SEARCH_SECONDS = 80
PLACEHOLDER = "Hello teammate. I am sending this short greeting."
ACTORS = ("A", "B")
RULES = (
    "Shared split-clue Minesweeper. Coordinates are [row,column], 1-based. "
    "A revealed digit counts mines in up to eight neighboring cells, including diagonals. "
    "Every opened cell is safe; an unavailable digit is private, not zero. "
    "The public mine count is exact. Normal zero flood-fill applies; each recorded "
    "opening revealed exactly its listed cell and no additional cells. "
    "The checkpoint openings were externally supplied, not selected by an informed player. "
    "Clue ownership is fixed for this supplied checkpoint. No flags or failed openings exist. "
    "You share the objective of opening a safe cell. This is one next-cell decision, "
    "with no earlier teammate text or adaptive turn. All listed closed cells are legal "
    "choices. Prefer a cell whose safety follows from available clues; if none can "
    "be established, make your best choice. Never invent unavailable digits."
)


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _canonical_mines(mines):
    """Canonical true layout under all eight symmetries of the 4x4 square."""
    def transforms(cell):
        r, c = cell[0] - 1, cell[1] - 1
        return ((r, c), (c, 3-r), (3-r, 3-c), (3-c, r),
                (r, 3-c), (3-c, 3-r), (3-r, c), (c, r))
    return min(tuple(sorted(transforms(cell)[k] for cell in mines)) for k in range(8))


def _history(workspace):
    paths = sorted(p for folder in ("minesweeper_experiments", "minesweeper_validation")
                   for p in (workspace / folder / "runs").glob("*/manifest.json"))
    _require(bool(paths), "minesweeper_historical_manifests_missing")
    layouts, sources = set(), []
    for path in paths:
        raw = path.read_bytes()
        manifest = json.loads(raw)
        _require(isinstance(manifest.get("states"), list), "historical_states_missing")
        for state in manifest["states"]:
            _require(state["public"]["width"] == state["public"]["height"] == 4,
                     "historical_board_size_unsupported")
            layouts.add(_canonical_mines(state["private_scoring"]["true_mines"]))
        sources.append({"path": str(path.relative_to(workspace)),
                        "sha256": hashlib.sha256(raw).hexdigest(),
                        "state_records": len(manifest["states"])})
    return layouts, sources


def _certificate(state, stratum):
    """Check the legal checkpoint and save complete per-view configuration sets."""
    public, scoring = state["public"], state["private_scoring"]
    cells = set(_cells(4, 4))
    opened = [tuple(cell) for cell in public["opened"]]
    closed = [tuple(cell) for cell in public["closed"]]
    mines = {tuple(cell) for cell in scoring["true_mines"]}
    _require(public["width"] == public["height"] == 4, "board_size_changed")
    _require(public["total_mines"] == len(mines) == 3, "mine_count_changed")
    _require(len(opened) == len(set(opened)) == 4 and len(closed) == len(set(closed)) == 12,
             "checkpoint_cell_count_invalid")
    _require(set(opened).isdisjoint(closed) and set(opened) | set(closed) == cells
             and mines <= set(closed), "checkpoint_partition_invalid")
    expected_history = [{"move": i, "action": "open", "cell": list(cell), "result": "opened"}
                        for i, cell in enumerate(opened, 1)]
    _require(public["history"] == expected_history, "public_history_changed")
    _require(set(public) == {"width", "height", "total_mines", "opened", "closed", "history"},
             "public_field_not_allowlisted")
    observed = {}
    for actor in ACTORS:
        _require(len(state["private_views"][actor]) == 2, "private_clue_count_changed")
        for clue in state["private_views"][actor]:
            _require(set(clue) == {"cell", "digit"}, "private_clue_extra_field")
            cell, digit = tuple(clue["cell"]), clue["digit"]
            _require(cell in opened and cell not in observed, "clue_partition_invalid")
            _require(type(digit) is int and 1 <= digit <= 8
                     and digit == len(mines & _neighbors(4, 4, cell)), "clue_digit_invalid")
            observed[cell] = digit
    _require(set(observed) == set(opened), "missing_clue")
    all_cells = list(_cells(4, 4))
    expected_masks = {"true_mine_mask": sum(1 << all_cells.index(cell) for cell in mines),
                      "opened_mask": sum(1 << all_cells.index(cell) for cell in opened),
                      "clue_ownership_mask_A": sum(1 << all_cells.index(tuple(clue["cell"]))
                                                   for clue in state["private_views"]["A"])}
    _require(all(scoring.get(key) == value for key, value in expected_masks.items()), "private_masks_changed")
    _require(len(scoring["all_opened_clues"]) == 4
             and {tuple(clue["cell"]): clue["digit"] for clue in scoring["all_opened_clues"]} == observed,
             "all_clue_metadata_changed")
    _require(scoring["stratum"] == stratum, "state_certificate_stratum_mismatch")
    # Positive true digits guarantee replay has no omitted zero flood-fill.
    configs = {actor: possible_configurations(state, None if actor == "joint" else actor)
               for actor in (*ACTORS, "joint")}
    true_config = tuple(sorted(mines))
    _require(all(true_config in values for values in configs.values()), "true_board_inconsistent")
    proofs = {actor: _configuration_proof(state, None if actor == "joint" else actor)
              for actor in (*ACTORS, "joint")}
    own = [len(proofs[actor]["provably_safe"]) for actor in ACTORS]
    if stratum == "complementary":
        _require(own == [0, 0] and len(proofs["joint"]["provably_safe"]) == 1,
                 "strict_complementarity_changed")
        validate_state(state)
    elif stratum == "independent_safe":
        _require(all(own), "own_safe_boundary_changed")
        validate_state(state)
    else:
        _require(stratum == "asymmetric_own_safe" and sorted(own) == [0, 1]
                 and len(proofs["joint"]["provably_safe"]) == 1, "paired_partition_invalid")
    return {"legal_replay": True, "no_zero_flood_fill": True,
            "true_board_consistent_with_each_view": True,
            "configuration_counts": {actor: len(values) for actor, values in configs.items()},
            "configurations": {actor: [[list(cell) for cell in config] for config in values]
                               for actor, values in configs.items()},
            "views": proofs,
            "provably_safe": {actor: proof["provably_safe"] for actor, proof in proofs.items()},
            "provably_mined": {actor: proof["provably_mined"] for actor, proof in proofs.items()},
            "stratum": stratum}


def _alternate_partition(state):
    clues = [clue for view in state["private_views"].values() for clue in view]
    joint = _configuration_proof(state)
    for ids in itertools.combinations(range(4), 2):
        other = copy.deepcopy(state)
        other["private_views"] = {"A": [clues[i] for i in ids],
                                  "B": [clues[i] for i in range(4) if i not in ids]}
        own = [len(_configuration_proof(other, actor)["provably_safe"]) for actor in ACTORS]
        if sorted(own) != [0, 1]:
            continue
        _require(_configuration_proof(other) == joint, "joint_proof_changed")
        return other
    return None


@lru_cache(maxsize=4)
def _pool(workspace_text, historical_signature):
    workspace = Path(workspace_text)
    historical, sources = _history(workspace)
    _require(_json(sources) == historical_signature, "history_changed_during_build")
    used = set(historical)
    rng, cells = random.Random(SEED), _cells(4, 4)
    found = {"natural_complementary": [], "natural_independent_safe": [], "paired": []}
    target = {"natural_complementary": 16, "natural_independent_safe": 8, "paired": 12}
    deadline, sampled = time.monotonic() + SEARCH_SECONDS, 0
    while sampled < SEARCH_CAP and time.monotonic() < deadline:
        if all(len(found[key]) == target[key] for key in found):
            break
        sampled += 1
        order = rng.sample(cells, 4)
        owned = frozenset(rng.sample(order, 4)[:2])
        mines = frozenset(rng.sample([cell for cell in cells if cell not in order], 3))
        canonical = _canonical_mines(mines)
        if canonical in used or any(not mines & _neighbors(4, 4, cell) for cell in order):
            continue
        state = _make_state(4, 4, order, owned, mines, 0, "candidate", sampled)
        own = [len(_configuration_proof(state, actor)["provably_safe"]) for actor in ACTORS]
        other = None
        if all(own):
            key, stratum = "natural_independent_safe", "independent_safe"
            if len(found[key]) == target[key]:
                continue
        elif own == [0, 0] and len(_configuration_proof(state)["provably_safe"]) == 1:
            stratum = "complementary"
            if len(found["natural_complementary"]) < target["natural_complementary"]:
                key = "natural_complementary"
            elif len(found["paired"]) < target["paired"]:
                key, other = "paired", _alternate_partition(state)
                if other is None:
                    continue
                # The identity of the own-safe receiver is balanced before any replies.
                safe_actor = ACTORS[len(found["paired"]) % 2]
                if not _configuration_proof(other, safe_actor)["provably_safe"]:
                    for item in (state, other):
                        item["private_views"] = {"A": item["private_views"]["B"],
                                                "B": item["private_views"]["A"]}
            else:
                continue
        else:
            continue
        state["private_scoring"]["stratum"] = stratum
        all_cells = list(cells)
        for item in (state, other):
            if item is None:
                continue
            item["private_scoring"]["clue_ownership_mask_A"] = sum(
                1 << all_cells.index(tuple(clue["cell"])) for clue in item["private_views"]["A"])
            item["private_scoring"]["all_opened_clues"] = [
                copy.deepcopy(clue) for actor in ACTORS for clue in item["private_views"][actor]]
        certificate = _certificate(state, stratum)
        item = {"state": state, "certificate": certificate,
                "candidate_index": sampled, "canonical_true_layout": canonical}
        if other is not None:
            for paired_state in (state, other):
                paired_state["private_scoring"]["clue_partition_rule"] = (
                    "Legal partitions selected offline after board sampling; own-safe identity balanced before responses.")
            other["private_scoring"]["stratum"] = "asymmetric_own_safe"
            other_certificate = _certificate(other, "asymmetric_own_safe")
            _require(certificate["configurations"]["joint"] == other_certificate["configurations"]["joint"],
                     "paired_joint_configurations_changed")
            item.update(alternate_state=other, alternate_certificate=other_certificate)
        found[key].append(item)
        used.add(canonical)
    _require(all(len(found[key]) == target[key] for key in found),
             "bounded_search_incomplete:" + _json({key: len(value) for key, value in found.items()}))
    return found, sources, len(historical), sampled


def _message_id(case_id, actor, version=1, partition=None):
    prefix = "" if partition is None else f"p{partition}/"
    return f"{case_id}/{prefix}message_{ACTORS[actor]}_v{version}"


def _slot(case, phase, *, kind, condition, group, rep, sender, receiver,
          version=1, dependencies=(), **extra):
    return {"slot_id": case["case_id"] + "/" + phase, "case_id": case["case_id"],
            "module": case["module"], "kind": kind, "condition": condition,
            "group": group, "rep": rep, "sender": sender, "receiver": receiver,
            "version": version, "dependencies": list(dependencies), **extra}


def build_module(workspace: Path, module: str) -> dict:
    _require(module in (NATURAL, FACT_ADVICE), "unknown_minesweeper_module")
    workspace = Path(workspace).resolve()
    _, historical_sources = _history(workspace)
    pool, sources, historical_count, candidates = _pool(str(workspace), _json(historical_sources))
    selected = (pool["natural_complementary"] + pool["natural_independent_safe"]
                if module == NATURAL else pool["paired"])
    cases, slots = [], []
    for index, original in enumerate(selected):
        item = copy.deepcopy(original)
        case_id = f"{module}.board_{index:02d}"
        case = {"case_id": case_id, "module": module,
                "stratum": item["certificate"]["stratum"] if module == NATURAL else "paired_information_partition",
                "data": {"state": item["state"], "certificate": item["certificate"]},
                "provenance": {"generator_seed": SEED, "candidate_index": item["candidate_index"],
                               "search_candidates": candidates, "historical_sources": copy.deepcopy(sources),
                               "historical_unique_true_layouts": historical_count,
                               "canonical_true_layout": item["canonical_true_layout"],
                               "freshness": "All eight symmetries of historical and selected true layouts excluded."}}
        if module == NATURAL:
            case["data"]["placeholder_selected"] = index in (0, 1, 16, 17)
            for actor in (0, 1):
                for version in (1, 2):
                    slots.append(_slot(case, f"message_{ACTORS[actor]}_v{version}", kind="message",
                                       condition="natural_message", group=0, rep=0,
                                       sender=actor, receiver=1-actor, version=version))
                own_id = _message_id(case_id, actor)
                for version in (1, 2):
                    for group in (1, 2):
                        for rep in (1, 2, 3):
                            slots.append(_slot(case, f"{ACTORS[actor]}/normal_v{version}_g{group}_r{rep}",
                                kind="decision", condition="normal", group=group, rep=rep,
                                sender=1-actor, receiver=actor, version=version,
                                dependencies=(own_id, _message_id(case_id, 1-actor, version)),
                                analysis_family="main", analysis_group=f"N{group}"))
                for condition, groups in (("masked", (1, 2)), ("full_clues", (1,)),
                                          ("placeholder", (1,) if case["data"]["placeholder_selected"] else ())):
                    for group in groups:
                        for rep in (1, 2, 3):
                            slots.append(_slot(case, f"{ACTORS[actor]}/{condition}_g{group}_r{rep}",
                                kind="decision", condition=condition, group=group, rep=rep,
                                sender=1-actor, receiver=actor,
                                dependencies=() if condition == "full_clues" else (own_id,),
                                analysis_family={"masked": "main", "full_clues": "capability", "placeholder": "placeholder"}[condition],
                                analysis_group=f"H{group}" if condition == "masked" else str(group)))
        else:
            case["data"]["partitions"] = [item["state"], item["alternate_state"]]
            case["data"]["certificates"] = [item["certificate"], item["alternate_certificate"]]
            for partition in (0, 1):
                for actor in (0, 1):
                    slots.append(_slot(case, f"p{partition}/message_{ACTORS[actor]}_v1", kind="message",
                                       condition="structured_recommendation", group=0, rep=0,
                                       sender=actor, receiver=1-actor, partition=partition))
                    own_id, partner_id = (_message_id(case_id, actor, partition=partition),
                                          _message_id(case_id, 1-actor, partition=partition))
                    for facts, advice in itertools.product((False, True), repeat=2):
                        condition = f"facts_{int(facts)}_advice_{int(advice)}"
                        for group in (1, 2):
                            for rep in (1, 2, 3):
                                slots.append(_slot(case, f"p{partition}/{ACTORS[actor]}/{condition}_g{group}_r{rep}",
                                    kind="decision", condition=condition, group=group, rep=rep,
                                    sender=1-actor, receiver=actor, partition=partition,
                                    include_facts=facts, include_advice=advice,
                                    dependencies=(own_id, partner_id) if advice else (own_id,),
                                    analysis_family="fact_advice"))
        cases.append(case)
    return {"cases": cases, "slots": slots}


def _state(case, slot):
    if case["module"] == FACT_ADVICE:
        return case["data"]["partitions"][slot["partition"]]
    return case["data"]["state"]


def _view(state, actor, full=False):
    clues = [clue for name in (ACTORS if full else (ACTORS[actor],))
             for clue in _facts(state, ACTORS.index(name))]
    return {"you_are": ACTORS[actor], "coordinate_system": "[row,column], 1-based",
            "public": {key: copy.deepcopy(state["public"][key])
                       for key in ("width", "height", "total_mines", "opened", "closed", "history")},
            "numeric_clues_available_to_you": sorted(clues, key=lambda clue: clue["cell"])}


def _facts(state, actor):
    return [{"cell": copy.deepcopy(clue["cell"]), "digit": clue["digit"]}
            for clue in state["private_views"][ACTORS[actor]]]


def _returned(response):
    return (isinstance(response, dict) and response.get("status") == "saved"
            and response.get("finish_reason") == "stop" and isinstance(response.get("content"), str))


def _recommendation(response):
    if not _returned(response):
        return False, None
    try:
        value = json.loads(response["content"])
    except (TypeError, ValueError):
        return False, None
    if not isinstance(value, dict) or set(value) != {"recommendation"}:
        return False, None
    choice = value["recommendation"]
    valid = choice is None or (isinstance(choice, list) and len(choice) == 2
                              and all(type(n) is int for n in choice))
    return valid, choice if valid else None


def _source_valid(case, response):
    if case["module"] == FACT_ADVICE:
        return _recommendation(response)[0]
    return _returned(response) and bool(response["content"].strip())


def prepare_input(case, slot, responses: dict) -> dict:
    _require(slot["case_id"] == case["case_id"] and slot["module"] == case["module"],
             "case_slot_mismatch")
    for dependency in slot["dependencies"]:
        if dependency not in responses:
            return {"skip": "dependency_unavailable:" + dependency}
        if not _source_valid(case, responses[dependency]):
            return {"skip": "dependency_failed_or_invalid:" + dependency}
    state = _state(case, slot)
    if slot["kind"] == "message":
        actor = slot["sender"]
        if case["module"] == NATURAL:
            instruction = "Write one useful plain-text message to your teammate, at most 200 words, before receiving any teammate text."
        else:
            instruction = ('Before receiving any teammate text, return only JSON {"recommendation":[row,column]} '
                           'with one proposed next cell, or {"recommendation":null} if you choose to give no advice. '
                           'Do not include reasons or revealed-digit facts in your response. Your observed digits will be transmitted separately and accurately by the environment.')
        return {"messages": [{"role": "system", "content": "Use only supplied observations. No tools."},
                             {"role": "user", "content": RULES + "\n" + _json(_view(state, actor)) + "\n" + instruction}]}
    actor = slot["receiver"]
    messages = [{"role": "system", "content": "Use only supplied observations and messages. No tools. Return only the requested JSON."},
                {"role": "user", "content": RULES + "\n" + _json(_view(state, actor, slot["condition"] == "full_clues"))
                 + '\nReturn only JSON {"choice":[row,column],"reason":"brief"}. Choose one listed closed cell using integer coordinates.'}]
    if slot["condition"] == "full_clues":
        return {"messages": messages}
    partition = slot.get("partition")
    own_id = _message_id(case["case_id"], actor, partition=partition)
    own = responses[own_id]
    if case["module"] == NATURAL:
        own_text = own["content"]
        inbox = []
        if slot["condition"] == "normal":
            partner = responses[_message_id(case["case_id"], 1-actor, slot["version"])]
            inbox = [{"sender": ACTORS[1-actor], "text": partner["content"]}]
        elif slot["condition"] == "placeholder":
            inbox = [{"sender": ACTORS[1-actor], "text": PLACEHOLDER}]
        else:
            _require(slot["condition"] == "masked", "unknown_natural_condition")
    else:
        own_text = _json({"facts": _facts(state, actor),
                          "recommendation": _recommendation(own)[1]})
        content = {}
        if slot["include_facts"]:
            content["facts"] = _facts(state, 1-actor)
        if slot["include_advice"]:
            partner = responses[_message_id(case["case_id"], 1-actor, partition=partition)]
            recommendation = _recommendation(partner)[1]
            # Active abstention provides no recommendation text, not a hidden replacement draw.
            if recommendation is not None:
                content["recommendation"] = recommendation
        inbox = [{"sender": ACTORS[1-actor], "content": content}]
    messages.extend([{"role": "user", "content": "Your own message, written before receiving teammate text:\n" + own_text},
                     {"role": "user", "content": "Incoming teammate messages:\n" + _json(inbox)}])
    return {"messages": messages}


def score_response(case, slot, response, responses: dict) -> dict:
    state = _state(case, slot)
    result = {"format_valid": False, "legal": False, "choice": None,
              "category": "INVALID", "exact_choice_category": "INVALID", "outcomes": {}}
    if slot["kind"] == "message":
        valid = _source_valid(case, response)
        result.update(format_valid=valid, legal=valid, category="MESSAGE" if valid else "INVALID",
                      exact_choice_category="MESSAGE" if valid else "INVALID")
        if case["module"] == FACT_ADVICE and valid:
            choice = _recommendation(response)[1]
            legal = choice is None or choice in state["public"]["closed"]
            certificate = case["data"]["certificates"][slot["partition"]]
            result.update(legal=legal, choice=choice, disabled=choice is None)
            result["outcomes"] = {"recommendation_present": choice is not None,
                "recommendation_joint_proven_safe": None if choice is None or not legal else choice in certificate["provably_safe"]["joint"],
                "recommendation_own_proven_safe": None if choice is None or not legal else choice in certificate["provably_safe"][ACTORS[slot["sender"]]],
                "recommendation_actually_safe": None if choice is None or not legal else choice not in state["private_scoring"]["true_mines"]}
        return result
    if not _returned(response):
        return result
    try:
        value = json.loads(response["content"])
    except (TypeError, ValueError):
        return result
    if not isinstance(value, dict):
        return result
    choice = value.get("choice")
    valid = (isinstance(choice, list) and len(choice) == 2 and all(type(n) is int for n in choice)
             and isinstance(value.get("reason"), str) and set(value) == {"choice", "reason"})
    result.update(format_valid=valid, choice=choice if valid else None)
    if not valid or choice not in state["public"]["closed"]:
        return result
    certificate = (case["data"]["certificates"][slot["partition"]]
                   if case["module"] == FACT_ADVICE else case["data"]["certificate"])
    joint = choice in certificate["provably_safe"]["joint"]
    own = choice in certificate["provably_safe"][ACTORS[slot["receiver"]]]
    actual = choice not in state["private_scoring"]["true_mines"]
    category = ("own_proven_safe" if own else "joint_proven_safe_not_own" if joint
                else "actually_safe_not_proven" if actual else "mine")
    result.update(legal=True, category=category, exact_choice_category=_json(choice))
    result["outcomes"] = {"joint_proven_safe": joint, "own_proven_safe": own, "actually_safe": actual}
    if case["module"] == FACT_ADVICE:
        available = None
        if slot["include_advice"]:
            partner = responses.get(_message_id(case["case_id"], 1-slot["receiver"], partition=slot["partition"]))
            source_valid, recommendation = _recommendation(partner)
            available = source_valid and recommendation is not None
        result["outcomes"].update(recommendation_channel_enabled=slot["include_advice"],
            recommendation_available=available,
            recommendation_delivered=bool(slot["include_advice"] and available))
    return result


def fake_responses(module_data) -> dict:
    """Deterministic offline fixtures; never represent model-generated evidence."""
    cases = {case["case_id"]: case for case in module_data["cases"]}
    responses = {}
    for slot in module_data["slots"]:
        case = cases[slot["case_id"]]
        state = _state(case, slot)
        if slot["kind"] == "message":
            content = (_json({"facts": _facts(state, slot["sender"]),
                              "fixture_version": slot["version"]}) if case["module"] == NATURAL else
                       _json({"recommendation": state["public"]["closed"][0]}))
        else:
            certificate = (case["data"]["certificates"][slot["partition"]]
                           if case["module"] == FACT_ADVICE else case["data"]["certificate"])
            content = _json({"choice": certificate["provably_safe"]["joint"][0], "reason": "offline fixture"})
        responses[slot["slot_id"]] = {"status": "saved", "finish_reason": "stop", "content": content}
    return responses


def validate_inputs(module_data) -> dict:
    """Exhaustive metadata, certificate and synthetic-input audit without calls."""
    cases, slots = module_data["cases"], module_data["slots"]
    module = cases[0]["module"]
    expected = {NATURAL: (24, 1128, 96), FACT_ADVICE: (12, 1200, 48)}[module]
    _require((len(cases), len(slots), sum(slot["kind"] == "message" for slot in slots)) == expected,
             "module_coverage_changed")
    case_map = {case["case_id"]: case for case in cases}
    slot_map = {slot["slot_id"]: slot for slot in slots}
    _require(len(case_map) == len(cases) and len(slot_map) == len(slots), "duplicate_ids")
    layouts, safe_identities = set(), []
    for case in cases:
        _require(case["module"] == module and case["case_id"].startswith(module + "."), "case_module_mismatch")
        states = case["data"].get("partitions", [case["data"]["state"]])
        certificates = case["data"].get("certificates", [case["data"]["certificate"]])
        key = _canonical_mines(states[0]["private_scoring"]["true_mines"])
        _require(key not in layouts, "duplicate_canonical_layout")
        layouts.add(key)
        for state, certificate in zip(states, certificates):
            _require(_certificate(state, certificate["stratum"]) == certificate, "certificate_changed")
        if module == FACT_ADVICE:
            _require(case["data"]["state"] == states[0] and case["data"]["certificate"] == certificates[0],
                     "paired_base_metadata_changed")
            _require(states[0]["public"] == states[1]["public"]
                     and states[0]["private_scoring"]["true_mines"] == states[1]["private_scoring"]["true_mines"]
                     and certificates[0]["configurations"]["joint"] == certificates[1]["configurations"]["joint"],
                     "paired_physical_state_changed")
            safe_identities.append(next(actor for actor in ACTORS if certificates[1]["provably_safe"][actor]))
    if module == FACT_ADVICE:
        _require(safe_identities.count("A") == safe_identities.count("B") == 6, "safe_identity_imbalance")
    else:
        _require(sum(case["stratum"] == "complementary" for case in cases) == 16
                 and sum(case["stratum"] == "independent_safe" for case in cases) == 8,
                 "natural_strata_changed")
        _require(all(sum(case["data"]["placeholder_selected"] and case["stratum"] == label for case in cases) == 2
                     for label in ("complementary", "independent_safe")), "placeholder_strata_changed")
    responses = fake_responses(module_data)
    for slot in slots:
        _require(slot["case_id"] in case_map and slot["module"] == module
                 and slot["slot_id"].startswith(slot["case_id"] + "/"), "slot_metadata_invalid")
        _require(all(dependency in slot_map and slot_map[dependency]["kind"] == "message"
                     and slot_map[dependency]["case_id"] == slot["case_id"] for dependency in slot["dependencies"]),
                 "dependency_metadata_invalid")
    repeats = {}
    forbidden = ("true_mines", "private_scoring", "configuration_counts", "provably_safe", "stratum", "generator_seed", "candidate_index")
    for slot in slots:
        prepared = prepare_input(case_map[slot["case_id"]], slot, responses)
        _require("messages" in prepared, "synthetic_dependency_failure")
        encoded = _json(prepared)
        _require(not any(word in encoded for word in forbidden), "offline_material_leaked")
        if slot["kind"] == "decision":
            key = (slot["case_id"], slot.get("partition"), slot["receiver"], slot["condition"], slot["version"])
            _require(key not in repeats or repeats[key] == encoded, "same_input_repetition_changed")
            repeats[key] = encoded
    return {"valid": True, "module": module, "cases": len(cases), "slots": len(slots),
            "messages": expected[2], "certificates_replayed": sum(len(case["data"].get("certificates", [0])) for case in cases),
            "same_input_groups_verified": True, "proof_material_excluded": True}
