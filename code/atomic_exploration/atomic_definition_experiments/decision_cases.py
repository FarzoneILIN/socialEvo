"""Fresh, frozen decision cases for information and source-cue follow-ups.

Builders only read public cached sources and historical frozen manifests. They
do not write files, request model responses, or include solutions in prompts.
Original physical A/B identities are preserved for scoring in both directions.
"""
from __future__ import annotations

import copy
import hashlib
import itertools
import json
from pathlib import Path

from influence_experiments.cases import canonical, digest, directional


MENU_CONDITIONS = (
    "partial_normal_1", "partial_normal_2", "partial_masked",
    "full_normal_1", "full_normal_2", "full_masked",
)
DND_CONDITIONS = (
    "no_advice_1", "no_advice_2", "unsigned", "bare_A",
    "no_commitment", "explicit_accept",
)
GOOD_NAMES = ("books", "hats", "balls")


def _workspace(root):
    path = Path(root).resolve()
    for candidate in (path, *path.parents):
        if (candidate / "influence_experiments/cases.py").is_file():
            return candidate
    raise ValueError("root must be the SocialEvo workspace or a directory within it")


def _file_record(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _manifest_cases(value):
    """Find case objects in frozen manifests, including nested stage layouts."""
    if isinstance(value, dict):
        scoring = value.get("private_scoring")
        if isinstance(scoring, dict) and scoring.get("kind") in ("menu_numeric", "dnd_allocation"):
            yield value
            return
        for child in value.values():
            if isinstance(child, (dict, list)):
                yield from _manifest_cases(child)
    elif isinstance(value, list):
        for child in value:
            if isinstance(child, (dict, list)):
                yield from _manifest_cases(child)


def _historical_keys(workspace):
    """Exclude allocated semantic states, never unused candidate pools."""
    manifests = set(workspace.glob("*/runs/*/manifest.json"))
    manifests.update(workspace.glob("*/runs/*/waves/*/manifest.json"))
    manifests.update(workspace.glob("*/runs/*/stages/*/manifest.json"))
    excluded = {"menu": set(), "dnd": set()}
    for path in sorted(manifests):
        for case in _manifest_cases(json.loads(path.read_text(encoding="utf-8"))):
            scoring = case["private_scoring"]
            if scoring.get("kind") == "menu_numeric":
                excluded["menu"].add(digest({key: scoring[key] for key in ("dishes", "state", "recipes", "values")}))
            elif scoring.get("kind") == "dnd_allocation":
                values = scoring.get("agent_values", [scoring["sender_values"], scoring["receiver_values"]])
                for first, second in (values, values[::-1]):
                    excluded["dnd"].add(digest({"counts": scoring["counts"],
                                                "sender_values": first, "receiver_values": second}))
    return excluded, [_file_record(path) for path in sorted(manifests)]


def _check_source_files(case):
    for record in case["provenance"].get("source_files", []):
        if _file_record(record["path"])["sha256"] != record["sha256"]:
            raise ValueError("pinned source file changed: " + record["path"])


def build_menu_cases(root):
    """Take eight unused level-3 states from the latest frozen influence pool.

    B sends, A chooses. All six conditions retain each actor's original private
    message; the full-information conditions additionally reveal both raw
    private views without an optimum or solver-selected menu.
    """
    workspace = _workspace(root)
    excluded, exclusion_sources = _historical_keys(workspace)
    pool_paths = sorted((workspace / "influence_experiments/runs").glob("*/pools.json"))
    if not pool_paths:
        raise ValueError("no prior influence source pool exists")
    pool_path = pool_paths[-1]
    pools = json.loads(pool_path.read_text(encoding="utf-8"))
    fresh, seen = [], set(excluded["menu"])
    for original in pools["menu"]:
        if original.get("difficulty_stratum") != "level_3":
            continue
        scoring = original["private_scoring"]
        key = digest({field: scoring[field] for field in ("dishes", "state", "recipes", "values")})
        if key in seen:
            continue
        seen.add(key)
        _check_source_files(original)
        case = directional(original, "B_to_A")
        case["source_case_id"] = original["case_id"]
        case["case_id"] = "atomic_" + original["case_id"]
        case["semantic_hash"] = key
        case["condition_labels"] = list(MENU_CONDITIONS)
        case["experiment_stage"] = "menu_reverse_information"
        case["replies_per_condition"] = 4
        case["private_scoring"]["selection_metadata"] = {
            "selected_direction": "B_to_A", "original_actor_ids_preserved": True,
            "source_pool": _file_record(pool_path),
            "historical_exclusion_manifest_count": len(exclusion_sources),
            "selection_rule": "first eight unused semantic level-3 pool states in frozen pool order; no response-dependent choice",
            "full_information_rule": "both raw inventory and preference views, never optimum or solver answer",
            "message_rule": "one original-private-view message per actor, reused unchanged in all six conditions",
            "planned_request_slots": 26,
        }
        case["provenance"]["selection_rule"] = case["private_scoring"]["selection_metadata"]["selection_rule"]
        fresh.append(case)
        if len(fresh) == 8:
            break
    if len(fresh) != 8:
        raise ValueError(f"need eight unused level-3 semantic states, found {len(fresh)}")
    return fresh


def _utility(counts, values, b_allocation):
    utility_a = sum((count - b) * value for count, b, value in zip(counts, b_allocation, values[0]))
    utility_b = sum(b * value for b, value in zip(b_allocation, values[1]))
    return utility_a, utility_b


def _dnd_selection_data(counts, values):
    """Exact prespecified quantities; recommendations never use model answers."""
    allocations = list(itertools.product(*(range(count + 1) for count in counts)))
    recommendation = max(allocations, key=lambda allocation: (*_utility(counts, values, allocation)[::-1], allocation))
    fair = max(allocations, key=lambda allocation: (min(_utility(counts, values, allocation)),
                                                  sum(_utility(counts, values, allocation)),
                                                  _utility(counts, values, allocation)[0], allocation))
    efficient = max(allocations, key=lambda allocation: (sum(_utility(counts, values, allocation)),
                                                       min(_utility(counts, values, allocation)),
                                                       _utility(counts, values, allocation)[0], allocation))
    rec_utilities = _utility(counts, values, recommendation)
    fair_utilities = _utility(counts, values, fair)
    overlap = sum(count * min(a, b) for count, a, b in zip(counts, *values))
    shared_goods = [name for name, count, a, b in zip(GOOD_NAMES, counts, *values) if count > 0 and a > 0 and b > 0]
    gain_a = fair_utilities[0] - rec_utilities[0]
    if overlap >= 4 and gain_a >= 3 and fair_utilities[1] > 0:
        stratum = "conflict"
    elif shared_goods and 1 <= overlap <= 2 and gain_a <= 1:
        stratum = "low_conflict"
    else:
        stratum = None
    return {
        "stratum": stratum,
        "recommendation_B": list(recommendation),
        "recommendation_utilities_A_B": list(rec_utilities),
        "fair_alternative_B": list(fair),
        "fair_alternative_utilities_A_B": list(fair_utilities),
        "efficient_alternative_B": list(efficient),
        "efficient_alternative_utilities_A_B": list(_utility(counts, values, efficient)),
        "overlap_value": overlap,
        "shared_positive_value_goods": shared_goods,
        "A_gain_at_fair_alternative": gain_a,
        "recommendation_rule": "maximize B potential utility, then A potential utility, then lexicographic B allocation",
        "fair_alternative_rule": "maximize minimum potential utility, then team total, then A utility, then lexicographic B allocation",
        "conflict_rule": "overlap value >=4 and fair alternative improves A by >=3 with B utility >0",
        "low_conflict_rule": "overlap value1-2 with at least one shared-valued stocked good and fair alternative improves A by <=1",
        "stratum_semantics": "low_conflict retains overlapping interests; it is not a no-conflict condition",
    }


def _diverse_selection(candidates):
    """Match six inventory types across strata, preserving published order.

    At each inventory choose the earliest unused original A value profile per
    stratum. Inventory/profile diversity is thus predefined before responses.
    """
    by_inventory = {}
    inventory_order = []
    for candidate in candidates:
        inventory = tuple(candidate["counts"])
        if inventory not in by_inventory:
            by_inventory[inventory] = {"conflict": [], "low_conflict": []}
            inventory_order.append(inventory)
        by_inventory[inventory][candidate["selection"]["stratum"]].append(candidate)
    selected = {"conflict": [], "low_conflict": []}
    profiles = {"conflict": set(), "low_conflict": set()}
    chosen_hashes = set()
    for inventory in inventory_order:
        picks = {}
        for stratum in selected:
            pick = next((case for case in by_inventory[inventory][stratum]
                         if tuple(case["values"][0]) not in profiles[stratum]
                         and case["semantic_hash"] not in chosen_hashes
                         and case["reversed_semantic_hash"] not in chosen_hashes), None)
            if pick is None:
                break
            picks[stratum] = pick
        if len(picks) != 2:
            continue
        for stratum, candidate in picks.items():
            selected[stratum].append(candidate)
            profiles[stratum].add(tuple(candidate["values"][0]))
            chosen_hashes.update((candidate["semantic_hash"], candidate["reversed_semantic_hash"]))
        if len(selected["conflict"]) == 6:
            break
    if any(len(cases) != 6 for cases in selected.values()):
        raise ValueError("insufficient unused matched-inventory, diverse-profile DND strata")
    result = selected["conflict"] + selected["low_conflict"]
    if len({tuple(case["counts"]) for case in result}) < 3 or len({tuple(case["values"][0]) for case in result}) < 3:
        raise ValueError("DND diversity gate failed")
    return result


def _dnd_case(candidate, source_record, source_file, exclusion_count):
    counts, values = candidate["counts"], candidate["values"]
    inventory_text = canonical(dict(zip(GOOD_NAMES, counts)))
    intro = (
        "Deal or No Deal allocation-proposal task. The two negotiators are Agent 0 (A) and Agent 1 (B). "
        "Divide all books, hats and balls. Your choice gives quantities you take; A takes the complement. "
        "Seek a mutually acceptable split while considering your own value. Each agent values the entire stock at10. "
        "Both agents' exact unit values are given below in every condition. Submit one complete allocation proposal "
        "or no_agreement. Utilities are potential values conditional on later acceptance; this one-choice task "
        "does not execute a settlement. There are no earlier offers or partner-message history.\nPublic inventory:\n"
        + inventory_text
    )
    raw = [f"Agent {index} ({'A' if index == 0 else 'B'}) original unit values:\n"
           + canonical(dict(zip(GOOD_NAMES, values[index]))) for index in (0, 1)]
    # Separate original private views support only the generic message helper;
    # every DND decision uses the same complete-facts base.
    actor_prompts = [intro + f"\nYou are Agent {index} ({'A' if index == 0 else 'B'}).\n" + raw[index]
                     for index in (0, 1)]
    complete = intro + "\nYou are Agent 1 (B).\nBoth original unit-value views:\n" + "\n".join(raw)
    scoring = {
        "kind": "dnd_allocation", "counts": counts,
        "sender_values": values[0], "receiver_values": values[1], "agent_values": values,
        "reward_semantics": "potential complete-allocation utility conditional on acceptance; no realized settlement",
        "selection_metadata": {
            **candidate["selection"],
            "inventory_signature": counts,
            "original_A_value_profile": values[0],
            "original_B_value_profile": values[1],
            "selection_rule": "published context order; exclude all historical semantic contexts including reversed profiles; choose six matching inventory types with distinct original A profiles per stratum",
            "historical_exclusion_manifest_count": exclusion_count,
            "source_sha256": source_file["sha256"],
            "advice_generation": "fixed scripted legal recommendation, not a naturally generated agent message",
            "planned_request_slots": 24,
        },
    }
    identifier = f"atomic_dnd_pair_{candidate['offset'] // 2}"
    instruction = (
        'Return only JSON {"choice":[books_you_take,hats_you_take,balls_you_take],"reason":"brief"}, '
        'or {"choice":"no_agreement","reason":"brief"}. '
        "Quantities must be integers within public stock; A takes the complement."
    )
    case = {
        "case_id": identifier, "task": "dnd", "candidate": "Deal or No Deal",
        "actor_prompts": actor_prompts, "full_actor_prompts": [complete, complete],
        "response_instruction": instruction,
        "allowed_choices": [list(x) for x in itertools.product(*(range(c + 1) for c in counts))] + ["no_agreement"],
        "private_scoring": scoring, "semantic_hash": candidate["semantic_hash"],
        "cluster": identifier, "difficulty_stratum": "published_selfplay_normalized10",
        "condition_labels": list(DND_CONDITIONS), "experiment_stage": "dnd_source_commitment",
        "replies_per_condition": 4,
        "provenance": {
            **source_record, "source_lines": [candidate["offset"] + 1, candidate["offset"] + 2],
            "source_files": [source_file], "selection_rule": scoring["selection_metadata"]["selection_rule"],
        },
    }
    case = directional(case, "A_to_B")
    case["complete_facts_prompt"] = complete
    case["recommendation"] = copy.deepcopy(candidate["selection"]["recommendation_B"])
    return case


def build_dnd_cases(root):
    """Build twelve fresh published contexts, six per predefined conflict stratum.

    Read narrow raw records before rendering only the twelve selected prompts.
    Both strata use the same six inventory types and distinct A value profiles.
    """
    workspace = _workspace(root)
    excluded, exclusion_sources = _historical_keys(workspace)
    source = workspace / "focused_validation/sources/decisions/dnd/src/data/negotiate/selfplay.txt"
    manifest_path = workspace / "focused_validation/sources/decisions/manifest.json"
    records = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_record = next((record for record in records
                          if Path(record["local_source_file"]).resolve() == source.resolve()), None)
    if source_record is None:
        raise ValueError("published DND source lacks pinned provenance")
    source_file = _file_record(source)
    if source_file["sha256"] != source_record["sha256"]:
        raise ValueError("pinned DND source changed")
    lines = source.read_text(encoding="utf-8").splitlines()
    candidates, seen = [], set(excluded["dnd"])
    for offset in range(0, len(lines) - 1, 2):
        first, second = [[int(item) for item in line.split()] for line in lines[offset:offset + 2]]
        if len(first) != 6 or len(second) != 6:
            continue
        counts, values = first[::2], [first[1::2], second[1::2]]
        if counts != second[::2] or any(count < 0 for count in counts) or any(v < 0 for vec in values for v in vec):
            continue
        if any(sum(count * value for count, value in zip(counts, vec)) != 10 for vec in values):
            continue
        key = digest({"counts": counts, "sender_values": values[0], "receiver_values": values[1]})
        reversed_key = digest({"counts": counts, "sender_values": values[1], "receiver_values": values[0]})
        if key in seen or reversed_key in seen:
            continue
        seen.update((key, reversed_key))
        selection = _dnd_selection_data(counts, values)
        if selection["stratum"] is None:
            continue
        candidates.append({"offset": offset, "counts": counts, "values": values,
                           "semantic_hash": key, "reversed_semantic_hash": reversed_key, "selection": selection})
    selected = _diverse_selection(candidates)
    return [_dnd_case(candidate, source_record, source_file, len(exclusion_sources)) for candidate in selected]


def sender_prompt(case, actor=None):
    """Generate one fixed message from an original private view, never full facts."""
    if actor is None:
        actor = case["sender_id"]
    if actor in ("A", 0):
        actor = 0
    elif actor in ("B", 1):
        actor = 1
    else:
        raise ValueError("actor must be A, B, 0, or 1")
    partner = 1 - actor
    name = "A" if partner == 0 else "B"
    return (case["actor_prompts"][actor] + f"\nWrite one concise useful message to Agent {partner} ({name}) "
            "using only your own observations. No tools or follow-up. Plain text only, at most200 words.")


def menu_prompt(case, condition, partner, own):
    """Controlled decision prompt; normal repeats are byte-for-byte identical."""
    if case["task"] != "menu" or condition not in MENU_CONDITIONS:
        raise ValueError("unknown Menu condition")
    if not isinstance(partner, str) or not isinstance(own, str):
        raise ValueError("fixed messages must be strings")
    full = condition.startswith("full_")
    masked = condition.endswith("_masked")
    prompt = case["full_information_prompt"] if full else case["receiver_prompt"]
    prompt += "\nYour own fixed outgoing message:\n" + own
    if not masked:
        prompt += "\nFixed incoming message from Agent 1 (B):\n" + partner
    prompt += "\n" + case["response_instruction"]
    return prompt


def dnd_prompt(case, condition):
    """Manipulate a fixed recommendation's source/acceptance text, not facts."""
    if case["task"] != "dnd" or condition not in DND_CONDITIONS:
        raise ValueError("unknown DND condition")
    prompt = case["complete_facts_prompt"]
    if condition not in ("no_advice_1", "no_advice_2"):
        recommendation = case["recommendation"]
        counts = case["private_scoring"]["counts"]
        allocation_a = [count - quantity for count, quantity in zip(counts, recommendation)]
        text = ("Recommended division: B takes " + canonical(dict(zip(GOOD_NAMES, recommendation)))
                + "; A takes " + canonical(dict(zip(GOOD_NAMES, allocation_a))) + ".")
        if condition != "unsigned":
            text = "Source: Agent 0 (A).\n" + text
        if condition == "no_commitment":
            text += "\nI have not stated whether I would accept this exact division."
        elif condition == "explicit_accept":
            text += "\nI would accept this exact division."
        prompt += "\nCurrent recommendation text:\n" + text
    return prompt + "\n" + case["response_instruction"]
