"""Historical-state exclusion and exact negotiation allocation quantities."""
from __future__ import annotations
import hashlib
import itertools
import json
from pathlib import Path
from task_support.cases import digest


GOOD_NAMES = ("books", "hats", "balls")


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
