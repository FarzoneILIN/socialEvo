"""Four pinned decision-task adaptations for the bounded one-message screen.

Upstream instances and numerical objectives are retained. These are decision
probes, never native benchmark completion scores. Only the explicit prompts are
model inputs; private_scoring and source snapshots are offline records.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
from pathlib import Path


SOURCE_DIR = Path(__file__).resolve().parents[1] / "sources" / "decisions"
SENDER_INSTRUCTION = (
    "Write one brief useful message to Agent 1 using only your own information. "
    "You cannot inspect their private information, ask follow-up questions, or use tools. "
    "Return only the message text, no JSON wrapper, at most 200 words."
)


def _dump(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _provenance(sample, adaptation):
    result = sample["provenance"].copy()
    result["local_source_file"] = str(Path(result["local_source_file"]).resolve())
    result.update(adaptation=adaptation, native_agent_count=2,
                  measurement_scope="one focal receiver decision; no full dialogue or agreement evaluation")
    extracted = SOURCE_DIR / "samples.json"
    result["source_files"] = [
        {"path": result["local_source_file"], "sha256": result["sha256"]},
        {"path": str(extracted), "sha256": hashlib.sha256(extracted.read_bytes()).hexdigest()},
    ]
    return result


def _common(case_id, candidate, sender, receiver, full, response, provenance, scoring):
    return {"case_id": case_id, "candidate": candidate, "sender_id": 0, "receiver_id": 1,
            "sender_prompt": sender + "\n\n" + SENDER_INSTRUCTION,
            "receiver_prompt": receiver, "full_information_prompt": full,
            "response_instruction": response, "provenance": provenance,
            "private_scoring": scoring}


def _assignment_optimum(table):
    n = len(table)
    return max((sum(table[r][c] for r, c in enumerate(cols)), list(cols))
               for cols in itertools.permutations(range(n)))


def _dialop_cases(samples):
    cases = []
    for i, sample in enumerate(samples, 1):
        table = sample["table"]
        n = len(table)
        # Upstream reset replaces values unknown to both players by prior mean 50.
        pooled = [[table[r][c] if sample["mask1"][r][c] or sample["mask2"][r][c]
                   else 50 for c in range(n)] for r in range(n)]
        optimum, best = _assignment_optimum(pooled)
        assert math.isclose(optimum, sample["best_assignment_reward"], abs_tol=1e-8)
        views = []
        for mask, scale in ((sample["mask1"], sample["scale1"]),
                            (sample["mask2"], sample["scale2"])):
            views.append([[int(table[r][c] * scale) if mask[r][c] else None
                           for c in range(n)] for r in range(n)])
        intro = (
            f"DialOp Assignment decision probe: assign {n} workers to {n} papers, maximizing joint matching quality. "
            "Rows are worker indices and columns paper indices, both starting at 0. Each worker must have exactly "
            "one paper, and each paper exactly one worker. You and your partner see different cells. null means unknown. "
            "Each private table uses a different positive scale, so compare relative values and align them using overlapping "
            "observations rather than adding raw numbers from different agents. Unknown to both has unscaled prior mean 50. "
            "There is one partner message, then Agent 1 proposes a complete matching. This evaluates the proposal's joint "
            "quality; it does not simulate proposal acceptance or a full dialogue."
        )
        sender = intro + "\nYou are Agent 0. Your private scaled table:\n" + _dump(views[0])
        receiver = intro + "\nYou are Agent 1. Your private scaled table:\n" + _dump(views[1])
        full = (intro + "\nYou are Agent 1. For this centralized capability reference, pooled observations "
                "are presented on the common unscaled 0-100 scale; all missing pooled cells are filled by prior mean 50:\n"
                + _dump(pooled))
        response = ('Return one JSON object {"choice": [paper_index_for_worker_0, ...], "reason": "brief"}. '
                    f"choice must contain {n} distinct integers from 0 to {n - 1}, in worker-index order.")
        cases.append(_common(
            f"dialop_assignment_row{i}_a0_to_a1", "DialOp Assignment", sender, receiver, full, response,
            _provenance(sample,
                "Published optimization.jsonl records 1 and 2, original 8x8 tables, masks and integer-scaled observations retained. "
                "Original named entities replaced by indices; a one-message complete matching proposal replaces negotiation. "
                "Normalized proposal score uses the pooled-information optimum (unknown cells=50), not omniscient hidden truth. "
                "Full-information LLM reference gets pooled unscaled values, not the optimum answer."),
            {"kind": "dialop_assignment", "pooled_table": pooled, "max_reward": optimum,
             "optimal_choice": best, "reward_semantics": "joint matching proposal quality, no agreement evaluated"}))
    return cases


def _menu_score_data(state, recipes, values, selection):
    used = [sum(recipes[d][j] for d in selection) for j in range(len(state))]
    feasible = all(a <= b + 1e-9 for a, b in zip(used, state))
    return feasible, sum(values[d] for d in selection) if feasible else 0.0


def _menu_optimum(state, recipes, values):
    possible = [d for d, recipe in enumerate(recipes) if all(a <= b + 1e-9 for a, b in zip(recipe, state))]
    # Exhaustive enumeration of every potentially feasible dish subset is exact.
    best_reward, best_menu = 0.0, []
    for flags in itertools.product((0, 1), repeat=len(possible)):
        selection = [d for d, flag in zip(possible, flags) if flag]
        feasible, reward = _menu_score_data(state, recipes, values, selection)
        if feasible and reward > best_reward + 1e-9:
            best_reward, best_menu = reward, selection
    return best_reward, best_menu


def _menu_cases(samples):
    cases = []
    for i, sample in enumerate(samples):
        ingredients, dishes = sample["ingredients"], sample["dishes"]
        recipes = sample["recipes"]
        state = [sample["game_ingredients_state"].get(name, 0) for name in ingredients]
        values = [(a + b) / 2 for a, b in zip(sample["agent_0_values"], sample["agent_1_values"])]
        optimum, best = _menu_optimum(state, recipes, values)
        assert math.isclose(optimum, sample["max_reward"], abs_tol=1e-7)
        assert all(sample["agent_0_obs"].get(name, 0) + sample["agent_1_obs"].get(name, 0)
                   == sample["game_ingredients_state"].get(name, 0) for name in ingredients)
        readable_recipes = {name: {ingredient: count for ingredient, count in zip(ingredients, row) if count}
                            for name, row in zip(dishes, recipes)}
        intro = (
            "Menu-Numeric decision probe: design a feasible menu for two equally important guests. "
            "Each dish can be chosen at most once. Total available ingredients are the SUM of Agent 0 and Agent 1's "
            "private ingredient partitions. A missing ingredient entry means zero in that agent's partition. "
            "Dish value is the AVERAGE of the two guests' numerical preferences, and menu value sums selected dish values. "
            "Exceeding available ingredients yields zero team value. After at most one message, Agent 1 proposes a menu. "
            "We evaluate this proposal's feasibility and value, not whether both agents eventually agree.\n"
            "Public recipes (ingredient quantities needed per dish):\n" + _dump(readable_recipes)
        )
        sender = (intro + "\nYou are Agent 0. Your private ingredient partition:\n" + _dump(sample["agent_0_obs"])
                  + "\nGuest 0 preferences:\n" + _dump(dict(zip(dishes, sample["agent_0_values"]))))
        receiver = (intro + "\nYou are Agent 1. Your private ingredient partition:\n" + _dump(sample["agent_1_obs"])
                    + "\nGuest 1 preferences:\n" + _dump(dict(zip(dishes, sample["agent_1_values"]))))
        full = (intro + "\nYou are Agent 1. Complete inventory for the capability reference:\n"
                + _dump(sample["game_ingredients_state"]) + "\nComplete average dish preferences:\n"
                + _dump(dict(zip(dishes, values))))
        response = ('Return one JSON object {"choice": ["dish name", ...], "reason": "brief"}. '
                    "Use only exact public dish names, without duplicates. The empty menu [] is allowed.")
        cases.append(_common(
            f"deliberative_menu_game{i:04d}_a0_to_a1", "Deliberative Menu-Numeric", sender, receiver, full, response,
            _provenance(sample,
                "Published Numeric level-1-and-2 test games game_0000.pkl and game_0001.pkl retained with original "
                "recipes, partitioned inventory and numerical preference vectors. Restricted data-only extraction maps "
                "legacy game/persona classes to inert placeholders; no upstream methods executed. "
                "One message and menu proposal replace multi-stage dialogue/agreement. Shared normalized proposal value "
                "uses exact exhaustive full-state subset optimization; individual economic rewards not invented."),
            {"kind": "menu_numeric", "dishes": dishes, "state": state, "recipes": recipes,
             "values": values, "max_reward": optimum, "optimal_choice": [dishes[d] for d in best],
             "reward_semantics": "shared feasible-menu proposal value, no joint agreement evaluated"}))
    return cases


def _dnd_cases(samples):
    cases = []
    names = ["books", "hats", "balls"]
    for i, sample in enumerate(samples, 1):
        counts, sender_values, receiver_values = sample["counts"], sample["sender_values"], sample["receiver_values"]
        assert sum(c * v for c, v in zip(counts, sender_values)) == 10
        assert sum(c * v for c, v in zip(counts, receiver_values)) == 10
        intro = (
            "Deal or No Deal allocation-proposal probe. Two agents divide books, hats and balls. "
            "Counts are public; unit values are private. Each agent values the entire inventory at 10. "
            "Your objective is your own value while seeking a mutually acceptable allocation. After one message from "
            "Agent 0, Agent 1 proposes a complete split or chooses no_agreement. This is one proposed allocation, "
            "not a completed negotiation: reported values are the potential utilities IF the split were accepted. "
            "The complementary allocation to Agent 0 is fully determined; no items may disappear.\nPublic inventory:\n"
            + _dump(dict(zip(names, counts)))
        )
        sender = intro + "\nYou are Agent 0. Your private unit values:\n" + _dump(dict(zip(names, sender_values)))
        receiver = intro + "\nYou are Agent 1. Your private unit values:\n" + _dump(dict(zip(names, receiver_values)))
        full = (receiver + "\nFor the capability reference, Agent 0's actual unit values are:\n"
                + _dump(dict(zip(names, sender_values))))
        response = ('Return one JSON object {"choice": [books_you_take, hats_you_take, balls_you_take], "reason": "brief"}, '
                    'or {"choice":"no_agreement","reason":"brief"}. Counts must be nonnegative integers within public '
                    "inventory limits. Every remaining item goes to Agent 0. Choose a reasonable proposal using available information.")
        cases.append(_common(
            f"dnd_selfplay_pair{i}_a0_to_a1", "Deal or No Deal", sender, receiver, full, response,
            _provenance(sample,
                "First two paired contexts in published selfplay.txt; all inventory and private unit values retained. "
                "Native multi-turn free negotiation and two independent final selections replaced by one message and "
                "receiver's structured complete allocation proposal. Masks remove sender language, retaining all inventory "
                "and legal allocation mechanics. Rewards are conditional allocation utilities only; acceptance and full-game "
                "reward unmeasured. There is no unique correct split under conflicting interests."),
            {"kind": "dnd_allocation", "counts": counts, "sender_values": sender_values,
             "receiver_values": receiver_values, "reward_semantics": "potential allocation utilities conditional on acceptance"}))
    return cases


def _arena_cases(samples):
    cases = []
    for i, sample in enumerate(samples, 1):
        cost, value, price = sample["seller_cost"], sample["buyer_value"], sample["price"]
        assert sample["quantity"] == 1 and 0 <= price <= sample["buyer_budget"]
        intro = (
            "NegotiationArena SellerBuyer terminal-decision probe. Agent 0 owns one object X and is the seller. "
            "Agent 1 is the buyer. Production cost and willingness to pay are private. A structured offer is frozen "
            "separately from accompanying seller language. This probe permits ACCEPT or REJECT at this decision only; "
            "there are no future counteroffers. Acceptance gives seller price minus cost and buyer value minus price. "
            "Rejection gives each zero incremental utility.\nFrozen structured offer: seller transfers 1 X; buyer pays "
            + str(price) + " ZUP. This offer remains visible even when seller language is unavailable."
        )
        sender = (intro + f"\nYou are Agent 0. Your actual production cost is {cost} ZUP. "
                  "You may explain this frozen offer, but cannot change its price or quantity.")
        receiver = (intro + f"\nYou are Agent 1. Your actual willingness to pay for X is {value} ZUP, "
                    f"and your budget is {sample['buyer_budget']} ZUP. Maximize your incremental utility.")
        full = receiver + f"\nFor the full-information capability reference, seller's actual production cost is {cost} ZUP."
        response = 'Return one JSON object {"choice":"accept" or "reject","reason":"brief"}.'
        cases.append(_common(
            f"negotiation_arena_logged_offer{i}_seller_to_buyer", "NegotiationArena SellerBuyer", sender, receiver, full, response,
            _provenance(sample,
                "Pinned paper_experiment_code game logs: first split_the_difference run and first section_one buysell run. "
                "Original costs, values, budget and first seller structured offers retained. "
                "Historical dialogue and persona prompts omitted; newly generated seller message explains unchanged offer. "
                "Decision moved to terminal ACCEPT/REJECT with no continuation; utility is this fixed transaction only, "
                "not original multi-round game performance. Masking removes accompanying language, never price or transaction mechanics."),
            {"kind": "arena_acceptance", "seller_cost": cost, "buyer_value": value, "price": price,
             "reward_semantics": "incremental utility of this fixed terminal transaction relative to no trade"}))
    return cases


def build_cases():
    samples = json.loads((SOURCE_DIR / "samples.json").read_text(encoding="utf-8"))
    return (_dialop_cases(samples["dialop"]) + _menu_cases(samples["menu"])
            + _dnd_cases(samples["dnd"]) + _arena_cases(samples["arena"]))


def build_skips():
    return []


def _integers(choice, count):
    return isinstance(choice, list) and len(choice) == count and all(type(v) is int for v in choice)


def score(case, choice):
    s = case["private_scoring"]
    kind = s["kind"]
    result = {"legal": False, "correct": None, "receiver_reward": None, "sender_reward": None,
              "team_reward": None, "choice": choice, "reward_scope": s["reward_semantics"]}
    if kind == "dialop_assignment":
        n = len(s["pooled_table"])
        if not _integers(choice, n) or set(choice) != set(range(n)):
            return result
        raw = sum(s["pooled_table"][r][c] for r, c in enumerate(choice))
        result.update(legal=True, correct=math.isclose(raw, s["max_reward"], abs_tol=1e-8),
                      team_reward=raw / s["max_reward"], raw_team_reward=float(raw))
    elif kind == "menu_numeric":
        if not isinstance(choice, list) or any(type(x) is not str or x not in s["dishes"] for x in choice) or len(set(choice)) != len(choice):
            return result
        selected = sorted(s["dishes"].index(x) for x in choice)
        feasible, raw = _menu_score_data(s["state"], s["recipes"], s["values"], selected)
        result.update(legal=True, feasible=feasible, correct=feasible and math.isclose(raw, s["max_reward"], abs_tol=1e-7),
                      team_reward=raw / s["max_reward"], raw_team_reward=float(raw),
                      choice=[s["dishes"][d] for d in selected])
    elif kind == "dnd_allocation":
        if isinstance(choice, str) and choice.lower().strip() == "no_agreement":
            return {**result, "legal": True, "choice": "no_agreement", "potential_receiver_utility": 0.0,
                    "potential_sender_utility": 0.0, "potential_team_utility": 0.0}
        if not _integers(choice, 3) or any(x < 0 or x > c for x, c in zip(choice, s["counts"])):
            return result
        receiver = float(sum(x * v for x, v in zip(choice, s["receiver_values"])))
        sender = float(sum((c - x) * v for c, x, v in zip(s["counts"], choice, s["sender_values"])))
        result.update(legal=True, potential_receiver_utility=receiver,
                      potential_sender_utility=sender, potential_team_utility=receiver + sender)
    elif kind == "arena_acceptance":
        if not isinstance(choice, str) or choice.strip().lower() not in ("accept", "reject"):
            return result
        choice = choice.strip().lower()
        receiver = float(s["buyer_value"] - s["price"]) if choice == "accept" else 0.0
        sender = float(s["price"] - s["seller_cost"]) if choice == "accept" else 0.0
        correct = receiver >= max(0, s["buyer_value"] - s["price"])
        result.update(legal=True, choice=choice, correct=correct,
                      receiver_reward=receiver, sender_reward=sender, team_reward=receiver + sender)
    else:
        raise ValueError(f"Unknown decision score kind: {kind}")
    return result
