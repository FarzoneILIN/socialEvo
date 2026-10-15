"""Fresh, source-pinned states and role-correct directed communication probes.

Only explicit prompts are model inputs. Private scores and legal kitchen
prefix/suffix witnesses remain offline. Historical packages are read-only.
"""
from __future__ import annotations

import copy
import hashlib
import io
import itertools
import json
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import urllib.request

from focused_validation.adapters import collab
from focused_validation.sources.decisions.extract_snapshots import RestrictedPublicDataUnpickler, plain
from quick_screen.adapters.decisions import _menu_optimum, _menu_score_data

ROOT = Path(__file__).resolve().parents[1]
DIRECTIONS = {"A_to_B": (0, 1), "B_to_A": (1, 0)}


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_record(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


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


def historical_fingerprints(workspace=ROOT):
    """Exclude frozen historical wave cases, including unused/failed slots.

    Candidate pools not allocated to a frozen wave are not experimental cases.
    Legacy Menu hashes conservatively exclude equal objectives even if private
    partitions differ. DND reversals are excluded too, preventing role relabeling.
    """
    excluded = {name: set() for name in ("menu", "dnd", "collab")}
    for case in _historical_cases(workspace):
        task, key = _legacy_fingerprint(case)
        if task:
            excluded[task].add(key)
        if task == "dnd":
            s = case["private_scoring"]
            excluded[task].add(digest({"counts": s["counts"], "sender_values": s["receiver_values"],
                                      "receiver_values": s["sender_values"]}))
    return excluded


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


def menu_pool(root, excluded):
    tree_path = ROOT / "quick_screen/sources/decisions/deliberative_tree.json"
    tree = json.loads(tree_path.read_text(encoding="utf-8"))
    paths = [x for x in tree["tree"] if x["path"].startswith("data/cooking/games/test_games/games/")
             and x["path"].endswith(".pkl")]
    # The main and confirmation waves select only the first, level_3 stratum.
    paths.sort(key=lambda x: ("level_3" not in x["path"], x["path"]))
    old_records = []
    for package in ("quick_screen", "focused_validation", "mechanism_experiments"):
        for cache in (ROOT/package).rglob("game_*.pkl"):
            old_records.append(cache)

    def load(entry):
        relative = entry["path"]
        difficulty = Path(relative).parent.name
        url = "https://raw.githubusercontent.com/wcx21/deliberative-collaboration-agents/"+tree["sha"]+"/"+relative
        dest = Path(root)/"sources/menu"/difficulty/Path(relative).name
        data = None
        for old in old_records:
            if old.name != dest.name:
                continue
            candidate = old.read_bytes()
            if hashlib.sha1(b"blob "+str(len(candidate)).encode()+b"\0"+candidate).hexdigest() == entry["sha"]:
                data = candidate
                break
        if dest.exists():
            data = dest.read_bytes()
        if data is None:
            with urllib.request.urlopen(url, timeout=40) as response:
                data = response.read(2_000_001)
            if len(data) > 2_000_000:
                raise ValueError("public_source_too_large")
        if hashlib.sha1(b"blob "+str(len(data)).encode()+b"\0"+data).hexdigest() != entry["sha"]:
            raise ValueError("pinned_git_blob_mismatch")
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            dest.write_bytes(data)
        record = RestrictedPublicDataUnpickler(io.BytesIO(data)).load()
        fields = ("ingredients", "dishes", "recipes", "game_ingredients_state", "agent_0_obs", "agent_1_obs",
                  "agent_0_values", "agent_1_values", "max_reward")
        sample = {k: plain(getattr(record, k)) for k in fields}
        sample.update(upstream_game_id=dest.stem, provenance={"source_url": url,
                      "source_commit": tree["sha"], "upstream_path": relative, "git_blob_sha1": entry["sha"],
                      "local_source_file": str(dest.resolve()), "sha256": hashlib.sha256(data).hexdigest(),
                      "source_files": [file_record(dest), file_record(tree_path)]})
        result = _menu_case(sample, difficulty)
        return result if result["semantic_hash"] not in excluded else None

    with ThreadPoolExecutor(max_workers=8) as executor:
        result = [x for x in executor.map(load, paths) if x]
    return result


def dnd_pool(excluded, limit=84):
    source = ROOT/"focused_validation/sources/decisions/dnd/src/data/negotiate/selfplay.txt"
    records = json.loads((ROOT/"focused_validation/sources/decisions/manifest.json").read_text(encoding="utf-8"))
    provenance = next(x for x in records if Path(x["local_source_file"]) == source)
    if file_record(source)["sha256"] != provenance["sha256"]:
        raise ValueError("pinned_dnd_source_changed")
    lines = source.read_text(encoding="utf-8").splitlines()
    result, seen = [], set(excluded)
    names = ["books", "hats", "balls"]
    for offset in range(0, len(lines)-1, 2):
        a, b = [[int(x) for x in line.split()] for line in lines[offset:offset+2]]
        counts, values = a[::2], [a[1::2], b[1::2]]
        if counts != b[::2] or any(sum(c*v for c, v in zip(counts, vec)) != 10 for vec in values):
            continue
        s = {"kind": "dnd_allocation", "counts": counts, "sender_values": values[0], "receiver_values": values[1],
             "agent_values": values, "reward_semantics": "potential complete-allocation utility conditional on acceptance"}
        key = digest({k: s[k] for k in ("counts", "sender_values", "receiver_values")})
        reversed_key = digest({"counts": counts, "sender_values": values[1], "receiver_values": values[0]})
        if key in seen or reversed_key in seen:
            continue
        intro = ("Deal or No Deal allocation-proposal task. Two agents divide books, hats and balls. Inventory is public; "
                 "unit values are private. Each agent values all stock at 10. Seek a mutually acceptable split while "
                 "considering your own value. The acting agent proposes one full allocation or no_agreement after a "
                 "single communication opportunity. Unallocated items go to the other agent. Values are potential utilities "
                 "IF accepted, not realized rewards. There is no uniquely correct allocation. No earlier partner text exists.\n"
                 "Public inventory:\n"+canonical(dict(zip(names, counts))))
        raw = [f"Agent {i} original private unit values:\n"+canonical(dict(zip(names, values[i]))) for i in (0, 1)]
        prompts = [intro+f"\nYou are Agent {i} ({'A' if i == 0 else 'B'}).\n"+raw[i] for i in (0, 1)]
        case = {"case_id": f"influence_dnd_pair_{offset//2}", "task": "dnd", "candidate": "Deal or No Deal",
                "actor_prompts": prompts, "full_actor_prompts": [p+"\nBoth original private views:\n"+"\n".join(raw) for p in prompts],
                "response_instruction": 'Return only JSON {"choice":[books_you_take,hats_you_take,balls_you_take],'
                                        '"reason":"brief"}, or {"choice":"no_agreement","reason":"brief"}. '
                                        "Quantities must be integers within public stock; your partner takes the complement.",
                "allowed_choices": [list(x) for x in itertools.product(*(range(c+1) for c in counts))]+["no_agreement"],
                "private_scoring": s, "semantic_hash": key, "cluster": f"pair_{offset//2}",
                "difficulty_stratum": "published_selfplay_normalized10", "provenance": {**provenance,
                    "source_lines": [offset+1, offset+2], "source_files": [file_record(source)],
                    "selection_rule": "Published file order; normalization10; semantic exclusion including swapped profiles."}}
        result.append(directional(case, "A_to_B"))
        seen.update((key, reversed_key))
        if len(result) >= limit:
            break
    return result


def _kitchen_case(state, expected, prefix, suffix, recipe, reference, schedule):
    actor = [collab.observation(state, i) for i in (0, 1)]
    complete = [collab.observation(state, i, "full_information") for i in (0, 1)]
    identifier = "influence_collab_"+recipe.stem+"_"+digest({"physical": state["physical"], "schedule": schedule})[:14]
    s = {"order": state["order"], "recipe_file": state["recipe_file"],
         "reference_file": str(reference.relative_to(collab.UPSTREAM)).replace("\\", "/"),
         "physical": state["physical"], "reference_next_choices": expected[1], "reference_choices_by_actor": expected,
         "prefix_witness": prefix, "suffix_witness": suffix, "followup_prespecified": False,
         "reference_correctness_kind": "published feasible next operation; alternative legal actions may also succeed",
         "reward_semantics": "exact native sparse macro reward; reference match separate from payoff"}
    paths = [recipe, reference, collab.UPSTREAM/"lib/overcooked_ai/overcooked_ai_py/mdp/overcooked_mdp.py",
             collab.UPSTREAM/"lib/overcooked_ai/overcooked_ai_py/data/layouts/new_env.layout"]
    case = {"case_id": identifier, "task": "collab", "candidate": "Collab-Overcooked",
            "actor_prompts": [x["prompt"]+"\nNo earlier partner text or plan list is provided." for x in actor],
            "full_actor_prompts": [x["prompt"]+"\nNo earlier partner text or plan list is provided." for x in complete],
            "response_instructions": [x["response_instruction"] for x in actor],
            "allowed_by_actor": [x["allowed_macros"] for x in actor], "private_scoring": s,
            "semantic_hash": digest({"order": s["order"], "physical": s["physical"]}),
            "state_sha256": digest(s["physical"]), "cluster": s["recipe_file"],
            "difficulty_stratum": "recipe_level_"+recipe.stem.split("_", 1)[0],
            "provenance": {"source_url": "https://github.com/YusaeMeow/Collab-Overcooked/tree/"+collab.COMMIT,
                 "source_commit": collab.COMMIT, "source_files": [file_record(x) for x in paths],
                 "selection_rule": "Published recipe/reference order; prespecified legal schedule variants; historical exclusion.",
                 "adaptation": "Native physical rules and120step horizon retained; deterministic legal macro navigation; "
                               "both actor directions; feasibility witnesses remain offline; reference is not unique correctness.",
                 "scripted_schedule": schedule}}
    return directional(case, "A_to_B")


def collab_pool(excluded, limit=64):
    result, seen = [], set(excluded)
    # Compare physical states without merely counting timer/diagnostic changes as
    # fresh task states. This is stricter than the legacy hash exclusion.
    def substantive(physical):
        return digest({k: physical[k] for k in ("players", "objects", "order_list")})
    prior_substantive = set()
    for case in _historical_cases():
        s = case.get("private_scoring", {})
        physical = s.get("physical", s.get("snapshot", {}).get("state"))
        if physical and "players" in physical:
            prior_substantive.add(substantive(physical))
    captures = []
    for recipe in sorted((collab.UPSTREAM/"src/prompts/recipe").glob("*.txt")):
        reference = collab.UPSTREAM/"src/prompts/reference"/(recipe.stem+"_ref.txt")
        if not reference.exists():
            continue
        for ref_name, seq in sorted(json.loads(reference.read_text(encoding="utf-8")).items()):
            if not isinstance(seq, dict) or not all("agent_"+str(i) in seq for i in (0, 1)):
                continue
            for schedule in ("chef_first", "alternating_priority", "assistant_first"):
                order = recipe.stem.split("_", 1)[1]
                state = {"protocol": "collab_native_macro_v1", "case_id": "offline", "order": order,
                         "recipe_file": str(recipe.relative_to(collab.UPSTREAM)).replace("\\", "/"),
                         "physical": collab._serialize(collab._mdp(order).get_standard_start_state()),
                         "initial_timestep": 0, "team_reward": 0, "delivered": False, "macro_count": 0,
                         "actions": [], "stopped_reason": None}
                positions, local = [0, 0], []
                for step in range(80):
                    if state.get("stopped_reason"):
                        break
                    expected = []
                    for actor in (0, 1):
                        choices = seq["agent_"+str(actor)]
                        action = collab._canonical(choices[positions[actor]]) if positions[actor] < len(choices) else None
                        expected.append([action] if action and collab._execute(state["physical"], order, actor, action)["legal"] else [])
                    if all(expected):
                        local.append((copy.deepcopy(state), expected, len(state["actions"])))
                    priority = (0, 1) if schedule == "chef_first" else ((1, 0) if schedule == "assistant_first" else (step%2, 1-step%2))
                    progressed = False
                    for actor in priority:
                        choices = seq["agent_"+str(actor)]
                        if positions[actor] >= len(choices):
                            continue
                        trial = collab.apply_action(state, actor, collab._canonical(choices[positions[actor]]))
                        if trial["legal"]:
                            state = trial["state"]
                            positions[actor] += 1
                            progressed = True
                            break
                    if not progressed:
                        trial = collab.apply_action(state, 0, "wait(1)")
                        if not trial["legal"]:
                            break
                        state = trial["state"]
                if not state["delivered"]:
                    continue
                for snapshot, expected, length in local:
                    key = digest({"order": order, "physical": snapshot["physical"]})
                    content_key = substantive(snapshot["physical"])
                    if key in seen or content_key in prior_substantive:
                        continue
                    case = _kitchen_case(snapshot, expected, state["actions"][:length], state["actions"][length:],
                                         recipe, reference, ref_name+":"+schedule)
                    captures.append(case)
                    seen.add(key)
                    prior_substantive.add(content_key)
    # Spread over recipes so tiny waves do not exhaust a single recipe.
    groups = {}
    for case in captures:
        groups.setdefault(case["cluster"], []).append(case)
    result = [group[i] for i in range(max((len(g) for g in groups.values()), default=0))
              for _, group in sorted(groups.items()) if i < len(group)]
    return result[:limit]


def prepare_pools(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    excluded = historical_fingerprints()
    pools = {"menu": menu_pool(root, excluded["menu"]), "dnd": dnd_pool(excluded["dnd"]),
             "collab": collab_pool(excluded["collab"])}
    if sum(c["difficulty_stratum"] == "level_3" for c in pools["menu"]) < 16:
        raise ValueError("insufficient_fresh_public_level3_menus")
    if len(pools["dnd"]) < 72 or len(pools["collab"]) < 16:
        raise ValueError("insufficient_fresh_states")
    (root/"pools.json").write_text(json.dumps(pools, ensure_ascii=False, indent=2), encoding="utf-8")
    return pools


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


def canonical_choice(case, choice):
    if isinstance(choice, dict):
        choice = choice.get("choice")
    if case["task"] == "menu" and isinstance(choice, list) and all(isinstance(x, str) for x in choice):
        # Keep duplicates so malformed duplicate menus remain invalid.
        return sorted(choice)
    if case["task"] == "collab":
        return collab._canonical(choice)
    return choice.strip().lower() if isinstance(choice, str) else choice


def recommendations_distinct(case):
    """Offline mechanical precondition for opposing-recommendation probes."""
    recommendations = case.get("recommendations")
    return bool(recommendations and len(recommendations) == 2
                and canonical_choice(case, recommendations[0]) != canonical_choice(case, recommendations[1]))


def messages(case, condition, partner_text=None, own_text=None, **kwargs):
    """One incoming message only; own sent text identical across conditions.

    Caller never passes partner history into this local decision interface.
    Advice source variants change only the source label, never the action text.
    """
    full = condition == "full_raw" or condition.startswith(("advice", "no_advice"))
    result = [{"role": "system", "content": "Use only supplied observations. No tools. Return only requested JSON; never invent missing facts."},
              {"role": "user", "content": (case["full_information_prompt"] if full else case["receiver_prompt"])
               +"\n\n"+case["response_instruction"]}]
    if own_text is not None:
        result.append({"role": "user", "content": "Your own message, written from your original private observation before any incoming message:\n"+own_text})
    if condition.startswith("advice"):
        index = 0 if condition.startswith("advice_X") else 1
        action_text = "Choose "+canonical(canonical_choice(case, case["recommendations"][index]))+"."
        neutral = condition.endswith("_neutral")
        label = "Recommendation (source unspecified):\n" if neutral else "Recommendation from "+case["sender"]+":\n"
        result.append({"role": "user", "content": label+action_text})
    elif condition.startswith("no_advice"):
        result.append({"role": "user", "content": "[No recommendation supplied.]"})
    elif condition != "full_raw":
        text = "[Partner message withheld.]" if condition == "masked" else (case["facts_text"] if condition == "facts" else partner_text)
        result.append({"role": "user", "content": "Message from "+case["sender"]+":\n"+(text if text is not None else "[Partner message unavailable.]")})
    return result


def score(case, choice):
    choice = canonical_choice(case, choice)
    s = case["private_scoring"]
    result = {"choice": choice, "legal": False, "correct": None, "receiver_reward": None, "sender_reward": None,
              "team_reward": None, "reward_scope": s["reward_semantics"], "acting_agent_id": case["receiver_id"]}
    if case["task"] == "menu":
        if not isinstance(choice, list) or any(type(x) is not str or x not in s["dishes"] for x in choice) or len(choice) != len(set(choice)):
            return result
        feasible, raw = _menu_score_data(s["state"], s["recipes"], s["values"], [s["dishes"].index(x) for x in choice])
        quality = raw/s["max_reward"] if s["max_reward"] > 0 else float(feasible and raw == 0)
        result.update(legal=True, feasible=feasible, correct=feasible and math.isclose(raw, s["max_reward"], abs_tol=1e-7),
                      normalized_quality=quality, team_reward=quality, raw_team_reward=raw)
    elif case["task"] == "dnd":
        if choice == "no_agreement":
            result.update(legal=True, potential_receiver_utility=0., potential_sender_utility=0., potential_team_utility=0.,
                          potential_A_utility=0., potential_B_utility=0., original_A_utility=0., original_B_utility=0.)
            return result
        if not isinstance(choice, list) or len(choice) != 3 or any(type(x) is not int or not 0 <= x <= c for x, c in zip(choice, s["counts"])):
            return result
        receiver = case["receiver_id"]
        allocations = [None, None]
        allocations[receiver] = choice
        allocations[1-receiver] = [c-x for c, x in zip(s["counts"], choice)]
        utilities = [float(sum(x*v for x, v in zip(allocations[i], s["agent_values"][i]))) for i in (0, 1)]
        result.update(legal=True, potential_receiver_utility=utilities[receiver], potential_sender_utility=utilities[1-receiver],
                      potential_team_utility=sum(utilities), potential_A_utility=utilities[0], potential_B_utility=utilities[1],
                      original_A_utility=utilities[0], original_B_utility=utilities[1],
                      allocation_A=allocations[0], allocation_B=allocations[1])
    elif case["task"] == "collab":
        if choice not in case["allowed_choices"]:
            return result
        trial = collab.apply_action(collab.start_episode(case), case["receiver_id"], choice)
        reference = choice in s["reference_choices_by_actor"][case["receiver_id"]]
        result.update(legal=trial["legal"], correct=None, reference_next_operation_match=reference,
                      team_reward=trial["team_reward"], elapsed_steps=trial["elapsed_steps"],
                      completed_operation=trial["completed_operation"], state_after_sha256=digest(trial["state"]["physical"]),
                      native_target_delivered=trial["state"]["delivered"],
                      score_kind="native sparse reward; reference matching is not unique correctness")
    else:
        raise ValueError("unknown_task")
    return result


def category(case, choice, scoring, error=None):
    if error or not scoring.get("legal"):
        return "INVALID"
    if case["task"] == "menu":
        return "optimal" if scoring["correct"] else ("feasible_suboptimal" if scoring["feasible"] else "infeasible")
    if case["task"] == "dnd":
        if scoring["choice"] == "no_agreement":
            return "no_agreement"
        value = scoring["potential_receiver_utility"]
        return "receiver_0_to3" if value <= 3 else ("receiver_over3_to7" if value <= 7 else "receiver_over7_to10")
    return "reference_match" if scoring["reference_next_operation_match"] else "other_legal"


def score_acceptance(case, proposal, acceptance_choice):
    potential = score(case, proposal)
    result = {"legal": False, "agreement": False, "choice": acceptance_choice, "sender_reward": None,
              "receiver_reward": None, "team_reward": None, "A_reward": None, "B_reward": None,
              "acting_agent_id": case["sender_id"], "reward_scope": "realized terminal payoff of this bounded offer"}
    if case["task"] != "dnd" or not potential["legal"]:
        return result
    if potential["choice"] == "no_agreement":
        result.update(legal=True, sender_reward=0., receiver_reward=0., team_reward=0., A_reward=0., B_reward=0.)
        return result
    if not isinstance(acceptance_choice, str) or acceptance_choice.strip().lower() not in ("accept", "reject"):
        return result
    accept = acceptance_choice.strip().lower() == "accept"
    result.update(legal=True, choice="accept" if accept else "reject", agreement=accept,
                  sender_reward=potential["potential_sender_utility"] if accept else 0.,
                  receiver_reward=potential["potential_receiver_utility"] if accept else 0.,
                  team_reward=potential["potential_team_utility"] if accept else 0.,
                  A_reward=potential["potential_A_utility"] if accept else 0.,
                  B_reward=potential["potential_B_utility"] if accept else 0.)
    return result
