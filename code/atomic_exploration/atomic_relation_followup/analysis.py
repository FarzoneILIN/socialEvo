"""Receipt-grounded descriptive contrasts and the frozen six primary checks."""
from __future__ import annotations

from collections import Counter, defaultdict
import datetime as dt
import json
import math
from pathlib import Path
import sqlite3

import numpy as np

from .protocol import POLICY, write

GROUPS = ("N1", "N2", "H1", "H2")
SAVED = {"saved", "success", "ok"}
NAMES = {"minesweeper_natural": "扫雷自然消息", "minesweeper_fact_advice": "扫雷事实与推荐",
         "menu": "菜单", "negotiation": "谈判", "kitchen": "厨房"}


def tv(a, b):
    a, b = Counter(a), Counter(b)
    na, nb = sum(a.values()), sum(b.values())
    if not na or not nb:
        return None
    return .5 * sum(abs(a[k]/na-b[k]/nb) for k in a.keys() | b.keys())


def distances(groups):
    n1, n2, h1, h2 = [groups.get(k, []) for k in GROUPS]
    if any(len(g) != 3 for g in (n1, n2, h1, h2)):
        return None
    cross = sum(tv(n, h) for n in (n1, n2) for h in (h1, h2))/4
    reference = (tv(n1, n2)+tv(h1, h2))/2
    return dict(raw_distance=cross, repeat_reference=reference, adjusted_distance=cross-reference,
                normal_repeat=tv(n1, n2), masked_repeat=tv(h1, h2))


def _numeric(v):
    return isinstance(v, (int, float, np.integer, np.floating)) and math.isfinite(float(v))


def _mean(values):
    values = [float(v) for v in values if _numeric(v)]
    return float(np.mean(values)) if values else None


def _interval(values):
    if not values:
        return [None, None]
    rng, a = np.random.default_rng(42), np.asarray(values, dtype=float)
    draws = rng.integers(0, len(a), size=(10000, len(a)))
    return [float(v) for v in np.percentile(a[draws].mean(axis=1), [2.5, 97.5])]


def _flatten(values, prefix=""):
    result = {}
    for key, value in values.items():
        name = prefix+str(key)
        if isinstance(value, dict):
            result.update(_flatten(value, name+"."))
        elif _numeric(value):
            result[name] = float(value)
    return result


def _cluster(case):
    data = case.get("data", {})
    for key in ("recipe_file", "recipe_id", "recipe_name"):
        if key in data:
            return str(data[key])
    return str(data.get("native_case", {}).get("private_scoring", {}).get("recipe_file", case["case_id"]))


def _category(entry):
    if entry["response"].get("status") not in SAVED or not entry["score"]:
        return None
    value = entry["score"].get("exact_choice_category")
    if value is not None:
        return str(value)
    return "INVALID" if entry["score"].get("format_valid") is False or entry["score"].get("legal") is False else None


def _entries(manifest, rows):
    cases = {c["case_id"]: c for c in manifest["cases"]}
    recorded = {row["slot"]: dict(row) for row in rows}
    entries = []
    for slot in manifest["slots"]:
        row = recorded.get(slot["slot_id"], {})
        response = json.loads(row["response_json"]) if row.get("response_json") else {}
        score = json.loads(row["score_json"]) if row.get("score_json") else {}
        entries.append(dict(slot=slot, case=cases[slot["case_id"]], response=response, score=score,
                            recorded=bool(row), outcomes=dict(score.get("outcomes") or {})))
    by_id = {e["slot"]["slot_id"]: e for e in entries}
    for accept in entries:
        slot = accept["slot"]
        if slot["kind"] != "accept" or slot.get("proposal_slot_id") not in by_id:
            continue
        proposal = by_id[slot["proposal_slot_id"]]
        if proposal["score"].get("legal") is not True or proposal["score"].get("choice") == "no_agreement":
            continue
        known = accept["response"].get("status") in SAVED and accept["score"].get("legal") is True
        if known:
            for key in ("agreement", "realized_A", "realized_B", "realized_team", "settlement_status"):
                proposal["outcomes"][key] = accept["outcomes"].get(key)
        else:
            proposal["outcomes"]["settlement_status"] = "acceptance_unobserved"
        proposal["outcomes"]["accepted_given_legal_allocation"] = accept["outcomes"].get("agreement") if known else None
    for entry in entries:
        entry["numeric"] = _flatten(entry["outcomes"])
    return entries


def _side(entries):
    categories = [_category(e) for e in entries]
    values = defaultdict(list)
    for e in entries:
        for key, value in e["numeric"].items():
            values[key].append(value)
    return dict(planned=len(entries), returned=sum(e["response"].get("status") in SAVED for e in entries),
                scored=sum(v is not None for v in categories), missing=sum(v is None for v in categories),
                categories=[v for v in categories if v is not None],
                category_counts=dict(Counter(v for v in categories if v is not None)),
                outcome_means={k: _mean(v) for k, v in values.items()},
                outcome_denominators={k: len(v) for k, v in values.items()})


def _contrast(case, receiver, family, name, left, right, version=1, partition=None, repeated=True):
    treatment, reference = _side(left), _side(right)
    groups = {}
    if repeated:
        for prefix, entries in (("N", left), ("H", right)):
            for group in (1, 2):
                groups[prefix+str(group)] = [v for e in entries if e["slot"].get("group") == group
                                            and (v := _category(e)) is not None]
        d = distances(groups) if len(left) == len(right) == 6 else None
    else:
        complete = len(left) and len(right) and not treatment["missing"] and not reference["missing"]
        d = {"raw_distance": tv(treatment["categories"], reference["categories"])} if complete else None
    changes = {}
    for metric in treatment["outcome_means"].keys() | reference["outcome_means"].keys():
        nl, nr = treatment["outcome_denominators"].get(metric, 0), reference["outcome_denominators"].get(metric, 0)
        full = nl == len(left) and nr == len(right) and nl and nr
        changes[metric] = dict(mean_change=treatment["outcome_means"][metric]-reference["outcome_means"][metric] if full else None,
            treatment_known_mean=treatment["outcome_means"].get(metric), reference_known_mean=reference["outcome_means"].get(metric),
            treatment_known=nl, reference_known=nr, treatment_planned=len(left), reference_planned=len(right))
    return dict(case_id=case["case_id"], module=case["module"], stratum=case.get("stratum"), receiver=receiver,
                version=version, partition=partition, family=family, contrast=name,
                cluster=_cluster(case) if case["module"] == "kitchen" else case["case_id"], distances=d, groups=groups,
                treatment=treatment, reference=reference, outcome_changes=changes,
                missing=treatment["missing"]+reference["missing"], exploratory=family != "main")


def _estimate(records, getter):
    clusters = defaultdict(list)
    for record in records:
        value = getter(record)
        if _numeric(value):
            clusters[record["cluster"]].append(float(value))
    values = [_mean(v) for v in clusters.values()]
    return dict(mean=_mean(values), interval=_interval(values), complete_clusters=len(values),
                complete_states=sum(len(v) for v in clusters.values()))


def _aggregate(records):
    fields = ("module", "stratum", "receiver", "version", "partition", "family", "contrast", "partition_role")
    groups = defaultdict(list)
    for record in records:
        groups[tuple(record.get(k) for k in fields)].append(record)
    results = []
    for key, members in sorted(groups.items(), key=lambda item: str(item[0])):
        item = dict(zip(fields, key))
        item.update(expected_states=len({r["case_id"] for r in members}),
                    expected_clusters=len({r["cluster"] for r in members}),
                    complete_behavior_states=sum(r.get("distances") is not None for r in members),
                    missing_choices=sum(r["missing"] for r in members), exploratory=True)
        names = set().union(*((r.get("distances") or {}).keys() for r in members))
        item["behavior"] = {k: _estimate(members, lambda r, metric=k: (r.get("distances") or {}).get(metric)) for k in names}
        metrics = set().union(*(r["outcome_changes"].keys() for r in members))
        item["outcome_changes"] = {k: _estimate(members, lambda r, metric=k: r["outcome_changes"].get(metric, {}).get("mean_change")) for k in metrics}
        results.append(item)
    return results


def _contrasts(entries):
    by_case = defaultdict(list)
    for e in entries:
        if e["slot"]["kind"] in ("decision", "observation"):
            by_case[(e["case"]["case_id"], e["slot"].get("receiver"))].append(e)
    records, cells, interactions = [], [], []
    for (_, receiver), members in by_case.items():
        case = members[0]["case"]
        main = [e for e in members if e["slot"].get("analysis_family") == "main" and e["slot"]["kind"] == "decision"]
        hidden = [e for e in main if e["slot"].get("analysis_group") in ("H1", "H2") and e["slot"].get("version", 1) in (None, 1)]
        normal = {}
        for version in (1, 2):
            normal[version] = [e for e in main if e["slot"].get("analysis_group") in ("N1", "N2") and e["slot"].get("version", 1) == version]
            if normal[version]:
                record = _contrast(case, receiver, "main", "normal_minus_hidden", normal[version], hidden, version)
                record["shared_hidden_version"] = 1
                records.append(record)
        if normal.get(2):
            records.append(_contrast(case, receiver, "message_version", "partner_v2_minus_v1", normal[2], normal[1], 2))
        families = defaultdict(lambda: {"N": [], "H": []})
        for e in members:
            slot = e["slot"]
            if slot["kind"] != "decision":
                continue
            memberships = [(slot.get("analysis_family"), slot.get("analysis_group"))]
            memberships += [(v["analysis_family"], v["analysis_group"]) for v in slot.get("additional_analysis_memberships", [])]
            for family, group in memberships:
                if family in ("source", "full_facts", "role") or str(family).startswith("restore_"):
                    if group and group[0] in ("N", "H"):
                        families[family][group[0]].append(e)
        for family, sides in families.items():
            left, right = sides["N"], sides["H"]
            if family == "source":
                left, right, name = right, left, "A_label_minus_unsigned"
            elif family.startswith("restore_"):
                right, name = hidden, family+"_minus_hidden"
            else:
                name = "full_normal_minus_full_hidden" if family == "full_facts" else "role_normal_minus_role_hidden"
            records.append(_contrast(case, receiver, family, name, left, right))
        condition = {name: [e for e in members if e["slot"].get("condition") == name]
                     for name in ("pressure_correct", "pressure_incorrect", "matched_correct", "matched_wrong", "placeholder", "original", "explicit")}
        for wrong, correct in (("pressure_incorrect", "pressure_correct"), ("matched_wrong", "matched_correct")):
            if condition[wrong]:
                records.append(_contrast(case, receiver, "pressure", "incorrect_minus_matched_correct", condition[wrong], condition[correct], repeated=False))
        if condition["placeholder"]:
            records.append(_contrast(case, receiver, "placeholder", "placeholder_minus_hidden", condition["placeholder"], hidden, repeated=False))
        if condition["explicit"]:
            records.append(_contrast(case, receiver, "observation", "explicit_minus_original", condition["explicit"], condition["original"], repeated=False))
        if case["module"] != "minesweeper_fact_advice":
            continue
        partition_records = {}
        for partition in (0, 1):
            grid = {(fact, advice): [e for e in members if e["slot"].get("partition") == partition
                    and e["slot"].get("include_facts") is fact and e["slot"].get("include_advice") is advice]
                    for fact in (False, True) for advice in (False, True)}
            sides = {key: _side(values) for key, values in grid.items()}
            for (fact, advice), side in sides.items():
                cells.append(dict(case_id=case["case_id"], receiver=receiver, partition=partition,
                                  include_facts=fact, include_advice=advice, **side))
            for name, left, right in (("facts_without_advice", (True, False), (False, False)),
                                     ("advice_with_facts", (True, True), (True, False)),
                                     ("facts_with_advice", (True, True), (False, True)),
                                     ("advice_without_facts", (False, True), (False, False))):
                record = _contrast(case, receiver, "fact_advice", name, grid[left], grid[right], partition=partition)
                records.append(record)
                partition_records[(partition, name)] = record
            names = set().union(*(side["outcome_means"].keys() for side in sides.values()))
            values = {}
            for metric in names:
                full = all(side["outcome_denominators"].get(metric) == side["planned"] == 6 for side in sides.values())
                y = {key: side["outcome_means"].get(metric) for key, side in sides.items()}
                values[metric] = {"mean_change": (y[(True, True)]-y[(True, False)]-y[(False, True)]+y[(False, False)]) if full else None}
            record = dict(case_id=case["case_id"], module=case["module"], stratum=case.get("stratum"), receiver=receiver,
                          version=1, partition=partition, family="fact_advice_interaction", contrast="facts_by_advice_interaction",
                          cluster=case["case_id"], distances=None, outcome_changes=values,
                          missing=sum(side["missing"] for side in sides.values()), exploratory=True)
            interactions.append(record)
            records.append(record)
        for name in ("facts_without_advice", "advice_with_facts"):
            before, after = partition_records[(0, name)], partition_records[(1, name)]
            bd, ad = before["distances"], after["distances"]
            changes = {}
            for metric in before["outcome_changes"].keys() | after["outcome_changes"].keys():
                x = before["outcome_changes"].get(metric, {}).get("mean_change")
                y = after["outcome_changes"].get(metric, {}).get("mean_change")
                changes[metric] = {"mean_change": y-x if _numeric(x) and _numeric(y) else None}
            certificates = case.get("data", {}).get("certificates", [])
            own_safe = bool(certificates[1]["provably_safe"]["A" if receiver == 0 else "B"]) if certificates else None
            records.append(dict(case_id=case["case_id"], module=case["module"], stratum=case.get("stratum"), receiver=receiver,
                version=1, partition=None, family="information_partition", contrast="asymmetric_minus_complementary_"+name,
                cluster=case["case_id"], partition_role="receiver_own_safe" if own_safe else "sender_own_safe" if own_safe is not None else "unspecified",
                distances={"adjusted_distance_change": ad["adjusted_distance"]-bd["adjusted_distance"]} if ad and bd else None,
                outcome_changes=changes, missing=before["missing"]+after["missing"], exploratory=True))
    return records, cells, interactions


def _conditions(entries):
    fields = ("module", "stratum", "kind", "sender", "receiver", "version", "partition", "condition", "analysis_family")
    groups = defaultdict(list)
    for e in entries:
        groups[(e["slot"]["module"], e["case"].get("stratum"), *(e["slot"].get(k) for k in fields[2:]))].append(e)
    results = []
    for key, members in sorted(groups.items(), key=lambda item: str(item[0])):
        values, settlements = defaultdict(list), Counter()
        for e in members:
            for metric, value in e["numeric"].items():
                values[metric].append(value)
            if e["outcomes"].get("settlement_status"):
                settlements[e["outcomes"]["settlement_status"]] += 1
        results.append(dict(zip(fields, key)) | dict(planned=len(members), recorded=sum(e["recorded"] for e in members),
            saved=sum(e["response"].get("status") in SAVED for e in members),
            failed=sum(e["response"].get("status") in ("failed", "unknown", "failure") for e in members),
            skipped=sum(e["response"].get("status") == "skipped" for e in members), unreturned=sum(not e["response"] for e in members),
            format_valid=sum(e["response"].get("status") in SAVED and e["score"].get("format_valid") is True for e in members),
            legal=sum(e["response"].get("status") in SAVED and e["score"].get("legal") is True for e in members),
            outcome_means={k: _mean(v) for k, v in values.items()}, outcome_denominators={k: len(v) for k, v in values.items()},
            settlement_counts=dict(settlements)))
    return results


def _primary(cases, state_results):
    results = []
    for module in ("minesweeper_natural", "menu", "negotiation", "kitchen"):
        for receiver in ((0, 1) if module in ("minesweeper_natural", "menu") else (1,)):
            eligible = [c for c in cases if c["module"] == module]
            if module == "minesweeper_natural":
                eligible = [c for c in eligible if c.get("stratum") == "complementary"]
            elif module == "menu":
                eligible = [c for c in eligible if c.get("stratum") == "level_3" or c.get("data", {}).get("difficulty_stratum") == "level_3"]
            ids = {c["case_id"] for c in eligible}
            complete = [v for v in state_results if v["case_id"] in ids and v["receiver"] == receiver and v["version"] == 1 and v["distances"]]
            estimate = _estimate(complete, lambda v: v["distances"]["adjusted_distance"])
            item = dict(module=module, receiver=receiver, expected_states=len(eligible), complete_states=len(complete),
                        complete_clusters=estimate["complete_clusters"], primary_message_version=1,
                        adjusted_distance=estimate["mean"], interval=estimate["interval"], p_value=None, holm_adjusted_p=None,
                        p_reason="incomplete_or_missing_first_version_groups", exploratory=True)
            if len(complete) == len(eligible) and complete:
                rng, distributions = np.random.default_rng(42), []
                for value in complete:
                    labels = [x for g in GROUPS for x in value["groups"][g]]
                    mapping = {label: i for i, label in enumerate(sorted(set(labels)))}
                    codes = np.asarray([mapping[x] for x in labels])
                    perms = np.asarray([rng.permutation(codes) for _ in range(9999)])
                    probs = [np.stack([(perms[:, start:start+3] == i).mean(axis=1) for i in range(len(mapping))], axis=1) for start in (0, 3, 6, 9)]
                    distance = lambda x, y: np.abs(x-y).sum(axis=1)*.5
                    cross = sum(distance(probs[n], probs[h]) for n in (0, 1) for h in (2, 3))/4
                    within = (distance(probs[0], probs[1])+distance(probs[2], probs[3]))/2
                    distributions.append((value["cluster"], cross-within))
                by = defaultdict(list)
                for cluster, values in distributions:
                    by[cluster].append(values)
                simulated = np.mean([np.mean(values, axis=0) for values in by.values()], axis=0)
                item.update(p_value=float((1+np.sum(simulated >= item["adjusted_distance"]-1e-12))/10000), p_reason=None)
            results.append(item)
    available = sorted(((i, r["p_value"]) for i, r in enumerate(results) if r["p_value"] is not None), key=lambda x: x[1])
    peak = 0.
    for rank, (i, p) in enumerate(available):
        peak = max(peak, min(1., p*(6-rank)))
        results[i]["holm_adjusted_p"] = peak
    return results


def summarize(manifest, rows, attempts):
    entries = _entries(manifest, rows)
    contrasts, cells, interactions = _contrasts(entries)
    states = [r for r in contrasts if r["family"] == "main"]
    modules = []
    for module in sorted({e["slot"]["module"] for e in entries}):
        members = [e for e in entries if e["slot"]["module"] == module]
        saved = [e for e in members if e["response"].get("status") in SAVED]
        decisions = [e for e in saved if e["slot"]["kind"] == "decision"]
        modules.append(dict(module=module, planned=len(members), receipts=sum(bool(e["response"]) for e in members),
            successful=len(saved), failed=sum(e["response"].get("status") in ("failed", "unknown", "failure") for e in members),
            skipped=sum(e["response"].get("status") == "skipped" for e in members),
            format_valid=sum(e["score"].get("format_valid") is True for e in saved), unscored_saved=sum(not e["score"] for e in saved),
            decision_successful=len(decisions), decision_format_valid=sum(e["score"].get("format_valid") is True for e in decisions),
            decision_legal=sum(e["score"].get("legal") is True for e in decisions)))
    sources = []
    for e in entries:
        if e["slot"]["module"] == "minesweeper_fact_advice" and e["slot"]["kind"] == "message":
            score, response = e["score"], e["response"]
            status = ("unreturned" if not response else "service_or_dependency_failure" if response.get("status") not in SAVED else
                      "unscored" if not score else "parse_invalid" if score.get("format_valid") is not True else
                      "active_abstention" if score.get("disabled") else "recommendation_present")
            sources.append(dict(case_id=e["case"]["case_id"], partition=e["slot"]["partition"], sender=e["slot"]["sender"],
                                status=status, legal=score.get("legal"), outcomes=score.get("outcomes", {})))
    attempts = [dict(a) for a in attempts]
    known = sum(a.get("known_credits") or 0 for a in attempts)*POLICY["reference_cny_per_credit"]
    starts = [dt.datetime.fromisoformat(a["started"]) for a in attempts if a.get("started")]
    ends = [dt.datetime.fromisoformat(a["finished"]) for a in attempts if a.get("finished")]
    return dict(scope="Exploratory local influence; not stable relations or self-organization",
        primary=_primary(manifest["cases"], states), state_effects=states, conditions=_conditions(entries),
        contrasts=contrasts, contrast_summary=_aggregate(contrasts), factorial_cells=cells, factorial_interactions=interactions,
        recommendation_sources=sources, modules=modules,
        counts=dict(planned_slots=len(manifest["slots"]), recorded_slots=sum(e["recorded"] for e in entries),
                    terminal_receipts=sum(bool(e["response"]) for e in entries), actual_attempts=len(attempts),
                    retries=sum(a.get("attempt", 1) > 1 for a in attempts),
                    unknown_billing=sum(a.get("known_credits") is None for a in attempts),
                    failed_attempts=sum(a.get("state") in ("error", "unknown") for a in attempts)),
        successful_usage_reference_cny=known, actual_request_window_minutes=(max(ends)-min(starts)).total_seconds()/60 if starts and ends else None,
        interval_scope="Conditional on saved messages/replies; whole state/board or kitchen recipe resampling. Missing outcome pairs excluded and counted.",
        recommendation_scope="Advice can carry inferences; active abstention is retained and technical missingness is distinct. Correctness strata are descriptive.")


def _number(value):
    return "未测／缺失" if value is None else f"{value:.3f}"


def _direction(receiver):
    return "B→A" if receiver == 0 else "A→B"


def write_report(root, manifest, verification):
    root = Path(root)
    if (root/"requests.sqlite3").exists():
        db = sqlite3.connect((root/"requests.sqlite3").resolve().as_uri()+"?mode=ro", uri=True)
        db.row_factory = sqlite3.Row
        try:
            rows, attempts = list(db.execute("SELECT * FROM slots")), list(db.execute("SELECT * FROM attempts"))
        finally:
            db.close()
    else:
        rows, attempts = [], []
    result = summarize(manifest, rows, attempts)
    state = json.loads((root/"state.json").read_text(encoding="utf-8")) if (root/"state.json").exists() else {}
    result.update(verification=verification, account_used_change_reference_cny=state.get("account_used_change_reference_cny"))
    write(root/"analysis.json", result)
    c = result["counts"]
    lines = ["# 原子关系探索补强：实验结果报告", "", "当前状态："+state.get("status", "未执行")+"。本轮是两agent局部探索，为后续重复交互提供测量依据。", "",
        f"计划{c['planned_slots']}请求位；收到{c['terminal_receipts']}回执（含失败和跳过）；实际尝试{c['actual_attempts']}次，含{c['retries']}次重试。",
        f"成功用量费用参考{_number(result['successful_usage_reference_cny'])}元；账户变化参考{_number(result['account_used_change_reference_cny'])}元。两者不能相加，{c['unknown_billing']}次尝试计费未知，均非核实账单。请求窗口{_number(result['actual_request_window_minutes'])}分钟。", "",
        "## 执行与有效性", "", "回执完整、输出有效和任务成功分别统计；依赖跳过不补抽。", "",
        "| 模块 | 回执／计划 | 成功返回 | 格式正确／成功 | 合法决定／返回决定 | 失败／跳过 |", "|---|---:|---:|---:|---:|---:|"]
    for r in result["modules"]:
        lines.append(f"| {NAMES[r['module']]} | {r['receipts']}/{r['planned']} | {r['successful']} | {r['format_valid']}/{r['successful']} | {r['decision_legal']}/{r['decision_successful']} | {r['failed']}/{r['skipped']} |")
    lines += ["", "## 第一版消息的六项预定比较", "", "动作距离衡量选择概率分布变化；调整距离扣除显示和隐藏双方的同输入波动，可能为负，不等于收益或社会地位。", "",
        "| 模块 | 方向 | 完整状态／预定 | 调整距离 | 状态／菜谱区间 | Holm调整p |", "|---|---|---:|---:|---|---:|"]
    for r in result["primary"]:
        lines.append(f"| {NAMES[r['module']]} | {_direction(r['receiver'])} | {r['complete_states']}/{r['expected_states']} | {_number(r['adjusted_distance'])} | {r['interval']} | {_number(r['holm_adjusted_p'])} |")
    lines += ["", "仅上表六项使用9,999次状态内置换和Holm校正。缺失或未评分的预定主组使正式p记NA；已评分无效返回保留INVALID。扫雷主检验限互补层，菜单限level3；厨房菜谱等权。", ""]
    metrics = {"minesweeper_natural": ("joint_proven_safe", "own_proven_safe", "actually_safe"),
               "menu": ("empty_menu", "normalized_quality"), "negotiation": ("potential_A", "potential_B", "accepted_given_legal_allocation", "realized_A", "realized_B"),
               "kitchen": ("conflict_after_plan", "receiver_operation_completed", "receiver_elapsed_steps", "recipe_progress.new_subgoal_count", "all_facts_correct"),
               "minesweeper_fact_advice": ("joint_proven_safe", "actually_safe", "recommendation_delivered")}
    explanations = {"minesweeper_natural": "联合可证明安全＝所有符合联合线索的棋盘都无雷；自身安全只用原始私有数字；实际无雷可能碰巧。互补盘的均匀联合安全命中参照1/12，实际无雷参照9/12。自身足够层只作边界。",
        "menu": "空菜单是合法弃选，保留在比较中。normalized_quality表示相对该状态最优菜单的得分。level3与level1/2分开解释。",
        "negotiation": "potential是提案若成交的潜在价值；realized是对应A接受或拒绝后的兑现价值。no_agreement按规则为零；接受缺失不填零。合法分配的接受率与全体协议率分开。",
        "kitchen": "实际伙伴P固定先执行。冲突、操作完成、耗时和新增局部菜谱目标分别测量；子目标不是累计进度，也不是完整送餐。只覆盖5菜谱、同一流程家族。",
        "minesweeper_fact_advice": "四格开关显式数字与推荐。坐标推荐也可能传递推断；主动无建议与技术缺失分开。两划分棋盘及联合安全格相同，私有数字确实变化。"}
    for module in NAMES:
        lines += ["## "+NAMES[module], "", explanations[module], "",
                  "后果以均值（已知分母）显示。布尔指标0—1表示比例；分母不同或缺失时不能把已知子集均值直接解释为因果收益。", "",
                  "| 类型／层／方向／版本／划分 | 条件 | 返回／计划 | 后果均值（分母） |", "|---|---|---:|---|"]
        for r in result["conditions"]:
            if r["module"] != module or r["kind"] not in ("decision", "observation"):
                continue
            values = [f"{key}={r['outcome_means'][key]:.3f}（n={r['outcome_denominators'][key]}）" for key in metrics[module] if key in r["outcome_means"]]
            label = f"{r['kind']}／{r['stratum']}／{_direction(r['receiver'])}／v{r['version']}／{r['partition']}"
            lines.append(f"| {label} | {r['condition']} | {r['saved']}/{r['planned']} | {'；'.join(values) or '后果未测／缺失'} |")
        lines += ["", "| 配对机制／方向／版本／划分 | 完整行为状态 | 调整距离或其变化 | 后果变化：处理减参照 |", "|---|---:|---:|---|"]
        for r in result["contrast_summary"]:
            if r["module"] != module:
                continue
            behavior = r["behavior"].get("adjusted_distance", r["behavior"].get("adjusted_distance_change", {}))
            values = [f"{key}={v['mean']:.3f}（{v['complete_states']}完整状态）" for key, v in r["outcome_changes"].items() if key in metrics[module] and v["mean"] is not None]
            label = f"{r['contrast']}／{r['stratum']}／{_direction(r['receiver'])}／v{r['version']}／{r['partition']}"
            lines.append(f"| {label} | {r['complete_behavior_states']}/{r['expected_states']} | {_number(behavior.get('mean'))} | {'；'.join(values) or '后果差未测／缺失'} |")
        if module == "minesweeper_fact_advice":
            counts = Counter(v["status"] for v in result["recommendation_sources"])
            lines += ["", "推荐来源覆盖："+ "；".join(f"{key}={value}" for key, value in sorted(counts.items()))+"。",
                      "事实×推荐交互是推荐作用在有／无数字时的差。同盘两方向、两划分不算独立棋盘；推荐正确性分层只是描述。"]
        lines.append("")
    lines += ["## 解释与下一步", "", "机制与第二版消息比较均为探索描述，没有新增p检验。区间条件于保存的消息和回复，重抽整个棋盘／状态或菜谱簇；共享隐藏基线保持关联。后果配对只在两边均完整已知时计算，缺失状态和分母保留。",
              "", "全数字也差时先查能力和表达；自然消息弱而标准事实有效时查信息传递与建议干扰。改选、收益改善、来源提示与局部执行后果分开，不能直接证明信任、持续关系、自组织或跨模型普遍性。",
              "", "详细组分布、原始距离、双方波动参照、配对区间及分母见analysis.json。小诊断子集、6个高难度菜单状态与5个厨房菜谱簇限制推广；不追加请求追求正结果。"]
    (root/"结果报告.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    return result
