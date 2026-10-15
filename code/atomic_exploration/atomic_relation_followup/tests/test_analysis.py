import copy
import json
import tempfile
from pathlib import Path
import unittest

from atomic_relation_followup import analysis as a


def case(cid="board", module="minesweeper_natural", stratum="complementary", **data):
    return {"case_id": cid, "module": module, "stratum": stratum, "data": data}


def row(slot, category="x", outcomes=None, **score_fields):
    score = dict(format_valid=True, legal=True, exact_choice_category=category, outcomes=outcomes or {})
    score.update(score_fields)
    return {"slot": slot["slot_id"], "response_json": json.dumps({"status": "saved"}), "score_json": json.dumps(score)}


def main(c, *, second=False, hidden_version=1, normal="n", hidden="h"):
    slots, rows = [], []
    for version in ((1, 2) if second else (1,)):
        for group in (1, 2):
            for rep in range(3):
                slot = dict(slot_id=f"{c['case_id']}/N_v{version}_g{group}_r{rep}", case_id=c["case_id"], module=c["module"],
                            kind="decision", sender=0, receiver=1, condition="normal", group=group, rep=rep, version=version,
                            analysis_family="main", analysis_group=f"N{group}")
                slots.append(slot)
                category = normal if version == 1 else hidden
                rows.append(row(slot, category, {"joint_proven_safe": category == "n"}))
    for group in (1, 2):
        for rep in range(3):
            slot = dict(slot_id=f"{c['case_id']}/H_g{group}_r{rep}", case_id=c["case_id"], module=c["module"],
                        kind="decision", sender=0, receiver=1, condition="hidden", group=group, rep=rep, version=hidden_version,
                        analysis_family="main", analysis_group=f"H{group}")
            slots.append(slot)
            rows.append(row(slot, hidden, {"joint_proven_safe": hidden == "n"}))
    return slots, rows


class AnalysisTests(unittest.TestCase):
    def summary(self, cases, slots, rows):
        return a.summarize({"cases": cases, "slots": slots}, rows, [])

    def primary(self, result, module="minesweeper_natural", receiver=1):
        return next(v for v in result["primary"] if v["module"] == module and v["receiver"] == receiver)

    def test_symmetric_distance_keeps_negative_noise_adjustment(self):
        d = a.distances({"N1": ["a"]*3, "N2": ["b"]*3, "H1": ["a"]*3, "H2": ["b"]*3})
        self.assertEqual((d["raw_distance"], d["repeat_reference"], d["adjusted_distance"]), (.5, 1, -.5))

    def test_hidden_none_is_shared_and_v2_does_not_pollute_primary(self):
        c = case()
        slots, rows = main(c, second=True, hidden_version=None)
        result = self.summary([c], slots, rows)
        primary = self.primary(result)
        self.assertEqual(primary["complete_states"], 1)
        self.assertEqual(primary["adjusted_distance"], 1)
        self.assertIsNotNone(primary["p_value"])
        v2 = next(v for v in result["state_effects"] if v["version"] == 2)
        self.assertEqual(v2["distances"]["adjusted_distance"], 0)
        self.assertEqual(v2["shared_hidden_version"], 1)
        self.assertEqual(len(result["primary"]), 6)

    def test_unissued_and_unscored_work_is_missing_not_invalid_or_zero(self):
        c = case()
        slots, rows = main(c)
        for partial in (rows[:-1], rows[:-1]+[{"slot": rows[-1]["slot"], "response_json": rows[-1]["response_json"], "score_json": None}]):
            result = self.summary([c], slots, partial)
            self.assertIsNone(self.primary(result)["p_value"])
            self.assertEqual(result["state_effects"][0]["missing"], 1)
            self.assertNotIn("INVALID", result["state_effects"][0]["groups"]["H2"])
        result = self.summary([c], slots, [])
        self.assertEqual(result["state_effects"][0]["missing"], 12)
        self.assertEqual(self.primary(result)["expected_states"], 1)

    def test_scored_invalid_saved_reply_stays_in_formal_groups(self):
        c = case()
        slots, rows = main(c)
        rows[0] = row(slots[0], "INVALID", format_valid=False, legal=False)
        result = self.summary([c], slots, rows)
        self.assertIn("INVALID", result["state_effects"][0]["groups"]["N1"])
        self.assertEqual(self.primary(result)["complete_states"], 1)
        self.assertIsNotNone(self.primary(result)["p_value"])

    def test_boundary_layer_excluded_from_minesweeper_primary(self):
        c, boundary = case(), case("boundary", stratum="independent_safe")
        s1, r1 = main(c)
        s2, r2 = main(boundary, normal="h")
        result = self.summary([c, boundary], s1+s2, r1+r2)
        self.assertEqual(self.primary(result)["expected_states"], 1)
        self.assertEqual(self.primary(result)["adjusted_distance"], 1)

    def test_kitchen_uses_recipe_equal_weight_and_nested_numeric_outcomes(self):
        cases = [case("k1", "kitchen", "sensitive", recipe_file="r1"),
                 case("k2", "kitchen", "sensitive", recipe_file="r1"),
                 case("k3", "kitchen", "boundary", recipe_file="r2")]
        slots, rows = [], []
        for c in cases:
            ss, rr = main(c, normal="n" if c["data"]["recipe_file"] == "r1" else "h")
            for r, slot in zip(rr, ss):
                score = json.loads(r["score_json"])
                score["outcomes"]["recipe_progress"] = {"new_subgoal_count": int(slot["condition"] == "normal")}
                r["score_json"] = json.dumps(score)
            slots += ss
            rows += rr
        result = self.summary(cases, slots, rows)
        primary = self.primary(result, "kitchen")
        self.assertEqual(primary["adjusted_distance"], .5)
        self.assertEqual(primary["complete_clusters"], 2)
        self.assertTrue(any("recipe_progress.new_subgoal_count" in v["outcome_means"] for v in result["conditions"]))

    def test_acceptance_is_linked_to_proposal_and_not_mixed_in_condition_summary(self):
        c = case("neg", "negotiation", "high_conflict")
        proposal = dict(slot_id="neg/proposal", case_id="neg", module="negotiation", kind="decision", sender=0, receiver=1,
                        condition="normal", version=1, group=1, rep=0, analysis_family="main", analysis_group="N1")
        accept = dict(proposal, slot_id="neg/accept", kind="accept", sender=1, receiver=0, proposal_slot_id="neg/proposal")
        rows = [row(proposal, "[1,0,1]", {"potential_A": 4, "potential_B": 6, "realized_A": None, "realized_B": None, "agreement": None}, choice=[1, 0, 1]),
                row(accept, '"reject"', {"realized_A": 0, "realized_B": 0, "agreement": False, "settlement_status": "rejected"})]
        result = self.summary([c], [proposal, accept], rows)
        p = next(v for v in result["conditions"] if v["kind"] == "decision")
        self.assertEqual(p["outcome_means"]["realized_B"], 0)
        self.assertEqual(p["settlement_counts"], {"rejected": 1})
        self.assertEqual(len(result["conditions"]), 2)
        missing = self.summary([c], [proposal, accept], rows[:1])
        p = next(v for v in missing["conditions"] if v["kind"] == "decision")
        self.assertNotIn("realized_B", p["outcome_means"])
        self.assertEqual(p["settlement_counts"], {"acceptance_unobserved": 1})

    def factorial(self):
        c = case("pair", "minesweeper_fact_advice", "paired", certificates=[{}, {"provably_safe": {"A": [[1, 1]], "B": []}}])
        slots, rows = [], []
        for partition in (0, 1):
            for receiver in (0, 1):
                for facts in (False, True):
                    for advice in (False, True):
                        good = partition == 1 or facts and not advice
                        for group in (1, 2):
                            for rep in range(3):
                                slot = dict(slot_id=f"pair/p{partition}/r{receiver}/f{facts}a{advice}/g{group}/{rep}", case_id="pair", module=c["module"],
                                    kind="decision", sender=1-receiver, receiver=receiver, condition=f"facts_{int(facts)}_advice_{int(advice)}",
                                    analysis_family="fact_advice", version=1, partition=partition, include_facts=facts, include_advice=advice, group=group, rep=rep)
                                slots.append(slot)
                                rows.append(row(slot, "safe" if good else "mine", {"joint_proven_safe": good, "actually_safe": good}))
        return c, slots, rows

    def test_fact_advice_cells_interaction_and_paired_partition_effect(self):
        c, slots, rows = self.factorial()
        result = self.summary([c], slots, rows)
        self.assertEqual(len(result["factorial_cells"]), 16)
        self.assertEqual(len(result["factorial_interactions"]), 4)
        effect = next(v for v in result["contrasts"] if v["contrast"] == "facts_without_advice" and v["partition"] == 0 and v["receiver"] == 0)
        self.assertEqual(effect["outcome_changes"]["joint_proven_safe"]["mean_change"], 1)
        effect = next(v for v in result["contrasts"] if v["contrast"] == "advice_with_facts" and v["partition"] == 0 and v["receiver"] == 0)
        self.assertEqual(effect["outcome_changes"]["joint_proven_safe"]["mean_change"], -1)
        interaction = next(v for v in result["factorial_interactions"] if v["partition"] == 0 and v["receiver"] == 0)
        self.assertEqual(interaction["outcome_changes"]["joint_proven_safe"]["mean_change"], -1)
        paired = next(v for v in result["contrasts"] if v["family"] == "information_partition" and v["receiver"] == 0 and v["contrast"].endswith("facts_without_advice"))
        self.assertEqual(paired["outcome_changes"]["joint_proven_safe"]["mean_change"], -1)
        self.assertEqual(paired["partition_role"], "receiver_own_safe")
        self.assertTrue(all(v["p_value"] is None for v in result["primary"]))

    def test_unknown_task_outcome_does_not_create_complete_paired_gain(self):
        c, slots, rows = self.factorial()
        target = next(i for i, s in enumerate(slots) if s["partition"] == 0 and s["receiver"] == 0 and s["include_facts"] and not s["include_advice"])
        score = json.loads(rows[target]["score_json"])
        score["outcomes"] = {}
        rows[target]["score_json"] = json.dumps(score)
        result = self.summary([c], slots, rows)
        effect = next(v for v in result["contrasts"] if v["contrast"] == "facts_without_advice" and v["partition"] == 0 and v["receiver"] == 0)
        self.assertIsNotNone(effect["distances"])
        self.assertIsNone(effect["outcome_changes"]["joint_proven_safe"]["mean_change"])
        self.assertEqual(effect["outcome_changes"]["joint_proven_safe"]["treatment_known"], 5)

    def test_recommendation_abstention_parse_failure_and_missing_remain_distinct(self):
        c, slots, rows = self.factorial()
        sources = []
        for index, (partition, sender) in enumerate(((0, 0), (0, 1), (1, 0), (1, 1))):
            s = dict(slot_id=f"pair/source{index}", case_id="pair", module=c["module"], kind="message", sender=sender, receiver=1-sender,
                     condition="structured_recommendation", version=1, partition=partition)
            sources.append(s)
        rows.append(row(sources[0], "MESSAGE", disabled=True))
        rows.append({"slot": sources[1]["slot_id"], "response_json": json.dumps({"status": "failed"}), "score_json": json.dumps({"format_valid": False})})
        rows.append(row(sources[2], "INVALID", format_valid=False))
        result = self.summary([c], slots+sources, rows)
        self.assertEqual({v["status"] for v in result["recommendation_sources"]},
                         {"active_abstention", "service_or_dependency_failure", "parse_invalid", "unreturned"})

    def test_restoration_and_source_signs_are_preserved(self):
        c = case("menu", "menu", "level_3")
        slots, rows = main(c, normal="h")
        for condition, family, prefix in (("raw_inventory", "restore_inventory", "N"), ("raw_both", "full_facts", "H"), ("full_normal", "full_facts", "N")):
            for group in (1, 2):
                for rep in range(3):
                    s = dict(slots[0], slot_id=f"menu/{condition}/{group}/{rep}", condition=condition, analysis_family=family,
                             analysis_group=prefix+str(group), group=group, rep=rep)
                    if condition == "raw_both":
                        s["additional_analysis_memberships"] = [{"analysis_family": "restore_both", "analysis_group": "N"+str(group)}]
                    slots.append(s)
                    rows.append(row(s, "good", {"joint_proven_safe": True}))
        result = self.summary([c], slots, rows)
        effect = next(v for v in result["contrasts"] if v["family"] == "restore_both")
        self.assertEqual(effect["outcome_changes"]["joint_proven_safe"]["mean_change"], 1)
        full = next(v for v in result["contrasts"] if v["family"] == "full_facts")
        self.assertEqual(full["outcome_changes"]["joint_proven_safe"]["mean_change"], 0)

    def test_empty_legal_menu_is_a_choice_not_invalid(self):
        c = case("menu", "menu", "level_3")
        slots, rows = main(c)
        rows[0] = row(slots[0], "[]", {"empty_menu": True, "normalized_quality": 0})
        result = self.summary([c], slots, rows)
        self.assertIn("[]", result["state_effects"][0]["groups"]["N1"])


if __name__ == "__main__":
    unittest.main()
