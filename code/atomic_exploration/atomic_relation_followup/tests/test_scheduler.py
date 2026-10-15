"""Bounded offline scheduler mocks; never construct a network client."""
import copy
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from atomic_relation_followup import runner
from atomic_relation_followup.protocol import BudgetStop


def slot(identifier, module="menu", kind="decision", deps=()):
    return {"slot_id": identifier, "case_id": module + "_case", "module": module,
            "kind": kind, "dependencies": list(deps)}


class Cursor:
    def __init__(self, value):
        self.value = value

    def fetchone(self):
        return self.value


class Database:
    def __init__(self, store):
        self.store = store

    def execute(self, query, args=()):
        if query.startswith("SELECT 1 FROM snapshots"):
            return Cursor((1,) if args[0] in self.store.snapshots else None)
        raise AssertionError("unexpected offline scheduler query: " + query)


class Store:
    def __init__(self, root, existing=None, budget_stop=False):
        self.root = Path(root)
        self.lock = threading.RLock()
        self.stop = None
        self.module_stops = {}
        self.limit = 8
        self.db = Database(self)
        self.snapshots = set()
        self.meta = {}
        self.saved = dict(existing or {})
        self.saved_scores = {}
        self.calls = []
        self.inflight = 0
        self.peak_inflight = 0
        self.budget_stop = budget_stop
        self.paid_counter = 0

    def responses(self):
        with self.lock:
            return copy.deepcopy(self.saved)

    def finished(self):
        with self.lock:
            return set(self.saved)

    def counts(self):
        return {"actual_requests": self.paid_counter}

    def snapshot(self, label, stage):
        self.snapshots.add(label)

    def get_meta(self, name):
        return self.meta.get(name)

    def set_meta(self, name, value):
        self.meta[name] = value

    def pause_module(self, module, reason):
        self.module_stops.setdefault(module, reason)

    def state(self, **extra):
        return {"counts": self.counts(), **extra}

    def save_score(self, definition, score):
        self.saved_scores[definition["slot_id"]] = score

    def finalize_quality(self, modules):
        self.quality_checked = set(modules)

    def skip(self, definition, reason):
        response = {"status": "skipped", "error": reason}
        self.saved[definition["slot_id"]] = response
        return response

    def ask(self, definition, messages, client):
        with self.lock:
            # The mock asserts dependencies have terminal receipts at dispatch.
            if not set(definition["dependencies"]) <= set(self.saved):
                raise AssertionError("dispatched_before_dependencies")
            self.calls.append(definition["slot_id"])
            if self.budget_stop:
                # Bound the pre-fix busy loop without ever making a paid call.
                if len(self.calls) >= 3:
                    self.stop = "offline_busy_loop_guard"
                raise BudgetStop("unclassified_pre_dispatch_fixture")
            self.inflight += 1
            self.peak_inflight = max(self.peak_inflight, self.inflight)
            self.paid_counter += 1  # synthetic attempt counter only
        time.sleep(0.015)
        response = {"status": "saved", "content": "offline fixture"}
        with self.lock:
            self.saved[definition["slot_id"]] = response
            self.inflight -= 1
        return response


def manifest(slots):
    modules = sorted({definition["module"] for definition in slots})
    return {"cases": [{"case_id": module + "_case", "module": module} for module in modules], "slots": slots}


class SchedulerTests(unittest.TestCase):
    def execute(self, store, definitions, prepare=None):
        def normal_input(case, definition, responses):
            return {"messages": [{"role": "user", "content": definition["slot_id"]}]}
        with patch.object(runner, "make_client", return_value=object()), \
                patch.object(runner, "wrapped_input", side_effect=prepare or normal_input), \
                patch.object(runner, "scored", side_effect=lambda c, s, r, rs: {"format_valid": True, "response_status": r["status"]}):
            runner._execute_stage(store, manifest(definitions), "A")

    def test_eight_lanes_are_shared_across_modules(self):
        definitions = [slot("menu_" + str(i)) for i in range(9)] + [slot("dnd_" + str(i), "negotiation") for i in range(9)]
        with tempfile.TemporaryDirectory() as folder:
            store = Store(folder)
            self.execute(store, definitions)
        self.assertEqual(len(store.calls), 18)
        self.assertEqual(len(set(store.calls)), 18)
        self.assertGreater(store.peak_inflight, 1)
        self.assertLessEqual(store.peak_inflight, 8)
        self.assertIsNone(store.stop)

    def test_dependencies_resume_and_structural_skip(self):
        definitions = [slot("A", kind="message"), slot("B", kind="message"),
                       slot("proposal", "negotiation", deps=("A", "B")),
                       slot("accept", "negotiation", "accept", ("proposal",))]
        def prepare(case, definition, responses):
            if definition["kind"] == "accept":
                return {"skip": "no_agreement"}
            return {"messages": [{"role": "user", "content": definition["slot_id"]}]}
        with tempfile.TemporaryDirectory() as folder:
            store = Store(folder, {"A": {"status": "saved", "content": "previously saved"}})
            self.execute(store, definitions, prepare)
        self.assertEqual(set(store.calls), {"B", "proposal"})
        self.assertEqual(store.saved["accept"], {"status": "skipped", "error": "no_agreement"})
        self.assertEqual(set(store.saved), {"A", "B", "proposal", "accept"})
        self.assertIsNone(store.stop)

    def test_unclassified_budget_stop_never_redispatches_forever(self):
        with tempfile.TemporaryDirectory() as folder:
            store = Store(folder, budget_stop=True)
            self.execute(store, [slot("pre_dispatch_exception")])
        self.assertEqual(len(store.calls), 1)
        self.assertIn("menu", store.module_stops)
        self.assertEqual(store.paid_counter, 0)
        self.assertNotIn("pre_dispatch_exception", store.saved)

    def test_input_exception_pauses_one_module_while_other_finishes(self):
        def prepare(case, definition, responses):
            if definition["module"] == "menu":
                raise ValueError("offline fixture input exception")
            return {"messages": [{"role": "user", "content": definition["slot_id"]}]}
        with tempfile.TemporaryDirectory() as folder:
            store = Store(folder)
            self.execute(store, [slot("bad"), slot("good", "negotiation")], prepare)
        self.assertIn("menu", store.module_stops)
        self.assertEqual(store.calls, ["good"])
        self.assertIsNone(store.stop)


if __name__ == "__main__":
    unittest.main()
