"""Independent offline runtime audit; no provider requests or real keys."""
import concurrent.futures as cf
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest import mock

from utils.client import ApiError, Reply
from atomic_relation_followup import runtime
from atomic_relation_followup import runner
from atomic_relation_followup.protocol import BudgetStop, POLICY, canonical, digest, stamp


MESSAGES = [{"role": "user", "content": "offline input"}]


def slot(index=0, module="kitchen"):
    return {"slot_id": f"{module}_{index}/decision", "case_id": f"{module}_{index}",
            "module": module, "kind": "decision", "dependencies": []}


class ScriptedClient:
    def __init__(self, errors=(), content='{"choice":"wait(1)"}', usage=None):
        self.errors = list(errors)
        self.content = content
        self.usage = usage or {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20}
        self.calls = 0
        self.lock = threading.Lock()

    def complete(self, messages):
        with self.lock:
            self.calls += 1
            error = self.errors.pop(0) if self.errors else None
        if error:
            raise ApiError(error)
        return Reply(self.content, self.usage, POLICY["model"], "stop")


class RuntimeAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.worker = runtime.WorkerLock(self.root).__enter__()
        self.ledgers = []

    def tearDown(self):
        for ledger in self.ledgers:
            try:
                ledger.db.close()
            except Exception:
                pass
        self.worker.__exit__(None, None, None)
        self.temp.cleanup()

    def ledger(self, slots):
        value = runtime.Ledger(self.root, {"slots": slots})
        self.ledgers.append(value)
        return value

    def seed_attempt(self, ledger, definition, number=1, state="inflight", error=None, reserve=.12):
        with ledger.db:
            ledger.db.execute("INSERT OR IGNORE INTO slots VALUES(?,?,?,?,?,?,?,?,?,?)", (
                definition["slot_id"], definition["case_id"], definition["module"], definition["kind"],
                "pending", canonical(MESSAGES), digest(MESSAGES), None, None, None))
            ledger.db.execute("INSERT INTO attempts(slot,module,stage,attempt,state,error,started,finished,reserved_credits) VALUES(?,?,?,?,?,?,?,?,?)", (
                definition["slot_id"], definition["module"], runtime.CAPS[definition["module"]][2], number,
                state, error, stamp(), None if state == "inflight" else stamp(), reserve))

    def test_retry_whitelist_max_two_and_exact_delays(self):
        definition = slot()
        ledger = self.ledger([definition])
        client = ScriptedClient(["http_500"] * 3)
        with mock.patch.object(runtime.time, "sleep") as sleep:
            response = ledger.ask(definition, MESSAGES, client)
        self.assertEqual(client.calls, 3)
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [2, 5])
        self.assertEqual(response["status"], "failed")
        self.assertEqual(ledger.counts()["retries"], 2)
        self.assertEqual(ledger.counts()["unknown_billing_attempts"], 3)

    def test_unpermitted_http_failures_never_retry(self):
        definitions = [slot(i) for i in range(2)]
        ledger = self.ledger(definitions)
        for definition, code in zip(definitions, ("http_408", "http_422")):
            client = ScriptedClient([code])
            with mock.patch.object(runtime.time, "sleep") as sleep:
                response = ledger.ask(definition, MESSAGES, client)
            self.assertEqual(client.calls, 1)
            self.assertEqual(response["error"], code)
            sleep.assert_not_called()

    def test_invalid_completed_reply_is_not_redrawn(self):
        definition = slot()
        ledger = self.ledger([definition])
        client = ScriptedClient(content="this completed output is invalid JSON")
        first = ledger.ask(definition, MESSAGES, client)
        second = ledger.ask(definition, MESSAGES, client)
        self.assertEqual(first, second)
        self.assertEqual(client.calls, 1)

    def test_recovery_stranded_five_slots_rebuilds_stop_without_sampling(self):
        definitions = [slot(i) for i in range(6)]
        ledger = self.ledger(definitions)
        for definition in definitions[:5]:
            self.seed_attempt(ledger, definition)
        ledger.db.close()
        restored = self.ledger(definitions)
        client = ScriptedClient()
        self.assertIn("kitchen", restored.module_stops)
        self.assertEqual(restored.counts()["unknown_billing_attempts"], 5)
        self.assertEqual(restored.counts()["actual_requests"], 5)
        with self.assertRaises(BudgetStop):
            restored.ask(definitions[5], MESSAGES, client)
        self.assertEqual(client.calls, 0)

    def test_recovery_preserves_launch_and_concurrency(self):
        definition = slot()
        ledger = self.ledger([definition])
        launch = ledger.launch.copy()
        ledger.set_meta("concurrency", 2)
        ledger.db.close()
        restored = self.ledger([definition])
        self.assertEqual(restored.launch, launch)
        self.assertEqual(restored.limit, 2)

    def test_cumulative_module_retry_cap_survives_recovery(self):
        definitions = [slot(0), slot(1)]
        with mock.patch.dict(runtime.CAPS, {"kitchen": (10, 1, "A")}):
            ledger = self.ledger(definitions)
            self.seed_attempt(ledger, definitions[0], 1, "error", "http_500")
            self.seed_attempt(ledger, definitions[0], 2, "error", "http_500")
            ledger.fail(definitions[0], "http_500")
            ledger.db.close()
            restored = self.ledger(definitions)
            client = ScriptedClient(["http_500"] * 3)
            with mock.patch.object(runtime.time, "sleep"):
                response = restored.ask(definitions[1], MESSAGES, client)
            self.assertEqual(response["error"], "retained_retry_cap")
            self.assertEqual(client.calls, 1)
            self.assertEqual(restored.counts(module="kitchen")["retries"], 1)

    def test_concurrent_actual_cap_is_atomic(self):
        definitions = [slot(i) for i in range(8)]
        ledger = self.ledger(definitions)
        client = ScriptedClient()
        with mock.patch.dict(POLICY, {"actual_request_cap": 3}):
            with cf.ThreadPoolExecutor(max_workers=8) as pool:
                futures = [pool.submit(ledger.ask, definition, MESSAGES, client) for definition in definitions]
                for future in futures:
                    try:
                        future.result()
                    except BudgetStop:
                        pass
        self.assertEqual(client.calls, 3)
        self.assertEqual(ledger.counts()["actual_requests"], 3)

    def test_concurrent_cost_reservations_block_extra_dispatch(self):
        definitions = [slot(i) for i in range(8)]
        ledger = self.ledger(definitions)
        reserved = (sum(len(m["content"].encode()) + 256 for m in MESSAGES) * 2.5 + POLICY["output_cap"] * 15) / 1e6
        release = threading.Event()
        client = ScriptedClient()
        original = client.complete
        def delayed(messages):
            reply = original(messages)
            release.wait(timeout=2)
            return reply
        client.complete = delayed
        with mock.patch.dict(POLICY, {"reference_cny_cap": reserved * .5 * 2.1}):
            with cf.ThreadPoolExecutor(max_workers=8) as pool:
                futures = [pool.submit(ledger.ask, definition, MESSAGES, client) for definition in definitions]
                # First blocked budget check supplies a bounded release point.
                for future in futures:
                    try:
                        future.result(timeout=3)
                    except BudgetStop:
                        release.set()
        self.assertEqual(client.calls, 2)
        self.assertEqual(ledger.counts()["actual_requests"], 2)

    def test_secret_input_rejected_and_output_redacted(self):
        definition = slot()
        secret = "offline-Audit-Key-with-slash/and+suffix"
        with mock.patch.dict(os.environ, {"SOCIALEVO_API_KEY": secret}):
            ledger = self.ledger([definition])
        client = ScriptedClient(content=json.dumps({"echo": secret}))
        with self.assertRaisesRegex(BudgetStop, "secret_in_input"):
            ledger.ask(definition, [{"role": "user", "content": secret}], client)
        self.assertEqual(client.calls, 0)
        response = ledger.ask(definition, MESSAGES, client)
        self.assertNotIn(secret, canonical(response))
        self.assertNotIn(secret, ledger.db.execute("SELECT response_json FROM slots").fetchone()[0])

    def test_usage_missing_stops_without_retry(self):
        definition = slot()
        ledger = self.ledger([definition])
        client = ScriptedClient(usage={"prompt_tokens": 10})
        response = ledger.ask(definition, MESSAGES, client)
        self.assertEqual(response["status"], "saved")
        self.assertEqual(ledger.stop, "successful_usage_missing")
        self.assertEqual(client.calls, 1)
        self.assertEqual(ledger.counts()["unknown_billing_attempts"], 1)

    def test_a_paused_module_does_not_stop_independent_a_modules(self):
        definitions = [slot(0), slot(1, "menu")]
        ledger = self.ledger(definitions)
        failed = ScriptedClient(["http_401"])
        ledger.ask(definitions[0], MESSAGES, failed)
        self.assertIn("kitchen", ledger.module_stops)
        client = ScriptedClient()
        response = ledger.ask(definitions[1], MESSAGES, client)
        self.assertEqual(response["status"], "saved")
        self.assertEqual(client.calls, 1)

    def test_stage_scheduler_shares_eight_lanes_and_round_robin_modules(self):
        modules = ("minesweeper_natural", "menu", "negotiation", "kitchen")
        definitions = [slot(i, module) for module in modules for i in range(8)]
        manifest = {"slots": definitions, "cases": [{"case_id": s["case_id"], "module": s["module"]} for s in definitions]}
        ledger = self.ledger(definitions)
        barrier = threading.Barrier(8)
        class ConcurrentClient(ScriptedClient):
            def __init__(self):
                super().__init__()
                self.entered = []
                self.active = 0
                self.max_active = 0
            def complete(self, messages):
                with self.lock:
                    self.entered.append(messages[0]["content"].split("_", 1)[0])
                    order = len(self.entered)
                    self.active += 1
                    self.max_active = max(self.max_active, self.active)
                if order <= 8:
                    barrier.wait(timeout=5)
                reply = super().complete(messages)
                with self.lock:
                    self.active -= 1
                return reply
        client = ConcurrentClient()
        def prepared(case, definition, responses):
            return {"messages": [{"role": "user", "content": definition["slot_id"]}]}
        def scored(case, definition, response, responses):
            return {"format_valid": True}
        with mock.patch.object(runner, "make_client", return_value=client), \
             mock.patch.object(runner, "wrapped_input", side_effect=prepared), \
             mock.patch.object(runner, "scored", side_effect=scored), \
             mock.patch.object(runtime, "allowance_snapshot", return_value={"status": "unavailable"}):
            runner._execute_stage(ledger, manifest, "A")
        self.assertEqual(client.calls, 32)
        self.assertEqual(client.max_active, 8)
        self.assertEqual(sorted(client.entered[:8]), sorted(["minesweeper", "menu", "negotiation", "kitchen"] * 2))
        self.assertEqual(ledger.counts()["inflight"], 0)

    def test_stage_b_is_not_dispatched_with_incomplete_a_gate(self):
        definitions = [slot(0, "menu"), slot(0, "minesweeper_fact_advice")]
        manifest = {"slots": definitions, "cases": [{"case_id": s["case_id"], "module": s["module"]} for s in definitions]}
        # run owns its lock; release this test fixture's already-held worker.
        self.worker.__exit__(None, None, None)
        (self.root / "manifest.json").write_text(canonical(manifest), encoding="utf-8")
        verification = {"ok": True, "stage_A_receipts_present": False, "all_planned_receipts_present": False}
        with mock.patch.object(runner, "verify", return_value=verification), \
             mock.patch.object(runner, "_execute_stage") as execute, \
             mock.patch.object(runner, "report"), \
             mock.patch.object(runtime, "allowance_snapshot", return_value={"status": "unavailable"}):
            runner.run(self.root)
        self.assertEqual([call.args[2] for call in execute.call_args_list], ["A"])
        receipt = json.loads((self.root / "completion_acceptance.json").read_text(encoding="utf-8"))
        self.assertEqual(receipt["status"], "not_completed")


if __name__ == "__main__":
    unittest.main()
