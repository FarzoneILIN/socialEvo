import tempfile
from pathlib import Path
import unittest

from utils.client import Reply, ApiError
from atomic_relation_followup.protocol import POLICY, BudgetStop
from atomic_relation_followup.runtime import Ledger, WorkerLock


class FakeClient:
    def __init__(self):self.calls=0
    def complete(self,messages):
        self.calls+=1
        return Reply('{"choice":"wait"}',dict(prompt_tokens=10,completion_tokens=10,total_tokens=20),POLICY['model'],'stop')


class RuntimeTests(unittest.TestCase):
    def slot(self):return dict(slot_id='kitchen_t/decision',case_id='kitchen_t',module='kitchen',kind='decision',dependencies=[])

    def test_saved_result_is_not_resampled_and_input_is_bound(self):
        with tempfile.TemporaryDirectory() as tmp,WorkerLock(Path(tmp)):
            slot=self.slot();ledger=Ledger(tmp,{'slots':[slot]});client=FakeClient()
            try:
                first=ledger.ask(slot,[{'role':'user','content':'test'}],client)
                self.assertEqual(first,ledger.ask(slot,[{'role':'user','content':'test'}],client))
                self.assertEqual(client.calls,1)
                with self.assertRaises(BudgetStop):ledger.ask(slot,[{'role':'user','content':'changed'}],client)
            finally:ledger.db.close()

    def test_stranded_attempt_becomes_unknown_and_is_never_retried(self):
        with tempfile.TemporaryDirectory() as tmp,WorkerLock(Path(tmp)):
            slot=self.slot();ledger=Ledger(tmp,{'slots':[slot]});client=FakeClient()
            messages=[{'role':'user','content':'test'}]
            from atomic_relation_followup.protocol import canonical,digest,stamp
            with ledger.db:
                ledger.db.execute('INSERT INTO slots VALUES(?,?,?,?,?,?,?,?,?,?)',(slot['slot_id'],slot['case_id'],'kitchen','decision','pending',canonical(messages),digest(messages),None,None,None))
                ledger.db.execute('INSERT INTO attempts(slot,module,stage,attempt,state,started,reserved_credits) VALUES(?,?,?,?,?,?,?)',(slot['slot_id'],'kitchen','A',1,'inflight',stamp(),.1))
            ledger.db.close();ledger=Ledger(tmp,{'slots':[slot]})
            try:
                response=ledger.ask(slot,messages,client)
                self.assertEqual(response['error'],'interrupted_billing_unknown');self.assertEqual(client.calls,0)
                self.assertEqual(ledger.counts()['unknown_billing_attempts'],1)
            finally:ledger.db.close()

    def test_lock_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(BudgetStop):Ledger(tmp,{'slots':[]})

    def test_structural_skip_has_no_attempt(self):
        with tempfile.TemporaryDirectory() as tmp,WorkerLock(Path(tmp)):
            slot=self.slot();ledger=Ledger(tmp,{'slots':[slot]})
            try:
                r=ledger.skip(slot,'no_agreement')
                self.assertEqual(r['status'],'skipped');self.assertEqual(ledger.counts()['actual_requests'],0)
            finally:ledger.db.close()

    def test_missing_metadata_uses_saved_usage_and_preserves_unknown_reserves(self):
        from unittest.mock import patch
        from atomic_relation_followup.protocol import stamp
        with tempfile.TemporaryDirectory() as tmp,WorkerLock(Path(tmp)):
            slot=self.slot();ledger=Ledger(tmp,{'slots':[slot]})
            try:
                with ledger.db:
                    ledger.db.execute('INSERT INTO attempts(slot,module,stage,attempt,state,started,finished,reserved_credits,known_credits) VALUES(?,?,?,?,?,?,?,?,?)',
                        ('prior','kitchen','A',1,'reply',stamp(),stamp(),200.0,.001))
                # An unavailable account endpoint does not make a known tiny reply
                # permanently consume its maximum output reservation.
                self.assertIsNone(ledger._budget_check(slot,1,.1))
                with ledger.db:
                    ledger.db.execute("UPDATE attempts SET known_credits=NULL,state='unknown'")
                with self.assertRaises(BudgetStop):ledger._budget_check(slot,1,.1)
            finally:ledger.db.close()

    def test_final_quality_checks_small_modules_below_rolling_minimum(self):
        with tempfile.TemporaryDirectory() as tmp,WorkerLock(Path(tmp)):
            slot=self.slot();ledger=Ledger(tmp,{'slots':[slot]})
            try:
                ledger.ask(slot,[{'role':'user','content':'fixture'}],FakeClient())
                ledger.save_score(slot,{'format_valid':False})
                self.assertNotIn('kitchen',ledger.module_stops)
                ledger.finalize_quality(['kitchen'])
                self.assertEqual(ledger.module_stops['kitchen'],'final_format_valid_rate_below_95pct')
            finally:ledger.db.close()


if __name__=='__main__':unittest.main()
