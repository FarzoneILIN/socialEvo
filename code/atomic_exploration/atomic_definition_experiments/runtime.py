"""Durable, bounded execution for the three-task atomic-definition queue.

This module owns execution only. Frozen prompts, scoring, scientific gates and
replay verification belong to the orchestrator. No response is redrawn because
it is empty, malformed, unsuccessful, or inconvenient for an experiment.
"""
from __future__ import annotations

import concurrent.futures
import copy
import datetime as dt
import json
import math
import os
from pathlib import Path
import threading
import time
from types import SimpleNamespace

from atomic_experiment.client import ApiError, ChatClient
from quick_screen.runner import Store, BudgetStop, canonical, digest, stamp, usage_cost as _base_usage_cost, write
from relation_experiments.storage import redact, usage_valid


POLICY = dict(
    actual_request_cap=650, unique_slot_cap=592, retry_cap=57,
    retries_per_slot=2, retry_delays_seconds=[2, 5],
    per_case_request_cap=650, per_candidate_request_cap=650,
    output_cap=8192, max_input_reference_tokens=20000,
    reference_cny_cap=30, reference_cny_per_credit=.5,
    model='gpt-5.4-2026-03-05', reasoning_effort='medium',
    timeout_seconds=300, concurrency=4, minimum_concurrency=2,
    seed=42, dispatch_minutes=180, max_distinct_failed_slots=5,
)
STAGE_SLOT_CAPS = {'T01': 208, 'T02': 288, 'T03': 96}
STAGE_RETRY_CAPS = {name: math.floor(count * .1)
                    for name, count in STAGE_SLOT_CAPS.items()}
RETRYABLE = frozenset({'http_429', 'http_500', 'http_502', 'http_503',
                       'http_504', 'transport_error_billing_unknown'})
FAILURE_STREAK_STOP = 'failed_attempts_5_distinct_slots_since_success'
_ACTIVE_LOCKS = {}
_LOCK_REGISTRY_GUARD = threading.Lock()


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _utc(value):
    parsed = dt.datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise BudgetStop('launch_time_missing_timezone')
    return parsed.astimezone(dt.timezone.utc)


def usage_cost(usage):
    """Keep historical prices; unavailable cache details use the full input price.

    Required token counters remain strictly validated. This calculation does not
    alter the raw usage saved with a reply or claim to be a provider invoice.
    """
    if not usage_valid(usage):
        return None
    priced_usage = dict(usage)
    if not isinstance(priced_usage.get('prompt_tokens_details'), dict):
        priced_usage['prompt_tokens_details'] = {}
    try:
        cost = _base_usage_cost(priced_usage)
    except (TypeError, ValueError, OverflowError):
        return None
    return cost if cost is not None and math.isfinite(cost) and cost >= 0 else None


class WorkerLock:
    """An OS lock must be held before recovery can mark stranded calls unknown."""

    def __init__(self, root):
        self.root = Path(root).resolve()
        self.path = self.root / 'worker.lock'
        self.file = None

    def __enter__(self):
        self.root.mkdir(parents=True, exist_ok=True)
        with _LOCK_REGISTRY_GUARD:
            if self.root in _ACTIVE_LOCKS:
                raise BudgetStop('another_worker_active')
            self.file = self.path.open('a+b')
            if self.path.stat().st_size == 0:
                self.file.write(b'1')
                self.file.flush()
            self.file.seek(0)
            try:
                if os.name == 'nt':
                    import msvcrt
                    msvcrt.locking(self.file.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(self.file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except (OSError, BlockingIOError):
                self.file.close()
                self.file = None
                raise BudgetStop('another_worker_active') from None
            _ACTIVE_LOCKS[self.root] = self
        return self

    def __exit__(self, *args):
        with _LOCK_REGISTRY_GUARD:
            if self.file is not None:
                try:
                    self.file.seek(0)
                    if os.name == 'nt':
                        import msvcrt
                        msvcrt.locking(self.file.fileno(), msvcrt.LK_UNLCK, 1)
                    else:
                        import fcntl
                        fcntl.flock(self.file.fileno(), fcntl.LOCK_UN)
                finally:
                    self.file.close()
                    self.file = None
                    _ACTIVE_LOCKS.pop(self.root, None)


def _lock_held(root):
    with _LOCK_REGISTRY_GUARD:
        return Path(root).resolve() in _ACTIVE_LOCKS


class QueueStore(Store):
    def __init__(self, root, manifest):
        root = Path(root).resolve()
        if not _lock_held(root):
            raise BudgetStop('worker_lock_required')
        self.manifest = copy.deepcopy(manifest)
        if manifest.get('policy', POLICY) != POLICY:
            raise BudgetStop('runtime_policy_mismatch')
        all_slots = manifest.get('slots', [])
        if len(all_slots) > POLICY['unique_slot_cap'] or len(set(all_slots)) != len(all_slots):
            raise BudgetStop('global_slot_plan_exceeds_cap')
        self.stage = self.wave = None
        self.whitelist = set()
        self.allowed = {}
        self.gates = {}
        self.limit = POLICY['concurrency']
        self.errors = set()
        self._active = 0
        launch_path = root / 'launch.json'
        if not launch_path.exists():
            now = dt.datetime.now(dt.timezone.utc)
            write(launch_path, dict(start_utc=now.isoformat(), dispatch_stop_utc=(
                now + dt.timedelta(minutes=POLICY['dispatch_minutes'])).isoformat()))
        self.launch = _read(launch_path)
        duration = _utc(self.launch['dispatch_stop_utc']) - _utc(self.launch['start_utc'])
        if duration <= dt.timedelta(0) or duration > dt.timedelta(minutes=POLICY['dispatch_minutes']):
            raise BudgetStop('launch_deadline_mismatch')
        super().__init__(root, POLICY)
        self.permit = threading.Condition(self.lock)
        state_path = root / 'state.json'
        if state_path.exists():
            old = _read(state_path)
            self.stop = self.stop or old.get('global_stop') or old.get('stop_reason')
            self.limit = old.get('concurrency', self.limit)
            if self.limit not in (2, 4):
                raise BudgetStop('saved_concurrency_mismatch')
        # Only an exclusively locked worker may reinterpret interrupted calls.
        with self.lock, self.db:
            stranded = list(self.db.execute("SELECT DISTINCT slot FROM attempts WHERE state='inflight'"))
            for row in stranded:
                self.db.execute("UPDATE attempts SET state='unknown',error='interrupted_billing_unknown',finished=? WHERE slot=? AND state='inflight'",
                                (stamp(), row['slot']))
                self.db.execute("UPDATE slots SET status='unknown',error='interrupted_billing_unknown' WHERE slot=?",
                                (row['slot'],))
            events = list(self.db.execute("SELECT slot,state,error FROM attempts WHERE state IN ('reply','error','unknown') ORDER BY COALESCE(finished,started),seq"))
            for row in events:
                if row['state'] == 'reply':
                    self.errors.clear()
                else:
                    self.errors.add(row['slot'])
                    self.limit = 2
                    if len(self.errors) >= POLICY['max_distinct_failed_slots']:
                        self.stop = self.stop or FAILURE_STREAK_STOP
                    if row['error'] in {'http_400', 'http_401', 'http_403', 'http_404'}:
                        self.stop = self.stop or 'nonretryable_' + row['error']
            for row in self.db.execute('SELECT response_json FROM slots WHERE response_json IS NOT NULL'):
                response = json.loads(row['response_json'])
                if usage_cost(response.get('usage')) is None:
                    self.stop = self.stop or 'successful_usage_missing'
                if response.get('model') != POLICY['model']:
                    self.stop = self.stop or 'returned_model_mismatch'

    def _definition(self, stage):
        stages = self.manifest.get('stages', {})
        if isinstance(stages, list):
            definition = next((entry for entry in stages
                               if entry.get('name', entry.get('stage')) == stage), {})
        else:
            definition = stages.get(stage, {})
        if not isinstance(definition, dict):
            raise BudgetStop('invalid_stage_definition')
        cases = definition.get('cases', [case for case in self.manifest.get('cases', [])
                                         if case.get('stage', case['case_id'].split('_')[0]) == stage])
        slots = definition.get('slots', [slot for slot in self.manifest.get('slots', [])
                                         if slot.startswith(stage + '_')])
        return cases, slots

    def stage_counts(self, stage=None):
        stage = stage or self.stage
        with self.lock:
            rows = list(self.db.execute("SELECT * FROM attempts WHERE substr(case_id,1,?)=? AND state!='not_sent'",
                                        (len(stage) + 1, stage + '_'))) if stage else []
            slots = self._definition(stage)[1] if stage else []
            retry_cap = min(STAGE_RETRY_CAPS.get(stage, 0), math.floor(len(slots) * .1))
            return dict(actual_requests=len(rows), retries=sum(row['attempt'] > 1 for row in rows),
                        planned_slots=len(slots), retry_cap=retry_cap,
                        completed_successful_requests=sum(row['state'] == 'reply' for row in rows))

    def set_stage(self, stage):
        if stage not in STAGE_SLOT_CAPS:
            raise BudgetStop('unplanned_stage')
        if not _lock_held(self.root):
            raise BudgetStop('worker_lock_required')
        cases, slots = self._definition(stage)
        if len(slots) > STAGE_SLOT_CAPS[stage] or len(set(slots)) != len(slots):
            raise BudgetStop('stage_slot_plan_exceeds_cap')
        # A smaller frozen stage still receives only floor(10% of its slots).
        retry_cap = min(STAGE_RETRY_CAPS[stage], math.floor(len(slots) * .1))
        by_id = {case['case_id']: case for case in cases}
        if (len(by_id) != len(cases) or any(not identifier.startswith(stage + '_') for identifier in by_id)
                or any(case.get('stage', stage) != stage for case in cases)):
            raise BudgetStop('stage_case_identity_mismatch')
        if any(slot.split('/', 1)[0] not in by_id for slot in slots):
            raise BudgetStop('stage_slot_without_case')
        if self.manifest.get('slots') and not set(slots) <= set(self.manifest['slots']):
            raise BudgetStop('stage_slots_outside_frozen_global_plan')
        with self.lock:
            if self._active or any(gate.locked() for gate in self.gates.values()):
                raise BudgetStop('stage_changed_with_active_jobs')
            used = self.stage_counts(stage)
            if used['retries'] > retry_cap or used['actual_requests'] > len(slots) + retry_cap:
                raise BudgetStop('stage_cumulative_cap_violation')
            self.stage = self.wave = stage
            self.whitelist = set(slots)
            self.allowed = by_id
            self.stage_retry_cap = retry_cap
        self.state()

    def counts(self):
        with self.lock:
            all_rows = list(self.db.execute('SELECT * FROM attempts'))
            rows = [row for row in all_rows if row['state'] != 'not_sent']
            return dict(actual_requests=len(rows), retries=sum(row['attempt'] > 1 for row in rows),
                        completed_successful_requests=sum(row['state'] == 'reply' for row in rows),
                        failed_attempts=sum(row['state'] in ('error', 'unknown') for row in rows),
                        unknown_billing_attempts=sum(row['known_credits'] is None for row in rows),
                        inflight=sum(row['state'] == 'inflight' for row in rows),
                        unique_slots=self.db.execute('SELECT COUNT(*) FROM slots').fetchone()[0],
                        known_token_reference_credits=sum(row['known_credits'] or 0 for row in rows),
                        conservatively_reserved_credits=sum(row['known_credits'] if row['known_credits'] is not None else row['reserved_credits'] for row in rows),
                        administratively_not_sent=len(all_rows) - len(rows))

    def check(self):
        if not _lock_held(self.root):
            raise BudgetStop('worker_lock_required')
        if self.stop:
            raise BudgetStop(self.stop)
        if dt.datetime.now(dt.timezone.utc) >= _utc(self.launch['dispatch_stop_utc']):
            self.stop = 'dispatch_180min_limit'
            raise BudgetStop(self.stop)

    def state(self, status='running', **extra):
        payload = dict(extra)
        payload.update(status=status, utc=stamp(), stage=self.stage, wave=self.wave,
                       stop_reason=self.stop, global_stop=self.stop, concurrency=self.limit,
                       counts=self.counts(), stage_counts=self.stage_counts())
        write(self.root / 'state.json', redact(payload, self.secrets))

    def failed(self, slot, error):
        with self.lock, self.db:
            old = self.db.execute('SELECT * FROM slots WHERE slot=?', (slot,)).fetchone()
            if old and old['response_json']:
                return json.loads(old['response_json'])
            if old and old['status'] in ('failed', 'unknown'):
                return dict(status='failed', error=old['error'])
            self.db.execute("UPDATE slots SET status='failed',error=? WHERE slot=?", (error, slot))
        return dict(status='failed', error=error)

    fail = failed

    def response(self, case, phase):
        with self.lock:
            row = self.db.execute('SELECT * FROM slots WHERE slot=?', (case['case_id'] + '/' + phase,)).fetchone()
            if row is None:
                return None
            if row['response_json']:
                return json.loads(row['response_json'])
            if row['status'] in ('failed', 'unknown'):
                return dict(status='failed', error=row['error'])
            return None

    def _acquire(self):
        with self.permit:
            while self._active >= self.limit:
                self.check()
                self.permit.wait(1)
            self.check()
            self._active += 1

    def _release(self):
        with self.permit:
            self._active -= 1
            self.permit.notify_all()

    def ask(self, case, phase, messages, client):
        slot = case['case_id'] + '/' + phase
        with self.lock:
            gate = self.gates.setdefault(slot, threading.Lock())
        with gate:
            return self._ask(case, phase, messages, client)

    def _ask(self, case, phase, messages, client):
        slot = case['case_id'] + '/' + phase
        frozen = self.allowed.get(case['case_id'])
        if (slot not in self.whitelist or frozen is None or
                case.get('stage', case['case_id'].split('_')[0]) != self.stage or
                case['candidate'] != frozen['candidate']):
            raise BudgetStop('unplanned_request')
        if redact(messages, self.secrets) != messages:
            raise BudgetStop('secret_in_input')
        bound = sum(len(message['content'].encode('utf-8')) + 256 for message in messages)
        reserve = (bound * 2.5 + POLICY['output_cap'] * 15) / 1e6
        next_attempt = 1
        with self.lock, self.db:
            row = self.db.execute('SELECT * FROM slots WHERE slot=?', (slot,)).fetchone()
            if row:
                if row['input_hash'] != digest(messages) or row['input_json'] != canonical(messages):
                    raise BudgetStop('resume_input_mismatch')
                if row['response_json']:
                    return json.loads(row['response_json'])
                if row['status'] in ('failed', 'unknown'):
                    return dict(status='failed', error=row['error'])
                last = self.db.execute('SELECT * FROM attempts WHERE slot=? ORDER BY attempt DESC LIMIT 1', (slot,)).fetchone()
                if last:
                    if last['state'] in ('inflight', 'unknown'):
                        raise BudgetStop('unrecovered_interrupted_slot')
                    if last['state'] != 'error' or last['error'] not in RETRYABLE or last['attempt'] >= 3:
                        return self.failed(slot, last['error'] or 'retained_nonretryable_attempt')
                    next_attempt = last['attempt'] + 1
            else:
                self.check()
                if self.db.execute('SELECT COUNT(*) FROM slots').fetchone()[0] >= POLICY['unique_slot_cap']:
                    self.stop = 'unique_slot_cap'
                    raise BudgetStop(self.stop)
                self.db.execute('INSERT INTO slots VALUES(?,?,?,?,?,?,?,?)',
                                (slot, case['case_id'], case['candidate'], canonical(messages),
                                 digest(messages), 'pending', None, None))
            if bound > POLICY['max_input_reference_tokens']:
                return self.failed(slot, 'input_bound_exceeded')
        for attempt in range(next_attempt, POLICY['retries_per_slot'] + 2):
            if attempt > 1:
                time.sleep(POLICY['retry_delays_seconds'][attempt - 2])
            self._acquire()
            try:
                with self.lock, self.db:
                    self.check()
                    counts = self.counts()
                    if counts['actual_requests'] >= POLICY['actual_request_cap']:
                        self.stop = 'actual_request_cap'
                        raise BudgetStop(self.stop)
                    if attempt > 1:
                        if counts['retries'] >= POLICY['retry_cap']:
                            return self.failed(slot, 'retained_global_retry_cap')
                        if self.stage_counts()['retries'] >= self.stage_retry_cap:
                            return self.failed(slot, 'retained_stage_retry_cap')
                    if (counts['conservatively_reserved_credits'] + reserve) * POLICY['reference_cny_per_credit'] > POLICY['reference_cny_cap']:
                        self.stop = 'reference_cost_cap'
                        raise BudgetStop(self.stop)
                    unobserved = self.db.execute("SELECT COALESCE(SUM(reserved_credits),0) FROM attempts WHERE seq>? OR state='inflight'",
                                                 (self.snapshot_attempt_watermark,)).fetchone()[0]
                    if (self.observed_spend_credits + unobserved + reserve) * POLICY['reference_cny_per_credit'] > POLICY['reference_cny_cap']:
                        self.stop = 'account_reference_plus_reserves_cap'
                        raise BudgetStop(self.stop)
                    self.db.execute('INSERT INTO attempts(slot,case_id,candidate,attempt,state,started,reserved_credits) VALUES(?,?,?,?,?,?,?)',
                                    (slot, case['case_id'], case['candidate'], attempt, 'inflight', stamp(), reserve))
                error = None
                try:
                    reply = client.complete(messages)
                except ApiError as exc:
                    error = exc.error_code
                except Exception:
                    error = 'client_error_billing_unknown'
                with self.lock, self.db:
                    if error is None:
                        known = usage_cost(reply.usage)
                        response = redact(dict(status='saved', content=reply.content,
                                               usage=reply.usage, model=reply.model,
                                               finish_reason=reply.finish_reason), self.secrets)
                        self.db.execute("UPDATE attempts SET state='reply',finished=?,known_credits=? WHERE slot=? AND attempt=?",
                                        (stamp(), known, slot, attempt))
                        self.db.execute("UPDATE slots SET status='saved',response_json=?,error=NULL WHERE slot=?",
                                        (canonical(response), slot))
                        if known is None:
                            self.stop = self.stop or 'successful_usage_missing'
                        if reply.model != POLICY['model']:
                            self.stop = self.stop or 'returned_model_mismatch'
                        self.errors.clear()
                    else:
                        self.db.execute("UPDATE attempts SET state='error',error=?,finished=? WHERE slot=? AND attempt=?",
                                        (error, stamp(), slot, attempt))
                        self.errors.add(slot)
                        self.limit = 2
                        if len(self.errors) >= POLICY['max_distinct_failed_slots']:
                            self.stop = self.stop or FAILURE_STREAK_STOP
                        if error in {'http_400', 'http_401', 'http_403', 'http_404'}:
                            self.stop = self.stop or 'nonretryable_' + error
                if error is None:
                    return response
                if self.stop or error not in RETRYABLE or attempt == POLICY['retries_per_slot'] + 1:
                    return self.failed(slot, error)
            finally:
                self._release()
        return self.failed(slot, 'retained_attempt_limit')


def make_client():
    return ChatClient(SimpleNamespace(temperature=None, token_parameter='max_completion_tokens',
                                     max_output_tokens=POLICY['output_cap'],
                                     reasoning_effort=POLICY['reasoning_effort'],
                                     timeout=POLICY['timeout_seconds']), model=POLICY['model'])


def jobs(store, items, callback):
    """Execute frozen (case, phase, prompt) items; callbacks own all interpretation."""
    service = make_client()
    iterator = iter(items)
    pending = {}
    exhausted = False
    checkpoint = store.counts()['actual_requests']
    last_notice = 0.
    with concurrent.futures.ThreadPoolExecutor(max_workers=POLICY['concurrency']) as pool:
        while pending or not exhausted:
            try:
                store.check()
            except BudgetStop:
                exhausted = True
            refresh = store.counts()['actual_requests'] - checkpoint >= 32
            if refresh and not pending:
                store.snapshot('budget_checkpoint_' + str(store.counts()['actual_requests']))
                checkpoint = store.counts()['actual_requests']
                refresh = False
            while not exhausted and not refresh and len(pending) < store.limit:
                try:
                    case, phase, prompt = next(iterator)
                except StopIteration:
                    exhausted = True
                    break
                pending[pool.submit(store.ask, case, phase, prompt, service)] = (case, phase)
            if not pending:
                break
            finished, _ = concurrent.futures.wait(pending, timeout=1,
                                                  return_when=concurrent.futures.FIRST_COMPLETED)
            for future in finished:
                case, phase = pending.pop(future)
                try:
                    response = future.result()
                    callback(case, phase, response)
                except BudgetStop as exc:
                    store.stop = store.stop or redact(str(exc), store.secrets)
                    exhausted = True
                except Exception as exc:
                    # Never save exception text or provider bodies.
                    write(store.root / 'internal_diagnostic.json',
                          dict(exception_type=type(exc).__name__, stage=store.stage))
                    store.stop = store.stop or 'callback_or_runtime_failure'
                    exhausted = True
            if finished:
                store.state()
                if time.monotonic() - last_notice >= 20 or not pending:
                    print(canonical(redact(dict(event='progress', stage=store.stage,
                                                counts=store.counts(), concurrency=store.limit,
                                                stop=store.stop), store.secrets)), flush=True)
                    last_notice = time.monotonic()
