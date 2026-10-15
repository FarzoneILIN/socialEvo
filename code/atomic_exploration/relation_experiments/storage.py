"""Durable request ledger; no unrecorded retry or response replacement."""

import json
import os
import sqlite3
import time
import urllib.parse
from pathlib import Path
from atomic_experiment.client import Reply, ApiError, safe_error_code
from task_experiments.protocol import canonical_json, digest
from task_experiments.recording import utc_now
from .design import RETRY_CODES, RETRY_DELAYS


class Stop(RuntimeError):
    """An adapter-owned reason safe to persist."""


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value, *, immutable=False):
    path = Path(path)
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    if immutable and path.exists():
        if read(path) != value:
            raise Stop('immutable_artifact_mismatch')
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.pending')
    with temp.open('w', encoding='utf-8', newline='\n') as file:
        file.write(text)
        file.flush()
        os.fsync(file.fileno())
    os.replace(temp, path)


def usage_valid(usage):
    keys = ('prompt_tokens', 'completion_tokens', 'total_tokens')
    return (isinstance(usage, dict) and
            all(type(usage.get(k)) is int and usage[k] >= 0 for k in keys) and
            usage['prompt_tokens'] + usage['completion_tokens'] == usage['total_tokens'])


def redact(value, secrets):
    if isinstance(value, str):
        for secret in secrets:
            if secret:
                value = value.replace(secret, '[REDACTED]')
        return value
    if isinstance(value, dict):
        return {redact(k, secrets): redact(v, secrets) for k, v in value.items()}
    if isinstance(value, list):
        return [redact(v, secrets) for v in value]
    return value


class Ledger:
    def __init__(self, root, *, readonly=False):
        self.root = Path(root)
        self.readonly = readonly
        path = self.root / 'requests.sqlite3'
        if readonly:
            self.db = sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True)
        else:
            self.db = sqlite3.connect(path)
            self.db.execute('PRAGMA journal_mode=WAL')
            self.db.execute('PRAGMA synchronous=FULL')
            self.db.executescript('''
                CREATE TABLE IF NOT EXISTS calls(
                    slot TEXT PRIMARY KEY, wave TEXT NOT NULL, input_json TEXT NOT NULL,
                    input_hash TEXT NOT NULL, context_json TEXT NOT NULL, cap INTEGER NOT NULL,
                    status TEXT NOT NULL, response_json TEXT, error TEXT);
                CREATE TABLE IF NOT EXISTS attempts(
                    seq INTEGER PRIMARY KEY AUTOINCREMENT, slot TEXT NOT NULL,
                    wave TEXT NOT NULL, attempt INTEGER NOT NULL, api INTEGER NOT NULL,
                    state TEXT NOT NULL, code TEXT, started TEXT NOT NULL, finished TEXT,
                    UNIQUE(slot, attempt));
            ''')
        self.db.row_factory = sqlite3.Row

    def close(self):
        self.db.close()

    def accounting(self, wave=None):
        where, args = (' WHERE wave=?', (wave,)) if wave else ('', ())
        calls = list(self.db.execute('SELECT * FROM calls' + where, args))
        attempts = list(self.db.execute('SELECT * FROM attempts' + where, args))
        replies = [json.loads(c['response_json']) for c in calls if c['response_json']]
        known = sum(r['usage']['total_tokens'] for r in replies if usage_valid(r['usage']))
        actual = sum(a['api'] for a in attempts)
        unknown = sum(a['api'] for a in attempts if a['state'] != 'reply')
        unknown += sum(not usage_valid(r['usage']) for r in replies
                       if r.get('actual_api_request'))
        return dict(logical_slots_started=len(calls), responses_saved=len(replies),
                    api_requests=actual, additional_retry_requests=sum(a['api'] for a in attempts if a['attempt'] > 1),
                    known_reported_tokens=known, unknown_billing_attempts=unknown,
                    usage_complete=unknown == 0,
                    unfinished_requests=sum(a['state'] == 'inflight' for a in attempts),
                    failed_attempts=sum(a['state'] == 'error' for a in attempts),
                    length_finish_count=sum(r['finish_reason'] == 'length' for r in replies),
                    empty_output_count=sum(not r['content'].strip() for r in replies))


class Session:
    def __init__(self, ledger, design, wave, client=None, *, replay=False, sleeper=time.sleep):
        self.ledger, self.design, self.wave, self.client = ledger, design, wave, client
        self.replay, self.sleeper = replay, sleeper
        key = getattr(client, 'key', '') or os.environ.get('SOCIALEVO_API_KEY', '')
        self.secrets = tuple({key, urllib.parse.quote(key, safe=''),
                              json.dumps(key, ensure_ascii=False)[1:-1]} - {''})
        self.seen = set()

    def ask(self, slot, messages, context, cap):
        slot = self.wave['wave_id'] + '/' + slot
        if slot in self.seen:
            raise Stop('duplicate_logical_slot')
        self.seen.add(slot)
        clean = redact(messages, self.secrets)
        if clean != messages:
            raise Stop('secret_in_model_input')
        db = self.ledger.db
        row = db.execute('SELECT * FROM calls WHERE slot=?', (slot,)).fetchone()
        if row:
            if (row['input_hash'] != digest(messages) or json.loads(row['input_json']) != messages or
                    json.loads(row['context_json']) != context or row['cap'] != cap):
                raise Stop('cached_input_mismatch')
            if row['response_json']:
                return self._reply(json.loads(row['response_json']))
            attempts = list(db.execute('SELECT * FROM attempts WHERE slot=? ORDER BY attempt', (slot,)))
            if self.replay:
                raise Stop('missing_recorded_response')
            if any(a['state'] == 'inflight' for a in attempts):
                raise Stop('inflight_request_billing_unknown')
            if row['status'] == 'failed':
                raise Stop(row['error'] or 'failed_slot')
        else:
            if self.replay:
                raise Stop('missing_recorded_response')
            if db.execute('SELECT COUNT(*) FROM calls WHERE wave=?', (self.wave['wave_id'],)).fetchone()[0] >= self.wave['logical_upper_bound']:
                raise Stop('wave_logical_ceiling')
            with db:
                db.execute('INSERT INTO calls VALUES(?,?,?,?,?,?,?,?,?)',
                           (slot, self.wave['wave_id'], canonical_json(messages), digest(messages),
                            canonical_json(context), cap, 'pending', None, None))
            attempts = []
        if attempts and (attempts[-1]['code'] not in RETRY_CODES or len(attempts) >= 3):
            raise Stop('recorded_failure_not_retryable')
        actual = self.design['backend'] == 'api'
        while True:
            attempt = len(attempts) + 1
            actual_count = db.execute('SELECT COALESCE(SUM(api),0) FROM attempts').fetchone()[0]
            if actual and actual_count >= self.design['actual_request_ceiling']:
                raise Stop('actual_request_ceiling')
            if attempt > 1:
                retries = db.execute('SELECT COALESCE(SUM(api),0) FROM attempts WHERE wave=? AND attempt>1',
                                     (self.wave['wave_id'],)).fetchone()[0]
                if actual and retries >= self.wave['extra_attempt_limit']:
                    raise Stop('wave_retry_ceiling')
                self.sleeper(RETRY_DELAYS[attempt - 2])
            with db:
                db.execute('INSERT INTO attempts(slot,wave,attempt,api,state,started) VALUES(?,?,?,?,?,?)',
                           (slot, self.wave['wave_id'], attempt, int(actual), 'inflight', utc_now()))
            self.client.config.max_output_tokens = cap
            try:
                reply = self.client.complete(messages)
            except ApiError as exc:
                code = safe_error_code(exc)
                with db:
                    db.execute('UPDATE attempts SET state=?,code=?,finished=? WHERE slot=? AND attempt=?',
                               ('error', code, utc_now(), slot, attempt))
                    if code not in RETRY_CODES or attempt >= 3:
                        db.execute('UPDATE calls SET status=?,error=? WHERE slot=?', ('failed', code, slot))
                attempts = list(db.execute('SELECT * FROM attempts WHERE slot=? ORDER BY attempt', (slot,)))
                if code not in RETRY_CODES or attempt >= 3:
                    raise Stop('service_failure_' + code) from None
                continue
            # Unexpected exceptions and interrupts leave the durable reservation unresolved.
            saved = dict(content=reply.content, usage=reply.usage, actual_model=reply.model,
                         finish_reason=reply.finish_reason, actual_api_request=actual)
            clean = redact(saved, self.secrets)
            clean['secret_redacted'] = clean != saved
            with db:
                db.execute('UPDATE attempts SET state=?,finished=? WHERE slot=? AND attempt=?',
                           ('reply', utc_now(), slot, attempt))
                db.execute('UPDATE calls SET status=?,response_json=? WHERE slot=?',
                           ('reply', canonical_json(clean), slot))
            return self._reply(clean)

    def _reply(self, saved):
        if saved.get('secret_redacted'):
            raise Stop('credential_echo_redacted')
        if saved['actual_api_request'] and not usage_valid(saved['usage']):
            raise Stop('successful_response_usage_incomplete')
        return Reply(saved['content'], saved['usage'], saved['actual_model'], saved['finish_reason'])
