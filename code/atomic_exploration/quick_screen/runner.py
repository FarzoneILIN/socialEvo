"""Durable one-message probes, strict cumulative limits, no outcome redraws."""
import argparse
import base64
import concurrent.futures
import hashlib
import importlib
import json
import math
import mimetypes
import os
import urllib.parse
from pathlib import Path
import sqlite3
import threading
import time
from types import SimpleNamespace

from atomic_experiment.client import ChatClient, ApiError
from lbf_upgrade.billing import allowance_snapshot
from relation_experiments.storage import usage_valid, redact

ROOT = Path(__file__).resolve().parents[1]
MODULES = ('information', 'decisions', 'actions')
CANDIDATES = ('HiddenBench', 'MutualFriends', 'SILO-BENCH', 'DialOp Assignment',
              'Deliberative Menu-Numeric', 'Deal or No Deal', 'NegotiationArena SellerBuyer',
              'Collab-Overcooked', 'COMMA')
POLICY = dict(actual_request_cap=500, per_case_request_cap=20,
              per_candidate_request_cap=50, retry_cap=50, retries_per_slot=2,
              output_cap=8192, max_input_reference_tokens=20000,
              reference_cny_cap=40, reference_cny_per_credit=.5,
              input_price_per_million=2.5, cached_price_per_million=.25,
              output_price_per_million=15, model='gpt-5.4-2026-03-05',
              reasoning_effort='medium', timeout_seconds=300,
              concurrency=2, seed=42)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode('utf-8')).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.pending')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                   allow_nan=False) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def stamp():
    import datetime
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def code_hashes():
    paths = list((ROOT / 'quick_screen').rglob('*.py'))
    paths += [ROOT / 'atomic_experiment/client.py', ROOT / 'lbf_upgrade/billing.py',
              ROOT / 'relation_experiments/storage.py']
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(paths) if 'tests' not in p.parts}


def prepare(root):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if (root / 'manifest.json').exists():
        raise RuntimeError('run_already_prepared')
    cases, skips = [], []
    for name in MODULES:
        module = importlib.import_module('quick_screen.adapters.' + name)
        for case in module.build_cases():
            case['adapter'] = name
            cases.append(case)
        skips.extend(module.build_skips() if hasattr(module, 'build_skips') else [])
    ids = [c['case_id'] for c in cases]
    if len(ids) != len(set(ids)) or len(cases) > 18:
        raise RuntimeError('case_coverage_invalid')
    counts = {}
    for c in cases:
        for k in ('case_id', 'candidate', 'sender_prompt', 'receiver_prompt',
                  'full_information_prompt', 'response_instruction', 'provenance',
                  'private_scoring'):
            if k not in c:
                raise RuntimeError('case_field_missing_' + k)
        counts[c['candidate']] = counts.get(c['candidate'], 0) + 1
    if any(n > 2 for n in counts.values()):
        raise RuntimeError('candidate_case_ceiling')
    covered = set(counts) | {s['candidate'] for s in skips}
    if covered != set(CANDIDATES):
        raise RuntimeError('candidate_coverage_mismatch')
    manifest = dict(protocol='socialevo.quick-screen.v1', created_utc=stamp(),
                    policy=POLICY, cases=cases, skips=skips,
                    code_hashes=code_hashes(),
                    scope='Exploratory decision-state adaptations; not full benchmark runs.')
    manifest['design_sha256'] = digest({k: v for k, v in manifest.items()
                                       if k != 'created_utc'})
    write(root / 'manifest.json', manifest)
    write(root / 'state.json', dict(status='ready', cases=len(cases), skipped=skips))
    print(canonical(dict(status='prepared', cases=len(cases), candidates=counts,
                         skips=skips, run_directory=str(root))), flush=True)


def usage_cost(usage):
    if not usage_valid(usage):
        return None
    cached = usage.get('prompt_tokens_details', {}).get('cached_tokens', 0)
    if type(cached) is not int or not 0 <= cached <= usage['prompt_tokens']:
        cached = 0
    return ((usage['prompt_tokens'] - cached) * 2.5 + cached * .25
            + usage['completion_tokens'] * 15) / 1000000


def decision_messages(case, partner_text=None, full=False):
    # The answer/scoring/provenance fields are intentionally never serialized here.
    prompt = case['full_information_prompt'] if full else case['receiver_prompt']
    texts = [{'type': 'text', 'text': prompt + '\n\n' + case['response_instruction']}]
    image_paths = case.get('full_info_image_paths' if full else 'receiver_image_paths', [])
    for image_path in image_paths:
        path = Path(image_path)
        if not path.is_absolute():
            path = ROOT / path
        mime = mimetypes.guess_type(str(path))[0] or 'image/png'
        data = base64.b64encode(path.read_bytes()).decode('ascii')
        texts.append({'type': 'image_url', 'image_url': {
            'url': 'data:' + mime + ';base64,' + data, 'detail': 'high'}})
    body = texts if image_paths else texts[0]['text']
    result = [{'role': 'system', 'content': 'Make the requested decision using only your supplied observations. Return only the requested JSON. Do not invent missing facts.'},
              {'role': 'user', 'content': body}]
    if not full:
        result.append({'role': 'user', 'content': 'Message from partner:\n' +
                       (partner_text if partner_text is not None else '[Partner message withheld.]')})
    return result


class BudgetStop(RuntimeError):
    pass


class Store:
    def __init__(self, root, policy=POLICY):
        self.root, self.policy = Path(root), policy
        self.lock = threading.RLock()
        self.db = sqlite3.connect(self.root / 'requests.sqlite3', check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.executescript('''
            CREATE TABLE IF NOT EXISTS slots(slot TEXT PRIMARY KEY, case_id TEXT,
                candidate TEXT, input_json TEXT, input_hash TEXT, status TEXT,
                response_json TEXT, error TEXT);
            CREATE TABLE IF NOT EXISTS attempts(seq INTEGER PRIMARY KEY AUTOINCREMENT,
                slot TEXT, case_id TEXT, candidate TEXT, attempt INTEGER,
                state TEXT, error TEXT, started TEXT, finished TEXT,
                reserved_credits REAL, known_credits REAL,
                UNIQUE(slot,attempt));
            CREATE TABLE IF NOT EXISTS snapshots(seq INTEGER PRIMARY KEY AUTOINCREMENT,
                label TEXT, payload TEXT);
        ''')
        self.db.commit()
        secret = os.environ.get('SOCIALEVO_API_KEY', '')
        self.secrets = tuple({secret, urllib.parse.quote(secret, safe=''),
                              json.dumps(secret, ensure_ascii=False)[1:-1]} - {''})
        self.stop = None
        self.observed_spend_credits = 0.0
        self.snapshot_attempt_watermark = 0
        self._restore_observed_budget()

    def _restore_observed_budget(self):
        """Retain the largest saved allowance delta and its safe prefix.

        Lower or delayed metadata must never reset an already observed peak.
        The prefix belonging to a lower snapshot cannot cover later attempts.
        """
        rows = list(self.db.execute('SELECT label,payload FROM snapshots ORDER BY seq'))
        usable = lambda value: type(value) in (int, float) and math.isfinite(value) and value >= 0
        saved = [(row['label'], json.loads(row['payload'])) for row in rows]
        first = next((s for label, s in saved if label == 'run_start'
                      and s.get('status') == 'available'
                      and usable(s.get('allowance', {}).get('total_used'))), None)
        if first is None:
            return
        initial = first['allowance']['total_used']
        for _, snapshot in saved:
            used = snapshot.get('allowance', {}).get('total_used')
            if snapshot.get('status') != 'available' or not usable(used):
                continue
            if used < initial:
                self.stop = 'usage_counter_decreased'
                continue
            delta = (used - initial) / 500000
            if delta > self.observed_spend_credits:
                self.observed_spend_credits = delta
                self.snapshot_attempt_watermark = snapshot.get('model_attempt_watermark_before_query', 0)
        if self.observed_spend_credits * self.policy['reference_cny_per_credit'] >= self.policy['reference_cny_cap']:
            self.stop = 'observed_token_allowance_budget'

    def counts(self):
        with self.lock:
            rows = list(self.db.execute('SELECT * FROM attempts'))
            return dict(actual_requests=len(rows), retries=sum(r['attempt'] > 1 for r in rows),
                        completed_successful_requests=sum(r['state'] == 'reply' for r in rows),
                        unknown_billing_attempts=sum(r['known_credits'] is None for r in rows),
                        known_token_reference_credits=sum(r['known_credits'] or 0 for r in rows),
                        conservatively_reserved_credits=sum(r['known_credits'] if r['known_credits'] is not None
                                                            else r['reserved_credits'] for r in rows))

    def snapshot(self, label):
        with self.lock:
            watermark = self.db.execute('SELECT COALESCE(MAX(seq),0) FROM attempts').fetchone()[0]
            first_inflight = self.db.execute("SELECT MIN(seq) FROM attempts WHERE state='inflight'").fetchone()[0]
            if first_inflight is not None:
                watermark = min(watermark, first_inflight - 1)
        snapshot = allowance_snapshot()
        snapshot['model_attempt_watermark_before_query'] = watermark
        with self.lock, self.db:
            self.db.execute('INSERT INTO snapshots(label,payload) VALUES(?,?)',
                            (label, canonical(snapshot)))
            self._restore_observed_budget()
        return snapshot

    def ask(self, case, phase, messages, client):
        slot = case['case_id'] + '/' + phase
        clean = redact(messages, self.secrets)
        if clean != messages:
            raise BudgetStop('secret_in_input')
        # UTF-8 byte count is a conservative text-token bound; images use a separate reserve.
        text_bytes = 0
        image_count = 0
        for message in messages:
            content = message['content']
            if isinstance(content, str):
                text_bytes += len(content.encode('utf-8'))
            else:
                for part in content:
                    if part['type'] == 'text':
                        text_bytes += len(part['text'].encode('utf-8'))
                    elif part['type'] == 'image_url':
                        image_count += 1
        input_bound = text_bytes + 10000 * image_count
        if input_bound > self.policy['max_input_reference_tokens']:
            return dict(status='failed', error='input_exceeds_fast_screen_scope')
        reserve = (input_bound * 2.5 + self.policy['output_cap'] * 15) / 1000000
        next_attempt = 1
        with self.lock, self.db:
            old = self.db.execute('SELECT * FROM slots WHERE slot=?', (slot,)).fetchone()
            if old:
                if old['input_hash'] != digest(messages):
                    raise BudgetStop('resume_input_mismatch')
                if old['response_json']:
                    return json.loads(old['response_json'])
                if old['status'] in ('failed', 'unknown'):
                    return dict(status='failed', error=old['error'])
                if self.db.execute("SELECT COUNT(*) FROM attempts WHERE slot=? AND state='inflight'", (slot,)).fetchone()[0]:
                    self.db.execute("UPDATE slots SET status='unknown',error='interrupted_billing_unknown' WHERE slot=?", (slot,))
                    return dict(status='failed', error='interrupted_billing_unknown')
                previous = self.db.execute('SELECT attempt,error FROM attempts WHERE slot=? ORDER BY attempt DESC LIMIT 1', (slot,)).fetchone()
                if previous:
                    next_attempt = previous['attempt'] + 1
                    if next_attempt > self.policy['retries_per_slot'] + 1:
                        self.db.execute("UPDATE slots SET status='failed',error='retained_attempt_limit' WHERE slot=?", (slot,))
                        return dict(status='failed', error='retained_attempt_limit')
            else:
                self.db.execute('INSERT INTO slots VALUES(?,?,?,?,?,?,?,?)',
                                (slot, case['case_id'], case['candidate'], canonical(messages),
                                 digest(messages), 'pending', None, None))
        retryable = {'http_429', 'http_500', 'http_502', 'http_503', 'http_504',
                     'transport_error_billing_unknown'}
        for attempt in range(next_attempt, self.policy['retries_per_slot'] + 2):
            if attempt > 1:
                time.sleep((2, 5)[attempt-2])
            with self.lock, self.db:
                if self.stop:
                    raise BudgetStop(self.stop)
                counts = self.counts()
                per_case = self.db.execute('SELECT COUNT(*) FROM attempts WHERE case_id=?', (case['case_id'],)).fetchone()[0]
                per_candidate = self.db.execute('SELECT COUNT(*) FROM attempts WHERE candidate=?', (case['candidate'],)).fetchone()[0]
                if (counts['actual_requests'] >= self.policy['actual_request_cap'] or
                    per_case >= self.policy['per_case_request_cap'] or
                    per_candidate >= self.policy['per_candidate_request_cap'] or
                    (attempt > 1 and counts['retries'] >= self.policy['retry_cap'])):
                    if counts['actual_requests'] >= self.policy['actual_request_cap']:
                        self.stop = 'global_actual_request_ceiling'
                    elif attempt > 1 and counts['retries'] >= self.policy['retry_cap']:
                        self.stop = 'global_retry_ceiling'
                    raise BudgetStop('request_or_retry_ceiling')
                if ((counts['conservatively_reserved_credits'] + reserve) *
                    self.policy['reference_cny_per_credit'] > self.policy['reference_cny_cap']):
                    self.stop = 'reference_cost_budget'
                    raise BudgetStop(self.stop)
                unobserved = self.db.execute("SELECT COALESCE(SUM(reserved_credits),0) FROM attempts WHERE seq>? OR state='inflight'", (self.snapshot_attempt_watermark,)).fetchone()[0]
                if ((self.observed_spend_credits + unobserved + reserve) *
                    self.policy['reference_cny_per_credit'] > self.policy['reference_cny_cap']):
                    self.stop = 'observed_cost_plus_outstanding_budget'
                    raise BudgetStop(self.stop)
                self.db.execute('INSERT INTO attempts(slot,case_id,candidate,attempt,state,started,reserved_credits) VALUES(?,?,?,?,?,?,?)',
                                (slot, case['case_id'], case['candidate'], attempt, 'inflight', stamp(), reserve))
            try:
                reply = client.complete(messages)
                credit = usage_cost(reply.usage)
                response = redact(dict(status='saved', content=reply.content,
                                       usage=reply.usage, model=reply.model,
                                       finish_reason=reply.finish_reason), self.secrets)
                with self.lock, self.db:
                    self.db.execute("UPDATE attempts SET state='reply',finished=?,known_credits=? WHERE slot=? AND attempt=?",
                                    (stamp(), credit, slot, attempt))
                    self.db.execute("UPDATE slots SET status='saved',response_json=? WHERE slot=?", (canonical(response), slot))
                if credit is None:
                    self.stop = 'successful_response_usage_missing'
                return response
            except ApiError as exc:
                error = exc.error_code
            except Exception:
                error = 'client_error_billing_unknown'
            with self.lock, self.db:
                self.db.execute("UPDATE attempts SET state='error',error=?,finished=? WHERE slot=? AND attempt=?",
                                (error, stamp(), slot, attempt))
                if error not in retryable or attempt > self.policy['retries_per_slot']:
                    self.db.execute("UPDATE slots SET status='failed',error=? WHERE slot=?", (error, slot))
                    return dict(status='failed', error=error)
        raise AssertionError('unreachable')


def parse_choice(response):
    if response.get('status') != 'saved':
        return None, response.get('error', 'request_failed')
    text = response.get('content', '').strip()
    if text.startswith('```'):
        lines = text.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == '```':
            text = '\n'.join(lines[1:-1])
    try:
        def reject_constant(value):
            raise ValueError('nonfinite_json')
        value = json.loads(text, parse_constant=reject_constant)
        if not isinstance(value, dict) or 'choice' not in value:
            raise ValueError()
        return value['choice'], None
    except (ValueError, TypeError):
        return None, 'invalid_choice_json'


def execute_case(case, store):
    path = store.root / 'cases' / (case['case_id'] + '.json')
    if path.exists():
        return json.loads(path.read_text(encoding='utf-8'))
    config = SimpleNamespace(temperature=None, token_parameter='max_completion_tokens',
                             max_output_tokens=POLICY['output_cap'], reasoning_effort='medium', timeout=300)
    client = ChatClient(config, model=POLICY['model'])
    module = importlib.import_module('quick_screen.adapters.' + case['adapter'])
    result = dict(case_id=case['case_id'], candidate=case['candidate'],
                  provenance=case['provenance'], status='running', phases={})
    try:
        source = store.ask(case, 'sender', [{'role': 'system', 'content': 'Send one concise, useful message to your partner using only your private observations. Do not invent facts. No tools.'},
                                          {'role': 'user', 'content': case['sender_prompt']}], client)
        result['phases']['sender'] = dict(status=source['status'], error=source.get('error'))
        if source['status'] != 'saved' or not source.get('content', '').strip():
            result['status'] = 'source_failed'
        else:
            text = source['content']
            order = ('normal_1', 'masked', 'normal_2', 'full_information')
            if int(hashlib.sha256(case['case_id'].encode()).hexdigest(), 16) % 2:
                order = ('masked', 'normal_1', 'full_information', 'normal_2')
            for phase in order:
                response = store.ask(case, phase, decision_messages(case, text if phase.startswith('normal') else None,
                                      full=phase == 'full_information'), client)
                choice, error = parse_choice(response)
                if error is None:
                    try:
                        scoring = module.score(case, choice)
                        if not isinstance(scoring, dict) or type(scoring.get('legal')) is not bool:
                            raise ValueError()
                    except Exception:
                        scoring, error = {}, 'score_failed'
                else:
                    scoring = {}
                result['phases'][phase] = dict(status=response['status'], choice=choice, error=error, scoring=scoring)
                if store.stop:
                    raise BudgetStop(store.stop)
            phases = result['phases']
            valid = all(phases[p].get('error') is None and phases[p].get('scoring', {}).get('legal')
                        for p in ('normal_1', 'normal_2', 'masked', 'full_information'))
            result['status'] = 'complete' if valid else 'completed_with_invalid_outputs'
            if valid:
                a, b, m = [canonical(phases[p]['scoring'].get('choice', phases[p]['choice']))
                           for p in ('normal_1', 'normal_2', 'masked')]
                result['exploratory'] = dict(normal_repeats_agree=a == b,
                    masked_changes_choice=m != a and m != b,
                    repeat_agreement_and_masked_change=a == b and m != a,
                    full_information_correct=phases['full_information']['scoring'].get('correct'),
                    note='Two normal samples and one masked sample are screening clues, not distribution estimates or significance tests.')
    except BudgetStop as exc:
        result.update(status='budget_stopped', stop_reason=str(exc))
    write(path, result)
    store.snapshot('case_end_' + case['case_id'])
    print(canonical(dict(event='case_finished', case_id=case['case_id'], candidate=case['candidate'],
                         status=result['status'], exploratory=result.get('exploratory'),
                         counts=store.counts())), flush=True)
    return result


def verify(root):
    root = Path(root)
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    if manifest['code_hashes'] != code_hashes():
        raise BudgetStop('frozen_code_changed')
    clean = {k: v for k, v in manifest.items() if k not in ('created_utc', 'design_sha256')}
    if digest(clean) != manifest['design_sha256']:
        raise BudgetStop('frozen_design_changed')
    for case in manifest['cases']:
        normal = decision_messages(case, 'FIXED_PARTNER_TEXT')
        repeat = decision_messages(case, 'FIXED_PARTNER_TEXT')
        masked = decision_messages(case)
        if normal != repeat or normal[:-1] != masked[:-1]:
            raise BudgetStop('directional_mask_mismatch')
        sources = case['provenance'].get('source_files', [])
        if not sources and case['provenance'].get('local_source_file'):
            sources = [dict(path=case['provenance']['local_source_file'], sha256=case['provenance'].get('sha256'))]
        for source in sources:
            path = Path(source['path'])
            if not path.is_absolute():
                path = ROOT / path
            if source.get('sha256') and hashlib.sha256(path.read_bytes()).hexdigest() != source['sha256']:
                raise BudgetStop('source_hash_mismatch')
    database = root / 'requests.sqlite3'
    if database.exists():
        db = sqlite3.connect(database.resolve().as_uri() + '?mode=ro', uri=True)
        db.row_factory = sqlite3.Row
        try:
            attempts = list(db.execute('SELECT * FROM attempts'))
            if len(attempts) > POLICY['actual_request_cap'] or sum(a['attempt'] > 1 for a in attempts) > POLICY['retry_cap']:
                raise BudgetStop('recorded_budget_violation')
            for candidate in CANDIDATES:
                if sum(a['candidate'] == candidate for a in attempts) > POLICY['per_candidate_request_cap']:
                    raise BudgetStop('recorded_candidate_ceiling')
            for case in manifest['cases']:
                recorded_case = root / 'cases' / (case['case_id'] + '.json')
                if not recorded_case.exists():
                    continue
                saved = json.loads(recorded_case.read_text(encoding='utf-8'))
                module = importlib.import_module('quick_screen.adapters.' + case['adapter'])
                source_row = db.execute('SELECT response_json FROM slots WHERE slot=?', (case['case_id'] + '/sender',)).fetchone()
                source_text = json.loads(source_row[0]).get('content', '') if source_row and source_row[0] else None
                if sum(a['case_id'] == case['case_id'] for a in attempts) > POLICY['per_case_request_cap']:
                    raise BudgetStop('recorded_case_ceiling')
                for phase, value in saved['phases'].items():
                    if phase == 'sender':
                        continue
                    row = db.execute('SELECT * FROM slots WHERE slot=?', (case['case_id'] + '/' + phase,)).fetchone()
                    if not row:
                        raise BudgetStop('missing_recorded_slot')
                    expected = decision_messages(case, source_text if phase.startswith('normal') else None,
                                                 full=phase == 'full_information')
                    if row['input_hash'] != digest(expected) or json.loads(row['input_json']) != expected:
                        raise BudgetStop('recorded_input_mismatch')
                    if row['response_json']:
                        response = json.loads(row['response_json'])
                        choice, error = parse_choice(response)
                        if error is None and value.get('error') is None:
                            if choice != value['choice'] or module.score(case, choice) != value['scoring']:
                                raise BudgetStop('replay_scoring_mismatch')
                        if not usage_valid(response.get('usage')):
                            raise BudgetStop('successful_response_usage_missing')
        finally:
            db.close()
    return dict(status='verified', cases=len(manifest['cases']), frozen_code=True, directional_mask=True)


def report(root):
    root = Path(root)
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    records = [json.loads(p.read_text(encoding='utf-8')) for p in sorted((root / 'cases').glob('*.json'))]
    store = Store(root)
    counts = store.counts()
    snapshots = [json.loads(r[0]) for r in store.db.execute('SELECT payload FROM snapshots ORDER BY seq')]
    usable = lambda value: type(value) in (int, float) and math.isfinite(value) and value >= 0
    used = [s.get('allowance', {}).get('total_used') for s in snapshots if s.get('status') == 'available'
            and usable(s.get('allowance', {}).get('total_used'))]
    baseline_row = next((json.loads(r[0]) for r in store.db.execute("SELECT payload FROM snapshots WHERE label='run_start' ORDER BY seq")
                         if json.loads(r[0]).get('status') == 'available'
                         and usable(json.loads(r[0]).get('allowance', {}).get('total_used'))), None)
    observed = (max(0, used[-1] - baseline_row['allowance']['total_used']) / 500000
                if baseline_row and len(used) >= 2 else None)
    summary = dict(scope=manifest['scope'], cases=records, skips=manifest['skips'], counts=counts,
                   known_token_reference_cny=counts['known_token_reference_credits'] * .5,
                   observed_key_allowance_change_credits=observed,
                   observed_key_allowance_reference_cny=observed * .5 if observed is not None else None,
                   peak_observed_key_allowance_reference_cny=store.observed_spend_credits * .5,
                   billing_note='Allowance changes are key-level, may include concurrent outside consumption, and are not an invoice. Failed request billing is retained as unknown.',
                   verification=verify(root))
    write(root / 'report.json', summary)
    lines = ['# SocialEvo 快速筛选结果', '',
             '本轮为公开任务的单消息、单决策片段改编；不是完整基准复现。每例仅两次正常回答、一次屏蔽回答和一次完整信息参照，不能作显著性或稳定效应结论。', '',
             '| 候选 | 完成实例 | 正常重复一致 | 一致且屏蔽后改变 | 全信息正确（有定义） | 状态 |',
             '|---|---:|---:|---:|---:|---|']
    for candidate in sorted({c['candidate'] for c in manifest['cases']} | {s['candidate'] for s in manifest['skips']}):
        rows = [r for r in records if r['candidate'] == candidate]
        eligible = [r for r in rows if r['status'] == 'complete']
        flags = [r.get('exploratory', {}) for r in eligible]
        defined = [r.get('phases', {}).get('full_information', {}).get('scoring', {}).get('correct') for r in rows]
        defined = [x for x in defined if type(x) is bool]
        capability = str(sum(defined)) + '/' + str(len(defined)) if defined else '不适用或未获得'
        skip = next((s['reason'] for s in manifest['skips'] if s['candidate'] == candidate), '')
        statuses = ', '.join(sorted({r['status'] for r in rows})) or skip
        lines.append('| ' + candidate + ' | ' + str(len(eligible)) + '/' + str(len(rows)) + ' | ' +
                     str(sum(f.get('normal_repeats_agree') is True for f in flags)) + ' | ' +
                     str(sum(f.get('repeat_agreement_and_masked_change') is True for f in flags)) + ' | ' +
                     capability + ' | ' + statuses + ' |')
    lines += ['', f"实际模型请求 {counts['actual_requests']} 次；其中重试 {counts['retries']} 次；未知计费尝试 {counts['unknown_billing_attempts']} 次。",
              f"成功输出用量参考 {summary['known_token_reference_cny']:.4f} 元；账户额度变化参考 {summary['observed_key_allowance_reference_cny']} 元，二者不是已确认账单。", '',
              '具体选择、合法性与即时收益见 report.json；团队收益与个人收益均只报告评分器原生存在的字段，空值不填零。', '',
              '完成仅指输出合法且能评分；不等于解题成功。DialOp 与 Menu 同时需看可行性和归一化提案质量，未达到精确最优不等于无法完成任务。', '',
              'Deal or No Deal 没有唯一正确分配。potential_*_utility 是对方接受这一分配时的条件性效用，不是成交收益。NegotiationArena 的固定报价接受片段不要求卖方私有成本；零变化不能淘汰完整谈判任务。', '',
              '候选是否扩大还需检查两例的能力参照、信息边界与正常回答波动；探索性变化不得解释为统计显著性。']
    for record in records:
        if record['candidate'] == 'Deal or No Deal':
            lines += ['', '条件性分配效用：' + record['case_id'], '',
                      '| 条件 | 接收者 | 发送者 | 双方合计 |', '|---|---:|---:|---:|']
            for phase in ('normal_1', 'normal_2', 'masked', 'full_information'):
                score = record['phases'].get(phase, {}).get('scoring', {})
                values = [score.get('potential_' + who + '_utility') for who in ('receiver', 'sender', 'team')]
                lines.append('| ' + phase + ' | ' + ' | '.join('未测得' if x is None else str(x) for x in values) + ' |')
    (root / '报告.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    store.db.close()
    return summary


def run(root):
    root = Path(root).resolve()
    verify(root)
    lock_path = root / 'runner.lock'
    descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(descriptor, str(os.getpid()).encode())
    os.close(descriptor)
    store = Store(root)
    try:
        initial = store.snapshot('run_start')
        initial_used = initial.get('allowance', {}).get('total_used')
        if (initial.get('status') != 'available' or type(initial_used) not in (int, float)
            or not math.isfinite(initial_used) or initial_used < 0):
            raise BudgetStop('allowance_metadata_unavailable')
        manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
        write(root / 'state.json', dict(status='running', started_utc=stamp()))
        with concurrent.futures.ThreadPoolExecutor(max_workers=POLICY['concurrency']) as pool:
            futures = [pool.submit(execute_case, case, store) for case in manifest['cases']]
            for future in concurrent.futures.as_completed(futures):
                future.result()
        store.snapshot('run_end')
        write(root / 'state.json', dict(status='stopped' if store.stop else 'completed',
                                       stop_reason=store.stop, finished_utc=stamp(), counts=store.counts()))
    except Exception as exc:
        store.stop = store.stop or ('run_error_' + type(exc).__name__)
        write(root / 'state.json', dict(status='stopped', stop_reason=store.stop,
                                       finished_utc=stamp(), counts=store.counts()))
        raise
    finally:
        store.db.close()
        lock_path.unlink(missing_ok=True)
    summary = report(root)
    print(canonical(dict(status='finished', counts=summary['counts'], report=str(root / '报告.md'))), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('prepare', 'verify', 'run', 'report'))
    parser.add_argument('root')
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args.root)
    elif args.command == 'verify':
        print(canonical(verify(args.root)))
    elif args.command == 'report':
        value = report(args.root)
        print(canonical(value['counts']))
    else:
        run(args.root)


if __name__ == '__main__':
    main()
