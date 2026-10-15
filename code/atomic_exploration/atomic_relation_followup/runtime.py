"""One durable DAG queue, shared concurrency and cumulative independent caps."""
from __future__ import annotations

import datetime as dt
import json
import math
import os
from pathlib import Path
import sqlite3
import threading
import time
from types import SimpleNamespace
import urllib.parse

from utils.client import ApiError, ChatClient
from utils.execution import WorkerLock, usage_cost, _lock_held
from utils.billing import allowance_snapshot
from utils.io import redact
from .protocol import POLICY, CAPS, BudgetStop, canonical, digest, stamp, write

RETRYABLE = frozenset({'http_429','http_500','http_502','http_503','http_504',
                       'transport_error_billing_unknown'})
TERMINAL = frozenset({'saved','failed','unknown','skipped'})


class Ledger:
    def __init__(self, root, manifest):
        self.root, self.manifest = Path(root), manifest
        if not _lock_held(self.root.resolve()):
            raise BudgetStop('worker_lock_required_before_recovery')
        self.lock = threading.RLock()
        self.db = sqlite3.connect(self.root/'requests.sqlite3', check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.executescript('''
          CREATE TABLE IF NOT EXISTS slots(slot TEXT PRIMARY KEY,case_id TEXT,module TEXT,
            kind TEXT,status TEXT,input_json TEXT,input_hash TEXT,response_json TEXT,
            score_json TEXT,error TEXT);
          CREATE TABLE IF NOT EXISTS attempts(seq INTEGER PRIMARY KEY AUTOINCREMENT,
            slot TEXT,module TEXT,stage TEXT,attempt INTEGER,state TEXT,error TEXT,
            started TEXT,finished TEXT,reserved_credits REAL,known_credits REAL,
            UNIQUE(slot,attempt));
          CREATE TABLE IF NOT EXISTS snapshots(seq INTEGER PRIMARY KEY AUTOINCREMENT,
            label TEXT,stage TEXT,payload TEXT,watermark INTEGER);
          CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY,value TEXT);
        ''')
        self.db.commit()
        key = os.environ.get('SOCIALEVO_API_KEY','')
        self.secrets = tuple({key, urllib.parse.quote(key,safe=''),
                              json.dumps(key,ensure_ascii=False)[1:-1]}-{''})
        self.stop = self.get_meta('global_stop')
        self.limit = self.get_meta('concurrency') or POLICY['concurrency']
        if self.limit not in (2,4,8):
            raise BudgetStop('saved_concurrency_invalid')
        self.module_stops = self.get_meta('module_stops') or {}
        self.launch = self.get_meta('launch')
        if self.launch is None:
            now=dt.datetime.now(dt.timezone.utc)
            self.launch={'start_utc':now.isoformat(),'dispatch_stop_utc':(
                now+dt.timedelta(minutes=POLICY['dispatch_minutes'])).isoformat()}
            self.set_meta('launch',self.launch)
            write(self.root/'launch.json',self.launch)
        with self.lock, self.db:
            # Interrupted paid attempts stay unknown; they are never sampled again.
            for row in list(self.db.execute("SELECT DISTINCT slot FROM attempts WHERE state='inflight'")):
                response={'status':'failed','error':'interrupted_billing_unknown'}
                self.db.execute("UPDATE attempts SET state='unknown',error=?,finished=? WHERE slot=? AND state='inflight'",
                                (response['error'],stamp(),row['slot']))
                self.db.execute("UPDATE slots SET status='unknown',error=?,response_json=? WHERE slot=?",
                                (response['error'],canonical(response),row['slot']))
        self.whitelist={s['slot_id']:s for s in manifest['slots']}
        # Recovery failures participate in the same persisted service gates.
        # Never restore full concurrency or erase a failed-slot streak.
        with self.lock, self.db:
            for module in CAPS:
                self._service_event(module, None)

    def get_meta(self,key):
        with self.lock:
            row=self.db.execute('SELECT value FROM meta WHERE key=?',(key,)).fetchone()
            return json.loads(row[0]) if row else None

    def set_meta(self,key,value):
        with self.lock,self.db:
            self.db.execute('INSERT OR REPLACE INTO meta VALUES(?,?)',(key,canonical(value)))

    def counts(self,module=None,stage=None):
        with self.lock:
            query='SELECT * FROM attempts'
            args=[]
            if module: query+=' WHERE module=?'; args=[module]
            elif stage: query+=' WHERE stage=?'; args=[stage]
            rows=list(self.db.execute(query,args))
            slots=list(self.db.execute('SELECT * FROM slots'+(' WHERE module=?' if module else ''),
                                       [module] if module else []))
            return dict(actual_requests=len(rows), retries=sum(r['attempt']>1 for r in rows),
                        successful_requests=sum(r['state']=='reply' for r in rows),
                        failed_attempts=sum(r['state'] in ('error','unknown') for r in rows),
                        inflight=sum(r['state']=='inflight' for r in rows),
                        unknown_billing_attempts=sum(r['known_credits'] is None for r in rows),
                        known_reference_credits=sum(r['known_credits'] or 0 for r in rows),
                        reserved_reference_credits=sum(r['known_credits'] if r['known_credits'] is not None
                                                       else r['reserved_credits'] for r in rows),
                        terminal_slots=sum(r['status'] in TERMINAL for r in slots),
                        skipped_slots=sum(r['status']=='skipped' for r in slots))

    def check(self,module):
        if self.stop: raise BudgetStop(self.stop)
        if module in self.module_stops: raise BudgetStop(self.module_stops[module])
        if dt.datetime.now(dt.timezone.utc)>=dt.datetime.fromisoformat(self.launch['dispatch_stop_utc']):
            self.stop='dispatch_720min_limit';self.set_meta('global_stop',self.stop)
            raise BudgetStop(self.stop)

    def pause_module(self,module,reason):
        with self.lock:
            self.module_stops.setdefault(module,reason)
            self.set_meta('module_stops',self.module_stops)

    def responses(self):
        with self.lock:
            return {r['slot']:json.loads(r['response_json']) for r in self.db.execute(
                'SELECT slot,response_json FROM slots WHERE response_json IS NOT NULL')}

    def scores(self):
        with self.lock:
            return {r['slot']:json.loads(r['score_json']) for r in self.db.execute(
                'SELECT slot,score_json FROM slots WHERE score_json IS NOT NULL')}

    def finished(self):
        with self.lock:
            return {r['slot'] for r in self.db.execute('SELECT slot,status FROM slots') if r['status'] in TERMINAL}

    def save_score(self,slot,score):
        clean=redact(score,self.secrets)
        with self.lock,self.db:
            self.db.execute('UPDATE slots SET score_json=? WHERE slot=?',(canonical(clean),slot['slot_id']))
            rows=list(self.db.execute("SELECT score_json FROM slots WHERE module=? AND status='saved' AND score_json IS NOT NULL",
                                      (slot['module'],)))
            valid=[json.loads(r[0]).get('format_valid') for r in rows]
            valid=[v for v in valid if isinstance(v,bool)]
            if len(valid)>=POLICY['rate_check_min_responses'] and sum(valid)/len(valid)<POLICY['minimum_format_valid_rate']:
                self.pause_module(slot['module'],'format_valid_rate_below_95pct')

    def skip(self,slot,reason):
        if slot['slot_id'] not in self.whitelist: raise BudgetStop('unplanned_skip')
        response={'status':'skipped','error':reason}
        with self.lock,self.db:
            row=self.db.execute('SELECT response_json,status FROM slots WHERE slot=?',(slot['slot_id'],)).fetchone()
            if row and row['status'] in TERMINAL: return json.loads(row['response_json'])
            self.db.execute('INSERT OR REPLACE INTO slots VALUES(?,?,?,?,?,?,?,?,?,?)',
                (slot['slot_id'],slot['case_id'],slot['module'],slot['kind'],'skipped',None,None,canonical(response),None,reason))
        return response

    def finalize_quality(self, modules):
        """Small modules still need the quality gate at the stage boundary."""
        with self.lock:
            for module in modules:
                rows=list(self.db.execute("SELECT score_json FROM slots WHERE module=? AND status='saved' AND score_json IS NOT NULL", (module,)))
                flags=[json.loads(row[0]).get('format_valid') for row in rows]
                flags=[flag for flag in flags if isinstance(flag,bool)]
                if flags and sum(flags)/len(flags)<POLICY['minimum_format_valid_rate']:
                    self.pause_module(module,'final_format_valid_rate_below_95pct')

    def snapshot(self,label,stage):
        with self.lock:
            watermark=self.db.execute('SELECT COALESCE(MAX(seq),0) FROM attempts').fetchone()[0]
            first=self.db.execute("SELECT MIN(seq) FROM attempts WHERE state='inflight'").fetchone()[0]
            if first is not None:watermark=min(watermark,first-1)
        value=redact(allowance_snapshot(),self.secrets)
        with self.lock,self.db:
            self.db.execute('INSERT INTO snapshots(label,stage,payload,watermark) VALUES(?,?,?,?)',
                            (label,stage,canonical(value),watermark))
        return value

    def _observed(self,stage=None):
        rows=list(self.db.execute('SELECT * FROM snapshots ORDER BY seq'))
        if stage=='B':rows=[r for r in rows if r['stage']=='B']
        values=[]
        for row in rows:
            value=json.loads(row['payload'])
            used=value.get('allowance',{}).get('total_used')
            if value.get('status')=='available' and type(used) in (int,float) and math.isfinite(used):
                values.append((used,row['watermark']))
        if not values:return 0.,0
        initial=values[0][0]
        if any(v[0]<initial for v in values):
            self.stop='account_usage_counter_decreased';self.set_meta('global_stop',self.stop)
            raise BudgetStop(self.stop)
        peak,watermark=max(values,key=lambda x:x[0])
        return (peak-initial)/500000,watermark

    def _budget_check(self,slot,attempt,reserve):
        module=slot['module'];stage=CAPS[module][2]
        totals=self.counts();counts=self.counts(module=module)
        if totals['actual_requests']>=POLICY['actual_request_cap']:
            self.stop='global_actual_request_cap';self.set_meta('global_stop',self.stop);raise BudgetStop(self.stop)
        if counts['actual_requests']>=sum(CAPS[module][:2]):
            self.pause_module(module,'module_actual_request_cap');raise BudgetStop('module_actual_request_cap')
        if attempt>1 and (totals['retries']>=POLICY['retry_cap'] or counts['retries']>=CAPS[module][1]):
            return 'retained_retry_cap'
        for selected,cap in ((None,POLICY['reference_cny_cap']),(stage,POLICY['stage_reference_cny_caps'][stage])):
            data=self.counts(stage=selected) if selected else totals
            observed,watermark=self._observed(selected)
            query="SELECT COALESCE(SUM(COALESCE(known_credits,reserved_credits)),0) FROM attempts WHERE (seq>? OR state='inflight')"
            args=[watermark]
            if selected:query+=' AND stage=?';args.append(selected)
            outstanding=self.db.execute(query,args).fetchone()[0]
            estimate=max(data['reserved_reference_credits'],observed+outstanding)
            if (estimate+reserve)*POLICY['reference_cny_per_credit']>cap:
                self.stop='global_reference_cost_cap' if selected is None else 'stage_'+selected+'_reference_cost_cap'
                self.set_meta('global_stop',self.stop);raise BudgetStop(self.stop)
        return None

    def _service_event(self,module,error):
        rows=list(self.db.execute("SELECT slot,state,error FROM attempts WHERE module=? ORDER BY COALESCE(finished,started) DESC,seq DESC",(module,)))
        streak=set()
        for row in rows:
            if row['state']=='reply':break
            if row['state'] in ('error','unknown'):streak.add(row['slot'])
        if len(streak)>=POLICY['failure_distinct_slots_without_success']:
            self.pause_module(module,'five_distinct_failed_slots_since_success')
        recent=[r for r in self.db.execute("SELECT state FROM attempts WHERE state!='inflight' ORDER BY seq DESC LIMIT ?",
                                           (POLICY['service_rolling_window'],))]
        last_reduce=self.get_meta('last_reduce_at') or 0
        total=self.counts()['actual_requests']
        if len(recent)==POLICY['service_rolling_window'] and total-last_reduce>=POLICY['service_rolling_window']:
            rate=sum(r['state'] in ('error','unknown') for r in recent)/len(recent)
            if rate>=POLICY['service_reduce_fraction'] and self.limit>2:
                self.limit//=2;self.set_meta('concurrency',self.limit);self.set_meta('last_reduce_at',total)
        if error in ('http_400','http_401','http_403','http_404'):
            self.pause_module(module,'nonretryable_'+error)

    def ask(self,slot,messages,client):
        if self.whitelist.get(slot['slot_id'])!=slot:raise BudgetStop('unplanned_request')
        if redact(messages,self.secrets)!=messages:raise BudgetStop('secret_in_input')
        bound=sum(len(m['content'].encode('utf-8'))+256 for m in messages)
        if bound>POLICY['max_input_reference_tokens']:raise BudgetStop('input_bound_exceeded')
        reserve=(bound*2.5+POLICY['output_cap']*15)/1e6
        with self.lock,self.db:
            self.check(slot['module'])
            row=self.db.execute('SELECT * FROM slots WHERE slot=?',(slot['slot_id'],)).fetchone()
            if row:
                if row['input_hash']!=digest(messages) or row['input_json']!=canonical(messages):raise BudgetStop('resume_input_mismatch')
                if row['status'] in TERMINAL:return json.loads(row['response_json'])
                last=self.db.execute('SELECT * FROM attempts WHERE slot=? ORDER BY attempt DESC LIMIT 1',(slot['slot_id'],)).fetchone()
                attempt=last['attempt']+1 if last else 1
                if last and (last['state']!='error' or last['error'] not in RETRYABLE):
                    return self.fail(slot,last['error'] or 'retained_attempt_failure')
            else:
                self.db.execute('INSERT INTO slots VALUES(?,?,?,?,?,?,?,?,?,?)',
                    (slot['slot_id'],slot['case_id'],slot['module'],slot['kind'],'pending',canonical(messages),digest(messages),None,None,None))
                attempt=1
        for number in range(attempt,POLICY['retries_per_slot']+2):
            if number>1:time.sleep(POLICY['retry_delays_seconds'][number-2])
            with self.lock,self.db:
                self.check(slot['module'])
                reason=self._budget_check(slot,number,reserve)
                if reason:return self.fail(slot,reason)
                self.db.execute('INSERT INTO attempts(slot,module,stage,attempt,state,started,reserved_credits) VALUES(?,?,?,?,?,?,?)',
                    (slot['slot_id'],slot['module'],CAPS[slot['module']][2],number,'inflight',stamp(),reserve))
            error=None
            try:reply=client.complete(messages)
            except ApiError as exc:error=exc.error_code
            except Exception:error='client_error_billing_unknown'
            with self.lock,self.db:
                if error is None:
                    credit=usage_cost(reply.usage)
                    response=redact(dict(status='saved',content=reply.content,usage=reply.usage,
                                         model=reply.model,finish_reason=reply.finish_reason),self.secrets)
                    self.db.execute("UPDATE attempts SET state='reply',finished=?,known_credits=? WHERE slot=? AND attempt=?",
                                    (stamp(),credit,slot['slot_id'],number))
                    self.db.execute("UPDATE slots SET status='saved',response_json=?,error=NULL WHERE slot=?",
                                    (canonical(response),slot['slot_id']))
                    if credit is None or reply.model!=POLICY['model']:
                        self.stop='successful_usage_missing' if credit is None else 'returned_model_mismatch'
                        self.set_meta('global_stop',self.stop)
                    self._service_event(slot['module'],None)
                    return response
                self.db.execute("UPDATE attempts SET state='error',error=?,finished=? WHERE slot=? AND attempt=?",
                                (error,stamp(),slot['slot_id'],number))
                self._service_event(slot['module'],error)
            if error not in RETRYABLE or number==POLICY['retries_per_slot']+1 or self.stop or slot['module'] in self.module_stops:
                return self.fail(slot,error)
        return self.fail(slot,'retained_attempt_limit')

    def fail(self,slot,error):
        response={'status':'failed','error':error}
        with self.lock,self.db:
            old=self.db.execute('SELECT response_json,status FROM slots WHERE slot=?',(slot['slot_id'],)).fetchone()
            if old and old['status'] in TERMINAL:return json.loads(old['response_json'])
            self.db.execute("UPDATE slots SET status='failed',response_json=?,error=? WHERE slot=?",
                            (canonical(response),error,slot['slot_id']))
        return response

    def state(self,status='running',**extra):
        with self.lock:
            counts=self.counts()
            observed,_=self._observed()
            value=dict(status=status,utc=stamp(),counts=counts,concurrency=self.limit,
                       module_stops=self.module_stops,global_stop=self.stop,
                       account_used_change_reference_cny=observed*.5,
                       modules={m:self.counts(module=m) for m in CAPS},**extra)
            write(self.root/'state.json',redact(value,self.secrets))
        return value


def make_client():
    return ChatClient(SimpleNamespace(temperature=None,token_parameter='max_completion_tokens',
        max_output_tokens=POLICY['output_cap'],reasoning_effort=POLICY['reasoning_effort'],
        timeout=POLICY['timeout_seconds']),model=POLICY['model'])
