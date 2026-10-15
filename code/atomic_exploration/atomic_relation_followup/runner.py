"""Freeze, execute and replay the approved exploratory programme."""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
from pathlib import Path
import random
import sqlite3
import time

from utils.io import redact, usage_valid
from .protocol import ROOT, CAPS, MODULES, POLICY, adapter, canonical, digest, stamp, write, BudgetStop, wrapped_input, scored
from .runtime import Ledger, WorkerLock, make_client, TERMINAL


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def code_hashes():
    """Hash the published implementation; private launcher scripts are not dependencies."""
    files = sorted(p for package in ('atomic_relation_followup', 'atomic_experiment', 'task_support', 'utils')
                   for p in (ROOT/package).rglob('*.py')
                   if not {'tests', 'sources'} & set(p.relative_to(ROOT/package).parts))
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def source_hashes(payloads):
    values={}
    def visit(value):
        if isinstance(value,dict):
            for path_key in ('path','local_source_file'):
                if isinstance(value.get(path_key),str) and isinstance(value.get('sha256'),str):
                    path=Path(value[path_key])
                    if not path.is_absolute():path=ROOT/path
                    if path.is_file():
                        actual=hashlib.sha256(path.read_bytes()).hexdigest()
                        if actual!=value['sha256']:raise BudgetStop('source_changed_before_freeze')
                        values[str(path.resolve())]=actual
            for item in value.values():visit(item)
        elif isinstance(value,list):
            for item in value:visit(item)
    visit(payloads)
    layout=ROOT/'task_support/sources/collab/upstream/lib/overcooked_ai/overcooked_ai_py/data/layouts/new_env.layout'
    if layout.exists():values[str(layout.resolve())]=hashlib.sha256(layout.read_bytes()).hexdigest()
    return values


def verify_manifest(manifest, *, files=True):
    errors=[]
    if manifest.get('policy')!=POLICY:errors.append('policy_mismatch')
    if manifest.get('design_sha256')!=digest({k:v for k,v in manifest.items() if k!='design_sha256'}):errors.append('design_hash_mismatch')
    cases={c['case_id']:c for c in manifest['cases']}
    slots={s['slot_id']:s for s in manifest['slots']}
    if len(cases)!=74 or len(slots)!=4136:errors.append('coverage_mismatch')
    if len(cases)!=len(manifest['cases']) or len(slots)!=len(manifest['slots']):errors.append('duplicate_identity')
    for module,(count,retries,stage) in CAPS.items():
        selected=[s for s in slots.values() if s['module']==module]
        if len(selected)!=count:errors.append('module_slot_count_'+module)
        payload=manifest['modules'][module]
        result=adapter(module).validate_inputs(payload)
        if result is False:errors.append('adapter_validation_'+module)
        if isinstance(result,dict) and result.get('ok',result.get('passed',True)) is False:errors.append('adapter_validation_'+module)
    for slot in slots.values():
        if slot['case_id'] not in cases or cases[slot['case_id']]['module']!=slot['module']:errors.append('slot_case_mismatch')
        if any(dep not in slots for dep in slot['dependencies']):errors.append('unknown_dependency')
        if slot['slot_id'] in slot['dependencies']:errors.append('self_dependency')
    # Verify the complete dependency graph is acyclic without model requests.
    done=set()
    for _ in range(10):
        ready={sid for sid,s in slots.items() if set(s['dependencies'])<=done}
        new=ready-done
        if not new:break
        done|=new
    if done!=set(slots):errors.append('dependency_cycle_or_depth')
    if files:
        for name,expected in manifest['code_hashes'].items():
            path=ROOT/name
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:errors.append('code_changed_'+name)
        for name,expected in manifest['source_hashes'].items():
            path=Path(name)
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:errors.append('source_changed')
    return dict(ok=not errors,errors=sorted(set(errors)),cases=len(cases),planned_slots=len(slots),
                offline_only=True)


def prepare(root):
    root=Path(root).resolve()
    if (root/'manifest.json').exists():raise BudgetStop('run_already_frozen')
    payloads={}
    for module in MODULES:
        print(canonical({'event':'offline_build','module':module}),flush=True)
        payload=adapter(module).build_module(ROOT,module)
        if len(payload['slots'])!=CAPS[module][0]:raise BudgetStop('built_slot_count_'+module)
        adapter(module).validate_inputs(payload)
        payloads[module]=payload
    manifest=dict(protocol='socialevo.atomic-relation-followup.v1',created_utc=stamp(),policy=POLICY,
                  modules=payloads,cases=[c for p in payloads.values() for c in p['cases']],
                  slots=[s for p in payloads.values() for s in p['slots']],
                  code_hashes=code_hashes(),source_hashes=source_hashes(payloads),
                  research_role='exploratory local relation measurement supporting later self-organization',
                  statistical_rules=read(ROOT/'atomic_relation_followup/design.json'))
    manifest['design_sha256']=digest(manifest)
    verification=verify_manifest(manifest)
    if not verification['ok']:raise BudgetStop('preflight_manifest_failed')
    root.mkdir(parents=True,exist_ok=True)
    write(root/'manifest.json',manifest)
    write(root/'preflight_verification.json',verification)
    write(root/'state.json',dict(status='prepared',utc=stamp(),planned_slots=4136,paid_calls=0,
                                 concurrency=8,design_sha256=manifest['design_sha256']))
    print(canonical(dict(event='prepared',run_directory=str(root),**verification)),flush=True)
    return manifest


def verify(root, *, files=True):
    root=Path(root).resolve();manifest=read(root/'manifest.json')
    result=verify_manifest(manifest,files=files)
    path=root/'requests.sqlite3'
    if not path.exists():return result
    db=sqlite3.connect(path.as_uri()+'?mode=ro',uri=True);db.row_factory=sqlite3.Row
    try:
        rows=list(db.execute('SELECT * FROM slots'))
        responses={r['slot']:json.loads(r['response_json']) for r in rows if r['response_json']}
        definitions={s['slot_id']:s for s in manifest['slots']};cases={c['case_id']:c for c in manifest['cases']}
        mismatches=[]
        for row in rows:
            if row['slot'] not in definitions:mismatches.append('unplanned_slot');continue
            slot=definitions[row['slot']];case=cases[slot['case_id']]
            if row['status']=='skipped':
                expected=wrapped_input(case,slot,responses)
                if 'skip' not in expected or expected['skip']!=responses.get(row['slot'],{}).get('error'):
                    mismatches.append('skip_replay_'+row['slot'])
            if row['input_json']:
                expected=wrapped_input(case,slot,responses)
                if 'messages' not in expected or canonical(expected['messages'])!=row['input_json'] or digest(expected['messages'])!=row['input_hash']:
                    mismatches.append('input_replay_'+row['slot'])
            if row['response_json'] and row['score_json']:
                expected=scored(case,slot,json.loads(row['response_json']),responses)
                if canonical(expected)!=row['score_json']:mismatches.append('score_replay_'+row['slot'])
            if row['status']=='saved':
                response=responses[row['slot']]
                if not usage_valid(response.get('usage')) or response.get('model')!=POLICY['model']:
                    mismatches.append('successful_usage_or_model')
        attempts=list(db.execute('SELECT * FROM attempts'))
        if len(attempts)>POLICY['actual_request_cap'] or sum(r['attempt']>1 for r in attempts)>POLICY['retry_cap']:
            mismatches.append('global_request_cap')
        for module,(slots,retries,stage) in CAPS.items():
            a=[r for r in attempts if r['module']==module]
            if len(a)>slots+retries or sum(r['attempt']>1 for r in a)>retries:mismatches.append('module_cap_'+module)
        if any(r['attempt']>3 for r in attempts):mismatches.append('per_slot_attempt_cap')
        inflight=sum(r['state']=='inflight' for r in attempts)
        missing=[s['slot_id'] for s in manifest['slots'] if s['slot_id'] not in responses]
        scored_ids={r['slot'] for r in rows if r['score_json']}
        terminal_ids={r['slot'] for r in rows if r['status'] in TERMINAL and r['response_json']}
        missing_scores=[s['slot_id'] for s in manifest['slots'] if s['slot_id'] not in scored_ids]
        result.update(ledger_replay_ok=not mismatches,replay_errors=mismatches,recorded_slots=len(rows),
                      terminal_inflight_clear=inflight==0,all_planned_receipts_present=not missing,
                      all_planned_scores_present=not missing_scores,
                      all_planned_terminal_receipts_present=all(s['slot_id'] in terminal_ids for s in manifest['slots']),
                      missing_scores=len(missing_scores),missing_slots=len(missing),actual_requests=len(attempts),
                      stage_A_receipts_present=all(s['slot_id'] in terminal_ids and s['slot_id'] in scored_ids
                                                  for s in manifest['slots'] if CAPS[s['module']][2]=='A'))
        result['ok']=result['ok'] and not mismatches
        return result
    finally:db.close()


def _rebuild_scores(store,manifest):
    responses=store.responses();scores=store.scores();cases={c['case_id']:c for c in manifest['cases']}
    for slot in manifest['slots']:
        if slot['slot_id'] in responses and slot['slot_id'] not in scores:
            store.save_score(slot,scored(cases[slot['case_id']],slot,responses[slot['slot_id']],responses))


def _execute_stage(store,manifest,stage):
    cases={c['case_id']:c for c in manifest['cases']}
    modules=[m for m in MODULES if CAPS[m][2]==stage]
    ordered={m:[s for s in manifest['slots'] if s['module']==m] for m in modules}
    rng=random.Random(POLICY['seed']+ord(stage))
    for values in ordered.values():rng.shuffle(values)
    client=make_client();pending={};active=set();cursor=0;last_notice=0.;last_snapshot=store.counts()['actual_requests']
    if not store.db.execute('SELECT 1 FROM snapshots WHERE label=?',('stage_'+stage+'_start',)).fetchone():
        store.snapshot('stage_'+stage+'_start',stage)
    with cf.ThreadPoolExecutor(max_workers=8) as executor:
        while True:
            responses=store.responses();finished=store.finished()
            changed=False
            # Continue independent modules if one pauses; never mark unpaid work as output failures.
            while not store.stop and len(pending)<store.limit:
                selected=None
                for offset in range(len(modules)):
                    index=(cursor+offset)%len(modules);module=modules[index]
                    if module in store.module_stops:continue
                    selected=next((s for s in ordered[module] if s['slot_id'] not in finished and s['slot_id'] not in active
                                   and set(s['dependencies'])<=finished),None)
                    if selected:
                        cursor=(index+1)%len(modules);break
                if selected is None:break
                sid=selected['slot_id'];case=cases[selected['case_id']]
                try:prepared=wrapped_input(case,selected,responses)
                except Exception as exc:
                    store.pause_module(selected['module'],'input_preparation_'+type(exc).__name__);continue
                if 'skip' in prepared:
                    response=store.skip(selected,prepared['skip']);responses[sid]=response;finished.add(sid)
                    store.save_score(selected,scored(case,selected,response,responses));changed=True;continue
                active.add(sid)
                pending[executor.submit(store.ask,selected,prepared['messages'],client)]=selected
            if not pending:
                complete=all(s['slot_id'] in finished for values in ordered.values() for s in values)
                if complete or store.stop or all(m in store.module_stops or all(s['slot_id'] in finished for s in ordered[m]) for m in modules):break
                # A reachable frozen graph with no ready work is a hard implementation error.
                store.stop='dependency_scheduler_stalled';store.set_meta('global_stop',store.stop);break
            done,_=cf.wait(pending,timeout=1,return_when=cf.FIRST_COMPLETED)
            for future in done:
                slot=pending.pop(future);active.remove(slot['slot_id']);case=cases[slot['case_id']]
                try:
                    response=future.result();responses[slot['slot_id']]=response
                    store.save_score(slot,scored(case,slot,response,responses))
                except BudgetStop:
                    if not store.stop and slot['module'] not in store.module_stops:
                        store.pause_module(slot['module'],'unclassified_worker_budget_stop')
                except Exception as exc:
                    store.pause_module(slot['module'],'score_or_worker_'+type(exc).__name__)
                    write(store.root/'internal_diagnostic.json',dict(utc=stamp(),module=slot['module'],exception_type=type(exc).__name__))
                changed=True
            count=store.counts()['actual_requests']
            if count-last_snapshot>=32 and not pending:
                store.snapshot('checkpoint_'+str(count),stage);last_snapshot=count
            if changed or time.monotonic()-last_notice>=30:
                store.state(stage=stage)
            if time.monotonic()-last_notice>=30 or (count>=64 and not store.get_meta('first_64_reported')):
                value=store.state(stage=stage)
                print(canonical(dict(event='progress',stage=stage,counts=value['counts'],concurrency=store.limit,
                                     module_stops=store.module_stops,global_stop=store.stop)),flush=True)
                if count>=64:store.set_meta('first_64_reported',True)
                last_notice=time.monotonic()
            # Drain at a bounded frequency for fresh budget metadata; do not grow reserves indefinitely.
            if count-last_snapshot>=32 and pending:
                for future in cf.as_completed(list(pending)):
                    slot=pending.pop(future);active.remove(slot['slot_id'])
                    try:
                        response=future.result();responses[slot['slot_id']]=response
                        store.save_score(slot,scored(cases[slot['case_id']],slot,response,responses))
                    except BudgetStop:
                        if not store.stop and slot['module'] not in store.module_stops:
                            store.pause_module(slot['module'],'unclassified_worker_budget_stop')
                    except Exception as exc:store.pause_module(slot['module'],'score_or_worker_'+type(exc).__name__)
                store.snapshot('checkpoint_'+str(store.counts()['actual_requests']),stage)
                last_snapshot=store.counts()['actual_requests'];store.state(stage=stage)
    store.finalize_quality(modules)
    store.snapshot('stage_'+stage+'_end',stage)
    store.state(stage=stage)


def run(root):
    root=Path(root).resolve();manifest=read(root/'manifest.json')
    preflight=verify(root)
    if not preflight['ok']:raise BudgetStop('frozen_inputs_or_replay_failed')
    if (root/'completion_acceptance.json').exists() and read(root/'completion_acceptance.json')['status']=='accepted':
        print(canonical({'event':'already_completed'}),flush=True);return
    with WorkerLock(root):
        store=Ledger(root,manifest)
        try:
            _rebuild_scores(store,manifest)
            if not store.db.execute("SELECT 1 FROM snapshots WHERE label='run_start'").fetchone():store.snapshot('run_start','A')
            for stage in ('A','B'):
                if store.stop:break
                if stage=='B':
                    gate=verify(root)
                    write(root/'stage_A_verification.json',gate)
                    if not gate['ok'] or not gate['stage_A_receipts_present'] or store.module_stops:
                        store.set_meta('stage_B_blocked','stage_A_technical_or_coverage_gate');break
                _execute_stage(store,manifest,stage)
            _rebuild_scores(store,manifest)
            validation=verify(root)
            completed=(validation['ok'] and validation.get('all_planned_receipts_present')
                       and validation.get('all_planned_scores_present')
                       and validation.get('all_planned_terminal_receipts_present')
                       and validation.get('terminal_inflight_clear') and not store.stop and not store.module_stops)
            value=store.state('completed' if completed else 'stopped',stage='terminal',verification=validation)
            receipt=dict(status='accepted' if completed else 'not_completed',utc=stamp(),verification=validation,
                         counts=value['counts'],module_stops=store.module_stops,global_stop=store.stop)
            write(root/'completion_acceptance.json',receipt)
            report(root)
            print(canonical(dict(event='terminal',**receipt)),flush=True)
        finally:store.db.close()


def report(root):
    from .analysis import write_report
    root=Path(root).resolve();manifest=read(root/'manifest.json')
    return write_report(root,manifest,verify(root))


def main():
    p=argparse.ArgumentParser(description='Frozen bounded exploratory atomic-relation follow-up')
    p.add_argument('command',choices=('plan','run','resume','verify','report'))
    p.add_argument('root');args=p.parse_args()
    try:
        if args.command=='plan':prepare(args.root)
        elif args.command in ('run','resume'):run(args.root)
        elif args.command=='verify':print(canonical(verify(args.root)),flush=True)
        else:report(args.root)
    except BudgetStop as exc:
        print(canonical(dict(event='bounded_stop',reason=str(exc))),flush=True);raise SystemExit(2) from None
