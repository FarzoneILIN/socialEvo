"""Inspect submitted historical examples and replay saved inputs without model calls."""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DATA = ROOT.parents[1]/'data/atomic_exploration'


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def original_hash(value, expected):
    return any(hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=ascii_mode,allow_nan=False).encode()).hexdigest()==expected
               for ascii_mode in (False,True))


def verify_data(data=DATA):
    manifest=json.loads((data/'manifest.json').read_text(encoding='utf-8'))
    for entry in manifest['files']:
        path=data/entry['path']
        if hashlib.sha256(path.read_bytes()).hexdigest()!=entry['sha256']:
            raise ValueError('Data hash mismatch: '+entry['path'])
    summary=json.loads((data/'summary.json').read_text(encoding='utf-8'))
    total=collections.Counter()
    for task,counts in summary.items():
        folder=data/task
        cases=rows(folder/'cases.jsonl'); plans=rows(folder/'planned_slots.jsonl')
        records=rows(folder/'requests.jsonl'); attempts=rows(folder/'attempts.jsonl')
        case_ids={(c['source_run'],c['case']['case_id']) for c in cases}
        plan_ids={(p['source_run'],p['slot_id']) for p in plans}
        record_ids={(r['source_run'],r['slot_id']) for r in records}
        assert len(cases)==counts['cases'] and len(case_ids)==len(cases)
        assert len(plans)==counts['planned_slots'] and len(plan_ids)==len(plans)
        assert len(records)==counts['recorded_slots'] and len(record_ids)==len(records)
        assert record_ids <= plan_ids
        assert all((r['source_run'],r['case_id']) in case_ids for r in records+plans)
        assert all((a['source_run'],a['slot']) in record_ids for a in attempts)
        statuses=dict(collections.Counter(r['status'] for r in records))
        assert statuses==counts['statuses'] and statuses.get('saved',0)==counts['saved_replies']
        assert len(attempts)==counts['actual_attempts']
        assert len(plans)-len(records)==counts['unrecorded_slots']
        for record in records:
            for key in ('input','response','score'):
                expected=record.get(key+'_json_sha256')
                if expected:assert original_hash(record[key],expected),(record['slot_id'],key)
            if record.get('input_json_sha256'):
                assert record['input_json_sha256']==record['input_hash']
        if task=='sudoku':
            main_ids={(p['source_run'],p['slot_id']) for p in plans if p['condition'] in ('both_visible','both_hidden','mask_A_to_B','mask_B_to_A')}
            assert not any((r['source_run'],r['slot_id']) in main_ids and r['status']=='saved' for r in records)
        total.update(cases=len(cases),recorded_slots=len(records),saved_replies=counts['saved_replies'],attempts=len(attempts))
    return {'status':'passed','files_checked':len(manifest['files']),**dict(total)}


def replay(data=DATA, kitchen=False):
    from atomic_relation_followup.protocol import adapter, scored
    from sudoku_preflight import protocol as sudoku
    result=collections.Counter(); mismatches=[]
    for task in ('menu','minesweeper','negotiation','kitchen','sudoku'):
        folder=data/task
        cases={(c['source_run'],c['case']['case_id']):c['case'] for c in rows(folder/'cases.jsonl')}
        plans={(p['source_run'],p['slot_id']):p for p in rows(folder/'planned_slots.jsonl')}
        records=rows(folder/'requests.jsonl')
        grouped=collections.defaultdict(dict)
        for record in records:
            if record.get('response') is not None:
                grouped[(record['source_run'],record['case_id'])][record['slot_id']]=record['response']
        for record in records:
            if record['source_run']=='definition_v1' or (task=='kitchen' and not kitchen):
                result['archived_records_not_rescored']+=1
                continue
            if record['source_run'] not in ('followup_v1','sudoku_preflight'):continue
            case=cases[(record['source_run'],record['case_id'])]
            plan=plans[(record['source_run'],record['slot_id'])]
            responses=grouped[(record['source_run'],record['case_id'])]
            module=sudoku if task=='sudoku' else adapter(plan['module'])
            prepared=module.prepare_input(case,plan,responses)
            if record.get('input') is not None:
                result['inputs_rebuilt']+=1
                if prepared.get('messages')!=record['input']:mismatches.append([record['slot_id'],'input'])
            if record.get('score') is not None:
                response=record.get('response') or {'status':record['status']}
                rebuilt=sudoku.score_response(case,plan,response,responses) if task=='sudoku' else scored(case,plan,response,responses)
                result['scores_rebuilt']+=1
                if rebuilt!=record['score']:mismatches.append([record['slot_id'],'score'])
    return {'status':'passed' if not mismatches else 'failed',**dict(result),'mismatches':mismatches,'real_model_calls':0}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=DATA)
    parser.add_argument('--replay',action='store_true',help='rebuild followup and Sudoku inputs and scores from saved replies')
    parser.add_argument('--kitchen',action='store_true',help='include native kitchen replay after prepare_kitchen.py')
    args=parser.parse_args()
    result={'data_integrity':verify_data(args.data)}
    if args.replay:result['replay']=replay(args.data,args.kitchen)
    print(json.dumps(result,ensure_ascii=True,indent=2))
    return 1 if result.get('replay',{}).get('status')=='failed' else 0


if __name__=='__main__':
    raise SystemExit(main())
