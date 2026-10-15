"""Run local tests, a scripted smoke run and saved-trajectory replay without API access."""
import argparse
import contextlib
import importlib
import io
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.dont_write_bytecode=True
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'tools'))


def local_connect(original):
    def connect(sock,address):
        if isinstance(address,tuple) and address[0] not in ('127.0.0.1','::1','localhost'):
            raise RuntimeError('offline verification blocks external network')
        return original(sock,address)
    return connect


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kitchen',action='store_true',help='also replay native kitchen examples after preparing the pinned resources')
    args=parser.parse_args()
    for name in list(os.environ):
        if name.startswith('SOCIALEVO_'):os.environ.pop(name)
    temp=ROOT/'test-output';temp.mkdir(exist_ok=True);tempfile.tempdir=str(temp)
    result={'real_model_calls':0,'scope':'local tests, scripted smoke, submitted data integrity and saved-reply replay','syntax_errors':[]}
    for directory in ('atomic_experiment','atomic_relation_followup','utils','task_support','sudoku_preflight','tools'):
        for path in (ROOT/directory).rglob('*.py'):
            if 'sources' in path.relative_to(ROOT).parts:continue
            try:compile(path.read_text(encoding='utf-8-sig'),str(path),'exec')
            except (SyntaxError,UnicodeError):result['syntax_errors'].append(path.relative_to(ROOT).as_posix())
    suite=unittest.TestSuite()
    for directory in ('atomic_experiment/tests','atomic_relation_followup/tests'):
        suite.addTests(unittest.TestLoader().discover(str(ROOT/directory),pattern='test_*.py'))
    stream=io.StringIO()
    with patch.object(socket.socket,'connect',new=local_connect(socket.socket.connect)), contextlib.redirect_stdout(stream):
        executed=unittest.TextTestRunner(stream=stream,verbosity=1).run(suite)
        result['tests']={'run':executed.testsRun,'failures':len(executed.failures),'errors':len(executed.errors),'skipped':len(executed.skipped)}
        from utils.client import MockClient
        from atomic_experiment.config import Config
        from atomic_experiment.runner import run
        with tempfile.TemporaryDirectory(prefix='core-smoke-') as temporary:
            directory,aborted=run(Config(backend='mock',instances=1,repeats=1),MockClient(),Path(temporary))
            summary=json.loads((directory/'summary.json').read_text(encoding='utf-8'))
            result['scripted_smoke']={'aborted':aborted,'conditions':sorted(summary['groups']),'paired_comparisons':len(summary['paired_differences'])}
        from review_data import verify_data,replay
        result['submitted_data']=verify_data()
        result['saved_reply_replay']=replay(kitchen=args.kitchen)
    result['ok']=(not result['syntax_errors'] and executed.wasSuccessful() and not aborted and result['saved_reply_replay']['status']=='passed')
    if not result['ok']:result['test_output']=stream.getvalue()
    (ROOT/'VERIFICATION.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=True,indent=2))
    return 0 if result['ok'] else 1


if __name__=='__main__':
    raise SystemExit(main())
