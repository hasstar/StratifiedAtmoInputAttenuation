"""A single local successor: waits for exact search identity, then acquires writer lock."""
from __future__ import annotations

import argparse
import ast
import contextlib
import msvcrt
import sys
import time
import traceback

import core as c
import pipeline as p


@contextlib.contextmanager
def queue_lock():
    with (c.OUT/'successor.lock').open('a+b') as handle:
        if handle.tell()==0:
            handle.write(b'0'); handle.flush()
        handle.seek(0)
        msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)
        try:
            yield
        finally:
            handle.seek(0); msvcrt.locking(handle.fileno(),msvcrt.LK_UNLCK,1)


def syntactic_and_contract_checks(source):
    for name in p.PIPE_SOURCES:
        if name.endswith('.py'):
            ast.parse((c.HERE/name).read_text(encoding='utf-8'))
    names=source['feature_names']
    states=[c.names_for_mask(i,names) for i in range(1024)]
    c.require(len({tuple(x) for x in states})==1024 and states[1023]==names and len(states[0])==50,'All layer masks')
    c.require(sum(range(2,157))*3==36735,'RFE evaluation count')
    # Pure combinatorial checks: no training, no timing pilot, no research predictions.
    for bit in range(10):
        pairs=[(i,i|(1<<bit)) for i in range(1024) if not i&(1<<bit)]
        c.require(len(pairs)==512 and all(set(states[a])<set(states[b]) for a,b in pairs),'Layer context mapping')
    contract=dict(status='PASS',research_fit_calls=0,timing_pilot=False,layer_states=1024,
                  rfe_states=156,permutation_evaluations=36735,
                  code_sha256={name:c.sha(c.HERE/name) for name in p.PIPE_SOURCES},
                  search_source_fingerprint=source['fingerprint'])
    path=c.OUT/'pipeline_preflight.json'
    if path.exists():
        c.require(c.read(path)==contract,'Continuation code changed after preflight')
    else:
        c.emit(path,contract)
    return contract


def search_identity_alive(pid,created):
    try:
        process=c.psutil.Process(pid)
        c.require(abs(process.create_time()-created)<.01,'Search PID has been reused; refuse succession')
        command=process.cmdline()
        c.require(any('Revision_031/train_search.py' in x.replace('\\','/') for x in command),'Search command identity differs')
        return process.is_running()
    except c.psutil.NoSuchProcess:
        return False


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--wait-pid',type=int,required=True)
    parser.add_argument('--wait-created',type=float,required=True)
    args=parser.parse_args()
    started=dict(pid=c.os.getpid(),process_created=c.psutil.Process().create_time(),started_utc=c.now(),
                 parent_search_pid=args.wait_pid,parent_search_created=args.wait_created,
                 command=sys.argv,role='local successor; no fitting until predecessor exits and writer lock acquired')
    with queue_lock():
        try:
            source=c.source_manifest()
            checks=syntactic_and_contract_checks(source)
            owner=c.read(c.OUT/'active_worker.json')
            c.require(owner['pid']==args.wait_pid and abs(owner['process_created']-args.wait_created)<.01 and owner['stage']=='formal-search','Search receipt does not match requested predecessor')
            c.emit(c.OUT/'queue_receipts'/f'{time.time_ns()}.json',dict(started,code_sha256=checks['code_sha256']))
            c.update(c.OUT/'successor_active.json',dict(started,status='WAITING_SEARCH',utc=c.now()))
            print('QUEUE_READY: exact search identity; full layer/RFE successor; no concurrent fit',flush=True)
            while search_identity_alive(args.wait_pid,args.wait_created):
                c.update(c.OUT/'successor_active.json',dict(started,status='WAITING_SEARCH',utc=c.now()))
                time.sleep(10)
            owner=c.read(c.OUT/'active_worker.json')
            c.require(owner['pid']==args.wait_pid and abs(owner['process_created']-args.wait_created)<.01 and owner['status']=='EXITED','Predecessor failed/interrupted; do not auto-retrain')
            c.require(c.read(c.OUT/'search_qa.json')['status']=='PASS' and c.read(c.OUT/'full156_qa.json')['status']=='PASS','Search numerical QA not complete')
            c.source_manifest()
            for name,digest in checks['code_sha256'].items():
                c.require(c.sha(c.HERE/name)==digest,'Continuation source changed while waiting')
            with c.writer_lock('layer-rfe'):
                c.update(c.OUT/'successor_active.json',dict(started,status='TRAINING_LAYER_RFE',utc=c.now()))
                p.run()
            c.update(c.OUT/'successor_active.json',dict(started,status='NUMERICS_COMPLETE_REPORT_PENDING',utc=c.now()))
        except BaseException:
            c.update(c.OUT/'successor_active.json',dict(started,status='FAILED',utc=c.now()))
            c.emit(c.OUT/'failures'/f'{time.time_ns()}_successor.json',dict(started,utc=c.now(),traceback=traceback.format_exc()))
            raise


if __name__=='__main__':
    main()
