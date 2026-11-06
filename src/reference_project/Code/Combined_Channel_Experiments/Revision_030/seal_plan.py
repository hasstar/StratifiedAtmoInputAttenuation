"""One-shot seal of the offline proposal after independent plan QA."""
import hashlib,json,msvcrt,sys
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[3]
CODE=Path(__file__).resolve().parent
OUT=ROOT/'Outputs/Experiments/Combined_Channel_Revision_030'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def emit(n,d):
    with (OUT/n).open('x',encoding='utf-8') as f:json.dump(d,f,ensure_ascii=False,indent=2);f.write('\n')
def now():return datetime.now(timezone.utc).isoformat()
with (OUT/'seal.claim').open('x',encoding='utf-8') as f:f.write(now())
with (OUT/'execution.lock').open('r+b') as lock:
    msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    try:
        assert read(OUT/'independent_qa.json')['status']=='PASS_PLAN_ONLY'
        assert not list(OUT.glob('*failure*'))
        for p,h in {**read(OUT/'sources_before.json'),**read(OUT/'code_before.json')}.items():assert sha(p)==h
        for name in ('PROTOCOL.md','APPROVAL_REQUEST.md','QA.md','HANDOFF.md','SOURCE_NOTES.md','COMMANDS.md'):assert (OUT/name).is_file()
        emit('seal_command.json',{'utc':now(),'argv':sys.argv,'executable':sys.executable})
        files={str(p.relative_to(ROOT)):{'sha256':sha(p),'bytes':p.stat().st_size} for folder in (CODE,OUT) for p in sorted(folder.iterdir()) if p.is_file() and p.name!='execution.lock'}
        emit('delivery_manifest.json',{'utc':now(),'files':files})
        emit('final_verification.json',{'status':'PASS_PLAN_ONLY','utc':now(),'manifest_sha256':sha(OUT/'delivery_manifest.json'),
            'sealed_files':len(files),'execution_lock':'RELEASED','authorization_status':'AWAITING_EXPLICIT_NEW_WEATHER_AUTHORIZATION',
            'weather_requests':0,'new_predictions':0,'new_scores':0,'all_seven_complete':False,'mts_status':'NOT_COMPUTED_USER_DECISION'})
        print('SEALED PLAN ONLY',len(files),sha(OUT/'delivery_manifest.json'))
    finally:
        lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1)
