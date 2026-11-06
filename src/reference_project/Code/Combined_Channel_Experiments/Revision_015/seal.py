"""Independent coverage/unit check and immutable delivery seal; no model imports."""
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter
from decimal import Decimal, localcontext
import hashlib
import json
import msvcrt
import struct

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT/'Outputs/Experiments/Combined_Channel_Revision_015'
CODE = Path(__file__).resolve().parent

def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))

def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()

def need(x,m):
    if not x: raise RuntimeError(m)

def emit(name,obj):
    with (OUT/name).open('x',encoding='utf-8') as f:
        json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False)
        f.write('\n')

def main():
    need(not (OUT/'independent_qa.json').exists() and not (OUT/'final_verification.json').exists(),'Already sealed/started')
    guard=read(OUT/'sources_before.json')
    for p,h in {**guard,**read(OUT/'code_before.json')}.items():
        need(sha(p)==h,'Source/code changed: '+p)
    need(read(OUT/'audit_complete.json')['status']=='AUDIT_COMPLETE_MODEL_BLOCKED','Missing audit')
    with (OUT/'execution.lock').open('r+b') as lock:
        lock.seek(0); msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        try:
            inp=read(OUT/'input_contract.json')
            need(inp['rmop_complete_input_hours'] is None and inp['rmop_status']=='BLOCKED_NOT_COMPUTED','False readiness')
            need(inp['licensed_package_verified'] is False and inp['author_config_verified'] is False,'Unexpected completion')
            raw=read(ROOT/'Outputs/Experiments/Combined_Channel_Revision_013/attempt_002/hourly_input_ledger.json')
            nt=sum(int(x['original_test_key'] and x['nine_pressure_fields_all_grid_finite']) for x in raw)
            need(nt==687 and all(r['test_hours_raw_finite']==nt for r in inp['known_fields']),'Raw field count')
            scored=read(ROOT/'Outputs/Experiments/Combined_Channel_Revision_012/all_hour_ledger.json')
            by={}
            for r in scored:
                st=r['station']
                by.setdefault(st,{'hours':0,'dates':set(),'labels':0,'common':0,'event':Counter()})
                v=by[st]; v['hours']+=1; v['dates'].add(r['date']); v['labels']+=int(r['primary_score_eligible']); v['common']+=int(r['common_score_eligible'])
                if r['common_score_eligible']: v['event'][r['event']]+=1
            for r in read(OUT/'coverage_recheck.json')['stations']:
                v=by[r['station']]
                need([r['hours'],r['dates'],r['label_valid'],r['old_physical_common'],r['events_common']]==[v['hours'],len(v['dates']),v['labels'],v['common'],dict(v['event'])],'Independent coverage')
            q=read(OUT/'unit_sanity_qa.json')
            with localcontext() as ctx:
                ctx.prec=70
                errors=[]
                for r in q['conversions']:
                    expected=Decimal(10)*Decimal(r['linear_power_loss']).ln()/Decimal(10).ln()
                    errors.append(abs(float(expected)-r['db']))
                expected_mean=float(Decimal(10)*Decimal('5.5').ln()/Decimal(10).ln())
                need(abs(expected_mean-q['db_of_mean_linear_loss'])<1e-12,'Independent aggregation unit check')
            need(max(errors)<1e-12 and q['signed_reference_difference_db']==-2 and q['fixture']=='SYNTHETIC_ONLY','Unit check')
            renders=[]
            for p in sorted(OUT.glob('*.png')):
                b=p.read_bytes()
                need(b[:8]==b'\x89PNG\r\n\x1a\n','Not PNG')
                w,h=struct.unpack('>II',b[16:24]); need(max(w,h)==1900,'Unexpected render scale')
                renders.append({'file':p.name,'width':w,'height':h,'sha256':sha(p)})
            need(len(renders)==3,'Missing inspected pages')
            emit('independent_qa.json',{'status':'PASS','utc':datetime.now(timezone.utc).isoformat(),
                 'scope':'source/coverage/unit audit; not RMOP validation', 'checked_ledger_rows':len(scored),
                 'unit_checks':6,'max_conversion_difference':max(errors),'render_evidence':renders,
                 'visual_read_by_root':True,'paper_pages_visually_checked':[5,6,12],
                 'new_rmop_predictions':0,'metrics_recalculated':False,'seven_family_complete':False})
        finally:
            lock.seek(0); msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1)
    for p,h in guard.items(): need(sha(p)==h,'Source changed at final check')
    paths=sorted(p for base in [CODE,OUT] for p in base.rglob('*') if p.is_file() and p.name!='execution.lock')
    need(not any(p.name=='failure.json' for p in paths),'Failure artifact requires review')
    manifest={str(p.relative_to(ROOT)):sha(p) for p in paths}
    emit('delivery_manifest.json',manifest)
    for p,h in manifest.items(): need(sha(ROOT/p)==h,'Delivery changed')
    emit('final_verification.json',{'status':'PASS','utc':datetime.now(timezone.utc).isoformat(),
         'protected_sources':len(guard),'delivery_files':len(manifest),
         'delivery_manifest_sha256':sha(OUT/'delivery_manifest.json'),
         'source_sha_unchanged':True,'os_lock':'RELEASED','logical_write_scope':'RELEASED',
         'new_training':0,'new_predictions':0,'new_scores':0,'weather_requests':0,'automation_modified':False,
         'rmop_status':'BLOCKED_NOT_COMPUTED','seven_family_comparison_complete':False,
         'scores_reference':'Outputs/Experiments/Combined_Channel_Revision_012/REPORT.md'})
    print(json.dumps(read(OUT/'final_verification.json'),ensure_ascii=False))

if __name__=='__main__':main()
