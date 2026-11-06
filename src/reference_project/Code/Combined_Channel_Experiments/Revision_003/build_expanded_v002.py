"""Resume new QA after a negative-control fixture type error; preserve attempt001.

No acquisition/data change. The legacy catalog stores accepted units as sets;
the test fixture needs an ordered unit. Decode acceptance still has all units.
"""
import traceback
import build_expanded as b
from common import *

def main():
    offline()
    with lock():
        require(not (OUT/'dataset_complete.json').exists(),'Dataset already complete')
        guard=read(OUT/'source_sha256_before.json');protected_verify(guard)
        def protect(path,expected=None):
            path=Path(path).resolve();digest=sha(path)
            if expected:require(digest==expected,'Frozen SHA mismatch: '+str(path))
            guard[str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)]=digest
            return path
        plan=read(ACQ/'plan.json');signed_check(plan)
        require(sha(ACQ/'plan.json')==PLAN_SHA and (ACQ/'STOP_REQUESTED.json').exists(),'Plan/stop')
        complete,maps,units,parents=b.select_scope(plan)
        require(complete==read(OUT/'cohort_manifest.json')['native_days'],'Scope unchanged from attempt001')
        # Identity membership is unchanged; deterministic unit choice only for negative tests.
        b.CATALOG={k:(v[0],v[1],v[2],tuple(sorted(v[3]))) for k,v in b.CATALOG.items()}
        emit(OUT/'build_launch_v002.json',{'started':now(),'pid':os.getpid(),'repair':'negative-test unit-set indexing only; no scientific change','code_sha256':{p.name:sha(p) for p in HERE.glob('*.py')}})
        frame=b.build_fields(plan,complete,maps,units,protect)
        print('field_gate_pass',len(frame),flush=True)
        frame=b.native_audit(frame,plan,complete,protect)
        frame.to_parquet(OUT/'expanded_dataset.parquet',index=False)
        frame.groupby(['station','period',frame.hour_utc.dt.strftime('%Y-%m')]).agg(hours=('hour_utc','size'),eligible_hours=('primary_score_eligible','sum')).to_csv(OUT/'coverage_by_month.csv')
        emit(OUT/'source_sha256_used.json',guard);protected_verify(guard)
        emit(OUT/'dataset_complete.json',{'status':'PASS','completed':now(),'station_hours':len(frame),'primary_eligible':int(frame.primary_score_eligible.sum()),'dataset_sha256':sha(OUT/'expanded_dataset.parquet'),'protected_sources':len(guard),'code_sha256':{p.name:sha(p) for p in HERE.glob('*.py')},'new_fits':0,'predictions':0,'downloads':0})
        print(read(OUT/'dataset_complete.json'),flush=True)

if __name__=='__main__':
    try:main()
    except Exception:
        emit(OUT/f'failure_build_v002_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json',{'time':now(),'traceback':traceback.format_exc()})
        raise
