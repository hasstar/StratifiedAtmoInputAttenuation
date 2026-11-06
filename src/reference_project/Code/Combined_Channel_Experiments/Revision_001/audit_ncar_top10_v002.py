"""Bind v009 local concurrency regression and immutable v008 subset migration."""
import json
import subprocess
import sys
import weather_download_top10_ncar_v009 as fix
import audit_ncar_top10_v001 as base

def main():
    d=fix.app
    plan=d.build_plan() if not d.PLAN.exists() else d.load_plan()
    cp=subprocess.run([sys.executable,'-B',str(d.HERE/'test_ncar_recovery_v002.py'),'-v'],capture_output=True,text=True,encoding='utf-8')
    result={'status':'PASS' if cp.returncode==0 else 'FAIL','plan_fingerprint':plan['fingerprint'],'code_sha256':d.SHA(d.CODE),
            'exit_code':cp.returncode,'stdout':cp.stdout,'stderr':cp.stderr,'scope':'Offline transport/concurrency/error/recovery controls'}
    if cp.returncode:
        d.io.immutable(d.DEST/'preflight_history'/(str(d.time.time_ns())+'_failed.json'),result);print(json.dumps(result));sys.exit(1)
    qa=base.audit(plan)
    all_keys=set()
    for item in plan['ncar_reusable']:
        record=fix.old_records(item);keys={tuple(k) for k,v in record['records']}
        d.require(not (all_keys&keys),'Migrated NCAR duplicate samples');all_keys.update(keys)
    qa.update(reused_ncar_subsets=len(plan['ncar_reusable']),reused_ncar_grid_field_hours=len(all_keys),
              source_v008_state_files_unchanged=True)
    if d.PLAN.exists():
        d.io.immutable(d.DEST/'preflight_qa.json',result);d.io.immutable(d.DEST/'plan_independent_qa.json',qa)
    else:
        d.io.immutable(d.DEST/'candidate_preflight.json',result);d.io.immutable(d.DEST/'candidate_plan_audit.json',qa)
    print(json.dumps(qa,indent=2))

if __name__=='__main__':main()
