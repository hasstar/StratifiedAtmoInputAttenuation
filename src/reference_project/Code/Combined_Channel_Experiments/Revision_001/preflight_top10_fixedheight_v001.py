"""Reproducible offline tests and independent static audit; never contacts CDS."""
import json
import re
import subprocess
import sys
import weather_download_top10_fixedheight_v007 as w


def main():
    plan=w.load_plan()
    w.require(not (w.DEST/'preflight_qa.json').exists(),'Do not overwrite frozen preflight QA')
    command_transport = (
        "import sys,unittest; sys.path.insert(0,'Code/Combined_Channel_Experiments/Revision_001'); "
        "import weather_download_v002 as w; w.DEST=w.EXTERNAL/'weather_top10_fixedheight_v007'; "
        "import test_weather_download_v002 as t; "
        "r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(t)); "
        "sys.exit(not r.wasSuccessful())")
    results=[]
    for label,args in [('fixedheight_61',[str(w.TEST)]),('transport_20',['-c',command_transport])]:
        result=subprocess.run([sys.executable,'-B',*args],cwd=w.ROOT,capture_output=True,text=True,encoding='utf-8',errors='replace')
        matches=re.findall(r'Ran (\d+) tests',result.stderr)
        report={'name':label,'returncode':result.returncode,'tests':int(matches[-1]) if matches else 0,
                'stdout':result.stdout,'stderr':result.stderr}
        results.append(report)
        if result.returncode:
            w.io.immutable(w.DEST/'preflight_history'/('failure_'+str(__import__('time').time_ns())+'.json'),report)
            raise RuntimeError('Offline tests failed: '+label)
    result=subprocess.run([sys.executable,'-B',str(w.AUDIT)],cwd=w.ROOT,capture_output=True,text=True,encoding='utf-8',errors='replace')
    audit={'name':'independent_static_audit','returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr}
    if result.returncode:
        w.io.immutable(w.DEST/'preflight_history'/('audit_failure_'+str(__import__('time').time_ns())+'.json'),audit)
        raise RuntimeError('Independent audit failed')
    report={'status':'PASS','checked_utc':w.io.now(),'plan_fingerprint':plan['fingerprint'],'code_sha256':w.SHA(w.CODE),
            'test_sha256':w.SHA(w.TEST),'tests_run':sum(r['tests'] for r in results),'results':results,'independent_audit':audit,
            'network_requests':0,'new_fit_calls':0,'predictions_executed':0,'full_field_QA_complete':False,
            'scope':'Offline transport/scope/packing/geometry/field gates and independent static plan QA only; no completed external evaluation.'}
    w.io.immutable(w.DEST/'preflight_qa.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('results','independent_audit')},indent=2))


if __name__=='__main__':main()
