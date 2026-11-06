"""Run actual offline controls with temp files confined to the new workspace."""
import contextlib
import io as streams
import json
import tempfile
import unittest
import weather_download_gbs_extend_v011 as d

def main():
    directory=d.DEST/'preflight_temp'; directory.mkdir(parents=True,exist_ok=True)
    tempfile.tempdir=str(directory)
    log=streams.StringIO()
    suite=unittest.defaultTestLoader.loadTestsFromNames(['test_gbs_extend_v011','test_ncar_recovery_v002'])
    with contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
        result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
    p=d.app.load_plan()
    record={'status':'PASS' if result.wasSuccessful() else 'FAIL','tests':result.testsRun,
            'errors':len(result.errors),'failures':len(result.failures),'plan_fingerprint':p['fingerprint'],
            'code_sha256':d.SHA(d.CODE),'test_sha256':d.SHA(d.HERE/'test_gbs_extend_v011.py'),
            'scope':'Offline controls only; live rain response verification is separate','log':log.getvalue()}
    d.io.immutable(d.DEST/'preflight_qa.json',record)
    print(json.dumps({k:v for k,v in record.items() if k!='log'}))
    if not result.wasSuccessful(): print(log.getvalue());raise SystemExit(1)

if __name__=='__main__': main()
