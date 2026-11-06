"""Single public launcher: full retraining, frozen replay, or figure-only work."""
from __future__ import annotations
import argparse,json,os,subprocess,sys,time,traceback
from pathlib import Path
import package_support as s

def execute(script,args,run,stage):
    log=run/(stage+'.log');command=[sys.executable,'-X','utf8','-B',str(s.PACKAGE/script),*map(str,args)]
    print(stage,flush=True)
    with log.open('x',encoding='utf-8')as f:
        f.write(json.dumps(dict(command=command,started_utc=s.now()))+'\n');f.flush()
        proc=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace')
        for line in proc.stdout:
            f.write(line);f.flush();print(line.rstrip(),flush=True)
        code=proc.wait()
    if code:raise RuntimeError(stage+' failed; see '+str(log))

def figures(logrun,results=None):
    if results:
        os.environ['TVT_FIGURE_INPUTS']=str((results/'summary/figure_inputs').absolute())
        if (results/'uk/all_hourly_predictions.parquet').exists():
            os.environ['TVT_UK_PREDICTIONS']=str((results/'uk/all_hourly_predictions.parquet').absolute())
    folder=s.path('figures')/('render_'+s.timestamp());folder.mkdir(parents=True)
    entries=['plot_fig01_measurement.py','plot_fig02_itu.py','plot_fig02_atmosphere.py','plot_fig05_tabm.py',
       'plot_fig06_rfe.py','plot_fig07_independent_rfe.py','plot_shap_k08.py','plot_fig08_uk.py',
       'plot_fig10_fast_temporal.py','plot_s01_combined_examples.py']
    for file in entries:execute('figsrc/'+file,['--out',folder/Path(file).stem],logrun,Path(file).stem)
    s.write(folder/'render_manifest.json',dict(scope=('RFE, external and temporal statistics use the new run; static artwork, ITU and SHAP retain published reference sources.' if results else 'Publication figures redrawn from frozen source data; physical schematics retain their verified vector exports.'),
        figure_inputs=s.rel(s.path('figure_inputs')),
        script_entries=entries,files=[dict(path=s.rel(p),sha256=s.sha(p))for p in folder.rglob('*.pdf')]))
    return folder

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--mode',choices=['full','frozen','figures'],default='full')
    a=ap.parse_args();stamp=s.timestamp();logrun=s.path('logs')/('run_'+stamp);logrun.mkdir(parents=True)
    result=s.path('new_runs')/stamp;start=time.perf_counter()
    with s.run_lock(a.mode):
        s.write(logrun/'run.json',dict(mode=a.mode,started_utc=s.now(),pid=os.getpid(),environment=s.environment()))
        try:
            if a.mode!='figures':
                execute('src/data_preparation/prepare.py',['--out',logrun/'inputs'],logrun,'01_prepare')
            if a.mode=='full':
                execute('src/data_preparation/rebuild_era5.py',['--out',result/'era5_extracted'],logrun,'01b_era5_interpolation')
                execute('src/hourly/train_rfe.py',['--output',result/'hourly'],logrun,'02_hourly_training')
                execute('src/fast/generator.py',['--out',result/'generator'],logrun,'03_generator_fit_and_cv')
                execute('src/evaluation/verify_reference.py',['--out',result/'uk','--skip-state-audit','--hourly-run',result/'hourly'],logrun,'04_external_inference')
                execute('src/fast/evaluate.py',['--out',result/'fast','--generator',result/'generator','--hourly-run',result/'hourly','--uk-predictions',result/'uk/all_hourly_predictions.parquet'],logrun,'05_second_scale_evaluation')
                execute('src/evaluation/summarize.py',['--fast',result/'fast','--hourly',result/'hourly','--uk',result/'uk','--out',result/'summary'],logrun,'06_new_result_tables')
            elif a.mode=='frozen':
                execute('src/evaluation/verify_reference.py',['--out',result/'verified'],logrun,'02_frozen_replay')
                execute('src/fast/evaluate.py',['--out',result/'fast'],logrun,'03_frozen_generator_evaluation')
                execute('src/evaluation/summarize.py',['--fast',result/'fast','--out',result/'summary'],logrun,'04_result_tables')
            rendered=figures(logrun,result if a.mode!='figures' else None)
            s.write(logrun/'completion.json',dict(status='COMPLETE',mode=a.mode,elapsed_seconds=time.perf_counter()-start,
                figures=s.rel(rendered),new_results=s.rel(result),full_training_executed=a.mode=='full',
                figure_scope='See render_manifest.json for the exact source of each figure.'))
        except BaseException as error:
            s.write(logrun/'failure.json',dict(status='FAILED_REQUIRES_INSPECTION',error=type(error).__name__,message=str(error),traceback=traceback.format_exc()))
            raise
if __name__=='__main__':main()
