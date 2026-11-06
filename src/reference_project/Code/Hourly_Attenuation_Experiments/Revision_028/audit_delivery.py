"""Independent final aggregation, ranking, date intervals and complete delivery."""
import importlib.util
import json
import sys
from pathlib import Path
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
spec=importlib.util.spec_from_file_location('weather28',HERE/'run_weather.py')
p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)
np=p.np

def run():
    m=p.manifest();report=p.read(p.OUT/'final_report.json')
    p.require(p.read(p.OUT/'independent_qa.json')['status']=='PASS','Independent QA')
    ranks=[];checks=0;maxerr=0.
    for row in report['model_rows']:
        name=row['model'];items=p.paired_rows(m,name)
        truth=np.asarray(items[0]['truth_db'])
        errors={case:np.asarray([r[f'{case}_prediction_db'] for r in items])-truth for case in p.CASES}
        results={}
        for group in p.v.WEATHER:
            mask=np.ones(len(truth),bool) if group=='overall' else np.asarray(items[0]['weather_masks'][group],bool)
            results[group]={}
            for case,err in errors.items():
                z=err[:,mask];vals={'rmse_db':float(np.sqrt((z*z).mean(axis=1)).mean()),'mae_db':float(np.abs(z).mean(axis=1).mean()),'bias_db':float(z.mean(axis=1).mean())}
                for key,value in vals.items():
                    d=abs(value-row['metrics'][group][case][key]);p.require(d<1e-12,'Report mean mismatch');maxerr=max(maxerr,d)
                results[group][case]=vals['rmse_db']
            if group in m['acceptance_groups']:
                dates=sorted({h[:10] for h,yes in zip(items[0]['hour_bjt'],mask) if yes})
                ix=[np.flatnonzero(mask & np.asarray([h[:10]==day for h in items[0]['hour_bjt']])) for day in dates]
                counts=np.asarray([len(z) for z in ix])
                sse={case:np.stack([(z[:,j]**2).sum(axis=1) for j in ix]) for case,z in errors.items()}
                draws=np.random.default_rng(m['bootstrap_seed']).multinomial(len(dates),np.repeat(1/len(dates),len(dates)),size=m['bootstrap_replicates'])
                diff=np.asarray([(np.sqrt((d[:,None]*sse['base']).sum(axis=0)/(d*counts).sum())-
                                  np.sqrt((d[:,None]*sse['full']).sum(axis=0)/(d*counts).sum())).mean() for d in draws])
                published=row['metrics'][group]['date_bootstrap']
                p.require(published['n_dates']==len(dates) and published['draw_sha256']==p.fp(draws.tolist()),'Date block identity')
                for key,probs in [('percentile_95_db',[.025,.975]),('bonferroni_three_endpoint_bounds_db',[.05/6,1-.05/6])]:
                    d=float(np.max(np.abs(np.quantile(diff,probs)-published[key])));p.require(d<1e-12,'Independent paired interval');maxerr=max(maxerr,d);checks+=1
        improved=all(results[g]['full']<results[g]['base'] for g in m['acceptance_groups'])
        p.require(improved==row['all_three_mean_improvements'],'Three-endpoint comparison')
        ranks.append((results['overall']['full'],row['metrics']['overall']['full']['mae_db'],name,improved))
    ranks.sort();winner=ranks[0]
    p.require([x[2] for x in ranks]==[r['model'] for r in report['model_rows']],'Pure overall full ranking')
    guard=winner[0]<=m['prior_full_overall_benchmark_rmse_db']
    p.require(report['full_overall_winner']==winner[2] and report['continuity_guard_pass']==guard and
              report['acceptance']==('MET' if winner[3] and guard else 'NOT_MET'),'Acceptance/report mismatch')
    files=['run_manifest.json','selection.json','selection_qa.json','train_qa.json','independent_qa.json','final_report.json','RESULTS.md','per_seed_weather_metrics.csv','hourly_predictions.csv']
    qa={'status':'PASS','fingerprint':m['fingerprint'],'source_files_unchanged':len(m['source_sha256']),'independent_interval_checks':checks,
        'max_aggregation_or_interval_error_db':maxerr,'winner':winner[2],'acceptance':report['acceptance'],
        'output_sha256':{f:p.sha(p.OUT/f) for f in files},'utc':p.now()}
    p.atomic(p.OUT/'delivery_qa.json',qa);print(json.dumps(qa,indent=2))

if __name__=='__main__':
    with p.lock('audit-delivery'):run()
