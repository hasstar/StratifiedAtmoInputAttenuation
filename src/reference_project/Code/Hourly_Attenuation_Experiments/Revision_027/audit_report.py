"""Independent report/paired date-interval audit; no fit or model selection."""
import importlib.util
import json
import sys
from pathlib import Path
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('paired27', HERE / 'run_pair.py')
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)
np = p.np

def run():
    m = p.manifest()
    report = p.read(p.OUT / 'final_report.json')
    p.require(p.read(p.OUT / 'independent_qa.json')['status'] == 'PASS', 'Independent gate')
    recomputed, interval_checks, max_error = [], 0, 0.
    for published in report['model_rows']:
        name = published['model']
        items = [p.read(p.record_path('evaluation', name, s)) for s in m['seeds']]
        truth = np.asarray(items[0]['truth_db'])
        errors = {case: np.asarray([r[f'{case}_prediction_db'] for r in items]) - truth
                  for case in ('base', 'full')}
        metrics = {}
        for group in p.v.WEATHER:
            mask = np.ones(len(truth), bool) if group == 'overall' else np.asarray(items[0]['weather_masks'][group], bool)
            metrics[group] = {}
            for case in errors:
                e = errors[case][:, mask]
                vals = {'rmse_db': np.sqrt(np.sum(e*e, axis=1)/e.shape[1]).mean(),
                        'mae_db': np.abs(e).mean(axis=1).mean(), 'bias_db': e.mean(axis=1).mean()}
                for key, value in vals.items():
                    difference = abs(float(value) - published['metrics'][group][case][key])
                    p.require(difference < 1e-12, 'Published mean differs')
                    max_error = max(max_error, difference)
                metrics[group][case] = float(vals['rmse_db'])
            seed_delta = np.sqrt((errors['base'][:,mask]**2).mean(axis=1)) - np.sqrt((errors['full'][:,mask]**2).mean(axis=1))
            p.require(all(abs(published['metrics'][group]['seed_deltas_db'][str(s)] - float(d)) < 1e-12
                          for s,d in zip(m['seeds'],seed_delta)), 'Published seed delta differs')
            if group in m['acceptance_groups']:
                dates = sorted({h[:10] for h, yes in zip(items[0]['hour_bjt'],mask) if yes})
                rows_by_day = [np.flatnonzero(mask & np.asarray([h[:10] == d for h in items[0]['hour_bjt']])) for d in dates]
                n = np.asarray([len(ix) for ix in rows_by_day])
                sums = {case: np.stack([(e[:, ix]**2).sum(axis=1) for ix in rows_by_day]) for case,e in errors.items()}
                draws = np.random.default_rng(m['bootstrap_seed']).multinomial(len(dates), np.repeat(1/len(dates),len(dates)), size=m['bootstrap_replicates'])
                delta = np.asarray([(np.sqrt((count[:,None]*sums['base']).sum(axis=0)/(count*n).sum()) -
                                     np.sqrt((count[:,None]*sums['full']).sum(axis=0)/(count*n).sum())).mean() for count in draws])
                expected = published['metrics'][group]['date_bootstrap']
                for key, probs in [('percentile_95_db',[.025,.975]), ('bonferroni_three_endpoint_bounds_db',[.05/6,1-.05/6])]:
                    difference = float(np.max(np.abs(np.quantile(delta,probs)-expected[key])))
                    p.require(difference < 1e-12, 'Published interval differs')
                    max_error = max(max_error,difference)
                    interval_checks += 1
                p.require(expected['n_dates'] == len(dates) and expected['draw_sha256'] == p.fp(draws.tolist()), 'Date draws differ')
        recomputed.append((metrics['overall']['full'],name,all(metrics[g]['full'] < metrics[g]['base'] for g in m['acceptance_groups'])))
    recomputed.sort()
    p.require([x[1] for x in recomputed] == [r['model'] for r in report['model_rows']], 'Full rank differs')
    p.require(report['acceptance'] == ('MET' if recomputed[0][2] else 'NOT_MET'), 'Acceptance differs')
    outputs = ['final_report.json','RESULTS.md','paired_hourly_predictions.csv','per_seed_weather_metrics.csv','fit_qa.json','independent_qa.json']
    qa = {'status':'PASS','fingerprint':m['fingerprint'],'report_model_rows':len(recomputed),'independent_interval_checks':interval_checks,
          'max_metric_or_interval_error_db':max_error,'winner':recomputed[0][1],'acceptance':report['acceptance'],
          'test_based_strategy_is_exploratory':True,'output_sha256':{n:p.sha(p.OUT/n) for n in outputs},
          'audit_script_sha256':p.sha(Path(__file__)), 'utc':p.now()}
    p.atomic(p.OUT/'report_qa.json',qa)
    print(json.dumps(qa,indent=2))

if __name__ == '__main__':
    with p.lock('audit-report'):
        run()
