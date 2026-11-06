"""Independent scalar metrics and preprocessing; no production metric imports."""
from __future__ import annotations

import math
import statistics

import core as c


def scores(record):
    truth, pred = record['truth_db'], record['prediction_db']
    c.require(len(truth) == len(pred) == len(record['hour_bjt']), 'Prediction record length')
    c.require(len(set(record['hour_bjt'])) == len(truth), 'Duplicate prediction keys')
    result = {}
    for group in c.WEATHER:
        indices = list(range(len(truth))) if group == 'overall' else [i for i,b in enumerate(record['weather_masks'][group]) if b]
        error = [float(pred[i])-float(truth[i]) for i in indices]
        c.require(all(math.isfinite(x) for x in error), 'Nonfinite error')
        n = len(error)
        result[group] = dict(n=n, n_dates=len({record['hour_bjt'][i][:10] for i in indices}),
            rmse_db=math.sqrt(math.fsum(e*e for e in error)/n) if n else None,
            mae_db=math.fsum(abs(e) for e in error)/n if n else None,
            bias_db=math.fsum(error)/n if n else None)
    return result


def compare_metrics(expected, actual):
    for group in c.WEATHER:
        for name, value in expected[group].items():
            other = actual[group][name]
            c.require(other is None if value is None else other is not None and abs(other-value) < 1e-10,
                      f'Independent metric mismatch {group}/{name}')


def check_record(record, data):
    c.require(record['hour_bjt'] == data.hour_bjt.astype(str).tolist(), 'Prediction key alignment differs')
    c.require(record['truth_db'] == data[c.TARGET].to_numpy(float).tolist(), 'Target differs')
    for group in c.WEATHER[1:]:
        c.require(record['weather_masks'][group] == data[group].astype(bool).tolist(), 'Weather mask differs')
    recalculated = scores(record)
    compare_metrics(recalculated, record['metrics'])
    return recalculated


def check_preprocess(prep, data, names):
    # Independent column-by-column standard-library implementation.
    for j, name in enumerate(names):
        raw = [float(x) for x in data[name]]
        finite = [x for x in raw if math.isfinite(x)]
        median = statistics.median(finite) if finite else 0.
        filled = [x if math.isfinite(x) else median for x in raw]
        mean = statistics.fmean(filled)
        scale = statistics.pstdev(filled)
        if scale < 1e-12:
            scale = 1.
        for key, value in [('median',median), ('mean',mean), ('scale',scale)]:
            c.require(math.isclose(prep[key][j], value, rel_tol=1e-10, abs_tol=1e-10), f'Train-only preprocessing mismatch {name}/{key}')
    y = [float(x) for x in data[c.TARGET]]
    c.require(math.isclose(prep['y_mean'],statistics.fmean(y),abs_tol=1e-12) and
              math.isclose(prep['y_scale'],max(statistics.pstdev(y),1e-6),abs_tol=1e-12), 'Target normalization differs')
    c.require(prep['fitted_hour_sha256'] == c.fingerprint(data.hour_bjt.astype(str).tolist()), 'Preprocess fitting keys differ')


def verify_job(result, train, hold=None):
    from train_search import load_model
    check_preprocess(result['preprocess'], train, result['feature_names'])
    net, bundle = load_model(result)
    maximum = 0.
    for data, key in [(train,'training'), (hold,'holdout')]:
        if data is None:
            continue
        check_record(result[key],data)
        x = c.transform(data,bundle['names'],bundle['preprocess'])
        pred = c.predict(net,x,bundle['preprocess'])
        delta = float(c.np.max(c.np.abs(pred - result[key]['prediction_db'])))
        c.require(delta < 1e-8, 'Saved weight replay differs')
        maximum = max(maximum,delta)
    return maximum


def verify(manifest):
    target = c.OUT/'search_qa.json'
    if target.exists():
        previous = c.read(target)
        for path,digest in previous['verified_files'].items():
            c.require(c.sha(c.OUT/path)==digest,'Previously verified search changed')
        return previous
    c.source_manifest()
    train,val = c.frame('train',manifest),c.frame('validation',manifest)
    objective_rows, maximum, verified = [],0.,{}
    for config in manifest['configs']:
        path = c.OUT/'jobs'/f'search_{config["id"]:02d}'/'result.json'
        record = c.read(path)
        c.require(record['config']==config and record['feature_names']==manifest['feature_names'], 'Candidate config/names changed')
        c.require(record['seed']==17 and not record['test_used_for_fit_or_selection'],'Invalid seed/test use')
        maximum=max(maximum,verify_job(record,train,val))
        metric=scores(record['holdout'])
        objective=math.sqrt(math.fsum(metric[g]['rmse_db']**2 for g in ('overall','rain','dry_overcast'))/3)
        c.require(abs(objective-record['best_validation_objective_db'])<1e-10,'Selected checkpoint objective differs')
        hist=record['round_history']
        c.require([r['epoch'] for r in hist]==list(range(1,len(hist)+1)),'Incomplete epoch history')
        best=min(hist,key=lambda r:(r['official_validation_objective_db'],r['epoch']))
        c.require(record['selected_epoch']==best['epoch'],'Epoch selected incorrectly')
        objective_rows.append((objective,config['id']))
        verified[str(path.relative_to(c.OUT))]=c.sha(path)
        verified[record['model_file']]=record['model_sha256']
    selection=c.read(c.OUT/'search_selection.json')
    c.require(selection['config']['id']==min(objective_rows)[1],'Candidate selection rule failed')
    verified['search_selection.json']=c.sha(c.OUT/'search_selection.json')
    qa=dict(status='PASS',scope='five full156 candidates, source/normalization/keys/metrics/checkpoint/selection',
            max_prediction_replay_error_db=maximum,verified_files=verified, utc=c.now(),test_rows_used=0,
            numerical_auditor='same agent; independent standard-library numerical implementation, not outside peer review')
    c.emit(target,qa)
    return qa


def verify_full156(manifest):
    target=c.OUT/'full156_qa.json'
    if target.exists():
        for name,digest in c.read(target)['verified_files'].items():
            c.require(c.sha(c.OUT/name)==digest,'Full156 evaluation changed')
        return
    test=c.frame('test',manifest,allow_test=True)
    files={}
    maximum=0.
    for path in sorted((c.OUT/'full156').glob('*_test.json')):
        result=c.read(path)
        check_record(result['result'],test)
        candidate=result['config']['id']
        saved=c.read(c.OUT/'jobs'/f'search_{candidate:02d}'/'result.json')
        c.require(result['model_sha256']==saved['model_sha256'] and result['fit_calls']==0,'Full156 model identity')
        from train_search import load_model
        net,bundle=load_model(saved)
        predicted=c.predict(net,c.transform(test,bundle['names'],bundle['preprocess']),bundle['preprocess'])
        delta=float(c.np.max(c.np.abs(predicted-result['result']['prediction_db'])))
        c.require(delta<1e-8,'Test checkpoint replay differs')
        maximum=max(maximum,delta)
        files[str(path.relative_to(c.OUT))]=c.sha(path)
    c.require(len(files)==len({0,c.read(c.OUT/'search_selection.json')['config']['id']}),'Missing full156 comparison')
    c.emit(target,dict(status='PASS',verified_files=files,n_test=687,utc=c.now(),new_fit_calls=0,max_prediction_replay_error_db=maximum))
