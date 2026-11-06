"""Independent output metrics and weight replay audit, no estimator fitting."""
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

def verify():
    m=p.manifest();frozen=p.selection(m)
    p.require(p.read(p.OUT/'train_qa.json')['status']=='PASS','Train QA gate')
    frames={s:p.v.data_frame(s,m,True) for s in ('train','validation','test')}
    sets=[set(f.hour_bjt.astype(str)) for f in frames.values()]
    p.require(all(sets[i].isdisjoint(sets[j]) for i in range(3) for j in range(i)),'Exact-hour overlap')
    frame=frames['test'];truth=frame[m['target']].to_numpy(float)
    checks=0;largest=0.;seed_records=[];hour_records=[]
    for case in p.CASES:
        for name in m['models']:
            plan=frozen['plans'][case][name]
            # Validate every search trial independently, not just the selected score.
            candidates=[]
            for i in range(8):
                row=p.read(p.path('trials',case,name,f'config{i}'))
                target=frames['validation'][m['target']].to_numpy(float)
                err=np.asarray(row['validation_prediction_db'])-target
                components=[np.dot(err,err)/len(err)]
                for group in ('rain','dry_overcast'):
                    eg=err[frames['validation'][group].to_numpy(bool)]
                    components.append(np.dot(eg,eg)/len(eg))
                score=float(np.sqrt(sum(components)/3))
                p.require(abs(score-row['validation_objective_db'])<1e-12,'Candidate objective mismatch')
                if row['fit_metadata']['solver_converged']:candidates.append((score,i))
            p.require(min(candidates)[1]==plan['config_id'],'Best validation configuration differs')
            for seed in m['seeds']:
                fit=p.read(p.path('training',case,name,f'seed{seed}'))
                row=p.read(p.path('test',case,name,f'seed{seed}'))
                p.require(row['fingerprint']==fit['fingerprint']==m['fingerprint'] and row['plan']==plan and
                          row['feature_names']==p.case_plan(m,case)['retained_names'] and row['fit_calls_during_evaluation']==0,'Record identity')
                p.require(row['hour_bjt']==frame.hour_bjt.astype(str).tolist() and np.array_equal(row['truth_db'],truth),'Test alignment')
                path=p.model_path(case,name,seed);p.require(row['model_sha256']==fit['model_sha256']==p.sha(path),'Model digest')
                bundle=p.joblib.load(path)
                pred=np.asarray(row['prediction_db']);actual=p.v.predict(bundle,p.v.raw(frame,p.case_plan(m,case)))
                delta=float(np.max(np.abs(actual-pred)));largest=max(largest,delta);p.require(delta<1e-10,'Test replay')
                # Scaler's stored mean must come only from transformed train rows.
                xtrain=p.v.raw(frames['train'],p.case_plan(m,case))
                transformed=bundle['imputer'].transform(xtrain[:,bundle['positions']])
                if bundle['scaler'] is not None:
                    p.require(np.allclose(bundle['scaler'].mean_,transformed.mean(axis=0),atol=1e-10,rtol=1e-10),'Scaler not train-fitted')
                p.require(not fit['test_read_for_fit'] and not fit['validation_changed_formal_fit'] and
                          all(r['validation_objective_db'] is None for r in fit['round_history']), 'Holdout formal optimization')
                for group in p.v.WEATHER:
                    mask=np.ones(len(frame),bool) if group=='overall' else frame[group].to_numpy(bool)
                    if group!='overall':p.require(np.array_equal(mask,row['weather_masks'][group]),'Weather identity')
                    err=pred[mask]-truth[mask]
                    vals={'rmse_db':float(np.sqrt(np.dot(err,err)/len(err))), 'mae_db':float(sum(abs(x) for x in err)/len(err)),
                          'bias_db':float(sum(err)/len(err))}
                    for key,value in vals.items():
                        p.require(abs(value-row['metrics'][group][key])<1e-12,'Independent metric mismatch');checks+=1
                    p.require(row['metrics'][group]['n_hours']==int(mask.sum()),'Weather count')
                    seed_records.append(dict(case=case,model=name,seed=seed,weather=group,n_hours=int(mask.sum()),**vals))
                for i,h in enumerate(row['hour_bjt']):
                    hour_records.append({'case':case,'model':name,'seed':seed,'hour_bjt':h,'truth_db':float(truth[i]),
                                         'prediction_db':float(pred[i]),'error_db':float(pred[i]-truth[i]),
                                         **{g:bool(row['weather_masks'][g][i]) for g in p.v.WEATHER if g!='overall'}})
                del bundle
    p.require(len(list((p.OUT/'trials').glob('*.json')))==160 and all(len(list((p.OUT/d).glob(pattern)))==60
              for d,pattern in [('training','*.json'),('models','*.joblib'),('test','*.json')]),'Artifact completeness')
    p.require(checks==900 and len(seed_records)==300 and len(hour_records)==41220,'Metric/hour count')
    p.pd.DataFrame(seed_records).to_csv(p.OUT/'per_seed_weather_metrics.csv',index=False,encoding='utf-8-sig')
    p.pd.DataFrame(hour_records).to_csv(p.OUT/'hourly_predictions.csv',index=False,encoding='utf-8-sig')
    result={'status':'PASS','fingerprint':m['fingerprint'],'candidate_records':160,'formal_records':60,'models':60,
            'test_records':60,'metric_checks':900,'seed_input_weather_rows':300,'hourly_predictions':41220,
            'max_replay_difference_db':largest,'test_model_fit_calls':0,
            'test_dates_overlapping_train':len(set(frame.hour_bjt.dt.strftime('%Y-%m-%d')) & set(frames['train'].hour_bjt.dt.strftime('%Y-%m-%d'))),
            'historical_test_access':True,'utc':p.now()}
    p.atomic(p.OUT/'independent_qa.json',result);print(json.dumps(result,indent=2))

if __name__=='__main__':
    with p.lock('independent-verify'):verify()
