"""Bounded symmetric50/156 weather-aware ten-family experiment."""
from __future__ import annotations
import argparse
import contextlib
import gc
import importlib.util
import json
import os
import sys
import time
import traceback
from pathlib import Path
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(HERE))
spec=importlib.util.spec_from_file_location('frozen26',HERE.parent/'Revision_026/run_test_benchmark.py')
v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)
import engine as e
np,pd,joblib=v.np,v.pd,v.joblib
read,sha,fp,atomic,require,now=v.read,v.sha,v.fp,v.atomic_json,v.require,v.now
OUT=ROOT/'Outputs/Experiments/Hourly_Attenuation_Revision_028_Weather_Balanced'
CASES=('base','full')

@contextlib.contextmanager
def lock(stage):
    import msvcrt
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'writer.lock').open('a+b') as f:
        if f.tell()==0:f.write(b'0');f.flush()
        f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
        atomic(OUT/'active_worker.json',{'pid':os.getpid(),'stage':stage,'command':sys.argv,'status':'RUNNING','started_utc':now()})
        try:yield
        except Exception:
            atomic(OUT/'failures'/f'{time.time_ns()}_{stage}.json',{'stage':stage,'traceback':traceback.format_exc(),'utc':now()})
            raise
        finally:
            atomic(OUT/'active_worker.json',{'pid':os.getpid(),'stage':stage,'status':'EXITED','finished_utc':now()})
            f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_UNLCK,1)

def case_plan(m,case):
    return dict(m,retained_names=m['base_names'] if case=='base' else m['retained_names'])

def manifest(create=False):
    parent=v.manifest()
    paths=[ROOT/p for p in parent['source_sha256']]
    paths += [v.OUT/'run_manifest.json',v.OUT/'final_report.json',v.OUT/'delivery_qa.json',
              HERE.parent/'Revision_027/run_pair.py']
    paths += [HERE/n for n in ('PROTOCOL.md','PARAMETERS.md','run_weather.py','engine.py','verify_results.py','audit_delivery.py')]
    paths += [OUT/'synthetic_qa.json',OUT/'preflight_qa.json']
    for name in ('synthetic_qa.json','preflight_qa.json'):
        require(read(OUT/name)['status']=='PASS',f'Preflight gate {name}')
    obj={'version':'rev028-bounded-symmetric-weather-direct-v1','source_sha256':{str(p.relative_to(ROOT)):sha(p) for p in dict.fromkeys(paths)},
         'parent_fingerprint':parent['fingerprint'],'original_names':parent['original_names'],'retained_names':parent['retained_names'],
         'base_names':[n for n in parent['retained_names'] if n.startswith('surface__')], 'models':parent['models'],'seeds':parent['seeds'],
         'grids':{name:e.grids(name) for name in parent['models']},'target':parent['target'],
         'n_train':2404,'n_validation':343,'n_test':687,'search_seed':17,'search_fits':160,'formal_fits':60,
         'objective':'sqrt((overallMSE+rainMSE+dryOvercastMSE)/3)','acceptance_groups':['overall','rain','dry_overcast'],
         'prior_full_overall_benchmark_rmse_db':0.28567316472555604,
         'bootstrap_replicates':4096,'bootstrap_seed':20261011,'test_used_for_fit':False,'historical_test_access':True,
         'new_study_exploratory':True,'package_versions':v.versions()}
    if create:
        train=v.data_frame('train',obj,events=True)
        require(len(obj['base_names'])==50 and len(obj['retained_names'])==156,'Input counts')
        obj['training_weather_counts']={g:int(len(train) if g=='overall' else train[g].sum()) for g in v.WEATHER}
        obj['train_weight_sha256']={policy:fp(e.weights(train,policy).tolist()) for policy in ('uniform','balanced')}
        obj['weight_summary']={policy:{'minimum':float(e.weights(train,policy).min()),'maximum':float(e.weights(train,policy).max()),
                                        'mean':float(e.weights(train,policy).mean())} for policy in ('uniform','balanced')}
    path=OUT/'run_manifest.json'
    if path.exists():
        old=read(path)
        for k in ('training_weather_counts','train_weight_sha256','weight_summary'):obj[k]=old[k]
        obj['fingerprint']=fp(obj)
        require(obj['fingerprint']==old['fingerprint'],'Frozen source/config fingerprint changed')
        return old
    require(create,'prepare required')
    obj['fingerprint']=fp(obj);obj['created_utc']=now();atomic(path,obj)
    return obj

def preprocess(train,validation,m,name,case):
    cm=case_plan(m,case)
    xt,xv=v.raw(train,cm),v.raw(validation,cm)
    if name in v.STRUCTURED and case=='base':
        positions=list(range(188));imputer=v.SimpleImputer(strategy='median',keep_empty_features=True)
        xf=imputer.fit_transform(xt[:,positions]);scaler=v.StandardScaler();xf=scaler.fit_transform(xf)
        bundle={'positions':positions,'imputer':imputer,'scaler':scaler}
    else:xf,bundle=v.fit_preprocess(xt,name,cm)
    return xt,xv,xf,v.transform(bundle,xv),bundle

def path(folder,case,name,tag):return OUT/folder/f'{case}_{name}_{tag}.json'
def model_path(case,name,seed):return OUT/'models'/f'{case}_{name}_seed{seed}.joblib'

def search(m):
    require(not (OUT/'selection.json').exists(),'Search frozen; cannot reopen')
    require(not (OUT/'test').exists(),'Test already opened')
    train,val=v.data_frame('train',m,True),v.data_frame('validation',m,True)
    require(set(train.hour_bjt).isdisjoint(set(val.hour_bjt)),'Hour leakage')
    for case in CASES:
        for name in m['models']:
            xt,xv,xf,xh,bundle=preprocess(train,val,m,name,case)
            positions=tuple(range(10)) if case=='full' else ()
            for config_id,choice in enumerate(m['grids'][name]):
                target=path('trials',case,name,f'config{config_id}')
                if target.exists():require(read(target)['fingerprint']==m['fingerprint'],'Trial resume digest');continue
                fitted,history,tp,vp,meta=e.fit(name,choice,17,positions,xf,train,xh,val)
                row={'fingerprint':m['fingerprint'],'case':case,'model':name,'config_id':config_id,'choice':choice,
                     'feature_names':case_plan(m,case)['retained_names'],'fit_metadata':meta,'round_history':history,
                     'train_hour_bjt':train.hour_bjt.astype(str).tolist(),'train_prediction_db':tp.tolist(),
                     'validation_hour_bjt':val.hour_bjt.astype(str).tolist(),'validation_prediction_db':vp.tolist(),
                     'train_metrics':v.weather_metrics(train,tp),'validation_metrics':v.weather_metrics(val,vp),
                     'validation_objective_db':e.objective(val,vp),'train_weight_sha256':m['train_weight_sha256'][choice['weight_policy']],
                     'test_read_for_fit':False,'utc':now()}
                atomic(target,row)
                print(f'SEARCH {case} {name} config={config_id} score={row["validation_objective_db"]:.6f} selected={meta["selected_round"]} fitted={meta["trained_rounds"]}',flush=True)
                del fitted;gc.collect()

def freeze(m):
    require(not (OUT/'test').exists(),'Cannot change selection after test')
    if (OUT/'selection.json').exists():
        saved=read(OUT/'selection.json');require(saved['fingerprint']==m['fingerprint'],'Frozen plan changed');return saved
    val=v.data_frame('validation',m,True)
    plans={}
    for case in CASES:
        plans[case]={}
        for name in m['models']:
            rows=[read(path('trials',case,name,f'config{i}')) for i in range(8)]
            for i,row in enumerate(rows):
                require(row['fingerprint']==m['fingerprint'] and row['choice']==m['grids'][name][i] and
                        row['feature_names']==case_plan(m,case)['retained_names'] and not row['test_read_for_fit'], 'Trial identity')
                score=e.objective(val,row['validation_prediction_db'])
                require(abs(score-row['validation_objective_db'])<1e-12,'Validation score recomputation')
                best=min(row['round_history'],key=lambda h:(h['validation_objective_db'],h['round'] or 0))
                require(abs(best['validation_objective_db']-score)<1e-10 and best['round']==row['fit_metadata']['selected_round'],'Best-round mismatch')
            eligible=[r for r in rows if r['fit_metadata']['solver_converged']]
            require(eligible,f'No converged candidate {case}/{name}')
            best=min(eligible,key=lambda r:(r['validation_objective_db'],r['config_id']))
            plans[case][name]={'config_id':best['config_id'],'choice':best['choice'],'formal_rounds':best['fit_metadata']['selected_round'],
                              'validation_objective_db':best['validation_objective_db'],
                              'formal_round_reason':'selected by validation three-MSE objective; fixed before formal refits and test'}
    require(len(list((OUT/'trials').glob('*.json')))==160,'Trial count')
    selection={'fingerprint':m['fingerprint'],'plans':plans,'selection_test_reads':0,'search_fits':160,'utc':now()}
    atomic(OUT/'selection.json',selection)
    qa={'status':'PASS','fingerprint':m['fingerprint'],'candidate_records':160,'selection_sha256':sha(OUT/'selection.json'),'test_opened':False,'utc':now()}
    atomic(OUT/'selection_qa.json',qa)
    return selection

def selection(m):
    qa=read(OUT/'selection_qa.json');saved=read(OUT/'selection.json')
    require(qa['status']=='PASS' and qa['fingerprint']==saved['fingerprint']==m['fingerprint'] and qa['selection_sha256']==sha(OUT/'selection.json'),'Selection gate')
    return saved

def refit(m):
    require(not (OUT/'test').exists(),'No formal refit after test')
    plans=selection(m)['plans']
    train,val=v.data_frame('train',m,True),v.data_frame('validation',m,True)
    for case in CASES:
        for name in m['models']:
            xt,xv,xf,_,base_bundle=preprocess(train,val,m,name,case)
            plan=plans[case][name];positions=tuple(range(10)) if case=='full' else ()
            for seed in m['seeds']:
                target=path('training',case,name,f'seed{seed}')
                if target.exists():
                    row=read(target);require(row['fingerprint']==m['fingerprint'] and row['model_sha256']==sha(model_path(case,name,seed)),'Formal checkpoint mismatch');continue
                model,history,_,_,meta=e.fit(name,plan['choice'],seed,positions,xf,train,fixed_rounds=plan['formal_rounds'])
                bundle=dict(base_bundle,model=model,model_name=name,seed=seed,case=case,fingerprint=m['fingerprint'],
                            retained_names=case_plan(m,case)['retained_names'],choice=meta['effective_choice'])
                v.save_model(model_path(case,name,seed),bundle)
                tp,vp=v.predict(bundle,xt),v.predict(bundle,xv)
                replay=v.predict(joblib.load(model_path(case,name,seed)),xv)
                delta=float(np.max(np.abs(replay-vp)));require(delta<1e-10,'Saved model replay')
                row={'fingerprint':m['fingerprint'],'case':case,'model':name,'seed':seed,'feature_names':bundle['retained_names'],
                     'plan':plan,'fit_metadata':meta,'round_history':history,'model_sha256':sha(model_path(case,name,seed)),
                     'selection_sha256':sha(OUT/'selection.json'),'train_weight_sha256':m['train_weight_sha256'][plan['choice']['weight_policy']],
                     'train_hour_bjt':train.hour_bjt.astype(str).tolist(),'train_prediction_db':tp.tolist(),'train_metrics':v.weather_metrics(train,tp),
                     'validation_hour_bjt':val.hour_bjt.astype(str).tolist(),'validation_prediction_db':vp.tolist(),'validation_metrics':v.weather_metrics(val,vp),
                     'validation_changed_formal_fit':False,'test_read_for_fit':False,'reload_max_delta_db':delta,'utc':now()}
                atomic(target,row)
                print(f'REFIT {case} {name} seed={seed} rounds={plan["formal_rounds"]}',flush=True)
                del model,bundle;gc.collect()

def verify_train(m):
    frozen=selection(m)
    train,val=v.data_frame('train',m,True),v.data_frame('validation',m,True)
    require(set(train.hour_bjt).isdisjoint(set(val.hour_bjt)),'Exact-hour overlap')
    largest=0.
    for case in CASES:
        for name in m['models']:
            plan=frozen['plans'][case][name]
            for seed in m['seeds']:
                row=read(path('training',case,name,f'seed{seed}'))
                require(row['fingerprint']==m['fingerprint'] and row['plan']==plan and
                        row['model_sha256']==sha(model_path(case,name,seed)) and not row['test_read_for_fit'] and
                        not row['validation_changed_formal_fit'] and row['fit_metadata']['solver_converged'],'Formal record invalid')
                require(row['train_hour_bjt']==train.hour_bjt.astype(str).tolist() and row['validation_hour_bjt']==val.hour_bjt.astype(str).tolist(),'Hours differ')
                require(row['feature_names']==case_plan(m,case)['retained_names'] and all(h['validation_objective_db'] is None for h in row['round_history']),'Formal holdout selection')
                require(row['train_weight_sha256']==fp(e.weights(train,plan['choice']['weight_policy']).tolist()),'Weight source')
                require(e.round_key(name) is None or len(row['round_history'])==plan['formal_rounds'],'Formal round count')
                bundle=joblib.load(model_path(case,name,seed));pred=v.predict(bundle,v.raw(val,case_plan(m,case)))
                delta=float(np.max(np.abs(pred-np.asarray(row['validation_prediction_db']))));largest=max(largest,delta)
                require(delta<1e-10,'Replay failed')
    require(len(list((OUT/'training').glob('*.json')))==len(list((OUT/'models').glob('*.joblib')))==60,'Training/model count')
    qa={'status':'PASS','fingerprint':m['fingerprint'],'selection_sha256':sha(OUT/'selection.json'),'search_fits':160,'formal_fits':60,
        'models':60,'test_fit_reads':0,'replay_max_difference_db':largest,'utc':now()}
    atomic(OUT/'train_qa.json',qa);return qa

def test(m):
    require(read(OUT/'train_qa.json')['status']=='PASS','Training gate not PASS')
    verify_train(m)
    frame=v.data_frame('test',m,True)
    require([int(len(frame) if g=='overall' else frame[g].sum()) for g in v.WEATHER]==[687,322,51,158,156],'Test weather counts')
    for case in CASES:
        x=v.raw(frame,case_plan(m,case))
        for name in m['models']:
            for seed in m['seeds']:
                target=path('test',case,name,f'seed{seed}')
                if target.exists():require(read(target)['fingerprint']==m['fingerprint'],'Test checkpoint changed');continue
                training=read(path('training',case,name,f'seed{seed}'))
                bundle=joblib.load(model_path(case,name,seed));pred=v.predict(bundle,x)
                row={'fingerprint':m['fingerprint'],'case':case,'model':name,'seed':seed,'feature_names':case_plan(m,case)['retained_names'],
                     'plan':training['plan'],'model_sha256':training['model_sha256'],'fit_calls_during_evaluation':0,
                     'hour_bjt':frame.hour_bjt.astype(str).tolist(),'truth_db':frame[m['target']].to_numpy(float).tolist(),
                     'prediction_db':pred.tolist(),'weather_masks':{g:frame[g].astype(bool).tolist() for g in v.WEATHER if g!='overall'},
                     'metrics':v.weather_metrics(frame,pred),'utc':now()}
                atomic(target,row);print(f'TEST {case} {name} seed={seed} RMSE={row["metrics"]["overall"]["rmse_db"]:.6f}',flush=True)

def paired_rows(m,name):
    result=[]
    for seed in m['seeds']:
        base=read(path('test','base',name,f'seed{seed}'));full=read(path('test','full',name,f'seed{seed}'))
        require(base['hour_bjt']==full['hour_bjt'] and base['truth_db']==full['truth_db'],'Pair alignment')
        result.append({'hour_bjt':base['hour_bjt'],'truth_db':base['truth_db'],'weather_masks':base['weather_masks'],
                       'base_prediction_db':base['prediction_db'],'full_prediction_db':full['prediction_db'],
                       'base_metrics':base['metrics'],'full_metrics':full['metrics']})
    return result

def report(m):
    qa=read(OUT/'independent_qa.json');require(qa['status']=='PASS' and qa['fingerprint']==m['fingerprint'],'Independent gate')
    spec=importlib.util.spec_from_file_location('paired27',HERE.parent/'Revision_027/run_pair.py')
    pair=importlib.util.module_from_spec(spec);spec.loader.exec_module(pair)
    rows=[]
    for name in m['models']:
        items=paired_rows(m,name);groups={}
        for group in v.WEATHER:
            row={case:{metric:float(np.mean([r[f'{case}_metrics'][group][metric] for r in items])) for metric in ('rmse_db','mae_db','bias_db')} for case in CASES}
            row['n_hours']=items[0]['base_metrics'][group]['n_hours']
            delta=[r['base_metrics'][group]['rmse_db']-r['full_metrics'][group]['rmse_db'] for r in items]
            row.update(base_minus_full_rmse_db=float(np.mean(delta)),seed_deltas_db=dict(zip(map(str,m['seeds']),delta)),improved_mean=row['full']['rmse_db']<row['base']['rmse_db'])
            if group in m['acceptance_groups']:row['date_bootstrap']=pair.date_interval(items,group,m)
            groups[group]=row
        rows.append({'model':name,'model_label':v.MODEL_LABELS[name],'metrics':groups,
                     'all_three_mean_improvements':all(groups[g]['improved_mean'] for g in m['acceptance_groups'])})
    rows.sort(key=lambda r:(r['metrics']['overall']['full']['rmse_db'],r['metrics']['overall']['full']['mae_db'],r['model']))
    for i,r in enumerate(rows,1):r['full_overall_rmse_rank']=i
    winner=rows[0];guard=winner['metrics']['overall']['full']['rmse_db']<=m['prior_full_overall_benchmark_rmse_db']
    result={'stage_status':'COMPLETE','fingerprint':m['fingerprint'],'seeds':m['seeds'],'model_rows':rows,
            'full_overall_winner':winner['model'],'acceptance':'MET' if winner['all_three_mean_improvements'] and guard else 'NOT_MET',
            'all_three_mean_improvements':winner['all_three_mean_improvements'],'continuity_guard_pass':guard,
            'historical_test_access':True,'fresh_blind_test':False,'test_based_family_ranking':True,
            'search_fits':160,'new_formal_fits':60,'test_fit_calls':0,'independent_qa':qa,'utc':now()}
    atomic(OUT/'final_report.json',result)
    lines=['# Revision 028 对称调参、天气感知直接训练结果','','同一687测试小时，三个种子分别算指标再平均，非集成预测。',
           f'全层总体第一名：{winner["model_label"]}；均值三项改善与旧基准连续性验收：{result["acceptance"]}。','',
           '|全层排名|模型|总体50→156|雨天50→156|干燥阴天50→156|三项均值改善|','|---:|---|---|---|---|---|']
    for r in rows:
        cells=[f'{r["metrics"][g]["base"]["rmse_db"]:.6f} → {r["metrics"][g]["full"]["rmse_db"]:.6f}' for g in m['acceptance_groups']]
        lines.append(f'|{r["full_overall_rmse_rank"]}|{r["model_label"]}|{"|".join(cells)}|{"是" if r["all_three_mean_improvements"] else "否"}|')
    lines += ['','所有参数、实际训练终止、最佳轮、逐轮曲线、逐种子逐小时预测及五种天气指标完整保存。',
              '基础与全层分别使用相同8候选预算。方案由验证目标确定，最终测试用于排名，没有用测试回调配置。',
              '这是一项有历史测试访问的探索性实验；三种子均值不保证每个种子都改善。保留同日训练重叠、日期区间和负结果。']
    (OUT/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'status':'COMPLETE','winner':winner['model'],'acceptance':result['acceptance']},indent=2))

def synthetic_frame(n,seed):
    rng=np.random.default_rng(seed)
    return pd.DataFrame({'clear_sky_attenuation_db':rng.normal(size=n),'rain':np.arange(n)%4==0,'dry_overcast':np.arange(n)%4==1})

def self_test():
    frame=synthetic_frame(40,5);w=e.weights(frame,'balanced');pred=np.arange(40)/30
    require(abs(w.mean()-1)<1e-12 and w.min()>0,'Weights normalize')
    error=pred-frame.clear_sky_attenuation_db.to_numpy()
    require(abs(np.mean(w*error**2)-e.objective(frame,pred)**2)<1e-12,'Weighted objective equivalence')
    require(all(len(e.grids(name))==8 for name in v.c.MODEL_NAMES),'Symmetric8-grid')
    require(all([c['weight_policy'] for c in e.grids(name)]==['uniform']*4+['balanced']*4 for name in v.c.MODEL_NAMES),'Control/extension counts')
    require(all(e.round_key(n) is not None for n in v.c.MODEL_NAMES if n in v.c.NEURAL_MODELS),'Neural rounds')
    values=[{'score':.1,'round':3},{'score':.1,'round':1}]
    require(min(values,key=lambda r:(r['score'],r['round']))['round']==1,'Earliest tie')
    for kind in ('bilstm','transformer','gcn'):
        for positions,width in [((),188),(tuple(range(10)),348)]:
            net=e.GatedProfileNet(kind,positions,.1);net.eval()
            with v.torch.no_grad():result=net(v.torch.zeros(4,width))
            require(result.shape==(4,) and v.torch.isfinite(result).all(),'Gate tensor shape')
    require(np.sqrt(np.mean(np.asarray([[1.,-1.],[-1.,1.]])**2,axis=1)).mean()==1,'Seed metric meaning')
    obj={'status':'PASS','synthetic_checks':13,'research_samples_read':0,'runner_sha256':sha(HERE/'run_weather.py'),'engine_sha256':sha(HERE/'engine.py'),'utc':now()}
    atomic(OUT/'synthetic_qa.json',obj);print(json.dumps(obj,indent=2))

def preflight():
    rng=np.random.default_rng(19);train,val=synthetic_frame(64,1),synthetic_frame(16,2)
    records=[]
    for case in CASES:
        for name in v.c.MODEL_NAMES:
            choice=e.grids(name)[4];key=e.round_key(name)
            if key:choice['estimator'][key]=2
            width=(348 if case=='full' else 188) if name in v.STRUCTURED else (156 if case=='full' else 50)
            xf,xv=rng.normal(size=(64,width)),rng.normal(size=(16,width))
            model,history,tp,vp,meta=e.fit(name,choice,17,tuple(range(10)) if case=='full' else (),xf,train,xv,val)
            require(len(tp)==64 and len(vp)==16 and np.isfinite(tp).all() and np.isfinite(vp).all(),'Weighted API prediction')
            require(abs(meta['weight_mean']-1)<1e-12 and (meta['selected_round'] is not None if key else True),'API selection')
            records.append({'case':case,'model':name,'fit_metadata':meta,'round_history':history})
            del model;gc.collect()
    obj={'status':'PASS','synthetic_weighted_fits':20,'research_samples_read':0,'records':records,'utc':now()}
    atomic(OUT/'preflight_qa.json',obj);print(json.dumps({'status':'PASS','synthetic_weighted_fits':20,'research_samples_read':0}))

def status():
    return {'fingerprint':read(OUT/'run_manifest.json').get('fingerprint') if (OUT/'run_manifest.json').exists() else None,
            'counts':{n:len(list((OUT/n).glob('*.json' if n!='models' else '*.joblib'))) for n in ('trials','training','models','test')},
            'qa':{n:read(OUT/n)['status'] if (OUT/n).exists() else None for n in ('synthetic_qa.json','preflight_qa.json','selection_qa.json','train_qa.json','independent_qa.json','delivery_qa.json')},
            'active_worker':read(OUT/'active_worker.json') if (OUT/'active_worker.json').exists() else None,
            'report_complete':(OUT/'final_report.json').exists()}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=('self-test','preflight','prepare','search','freeze','refit','verify-train','test','report','status'))
    args=parser.parse_args()
    if args.command=='status':print(json.dumps(status(),indent=2));return
    with lock(args.command):
        if args.command=='self-test':self_test();return
        if args.command=='preflight':preflight();return
        m=manifest(create=args.command=='prepare')
        if args.command=='prepare':print(json.dumps({'status':'FROZEN','fingerprint':m['fingerprint']},indent=2))
        elif args.command=='search':search(m)
        elif args.command=='freeze':print(json.dumps(freeze(m),indent=2))
        elif args.command=='refit':refit(m)
        elif args.command=='verify-train':print(json.dumps(verify_train(m),indent=2))
        elif args.command=='test':test(m)
        else:report(m)

if __name__=='__main__':main()
