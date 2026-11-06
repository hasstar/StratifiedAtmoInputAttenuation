"""Read-only source verification, synthetic correctness tests and initial freeze."""
import ast, importlib.metadata, os, platform, sys, time
import runtime as r
import verify

def source_paths():
    paths=set(r.HERE.glob('*.py'))|set(r.HERE.glob('*.md'))
    for revision in ('014','026','028','029'):
        paths.update((r.HERE.parent/f'Revision_{revision}').glob('*.py'))
    paths.add(r.HERE.parent/'Revision_031/core.py')
    paths.update(r.c.VENDOR.glob('*.py')); paths.update(r.c.VENDOR.glob('*LICENSE*'))
    paths.add(r.ROOT/'Code/Combined_Channel_Experiments/Revision_002/run_comparison.py')
    paths.add(r.c.FEATURE_SOURCE)
    paths.update(r.DATA/x for x in ('train.parquet','validation.parquet','test.parquet','hourly_sample_status.parquet'))
    paths.update(r.P28/x for x in ('selection.json','run_manifest.json','independent_qa.json','delivery_qa.json'))
    paths.update(r.P29/x for x in ('run_manifest.json','train_qa.json','evaluation_qa.json','completion.json'))
    paths.update(r.P31/x for x in ('source_manifest.json','search_selection.json','search_qa.json','full156_qa.json','layer/states/1023.json','layer/qa/1023.json'))
    for p in (r.P29/'rfe').rglob('*'):
        if p.is_file() and p.suffix in ('.json','.joblib'): paths.add(p)
    for name in r.MODELS:
        if name in r.legacy.MODEL_NAMES:
            paths.update(r.P28/f'{part}/full_{name}_seed17.{ext}' for part,ext in (('models','joblib'),('training','json'),('test','json')))
    state=r.read(r.P31/'layer/states/1023.json')
    for role in [*state['folds'],state['full_fit']]:
        paths.add(r.P31/role['result_file']); paths.add(r.P31/role['model_file'])
    paths.update(r.ROOT/p for p in (
        'Materials/Original_Submission/English_2026_TVT.pdf',
        'Code/Combined_Channel_Experiments/Revision_011/PROTOCOL.md',
        'Outputs/Experiments/Combined_Channel_Revision_011/predictions_local.parquet',
        'Outputs/Experiments/Combined_Channel_Revision_011/final_verification.json',
        'Outputs/Experiments/Combined_Channel_Revision_011/fixed_reference_parameters.json',
        'Outputs/Experiments/Combined_Channel_Revision_027/excluded_hours.json'))
    return sorted(paths)

def synthetic():
    for path in r.HERE.glob('*.py'): ast.parse(path.read_text(encoding='utf-8'))
    cases=[([1,2,3],[1,2,3]),([1,2,3],[3,2,1]),([1,1],[2,2]),([2],[1]),([],[]),([1,2,3],[2,2,2])]
    for y,p in cases:
        a=r.metric(y,p); b=verify.scalar_metrics(y,p)
        for key,v in b.items(): verify.equal(v,a[key],'Synthetic metric '+key)
    r.require(r.metric([1,2,3],[3,2,1])['r2']==-3 and r.metric([1,2,3],[3,2,1])['pearson_r']==-1,'R2 not correlation square')
    names=['surface__surface_pressure','p700__temperature']
    records=[]
    for name in r.MODELS:
        r.torch.manual_seed(17)
        if name in r.NN:
            net=r.make_nn(name,names)
            width=204 if name in r.STRUCTURED else 2
            x=r.torch.randn(8,width); pred=net(x); loss=pred.square().mean(); loss.backward()
            r.require(r.torch.isfinite(pred).all() and any(p.grad is not None for p in net.parameters()),'Synthetic forward/backward '+name)
            # A single-feature surface-only network must remain valid without any pressure layer.
            single=r.make_nn(name,[names[0]]); out=single(r.torch.zeros(2,188 if name in r.STRUCTURED else 1))
            r.require(r.torch.isfinite(out).all(),'Synthetic k1 '+name)
        else:
            choice=r.config(name); cfg=dict(choice['choice']['estimator'])
            key=r.engine.round_key(name)
            if key: cfg[key]=2
            estimator=r.legacy.make_estimator(name,cfg,17,2,())
            x=r.np.arange(40,dtype=float).reshape(20,2)/40; y=x[:,0]*.2
            estimator.fit(x,y,sample_weight=r.np.ones(20))
            r.require(r.np.isfinite(estimator.predict(x)).all(),'Synthetic classical '+name)
        records.append(dict(model=name,forward_fit_predict='PASS',research_data_fit_calls=0))
    return dict(status='PASS',models=records,metric_edge_cases=len(cases),research_training_calls=0,timing_pilot=False)

def prepare():
    r.OUT.mkdir(parents=True,exist_ok=True)
    r.require(not (r.OUT/'manifest.json').exists(),'Manifest already frozen; do not prepare again')
    tests=synthetic(); data=r.frames(); names=r.key_names(); old=r.old_manifest()
    r.require(len(names)==156 and names==old['retained_names'],'Frozen156 identity')
    r.require(r.read(r.P29/'train_qa.json')['status']=='PASS','Prior RFE train QA')
    r.require(r.read(r.P29/'evaluation_qa.json')['status']=='PASS','Prior RFE scoring QA')
    r.require(r.read(r.P31/'search_qa.json')['status']=='PASS' and r.read(r.P31/'full156_qa.json')['status']=='PASS','TabM source QA')
    r.require(r.read(r.P28/'independent_qa.json')['status']=='PASS','Legacy source QA')
    folds=r.r29.folds(data['train']); prior=r.read(r.P29/'rfe/states/k_0156.json')
    for f,p in zip(folds,prior['folds']):
        r.require(f['fit_index_sha256']==p['fit_index_sha256'] and f['hold_index_sha256']==p['hold_index_sha256'],'Prior exact fold match')
    expected=[687,322,51,158,156]
    for g,n in zip(r.WEATHER,expected):
        actual=len(data['test']) if g=='overall' else int(data['test'][g].sum()); r.require(actual==n,'Weather population '+g)
    r.require(int(r.legal_mask(data['test']).sum())==684,'Physical common population')
    for name in r.MODELS:
        # Preprocessing identity tests use fitting inputs only, no learned regressor or timed training.
        for subset in (names,[names[0]],[next(x for x in names if x.startswith('p700__'))]):
            x,b=r.prep_fit(name,data['train'],subset)
            r.require(r.np.isfinite(x).all() and r.np.array_equal(x,r.transformed(b,data['train'])),'Preprocess round trip '+name)
    paths=source_paths(); r.require(all(p.exists() for p in paths),'Required source missing')
    hashes={r.rel(p):r.sha(p) for p in paths}
    versions={x:importlib.metadata.version(x) for x in ('numpy','pandas','torch','scikit-learn','catboost','joblib','psutil')}
    payload=dict(version='T014-multi-model-rfe-v1',models=list(r.MODELS),feature_names=names,original_names=old['original_names'],target=r.TARGET,
                 source_sha256=hashes,model_configs={n:r.config(n) for n in r.MODELS},model_seed=17,
                 folds=[{key:val for key,val in f.items() if key not in ('fit','hold')} for f in folds],
                 rfe_states_per_model=156,total_states=1560,fit_roles=6240,permutation_evaluations=367350,
                 permutation_namespace=r.PERM_NAMESPACE,environment=dict(python=sys.version,executable=sys.executable,platform=platform.platform(),packages=versions,cuda_available=r.torch.cuda.is_available(),threads=2),
                 no_weather_network=True,historical_test_access=True)
    payload['fingerprint']=r.fp(payload); payload['created_utc']=r.now()
    r.emit(r.OUT/'synthetic_qa.json',dict(tests,source_sha256={r.rel(p):r.sha(p) for p in r.HERE.glob('*.py')},utc=r.now()))
    r.emit(r.OUT/'manifest.json',payload)
    r.emit(r.OUT/'preflight_qa.json',dict(status='PASS',manifest=payload['fingerprint'],sources_hashed=len(hashes),folds=3,train_n=2404,validation_n=343,test_n=687,test_dates=data['test'].hour_bjt.dt.date.nunique(),weather_counts=expected,common_itu_n=684,research_fit_calls=0,utc=r.now()))
    r.emit(r.OUT/'status.json',dict(status='PREPARED_NOT_STARTED',utc=r.now()))
    print('PREPARED',payload['fingerprint'],'pinned',len(hashes),'sources')

if __name__=='__main__': prepare()
