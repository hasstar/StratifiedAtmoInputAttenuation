"""Independent row/metric checks and direct forward replay, no evaluator import."""
import hashlib
import json
import math
import sys
import unittest
from pathlib import Path
from datetime import datetime, timezone

sys.dont_write_bytecode = True
import joblib
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT/'Outputs/Experiments/Combined_Channel_Revision_001/external/validation_gbs_48h_v010'
EVAL = BASE/'evaluation_002'
OUT = BASE/'evaluation_independent_qa_001'
MODEL = ROOT/'Outputs/Experiments/Hourly_Attenuation_Revision_029_BiLSTM53_Layer_RFE/rfe/models/k_0010.joblib'
MODEL_SHA = 'b9eb8b7aa0f64b330dd0e6a23ac605afdceab8b7e34613011228a5015054c9e5'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for x in iter(lambda:f.read(1024*1024),b''):h.update(x)
    return h.hexdigest()


def read(path):
    with Path(path).open(encoding='utf-8-sig') as f:return json.load(f)


def need(test, message):
    if not test: raise ValueError(message)


def calculate(errors):
    errors = [float(x) for x in errors]
    need(all(math.isfinite(x) for x in errors), 'Nonfinite error')
    n=len(errors)
    return {'rmse_db': math.sqrt(math.fsum(x*x for x in errors)/n) if n else None,
            'mae_db':math.fsum(abs(x) for x in errors)/n if n else None,
            'bias_db':math.fsum(errors)/n if n else None}


def same_keys(actual, expected):
    keys=list(actual)
    need(len(set(keys))==len(keys) and set(keys)==set(expected), 'Changed/duplicate cohort')


class NegativeControls(unittest.TestCase):
    def test_known_numbers(self):
        got=calculate([-2,0,1]);self.assertAlmostEqual(got['rmse_db'],math.sqrt(5/3));self.assertEqual(got['mae_db'],1);self.assertAlmostEqual(got['bias_db'],-1/3)
    def test_sign(self):self.assertEqual(calculate([2])['bias_db'],2)
    def test_nan(self):
        with self.assertRaises(ValueError):calculate([float('nan')])
    def test_inf(self):
        with self.assertRaises(ValueError):calculate([float('inf')])
    def test_empty(self):self.assertEqual(calculate([]),dict(rmse_db=None,mae_db=None,bias_db=None))
    def test_duplicate(self):
        with self.assertRaises(ValueError):same_keys([1,1],[1])
    def test_missing(self):
        with self.assertRaises(ValueError):same_keys([1],[1,2])
    def test_shift(self):
        with self.assertRaises(ValueError):same_keys([1,3],[1,2])
    def test_cohort(self):same_keys([2,1],[1,2])
    def test_longitude_equivalence(self):self.assertEqual((358.56298828125+180)%360-180,-1.43701171875)


def main():
    OUT.mkdir(exist_ok=False)
    before={}
    def guard(path, digest=None):
        p=Path(path); actual=sha(p)
        need(digest is None or actual==digest,'Changed source '+str(p))
        before[str(p.resolve())]=actual
        return p
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(NegativeControls)
    controls=unittest.TextTestRunner(verbosity=1).run(suite)
    need(controls.wasSuccessful(),'Independent negative controls failed')
    scope=read(guard(BASE/'scope_manifest.json','43864c01e313b768ca21a8cbda396cda7a9a3dfbab71975f524a80089333f052'))
    metrics=read(guard(EVAL/'hourly_metrics.json'))
    receipt=read(guard(EVAL/'inference_receipt.json'))
    gate=read(guard(EVAL/'compatibility_gate.json'))
    guard(ROOT/'Code/Combined_Channel_Experiments/Revision_001/evaluate_gbs_frozen_top10_v002.py',receipt['code_sha256'])
    guard(EVAL/'PROTOCOL.md',metrics['selection_protocol_sha256'])
    need(receipt['new_fit_calls']==0 and receipt['prediction_calls']==1 and receipt['forbidden_call_attempts']==[],'Not pure inference')
    need(gate['status']=='PASS_FOR_AUTHORIZED_APPROXIMATE_ZERO_SHOT_TRIAL','Compatibility gate not passed')
    for p,digest in read(guard(EVAL/'read_only_inputs.json')).items():guard(p,digest)
    pred=pd.read_parquet(guard(ROOT/metrics['predictions']['path'],metrics['predictions']['sha256']))
    expected={(s,pd.Timestamp(t,unit='s',tz='UTC')) for s in scope['stations'] for t in scope['hours_utc_epoch_seconds']}
    same_keys(zip(pred.station,pred.hour_utc),expected)
    need(len(pred)==1872 and np.isfinite(pred.prediction_db).all(),'Prediction scope/nonfinite')
    labels=pd.read_parquet(guard(ROOT/'Outputs/Datasets/ClearSky_Alignment_Revision_004/ceda_gbs_hourly_attenuation.parquet'))
    joined=pred.merge(labels,on=['hour_utc'],suffixes=('_pred','_frozen'))
    joined=joined[joined.station==joined.site]
    need(len(joined)==1872,'Independent label join count')
    a=joined.truth_db.to_numpy(float);b=joined.clear_sky_attenuation_db.to_numpy(float)
    need(np.array_equal(a,b,equal_nan=True),'Truth not exact frozen native labels')
    need(np.array_equal(joined.valid_seconds_pred,joined.valid_seconds_frozen),'Coverage source differs')
    primary=np.isfinite(pred.truth_db)&(pred.valid_seconds>=3420)
    finite=np.isfinite(pred.truth_db)
    need(np.array_equal(primary,pred.primary_score_eligible) and np.array_equal(finite,pred.finite_target_diagnostic_eligible),'Scoring eligibility differs')
    need(int(primary.sum())==1847 and int(finite.sum())==1856,'Scorable counts differ')
    recomputed={}
    for key,mask in [('primary_coverage_ge_95',primary),('all_finite_targets_diagnostic',finite)]:
        recomputed[key]={}
        for site,reported in metrics[key].items():
            chosen=pred[mask & (np.ones(len(pred),bool) if site.startswith('pooled') else pred.station.eq(site))]
            r=calculate([p-y for p,y in zip(chosen.prediction_db,chosen.truth_db)])
            r.update(n_station_hours=len(chosen),n_dates=len(set(chosen.hour_utc.dt.date)),n_stations=len(set(chosen.station)),n_unique_utc_hours=len(set(chosen.hour_utc)))
            for k,v in r.items():
                need((v==reported[k]) if isinstance(v,int) or v is None else abs(v-reported[k])<1e-12,'Metric mismatch '+site+'/'+k)
            recomputed[key][site]=r
    stage='independent_direct_network_forward'
    guard(MODEL,MODEL_SHA)
    sys.path.insert(0,str(ROOT/'Code/Hourly_Attenuation_Experiments/Revision_014'))
    import models
    def denied(*args,**kwargs):raise RuntimeError('Fit/backward forbidden in independent evaluation QA')
    models.TorchRegressor.fit=denied
    torch.Tensor.backward=denied
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    SimpleImputer.fit=SimpleImputer.fit_transform=denied
    StandardScaler.fit=StandardScaler.fit_transform=StandardScaler.partial_fit=denied
    bundle=joblib.load(MODEL)
    model=bundle['model']
    need(not model.net.training,'Network mode changed')
    weights={k:v.detach().clone() for k,v in model.net.state_dict().items()}
    table=read(guard(ROOT/scope['local_acquisition_table']['path'],scope['local_acquisition_table']['sha256']))
    manifest=read(guard(ROOT/'Outputs/Experiments/Hourly_Attenuation_Revision_029_BiLSTM53_Layer_RFE/run_manifest.json'))
    entries={(r['station'],pd.Timestamp(r['utc'])):r for r in table['rows']}
    slots={name:bundle['positions'].index(manifest['original_names'].index(name)) for name in gate['feature_names']}
    # Direct saved native-layout construction, no evaluator raw/transform/predict helper.
    x=np.repeat(np.asarray(bundle['imputer'].statistics_)[None,:],len(pred),axis=0)
    for n,r in enumerate(pred.itertuples(index=False)):
        row=entries[(r.station,r.hour_utc)]
        for name,slot in slots.items():x[n,slot]=row['values'][name]
    x=(x-bundle['scaler'].mean_)/bundle['scaler'].scale_
    torch.set_num_threads(2)
    replay=[]
    with torch.inference_mode():
        for start in range(0,len(x),127):
            y=model.net(torch.tensor(x[start:start+127],dtype=torch.float32)).numpy()
            replay.extend((y.astype(float)*model.target_scale+model.target_mean).tolist())
    delta=np.abs(np.asarray(replay)-pred.prediction_db.to_numpy())
    need(np.isfinite(delta).all() and delta.max()<1e-5,'Direct-network replay mismatch')
    need(all(torch.equal(v,model.net.state_dict()[k]) for k,v in weights.items()),'Replay mutated weights')
    for p,digest in before.items():need(sha(p)==digest,'Input modified during independent QA')
    result={'status':'PASS_HOURLY_EVALUATION_ONLY','completed_utc':datetime.now(timezone.utc).isoformat(),
            'negative_controls':controls.testsRun,'prediction_rows':1872,'primary_valid_hours':1847,'all_finite_target_hours':1856,
            'missing_native_label_hours':16,'lower_coverage_finite_hours':9,'recomputed_metrics':recomputed,
            'direct_network_replay_rows':1872,'replay_batch_size':127,'max_replay_abs_difference_db':float(delta.max()),
            'replay_parameter_update_calls':0,'new_fit_calls':0,'original_source_files_unchanged':True,
            'weather_group_qa':'NOT_ASSESSED','probabilistic_fast_component_qa':'NOT_ASSESSED',
            'broad_generalization_claim':False,'T012_complete':False,'verifier_sha256':sha(__file__),
            'source_sha256':before}
    with (OUT/'qa.json').open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps({k:result[k] for k in ('status','negative_controls','prediction_rows','primary_valid_hours','max_replay_abs_difference_db')},indent=2))


if __name__=='__main__':main()
