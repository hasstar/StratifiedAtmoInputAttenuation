"""Existing saved k=10 only: load/check, no fitting, inference or selection."""
from __future__ import annotations
import csv
import importlib.util
import json
import sys
from pathlib import Path
sys.dont_write_bytecode = True
import weather_download_v002 as io
HERE, ROOT = io.HERE, io.ROOT
DEST = io.EXTERNAL / 'weather_top10_los_v004'


def inventory():
    folder = ROOT / 'Outputs/Experiments/Hourly_Attenuation_Revision_029_BiLSTM53_Layer_RFE'
    state_path, model_path = folder/'rfe/states/k_0010.json', folder/'rfe/models/k_0010.joblib'
    manifest_path, rank_path = folder/'run_manifest.json', folder/'rfe_survival_rank.csv'
    state, manifest = io.read(state_path), io.read(manifest_path)
    with rank_path.open(encoding='utf-8-sig', newline='') as f: rank=list(csv.DictReader(f))
    names = [r['feature_name'] for r in rank[:10]]
    io.require(len(names)==len(set(names))==10 and set(names)==set(state['feature_names']), 'Natural Top10 mismatch')
    io.require(io.sha(model_path)==state['full_fit']['model_sha256']=='b9eb8b7aa0f64b330dd0e6a23ac605afdceab8b7e34613011228a5015054c9e5', 'Frozen k10 SHA mismatch')
    spec=importlib.util.spec_from_file_location('top10_readonly_runtime', ROOT/'Code/Hourly_Attenuation_Experiments/Revision_029/runtime.py')
    runtime=importlib.util.module_from_spec(spec); spec.loader.exec_module(runtime)
    bundle=runtime.joblib.load(model_path)
    io.require(bundle['retained_names']==state['feature_names'], 'Model feature order differs')
    io.require(list(bundle['active_positions'])==[1,3,4,8,9] and bundle['model_name']=='bilstm' and bundle['seed']==17, 'Existing layout differs')
    positions=[int(x) for x in bundle['positions']]
    imputer, scaler, model=bundle['imputer'], bundle['scaler'], bundle['model']
    io.require(len(positions)==imputer.n_features_in_==scaler.n_features_in_==204, 'Native structural layout differs')
    io.require(runtime.np.isfinite(imputer.statistics_).all() and runtime.np.isfinite(scaler.mean_).all()
               and runtime.np.isfinite(scaler.scale_).all() and (scaler.scale_>0).all(), 'Invalid frozen preprocessing')
    io.require(model.target_mean==state['full_fit']['fit_metadata']['target_mean'] and
               model.target_scale==state['full_fit']['fit_metadata']['target_scale'], 'Target transform differs')
    slots=[]
    for name in state['feature_names']:
        original_slot=manifest['original_names'].index(name); slot=positions.index(original_slot)
        slots.append({'name':name,'original_slot':original_slot,'native_slot':slot,
                      'imputer_statistic':float(imputer.statistics_[slot]),'scaler_mean':float(scaler.mean_[slot]),
                      'scaler_scale':float(scaler.scale_[slot])})
    sources=[state_path,model_path,manifest_path,rank_path,Path(__file__),ROOT/'Code/Hourly_Attenuation_Experiments/Revision_029/runtime.py']
    for name in ('evaluation_qa.json','train_qa.json','report_qa.json'):
        p=folder/name; io.require(io.read(p)['status']=='PASS','Existing k10 QA not PASS'); sources.append(p)
    report={'status':'PASS_EXISTING_CHECKPOINT_AND_PREPROCESSING_ONLY','checked_utc':io.now(),
            'feature_names':state['feature_names'],'natural_rank_top10':names,'feature_count':10,
            'checkpoint':str(model_path.relative_to(ROOT)),'checkpoint_sha256':io.sha(model_path),
            'recorded_weights_sha256':state['full_fit']['model_weights_sha256'],
            'family':'BiLSTM','seed':17,'epochs':53,'source_manifest_fingerprint':manifest['fingerprint'],
            'source_training':'LOCAL Signal_Detectability_Revision_003, 2404 training hours; not GBS or OpenSat training',
            'historical_test_access':True,'new_fit_calls':0,'predictions_executed':0,
            'target':manifest['target'],'target_time_scale':'hourly relative-clear-sky attenuation dB',
            'preprocessing':{'imputer':type(imputer).__name__,'scaler':type(scaler).__name__,
                'native_layout_slots':204,'positions':positions,'selected_slots':slots,
                'target_mean':float(model.target_mean),'target_scale':float(model.target_scale),
                'note':'Saved k=10 native layout; do not pad or alter a 156-input model.'},
            'external_feature_geometry_units_target_scale_compatibility':'PENDING full-field QA and target compatibility; not an external evaluation',
            'other_families':'Not established compatible by this inventory; no new fitting authorized',
            'source_sha256':{str(p.relative_to(ROOT)):io.sha(p) for p in sources}}
    io.require(io.sha(model_path)==state['full_fit']['model_sha256'],'Model changed during inventory')
    io.DEST=DEST;io.immutable(DEST/'model_inventory.json',report)
    print(json.dumps({k:report[k] for k in ('status','feature_names','checkpoint_sha256','new_fit_calls')},indent=2))


if __name__=='__main__': inventory()
