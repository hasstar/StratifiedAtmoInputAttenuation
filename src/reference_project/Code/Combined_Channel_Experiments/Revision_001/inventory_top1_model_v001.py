"""Read an existing k=1 checkpoint and its frozen preprocessing; never fit."""
from __future__ import annotations
import csv
import importlib.util
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
import weather_download_v002 as io


def inventory():
    folder = ROOT / 'Outputs/Experiments/Hourly_Attenuation_Revision_029_BiLSTM53_Layer_RFE'
    state_path = folder / 'rfe/states/k_0001.json'
    model_path = folder / 'rfe/models/k_0001.joblib'
    manifest_path = folder / 'run_manifest.json'
    rank_path = folder / 'rfe_survival_rank.csv'
    state, manifest = io.read(state_path), io.read(manifest_path)
    with rank_path.open(encoding='utf-8-sig', newline='') as f:
        rank = list(csv.DictReader(f))
    io.require(rank[0]['feature_name'] == 'p1__geopotential', 'RFE survivor changed')
    io.require(state['feature_names'] == ['p1__geopotential'] and state['n_features'] == 1,
               'Not the existing one-feature model')
    io.require(io.sha(model_path) == state['full_fit']['model_sha256'], 'Checkpoint SHA mismatch')
    spec = importlib.util.spec_from_file_location('top1_readonly_runtime',
        ROOT / 'Code/Hourly_Attenuation_Experiments/Revision_029/runtime.py')
    runtime = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runtime)
    bundle = runtime.joblib.load(model_path)
    io.require(bundle['retained_names'] == state['feature_names'], 'Bundle feature order differs')
    io.require(list(bundle['active_positions']) == [9], 'Unexpected pressure layout')
    io.require(bundle['model_name'] == 'bilstm' and bundle['seed'] == 17, 'Unexpected model')
    positions = [int(v) for v in bundle['positions']]
    imputer, scaler = bundle['imputer'], bundle['scaler']
    io.require(len(positions) == 204 and imputer.n_features_in_ == scaler.n_features_in_ == 204,
               'Native one-feature structural layout differs')
    io.require(runtime.np.isfinite(imputer.statistics_).all() and
               runtime.np.isfinite(scaler.mean_).all() and
               runtime.np.isfinite(scaler.scale_).all() and (scaler.scale_ > 0).all(),
               'Existing preprocessing invalid')
    model = bundle['model']
    io.require(model.target_mean == state['full_fit']['fit_metadata']['target_mean'] and
               model.target_scale == state['full_fit']['fit_metadata']['target_scale'],
               'Target transform differs')
    feature_slot = manifest['original_names'].index('p1__geopotential')
    used_slot = positions.index(feature_slot)
    sources = [state_path, model_path, manifest_path, rank_path,
        folder / 'evaluation_qa.json', folder / 'train_qa.json', folder / 'report_qa.json',
        ROOT / 'Code/Hourly_Attenuation_Experiments/Revision_029/runtime.py', Path(__file__)]
    for name in ('evaluation_qa.json', 'train_qa.json', 'report_qa.json'):
        io.require(io.read(folder / name)['status'] == 'PASS', 'Existing model QA not PASS')
    report = {
        'status': 'PASS_EXISTING_CHECKPOINT_AND_PREPROCESSING_ONLY', 'checked_utc': io.now(),
        'feature_names': state['feature_names'], 'feature_count': 1,
        'family': 'BiLSTM', 'seed': 17, 'epochs': 53,
        'checkpoint': str(model_path.relative_to(ROOT)), 'checkpoint_sha256': io.sha(model_path),
        'recorded_weights_sha256': state['full_fit']['model_weights_sha256'],
        'source_manifest_fingerprint': manifest['fingerprint'],
        'source_training': 'LOCAL Signal_Detectability_Revision_003, 2404 training hours; not GBS/OpenSat fitting',
        'target': manifest['target'], 'target_time_scale': 'hourly relative-clear-sky attenuation dB',
        'pressure_axis_not_time_sequence': True, 'new_fit_calls': 0,
        'preprocessing': {'imputer': type(imputer).__name__, 'scaler': type(scaler).__name__,
            'native_layout_slots': 204, 'positions': positions,
            'original_feature_slot': feature_slot, 'selected_layout_slot': used_slot,
            'feature_median': float(imputer.statistics_[used_slot]),
            'feature_mean': float(scaler.mean_[used_slot]),
            'feature_scale': float(scaler.scale_[used_slot]),
            'target_mean': float(model.target_mean), 'target_scale': float(model.target_scale),
            'note': 'Use this k=1 checkpoint native layout only; no padding of a 156-feature model; all parameters already fitted and frozen'},
        'external_target_units_time_reference_compatibility': 'PENDING; this inventory is not external-evaluation QA',
        'other_families': 'No compatible checkpoint established by this inventory; no replacement training authorized',
        'source_sha256': {str(p.relative_to(ROOT)): io.sha(p) for p in sources}}
    io.require(io.sha(model_path) == state['full_fit']['model_sha256'], 'Checkpoint mutated during read')
    dest = ROOT / 'Outputs/Experiments/Combined_Channel_Revision_001/external/weather_top1_v003'
    io.DEST = dest
    io.immutable(dest / 'model_inventory.json', report)
    print(json.dumps({k: report[k] for k in ('status', 'feature_names', 'checkpoint_sha256', 'new_fit_calls', 'preprocessing')}, indent=2))


if __name__ == '__main__':
    inventory()
