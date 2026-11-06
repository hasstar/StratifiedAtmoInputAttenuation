"""Fixed-scope native-label audit, compatibility gate and zero-training inference.

Original sources read-only. Independent target audit uses native NetCDF bytes.
Only saved model classes are imported, not historical run/training modules.
"""
from __future__ import annotations
import csv
import hashlib
import json
import math
import re
import sys
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.dont_write_bytecode = True
import joblib
import netCDF4
import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / 'Outputs/Experiments/Combined_Channel_Revision_001/external/validation_gbs_48h_v010'
OUT = BASE / 'evaluation_001'
MODEL_DIR = ROOT / 'Outputs/Experiments/Hourly_Attenuation_Revision_029_BiLSTM53_Layer_RFE'
MODEL = MODEL_DIR / 'rfe/models/k_0010.joblib'
MODEL_SHA = 'b9eb8b7aa0f64b330dd0e6a23ac605afdceab8b7e34613011228a5015054c9e5'
GBS = ROOT / '外部样例数据CEDA/GBS_20p7GHz'
ALIGN = ROOT / 'Outputs/Datasets/ClearSky_Alignment_Revision_004'
LOCAL = ROOT / 'Outputs/Datasets/Signal_Detectability_Revision_003'
FEATURE_METADATA = Path(r'E:\0_Workflow_System\01_PhD\Data\era5_campaign_20250930_20260820\csv\feature_metadata.json')
MODEL_CLASSES = ROOT / 'Code/Hourly_Attenuation_Experiments/Revision_014/models.py'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    with Path(path).open(encoding='utf-8-sig') as f:
        return json.load(f)


def emit(name, value):
    p = OUT / name
    with p.open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
    return {'path': str(p.relative_to(ROOT)), 'sha256': sha(p)}


def weights_sha(model):
    h = hashlib.sha256()
    for name, value in sorted(model.net.state_dict().items()):
        h.update(name.encode())
        h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def preprocessing_sha(bundle):
    h = hashlib.sha256()
    for value in (bundle['imputer'].statistics_, bundle['scaler'].mean_, bundle['scaler'].scale_,
                  bundle['scaler'].var_, np.array(bundle['positions']),
                  np.array([bundle['model'].target_mean, bundle['model'].target_scale])):
        h.update(np.asarray(value).tobytes())
    return h.hexdigest()


def audit_native_labels(selected, scope, protect):
    manifest_path = protect(GBS / 'source_manifest.csv')
    with manifest_path.open(encoding='utf-8-sig', newline='') as f:
        manifest = {r['relative_path']: r for r in csv.DictReader(f)}
    links = {r['id']: r for r in scope['links']}
    checks = []
    maximum_delta = 0.
    for relative, rows in selected.groupby('source_file', sort=True):
        path = protect(GBS / relative)
        blob = path.read_bytes()
        source = manifest[relative]
        require(len(blob) == int(source['bytes']) and hashlib.md5(blob).hexdigest() == source['md5'], f'Native file differs from frozen source manifest: {relative}')
        require(len(rows) == 24 and rows.site.nunique() == 1 and source['site'] == rows.site.iloc[0], 'Native site/day identity')
        station = links[source['site']]
        day = pd.Timestamp(source['date_utc'], tz='UTC')
        with netCDF4.Dataset('native_gbs.nc', memory=blob) as ds:
            ds.set_auto_mask(False)
            def scalar(name, units):
                v = ds.variables[name]
                require(v.units == units, 'Native scalar units: ' + name)
                return float(v[...])
            require(abs(scalar('frequency', 'GHz') - 20.7) < 1e-4, 'Native carrier differs')
            require(scalar('latitude_receiver', 'degrees_north') == station['latitude_deg_n'], 'Native latitude differs')
            require(scalar('longitude_receiver', 'degrees_east') == station['longitude_deg_e'], 'Native longitude differs')
            require(scalar('receiver_altitude_amsl', 'm') == station['height_amsl_m'], 'Native antenna height differs')
            require(-scalar('satellite_position', 'degrees_west') == station['satellite_longitude_deg_e'], 'Native satellite longitude differs')
            signal, time = ds.variables['clearsky_attenuation'], ds.variables['time']
            require(signal.units == 'dB' and float(signal.integration_period) == 1.0, 'Native target unit/time scale')
            require('relative to clear sky' in signal.long_name.lower(), 'Native target definition')
            origin = re.match(r'seconds since (\d+)-(\d+)-(\d+) 0:0:0\.0+ \+0:00$', time.units)
            require(origin is not None and tuple(map(int, origin.groups())) == (day.year, day.month, day.day), 'Native UTC date origin')
            require('end of integration' in time.long_name.lower(), 'Native integration alignment')
            t = np.asarray(time[:], dtype=np.float64)
            y = np.asarray(signal[:], dtype=np.float64)
            require(t.ndim == y.ndim == 1 and t.shape == y.shape, 'Native arrays shape')
            valid = np.isfinite(y) & (y != signal._FillValue) & (y >= signal.valid_min) & (y <= signal.valid_max)
            valid &= np.isfinite(t) & (t > 0) & (t <= 86400)
            require(np.all(t[valid] == np.rint(t[valid])) and len(set(t[valid].tolist())) == int(valid.sum()), 'Native valid second timestamps not unique/integral')
            for r in rows.itertuples(index=False):
                hour = int((r.hour_utc - day).total_seconds() // 3600)
                require(0 <= hour < 24, 'Native hour outside source day')
                mask = valid & (t > hour * 3600) & (t <= (hour + 1) * 3600)
                values = y[mask]
                n = len(values)
                require(n <= 3600 and n == int(r.valid_seconds), 'Frozen valid-second count mismatch')
                require(abs(r.coverage_fraction - n/3600) < 1e-14 and bool(r.coverage_ge_95) == (n >= 3420), 'Frozen coverage rule differs')
                if n:
                    reproduced = float(-10 * np.log10(np.mean(np.exp(-values * math.log(10)/10))))
                    delta = abs(reproduced - r.clear_sky_attenuation_db)
                    require(math.isfinite(r.clear_sky_attenuation_db) and delta < 1e-10, 'Native label reproduction failed')
                    maximum_delta = max(maximum_delta, delta)
                    require(r.target_status == 'native_ceda_gbs', 'Native target status')
                else:
                    require(pd.isna(r.clear_sky_attenuation_db) and r.target_status == 'no_native_attenuation', 'Missing native target not preserved')
                require(r.clear_reference_method == 'source_native_clear_sky_attenuation', 'New reference not allowed')
                require(int(np.count_nonzero(values >= 3)) == r.seconds_ge_3db and int(np.count_nonzero(values >= 10)) == r.seconds_ge_10db, 'Native diagnostic count differs')
        checks.append({'source': str(path.relative_to(ROOT)), 'sha256': sha(path), 'md5_matches_original_manifest': True,
                       'site': source['site'], 'date': source['date_utc'], 'hours': 24,
                       'valid_seconds': int(rows.valid_seconds.sum())})
    require(len(checks) == 78, 'Native source file count')
    counts = {}
    for station, rows in selected.groupby('site'):
        finite = np.isfinite(rows.clear_sky_attenuation_db)
        counts[station] = {'weather_complete_hours': len(rows), 'finite_target_hours': int(finite.sum()),
                           'coverage_ge_95_finite_hours': int((finite & rows.coverage_ge_95).sum()),
                           'missing_target_hours': int((~finite).sum()),
                           'lower_coverage_finite_hours': int((finite & ~rows.coverage_ge_95).sum())}
    evidence = emit('native_label_source_evidence.json', checks)
    return {'status': 'PASS', 'native_files': 78, 'station_hours_checked': len(selected),
            'max_reproduced_label_difference_db': maximum_delta, 'source_evidence': evidence,
            'counts': counts, 'new_reference_fits': 0, 'labels_rewritten': False,
            'native_end_of_integration_alignment': 't in (3600h,3600(h+1)], t=0 excluded',
            'aggregation': '-10 log10(mean linear transmittance), preserved source native clear-sky reference'}


def metrics(frame, mask):
    x = frame[mask]
    error = x.prediction_db.to_numpy(float) - x.truth_db.to_numpy(float)
    require(np.isfinite(error).all(), 'Nonfinite scoring error')
    return {'n_station_hours': len(x), 'n_dates': int(x.hour_utc.dt.date.nunique()),
            'n_stations': int(x.station.nunique()), 'n_unique_utc_hours': int(x.hour_utc.nunique()),
            'rmse_db': None if not len(x) else float(np.sqrt(np.mean(error*error))),
            'mae_db': None if not len(x) else float(np.mean(np.abs(error))),
            'bias_db': None if not len(x) else float(np.mean(error))}


def main():
    require(OUT.exists() and not (OUT/'hourly_metrics.json').exists() and not (OUT/'failure.json').exists(), 'Output missing or already frozen')
    guarded = {}
    forbidden = []
    stage = 'input_gates'
    def protect(path, expected=None):
        p = Path(path).resolve()
        digest = sha(p)
        require(expected is None or digest == expected, 'Input SHA mismatch: ' + str(p))
        guarded[str(p)] = digest
        return p
    def deny(*args, **kwargs):
        forbidden.append('FORBIDDEN_TRAINING_CALL')
        raise RuntimeError('Training/refitting/optimizer/backward disabled in external inference')
    try:
        protect(OUT/'PROTOCOL.md')
        scope = read(protect(BASE/'scope_manifest.json', '43864c01e313b768ca21a8cbda396cda7a9a3dfbab71975f524a80089333f052'))
        qa = read(protect(BASE/'field_audit_001/actual_field_qa.json'))
        require(qa['status'] == 'PASS' and qa['scope_sha256'] == sha(BASE/'scope_manifest.json'), 'Independent field gate')
        protect(ROOT/'Code/Combined_Channel_Experiments/Revision_001/audit_gbs_48h_actual_fields_v001.py', qa['audit_code_sha256'])
        source_sha = read(protect(ROOT/qa['read_only_sha_evidence']['path'], qa['read_only_sha_evidence']['sha256']))
        for p, digest in source_sha.items():
            protect(ROOT/p, digest)
        controls = read(protect(BASE/'field_audit_001/negative_controls.json'))
        require(controls['status'] == 'PASS' and controls['tests'] == 24 and controls['audit_code_sha256'] == qa['audit_code_sha256'], 'Independent audit controls gate')
        table = read(protect(ROOT/scope['local_acquisition_table']['path'], scope['local_acquisition_table']['sha256']))
        names = scope['feature_names']
        alignment_manifest = read(protect(ALIGN/'build_manifest.json'))
        labels_path = protect(ALIGN/'ceda_gbs_hourly_attenuation.parquet', alignment_manifest['output_sha256']['ceda_gbs_hourly_attenuation.parquet'])
        labels = pd.read_parquet(labels_path)
        require(not labels.duplicated(['site', 'hour_utc']).any(), 'Duplicate frozen labels')
        labels = labels[labels.site.isin(scope['stations']) & (labels.hour_utc.astype('int64') // 10**9).isin(scope['hours_utc_epoch_seconds'])].copy()
        require(len(labels) == 1872, 'Frozen target scope count')
        stage = 'native_label_audit'
        label_qa = audit_native_labels(labels, scope, protect)
        label_ref = emit('native_label_qa.json', label_qa)
        stage = 'compatibility'
        inventory = read(protect(ROOT/'Outputs/Experiments/Combined_Channel_Revision_001/external/weather_top10_los_v004/model_inventory.json'))
        for p, digest in inventory['source_sha256'].items():
            protect(ROOT/p, digest)
        state = read(protect(MODEL_DIR/'rfe/states/k_0010.json'))
        manifest = read(protect(MODEL_DIR/'run_manifest.json'))
        data_manifest = read(protect(LOCAL/'manifest.json'))
        require(data_manifest['target'] == manifest['target'] == 'clear_sky_attenuation_db', 'Target identity differs')
        require(data_manifest['clear_reference_dbm'] == -76.19362933508154, 'Frozen LOCAL reference differs')
        require(manifest['n_train'] == 2404 and manifest['model_seed'] == 17 and manifest['epochs'] == 53, 'Model source provenance differs')
        metadata = read(protect(FEATURE_METADATA))
        require('bilinear' in metadata['pressure_interpolation'] and 'intersection' in metadata['pressure_interpolation'], 'Original feature geometry identity')
        expected_units = {'geopotential':'m**2 s**-2', 'u_component_of_wind':'m s**-1', 'ozone_mass_mixing_ratio':'kg kg**-1',
                          'relative_humidity':'%', 'convective_available_potential_energy':'J kg**-1',
                          'maximum_2m_temperature_since_previous_post_processing':'K'}
        units = {}
        for name in names:
            prefix, field = name.split('__')
            source = metadata['surface_features' if prefix == 'surface' else 'pressure_features'][field]
            require(source['units'] == expected_units[field], 'Original model feature units differ')
            units[name] = source['units']
        protect(MODEL, MODEL_SHA)
        protect(MODEL_CLASSES)
        sys.path.insert(0, str(MODEL_CLASSES.parent))
        import models
        require(Path(models.__file__).resolve() == MODEL_CLASSES.resolve(), 'Incorrect saved model class module')
        models.TorchRegressor.fit = deny
        SimpleImputer.fit = SimpleImputer.fit_transform = deny
        StandardScaler.fit = StandardScaler.partial_fit = StandardScaler.fit_transform = deny
        torch.Tensor.backward = deny
        torch.optim.AdamW.step = torch.optim.Adam.step = torch.optim.SGD.step = deny
        bundle = joblib.load(MODEL)
        model = bundle['model']
        require(bundle['retained_names'] == names == state['feature_names'] and bundle['kind'] == 'rfe' and bundle['key'] == 10 and bundle['fold'] == 0, 'Wrong model/feature set')
        require(bundle['fingerprint'] == manifest['fingerprint'] and bundle['model_name'] == 'bilstm' and bundle['seed'] == 17, 'Bundle identity differs')
        require(len(bundle['positions']) == model.n_features == 268 and tuple(bundle['active_positions']) == (1,3,4,8,9), 'Native model layout differs')
        require(not model.net.training, 'Saved network not in eval mode')
        require(model.target_mean == state['full_fit']['fit_metadata']['target_mean'] and model.target_scale == state['full_fit']['fit_metadata']['target_scale'], 'Target transform changed')
        weight_before, pre_before = weights_sha(model), preprocessing_sha(bundle)
        require(weight_before == state['full_fit']['model_weights_sha256'], 'Saved weights digest differs')
        rows = table['rows']
        raw = np.full((len(rows), len(manifest['original_names'])), np.nan, dtype=float)
        native_slots = []
        for name in names:
            position = manifest['original_names'].index(name)
            raw[:, position] = [r['values'][name] for r in rows]
            native_slots.append(bundle['positions'].index(position))
        native = raw[:, bundle['positions']]
        require(np.isfinite(native[:, native_slots]).all(), 'Missing actual selected feature')
        inactive = [i for i in range(268) if i not in native_slots]
        require(np.isnan(native[:, inactive]).all(), 'Unexpected extra variable model input')
        imputed = bundle['imputer'].transform(native)
        scaled = bundle['scaler'].transform(imputed)
        require(scaled.shape == (1872, 268) and np.isfinite(scaled).all(), 'Frozen transformation failure')
        require(np.all(imputed[:, native_slots] == native[:, native_slots]), 'Selected measurements were imputed or modified')
        require(np.all(scaled[:, inactive] == 0), 'Inactive trained structural slots differ from saved zero layout')
        compatibility = {'status': 'PASS_FOR_AUTHORIZED_APPROXIMATE_ZERO_SHOT_TRIAL',
                         'scope_sha256': sha(BASE/'scope_manifest.json'), 'field_qa_sha256': sha(BASE/'field_audit_001/actual_field_qa.json'),
                         'native_label_qa': label_ref, 'checkpoint_sha256': MODEL_SHA, 'weights_sha256': weight_before,
                         'preprocessing_sha256': pre_before, 'model': 'Rev029 k_0010 BiLSTM', 'seed':17, 'epochs':53,
                         'actual_fit_source': 'LOCAL Signal_Detectability_Revision_003, 2404 hours; no GBS fitting',
                         'feature_names': names, 'units': units, 'native_layout_slots':268, 'native_selected_slots': native_slots,
                         'selected_feature_missing_values':0, 'inactive_slots':258,
                         'native_layout_explanation': 'Original k=10 design masks all unselected variables before saved imputer/scaler. No 156-feature model is padded or changed.',
                         'target_name':'clear_sky_attenuation_db', 'target_units':'dB', 'target_scale':'hourly linear-transmittance-relative-clear-sky attenuation',
                         'target_mean':model.target_mean, 'target_scale_transform':model.target_scale,
                         'local_frozen_reference_dbm':data_manifest['clear_reference_dbm'], 'gbs_reference':'source-native, unchanged',
                         'gbs_frequency_ghz':20.7,
                         'domain_shift_caveats':['fixed four heights + nearest LOS grid versus original hourly isobar bilinear sampling',
                                                'different site/climate/instrument/frequency domain; no physical band conversion',
                                                'LOCAL empirical training reference versus GBS native clear-sky reference; no bias/scale recalibration'],
                         'weather_groups':'NOT_ESTABLISHED; attenuation thresholds are not rainfall diagnostics',
                         'fast_component':'NOT_ESTABLISHED in this hourly trial',
                         'other_model_families':'No compatible natural-Top10 checkpoint established here; no retraining',
                         'prediction_calls_before_gate':0, 'new_fit_calls':0,
                         'frozen_scaler_diagnostic':{name:{'min_z':float(scaled[:,j].min()),'max_z':float(scaled[:,j].max()),
                                                         'abs_z_gt5_count':int((np.abs(scaled[:,j])>5).sum())} for name,j in zip(names,native_slots)}}
        emit('compatibility_gate.json', compatibility)
        stage = 'frozen_inference'
        torch.set_num_threads(2)
        with torch.inference_mode():
            prediction = model.predict(scaled)
        require(prediction.shape == (1872,) and np.isfinite(prediction).all(), 'Invalid model predictions')
        require(weights_sha(model) == weight_before and preprocessing_sha(bundle) == pre_before and not forbidden, 'Inference mutated model/preprocessing or attempted training')
        result = pd.DataFrame({'station':[r['station'] for r in rows], 'hour_utc':pd.to_datetime([r['utc'] for r in rows],utc=True),
                               'prediction_db':prediction})
        truth = labels.rename(columns={'site':'station','clear_sky_attenuation_db':'truth_db'})
        result = result.merge(truth[['station','hour_utc','truth_db','valid_seconds','coverage_fraction','coverage_ge_95','target_status','source_file']],
                              on=['station','hour_utc'],how='left',validate='one_to_one')
        result['primary_score_eligible'] = np.isfinite(result.truth_db) & result.coverage_ge_95
        result['finite_target_diagnostic_eligible'] = np.isfinite(result.truth_db)
        require(len(result)==1872, 'Output scope differs')
        metric_report = {'model':'Rev029 k_0010 BiLSTM', 'checkpoint_sha256':MODEL_SHA, 'dataset':'GBS England', 'frequency_ghz':20.7,
                         'bias_sign':'prediction minus truth', 'selection_protocol_sha256':sha(OUT/'PROTOCOL.md'),
                         'primary_coverage_ge_95':{}, 'all_finite_targets_diagnostic':{}}
        for station in ['pooled_same_source_same_frequency'] + scope['stations']:
            m = np.ones(len(result),bool) if station.startswith('pooled') else result.station.eq(station)
            metric_report['primary_coverage_ge_95'][station] = metrics(result,m & result.primary_score_eligible)
            metric_report['all_finite_targets_diagnostic'][station] = metrics(result,m & result.finite_target_diagnostic_eligible)
        for p,digest in guarded.items():
            require(sha(p)==digest, 'Read-only input changed: ' + p)
        output_table = OUT/'hourly_predictions.parquet'
        require(not output_table.exists(), 'Immutable prediction output exists')
        result.to_parquet(output_table,index=False)
        metric_report['predictions'] = {'path':str(output_table.relative_to(ROOT)), 'sha256':sha(output_table)}
        metric_report['new_fit_calls'] = 0
        metric_report['prediction_calls'] = 1
        metric_report['prediction_rows'] = 1872
        metric_report['independent_evaluation_qa'] = 'PENDING'
        metric_report['T012_complete'] = False
        emit('hourly_metrics.json',metric_report)
        emit('read_only_inputs.json',guarded)
        emit('inference_receipt.json',{'status':'PASS_PURE_INFERENCE_PENDING_INDEPENDENT_EVALUATION_QA',
                                     'completed_utc':datetime.now(timezone.utc).isoformat(), 'code_sha256':sha(__file__),
                                     'checkpoint_sha256_before_after':MODEL_SHA, 'weights_sha256_before_after':weight_before,
                                     'preprocessing_sha256_before_after':pre_before, 'new_fit_calls':0,'forbidden_call_attempts':forbidden,
                                     'prediction_calls':1,'prediction_rows':1872,'new_downloads':0,
                                     'labels_rewritten':False,'new_clear_sky_reference_fits':0,'new_cdf_fits':0})
        print(json.dumps(metric_report['primary_coverage_ge_95'],ensure_ascii=False,indent=2))
    except Exception as e:
        emit('failure.json',{'status':'FAIL','stage':stage,'error':repr(e),'traceback':traceback.format_exc(),
                             'forbidden_call_attempts':forbidden,'inputs_hashed':guarded})
        raise


if __name__ == '__main__':
    main()
