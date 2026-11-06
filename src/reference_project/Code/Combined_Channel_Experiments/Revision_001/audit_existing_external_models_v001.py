"""Metadata-only existing-checkpoint input gate; no model/network imports or calls."""
from __future__ import annotations
import hashlib
import json
import re
import subprocess
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / 'Outputs/Experiments'
BASE = EXP / 'Combined_Channel_Revision_001/external/validation_gbs_48h_v010'
OUT = BASE / 'model_compatibility_001'
SOURCES = {}
CACHE = {}
SCHEMAS = {}
RFE = {}


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def rel(path):
    return path.relative_to(ROOT).as_posix()


def read(path, cache=False):
    key = rel(path)
    if cache and key in CACHE:
        return CACHE[key]
    b = path.read_bytes()
    digest = hashlib.sha256(b).hexdigest()
    if key in SOURCES:
        assert SOURCES[key] == digest, ('Input changed during audit', key)
    SOURCES[key] = digest
    obj = json.loads(b.decode('utf-8-sig'))
    if cache:
        CACHE[key] = obj
    return obj


def immutable(path, obj):
    with path.open('x', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def gate(names, expected, role='formal'):
    if role != 'formal':
        return 'EXCLUDED_' + role.upper()
    if len(names) != len(set(names)):
        return 'INVALID_DUPLICATE_FEATURE_NAMES'
    if len(names) == len(expected) and set(names) == set(expected):
        return 'EXACT_TOP10_INPUT_ONLY'
    if names and set(names) < set(expected):
        return 'SUBSET_DIFFERENT_DESIGN_NOT_AUTHORIZED'
    return 'INCOMPATIBLE_INPUT_FEATURES'


def schema(names, expected):
    key = hashlib.sha256(json.dumps(names, ensure_ascii=False).encode()).hexdigest()
    SCHEMAS.setdefault(key, {'feature_names': names,
                            'missing_from_current_top10': sorted(set(names) - set(expected)),
                            'unused_current_top10': sorted(set(expected) - set(names)),
                            'feature_count': len(names)})
    return key


def rfe_state(folder, path):
    rev = int(re.search(r'Revision_(\d+)', folder.name).group(1))
    k = int(re.search(r'k_(\d+)', path.name).group(1))
    key = (rev, k)
    if key in RFE:
        return RFE[key]
    statepath = folder / ('states' if rev in (21, 24) else 'rfe/states') / (
        f'k_{k:04d}.json' if rev == 29 else f'k_{k:03d}.json')
    d = read(statepath)
    names = d.get('feature_names', d.get('retained_feature_names'))
    assert names and len(names) == len(set(names)) == k, statepath
    saved = d.get('model_sha256')
    if rev == 29:
        saved = d['full_fit']['model_sha256']
    item = {'names': names, 'source': rel(statepath), 'saved_sha': saved,
            'by_seed': d.get('model_sha256_by_seed', {}), 'k': k}
    RFE[key] = item
    return item


def classify(path, expected):
    relative = path.relative_to(EXP)
    folder = EXP / relative.parts[0]
    match = re.search(r'^Hourly_Attenuation_Revision_(\d+)', folder.name)
    row = {'path': rel(path), 'bytes': path.stat().st_size,
           'mtime_ns': path.stat().st_mtime_ns, 'role': 'formal'}
    if not match:
        return dict(row, status='UNRESOLVED_LAYOUT')
    rev = int(match.group(1))
    row.update(revision=rev, study=folder.name)
    if 'synthetic_api_' in path.as_posix():
        return dict(row, family='bilstm', role='synthetic', status='EXCLUDED_SYNTHETIC')
    if '/fold_models/' in path.as_posix():
        row['role'] = 'fold'
    seed = re.search(r'_seed(\d+)', path.stem)
    if seed:
        row['seed'] = int(seed.group(1))
    if rev in (15, 16):
        d = read(path)
        if 'learner' not in d:
            return dict(row, status='SIDECAR_NOT_CHECKPOINT')
        count = int(d['learner']['learner_model_param']['num_feature'])
        row.update(family='xgboost', feature_count=count,
                   feature_evidence=rel(path), semantic_names_not_embedded=True,
                   status='INCOMPATIBLE_INPUT_DIMENSION' if count != 10 else 'UNRESOLVED_FEATURE_IDENTITY',
                   checkpoint_sha256=SOURCES[rel(path)])
        return row
    manifest = read(folder / 'run_manifest.json', cache=True)
    names, source, saved = None, rel(folder / 'run_manifest.json'), None
    if rev == 14:
        row['family'] = 'xgboost'
        sidecar = path.with_suffix('.json')
        d = read(sidecar)
        names = d['features']
        saved = d['model_sha256']
        source = rel(sidecar)
        if '/residual/' in path.as_posix():
            base = read(folder / 'layers/heldout_all_masks/models/base188.json', cache=True)
            names = base['features'] + names
            row['requires_frozen_base_predictor'] = True
            row['note'] = 'Semantic inputs for complete base-plus-residual prediction, not residual alone.'
    elif rev == 19:
        row['family'] = path.stem.split('_seed')[0]
        names = manifest['retained_features']
        record = read(folder / 'test_models' / (path.stem + '.json'))
        saved = record['fitted_model_sha256']
        row['note'] = 'Required retained semantic inputs; structured models additionally have native masked layout.'
    elif rev in (21, 24) or (rev in (25, 29) and '/rfe/' in path.as_posix()):
        item = rfe_state(folder, path)
        names, source, saved = item['names'], item['source'], item['saved_sha']
        row['k'] = item['k']
        row['family'] = {21: 'xgboost', 24: 'svr_rbf', 25: 'bilstm', 29: 'bilstm'}[rev]
        if rev == 21:
            saved = item['by_seed'].get(str(row['seed']))
        if row['role'] == 'fold':
            saved = None  # full-fit state hash must not be attributed to a fold file
    elif rev in (23, 25, 29):
        row['family'] = 'svr_rbf' if rev == 23 else 'bilstm'
        mask = int(re.search(r'mask_(\d+)', path.stem).group(1))
        if rev == 23:
            lm = read(folder / 'layer_manifest.json', cache=True)
            allnames, base_names, levels = lm['all_names'], lm['base_names'], lm['levels_hpa']
            source = rel(folder / 'layer_manifest.json')
        else:
            allnames = manifest['retained_names']
            base_names = [n for n in allnames if n.startswith('surface__')]
            levels = manifest['levels_hpa']
        selected = {int(level) for i, level in enumerate(levels) if mask & (1 << i)}
        names = [n for n in allnames if n in base_names or
                 (n.startswith('p') and int(n.split('__')[0][1:]) in selected)]
        row['semantic_names_reconstructed_from_frozen_layer_manifest'] = True
        row['mask'] = mask
    elif rev in (26, 27, 28):
        recordpath = folder / 'training' / (path.stem + '.json')
        record = read(recordpath)
        names = record['feature_names']
        source = rel(recordpath)
        saved = record.get('model_sha256')
        row['family'] = record.get('model', path.stem.split('_seed')[0].removeprefix('base_').removeprefix('full_'))
        row['case'] = record.get('case', 'full' if rev == 26 else 'base')
    else:
        return dict(row, status='UNRESOLVED_LAYOUT')
    assert names and isinstance(names, list) and all(isinstance(n, str) for n in names)
    row.update(feature_schema=schema(names, expected), feature_count=len(names),
               missing_feature_count=len(set(names) - set(expected)), feature_evidence=source,
               status=gate(names, expected, row['role']), recorded_checkpoint_sha256=saved,
               target_recorded=manifest.get('target', 'NOT_CHECKED_AFTER_FAILED_FEATURE_GATE'))
    # Actual hash binding for all formal ten-column candidates and latest comparison models.
    if row['role'] == 'formal' and (len(names) == 10 or rev in (26, 27, 28)):
        digest = sha(path)
        row['checkpoint_sha256'] = digest
        SOURCES[rel(path)] = digest
        assert saved == digest, ('Checkpoint hash mismatch', path, saved, digest)
        row['recorded_hash_matches_actual'] = True
    return row


def main():
    assert OUT.exists() and not (OUT / 'inventory.json').exists(), 'Immutable destination already used'
    state = read(EXP / 'Hourly_Attenuation_Revision_029_BiLSTM53_Layer_RFE/rfe/states/k_0010.json', cache=True)
    expected = state['feature_names']
    assert len(expected) == len(set(expected)) == 10
    read(BASE / 'scope_manifest.json')
    read(BASE / 'handoff_013.json')
    extensions = ('.joblib', '.pkl', '.pt', '.pth', '.cbm', '.ubj', '.onnx',
                  '.bin', '.model', '.npz', '.h5', '.keras', '.safetensors', '.ckpt')
    args = ['rg', '--files', str(EXP)]
    for ext in (*extensions, '.json'):
        args += ['-g', '*' + ext]
    listed = sorted(Path(p) for p in subprocess.check_output(args, text=True).splitlines())
    # NPZ etc. also store evaluation/data arrays; only inspect such formats in
    # explicitly model-labelled directories, while traditional checkpoint files
    # are inventoried regardless of their directory name.
    primary = ('.joblib', '.pkl', '.pt', '.pth', '.cbm', '.ubj', '.onnx')
    model_dirs = ('models', 'fitted_models', 'layer_models', 'fold_models', 'checkpoints', 'weights')
    paths = [p for p in listed if p.suffix in primary or
             (p.suffix in extensions and any(s in model_dirs for s in p.relative_to(EXP).parts[:-1]))]
    json_candidates = [p for p in listed if p.suffix == '.json' and
                       any(s in ('models', 'fitted_models', 'layer_models', 'checkpoints', 'weights')
                           for s in p.relative_to(EXP).parts[:-1])]
    json_model_paths, sidecars = [], []
    for p in json_candidates:
        # JSONs under model directories include XGBoost serialized trees and sidecar receipts.
        d = read(p)
        if 'learner' in d and 'learner_model_param' in d['learner']:
            json_model_paths.append(p)
        else:
            sidecars.append(rel(p))
    paths = sorted(set(paths + json_model_paths))
    rows = [classify(p, expected) for p in paths]
    unresolved = [r for r in rows if r['status'].startswith('UNRESOLVED')]
    assert not unresolved, unresolved
    controls = {
        'same_names_same_order': gate(expected, expected) == 'EXACT_TOP10_INPUT_ONLY',
        'same_names_reordered': gate(expected[::-1], expected) == 'EXACT_TOP10_INPUT_ONLY',
        'same_dimension_different_field': gate(expected[:-1] + ['not_requested'], expected) == 'INCOMPATIBLE_INPUT_FEATURES',
        'duplicate_name_rejected': gate(expected[:-1] + [expected[0]], expected) == 'INVALID_DUPLICATE_FEATURE_NAMES',
        'fewer_inputs_not_same_design': gate(expected[:1], expected) == 'SUBSET_DIFFERENT_DESIGN_NOT_AUTHORIZED',
        'extra_input_rejected': gate(expected + ['not_requested'], expected) == 'INCOMPATIBLE_INPUT_FEATURES',
        'fold_not_formal': gate(expected, expected, 'fold') == 'EXCLUDED_FOLD',
        'synthetic_not_formal': gate(expected, expected, 'synthetic') == 'EXCLUDED_SYNTHETIC',
        'empty_not_compatible': gate([], expected) == 'INCOMPATIBLE_INPUT_FEATURES',
    }
    assert all(controls.values())
    for p, digest in SOURCES.items():
        assert sha(ROOT / p) == digest, ('Protected input changed', p)
    for r in rows:
        p = ROOT / r['path']
        assert p.stat().st_size == r['bytes'] and p.stat().st_mtime_ns == r['mtime_ns'], p
    grouped = defaultdict(list)
    for r in rows:
        grouped[(r['study'], r['family'], r['role'], r['status'])].append(r)
    groups = [{'study': k[0], 'family': k[1], 'role': k[2], 'status': k[3],
               'checkpoints': len(v),
               'feature_counts': sorted({r['feature_count'] for r in v if 'feature_count' in r}),
               'minimum_missing_features': min((r['missing_feature_count'] for r in v if 'missing_feature_count' in r), default=None)}
              for k, v in sorted(grouped.items())]
    result = {'version': 'existing_models_metadata_gate_v001', 'created_local': datetime.now().astimezone().isoformat(),
              'status': 'PASS_INVENTORY_WITH_COMPARISON_GAPS', 'search_root': rel(EXP),
              'search_extensions': extensions, 'json_model_directory_search': True,
              'checkpoint_count': len(rows), 'json_actual_models': len(json_model_paths),
              'json_sidecar_count': len(sidecars), 'sidecars_not_counted_as_models': sidecars,
              'formal_exact_top10': [r for r in rows if r['status'] == 'EXACT_TOP10_INPUT_ONLY'],
              'different_design_subset_candidates': [r for r in rows if r['status'] == 'SUBSET_DIFFERENT_DESIGN_NOT_AUTHORIZED'],
              'formal_ten_column_candidates': [r for r in rows if r['role'] == 'formal' and r.get('feature_count') == 10],
              'expected_top10': expected, 'groups': groups, 'status_counts': dict(Counter(r['status'] for r in rows)),
              'unresolved': unresolved, 'controls': controls, 'rows': rows, 'feature_schemas': SCHEMAS,
              'source_sha256': SOURCES, 'source_hashes_rechecked_unchanged': True,
              'checkpoint_stat_rechecked_unchanged': True,
              'limits': ['Only declared search boundary; not a claim about unregistered storage.',
                         'Semantic metadata gate; most checkpoints not deserialized or hashed.',
                         'Incompatible inputs stop before complete preprocessing/target/weight QA.',
                         'Subsets are different designs; no automatic choice or prediction.',
                         'Exact existing formal Top10 BiLSTM was already independently validated; not rerun.'],
              'model_deserializations': 0, 'fit_calls': 0, 'predict_calls': 0, 'downloads': 0,
              'T012_complete': False}
    immutable(OUT / 'inventory.json', result)
    summary = {k: result[k] for k in ('status', 'checkpoint_count', 'json_actual_models', 'json_sidecar_count',
               'status_counts', 'formal_exact_top10', 'formal_ten_column_candidates',
               'different_design_subset_candidates', 'controls', 'limits')}
    summary['inventory_sha256'] = sha(OUT / 'inventory.json')
    summary['audit_script_sha256'] = sha(Path(__file__))
    immutable(OUT / 'summary.json', summary)
    print(json.dumps({'status': result['status'], 'checkpoint_count': len(rows),
                      'exact_formal_top10': len(result['formal_exact_top10']),
                      'subset_different_design': len(result['different_design_subset_candidates']),
                      'status_counts': result['status_counts'], 'protected_sources': len(SOURCES)}, indent=2))


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        if OUT.exists() and not (OUT / 'failure.json').exists():
            immutable(OUT / 'failure.json', {'type': type(exc).__name__, 'message': str(exc),
                                           'fit_calls': 0, 'predict_calls': 0, 'downloads': 0})
        raise
