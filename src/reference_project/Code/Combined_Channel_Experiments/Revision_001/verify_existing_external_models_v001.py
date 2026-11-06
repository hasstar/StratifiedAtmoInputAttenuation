"""Independent local source/feature check; does not import producer or models."""
from __future__ import annotations
import ast
import hashlib
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / 'Outputs/Experiments'
BASE = EXP / 'Combined_Channel_Revision_001/external/validation_gbs_48h_v010'
OUT = BASE / 'model_compatibility_001'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def main():
    assert not (OUT / 'independent_qa.json').exists()
    a = read(OUT / 'inventory.json')
    summary = read(OUT / 'summary.json')
    scope = read(BASE / 'scope_manifest.json')
    expected = tuple(scope['feature_names'])
    assert len(expected) == len(set(expected)) == 10
    assert set(expected) == set(a['expected_top10'])
    assert digest(OUT / 'inventory.json') == summary['inventory_sha256']
    primary = {'.joblib', '.pkl', '.pt', '.pth', '.cbm', '.ubj', '.onnx'}
    secondary = {'.bin', '.model', '.npz', '.h5', '.keras', '.safetensors', '.ckpt'}
    model_dirs = {'models', 'fitted_models', 'layer_models', 'fold_models', 'checkpoints', 'weights'}
    found = set()
    json_models = set()
    sidecar_count = 0
    for p in EXP.rglob('*'):
        if not p.is_file():
            continue
        parts = set(p.relative_to(EXP).parts[:-1])
        if p.suffix in primary or (p.suffix in secondary and model_dirs & parts):
            found.add(p.relative_to(ROOT).as_posix())
        elif p.suffix == '.json' and (model_dirs - {'fold_models'}) & parts:
            item = read(p)
            if isinstance(item.get('learner'), dict) and 'learner_model_param' in item['learner']:
                json_models.add(p.relative_to(ROOT).as_posix())
            else:
                sidecar_count += 1
    found |= json_models
    rows = a['rows']
    assert len(rows) == len({r['path'] for r in rows})
    assert {r['path'] for r in rows} == found
    assert len(rows) == a['checkpoint_count']
    assert len(json_models) == a['json_actual_models']
    assert sidecar_count == a['json_sidecar_count']
    schema_cache, source_cache = {}, {}
    candidate_checks, layer_checks = [], 0
    for row in rows:
        path = ROOT / row['path']
        assert (path.stat().st_size, path.stat().st_mtime_ns) == (row['bytes'], row['mtime_ns'])
        if row['role'] == 'synthetic':
            assert 'synthetic_api_' in row['path'] and row['status'] == 'EXCLUDED_SYNTHETIC'
            continue
        if row['path'] in json_models:
            d = read(path)
            assert row['feature_count'] == int(d['learner']['learner_model_param']['num_feature']) > 10
            assert row['status'] == 'INCOMPATIBLE_INPUT_DIMENSION'
            continue
        s = a['feature_schemas'][row['feature_schema']]
        names = tuple(s['feature_names'])
        assert len(set(names)) == len(names) == row['feature_count']
        assert set(s['missing_from_current_top10']) == set(names) - set(expected)
        assert row['missing_feature_count'] == len(set(names) - set(expected))
        evidence = row['feature_evidence']
        if evidence not in source_cache:
            d = read(ROOT / evidence)
            # Retain just the required feature information, not scores or histories.
            source_cache[evidence] = {k: d[k] for k in ('feature_names', 'retained_feature_names',
                'retained_features', 'features', 'base_names', 'retained_names', 'all_names', 'levels_hpa') if k in d}
        d = source_cache[evidence]
        if row.get('semantic_names_reconstructed_from_frozen_layer_manifest'):
            raw = d.get('all_names', d.get('retained_names'))
            base = d.get('base_names', [n for n in raw if n.startswith('surface__')])
            selected = [d['levels_hpa'][i] for i in range(len(d['levels_hpa']))
                        if row['mask'] // (2 ** i) % 2]
            rebuilt = [n for n in raw if n in base or any(n.startswith('p' + str(l) + '__') for l in selected)]
            assert list(names) == rebuilt and len(base) >= 50
            layer_checks += 1
        elif row['revision'] == 14:
            rebuilt = d['features']
            if row.get('requires_frozen_base_predictor'):
                folder = EXP / row['study']
                base = read(folder / 'layers/heldout_all_masks/models/base188.json')['features']
                rebuilt = base + rebuilt
            assert list(names) == rebuilt and len(names) >= 188
        else:
            original = next(d[k] for k in ('feature_names', 'retained_feature_names', 'retained_features') if k in d)
            assert list(names) == original
        if row['role'] == 'fold':
            assert '/fold_models/' in row['path'] and row['status'] == 'EXCLUDED_FOLD'
        elif set(names) == set(expected):
            assert len(names) == 10 and row['status'] == 'EXACT_TOP10_INPUT_ONLY'
        elif set(names) < set(expected):
            assert row['status'] == 'SUBSET_DIFFERENT_DESIGN_NOT_AUTHORIZED'
        else:
            assert row['status'] == 'INCOMPATIBLE_INPUT_FEATURES'
        if row.get('recorded_hash_matches_actual'):
            actual = digest(path)
            assert actual == row['checkpoint_sha256'] == row['recorded_checkpoint_sha256']
        if row['role'] == 'formal' and len(names) == 10:
            candidate_checks.append({'path': row['path'], 'family': row['family'],
                                     'missing_fields': sorted(set(names) - set(expected)),
                                     'sha256': digest(path)})
    exact = [r for r in rows if r['status'] == 'EXACT_TOP10_INPUT_ONLY']
    assert len(exact) == 1 and exact[0]['family'] == 'bilstm'
    assert exact[0]['checkpoint_sha256'] == 'b9eb8b7aa0f64b330dd0e6a23ac605afdceab8b7e34613011228a5015054c9e5'
    assert Counter(r['status'] for r in rows) == a['status_counts']
    for relative, expected_digest in a['source_sha256'].items():
        assert digest(ROOT / relative) == expected_digest, relative
    # Independent policy checker, including same dimension wrong identity.
    def same_design(fields, formal=True):
        return formal and len(fields) == 10 and len(set(fields)) == 10 and not (set(fields) ^ set(expected))
    tests = [same_design(expected), same_design(expected[::-1]),
             not same_design(expected[:-1] + ('wrong_field',)),
             not same_design(expected[:-1] + (expected[0],)),
             not same_design(expected[:1]), not same_design(expected + ('extra',)),
             not same_design(expected, False), not same_design(())]
    assert all(tests)
    script = ROOT / 'Code/Combined_Channel_Experiments/Revision_001/audit_existing_external_models_v001.py'
    tree = ast.parse(script.read_text(encoding='utf-8'))
    forbidden = {'fit', 'partial_fit', 'fit_transform', 'predict', 'backward', 'step',
                 'urlopen', 'request', 'get_results', 'retrieve', 'load_model'}
    called = {n.func.attr for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
    assert not (called & forbidden)
    imports = [n.names[0].name for n in ast.walk(tree) if isinstance(n, ast.Import)]
    assert not set(imports) & {'joblib', 'pickle', 'torch', 'requests', 'xgboost', 'sklearn'}
    payload = {'status': 'PASS_INDEPENDENT_METADATA_INPUT_GATE', 'created_local': datetime.now().astimezone().isoformat(),
               'checkpoint_paths_independently_enumerated': len(found), 'actual_json_models': len(json_models),
               'sidecar_metadata_not_checkpoints': sidecar_count, 'all_row_feature_gates_recomputed': len(rows),
               'layer_manifest_semantic_reconstructions': layer_checks, 'source_hashes_rechecked': len(a['source_sha256']),
               'formal_ten_column_candidates': candidate_checks, 'formal_exact_top10_models': 1,
               'different_design_subsets': len(a['different_design_subset_candidates']),
               'independent_controls_passed': len(tests), 'producer_controls_passed': len(a['controls']),
               'no_model_deserialization_fit_prediction_or_network': True,
               'audit_inventory_sha256': digest(OUT / 'inventory.json'), 'producer_sha256': digest(script),
               'checker_sha256': digest(Path(__file__)), 'T012_complete': False,
               'limits': 'Metadata/actual candidate hash gate only; incompatible inputs prevent full preprocessing/target QA. No claim about unregistered storage.'}
    with (OUT / 'independent_qa.json').open('x', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write('\n')
    print(json.dumps({k: payload[k] for k in ('status', 'checkpoint_paths_independently_enumerated',
                      'formal_exact_top10_models', 'different_design_subsets', 'source_hashes_rechecked')}, indent=2))


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        with (OUT / 'independent_failure_v001.json').open('x', encoding='utf-8') as f:
            json.dump({'type': type(exc).__name__, 'message': str(exc)}, f, indent=2)
        raise
