"""One-shot metadata publication and hash seal; no model, weather, or labels."""
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
CODE = Path(__file__).resolve().parent
OUT = ROOT / 'Outputs/Experiments/Combined_Channel_Revision_021'
PREV = ROOT / 'Outputs/Experiments/Combined_Channel_Revision_020'
checks = []


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()


def need(v, name):
    checks.append({'name': name, 'pass': bool(v)})
    if not v:
        raise AssertionError(name)


def emit(name, value):
    with (OUT / name).open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def main():
    need(not (OUT / 'final_verification.json').exists(), 'new_seal_target')
    for name in ['PROTOCOL.md', 'REPORT.md', 'QA.md', 'HANDOFF.md', 'MODEL_INPUT_MATRIX.md',
                 'FORMULA_INPUT_AUDIT.md', 'SOURCES.md', 'COMMANDS.md', 'CHANGE_RECORD.md',
                 'QA_ATTEMPT_001_INTERRUPTED.md']:
        need((OUT / name).is_file(), name)
    independent = read(OUT / 'independent_qa.json')
    summary = read(OUT / 'diagnostic_summary.json')
    ledger = read(OUT / 'hourly_input_ledger.json')
    context = read(OUT / 'rain_context_ledger.json')
    events = read(OUT / 'rain_events.json')
    need(independent['status'] == 'PASS', 'independent_qa_pass')
    need(read(OUT / 'unit_checks.json')['n'] == 19, '19_boundary_tests')
    need(len(ledger) == 7800 and len(context) == 7823 and len(events) == 253, 'diagnostic_sizes')
    for label, rows in [('campaign', ledger), ('test', [r for r in ledger if r['test_member']])]:
        expected = {'n': len(rows),
            'rain_invalid_hours': sum(not r['all_corners_liquid_nonnegative'] for r in rows),
            'deg0l_missing_hours': sum(not r['deg0l_all_corners_finite'] for r in rows),
            'deg0l_any_zero_hours': sum(r['deg0l_zero_corner_count'] > 0 for r in rows),
            'wet_hours': sum(r['wet_event_id'] >= 0 for r in rows),
            'complete_wet_event_hours': sum(r['wet_event_complete'] is True for r in rows),
            'censored_wet_event_hours': sum(r['wet_event_complete'] is False for r in rows),
            'dry_hours': sum(r['site_rain_mm_per_h'] == 0 for r in rows)}
        need(expected == summary[label], label + '_summary_from_rows')
    for rows, expected in [(ledger, [1142, 6646, 12]), ([r for r in ledger if r['test_member']], [42, 645, 0])]:
        c = Counter(r['station_crossings'] for r in rows)
        need([c[0], c[1], sum(v for k, v in c.items() if k > 1)] == expected, 'reported_crossing_counts')
    need(sum(e['complete'] for e in events) == 212, '212_complete_events')
    need(independent['numeric_crossing_candidates'] == 33514, '33514_candidates')
    original = (CODE / 'verify_independent.py').read_text(encoding='utf-8')
    modified = (CODE / 'verify_independent_v002.py').read_text(encoding='utf-8')
    change_from = "npz = np.load(OUT / 'zero_crossing_diagnostics.npz', allow_pickle=False)"
    change_to = "with np.load(OUT / 'zero_crossing_diagnostics.npz', allow_pickle=False) as archive:\n    npz = {key: archive[key] for key in archive.files}"
    need(original.replace(change_from, change_to) == modified, 'qa_fix_only_npz_eager_cache')
    need('import diagnose_inputs' not in modified and 'import rev021_diag' not in modified, 'independent_no_production_import')
    guards = read(OUT / 'computation_sources_before.json')
    changed = [p for p, digest in guards.items() if sha(p) != digest]
    need(not changed and len(guards) == 1081, '1081_protected_sources_unchanged')
    old_manifest = read(PREV / 'delivery_manifest.json')['files']
    need(all(sha(ROOT / p) == v['sha256'] for p, v in old_manifest.items()), 'rev020_seal_unchanged')
    rev011_code = [p for p in guards if 'Code\\Combined_Channel_Experiments\\Revision_011\\' in p and p.endswith('.py')]
    rev011_pred = [p for p in guards if 'Combined_Channel_Revision_011\\predictions_' in p and p.endswith('.parquet')]
    need(len(rev011_code) == 5 and len(rev011_pred) == 2, 'rev011_five_code_two_prediction_hashes')
    emit('reuse_check.json', {'status': 'PASS', 'revision011_code': {p: guards[p] for p in rev011_code},
         'revision011_predictions': {p: guards[p] for p in rev011_pred},
         'new_model_evaluations': 0, 'scores_source': '../Combined_Channel_Revision_012/seven_family_rankings.json',
         'previous_historical_authorized_exception': read(PREV / 'reuse_check.json')['authorized_exceptions']})
    status = read(PREV / 'model_status.json')
    status.update({'revision': '021', 'scope': 'LOCAL_REAL_INPUT_DIAGNOSTICS_WITH_INDEPENDENT_QA',
                   'new_weather_diagnostic_runs': 1, 'new_real_predictions': 0,
                   'new_real_physics_runs': 0, 'new_real_scoring': 0, 'new_training': 0,
                   'new_weather_requests': 0, 'current_required_work_complete': False})
    esst = next(m for m in status['models'] if m['id'] == 'e_sst')
    esst.update({'implementation': 'HOURLY_KERNEL_AND_LOCAL_INPUT_DIAGNOSTICS_ONLY_SITE_ADAPTATION_PENDING',
                 'real_input_diagnostic_qa': 'PASS_LOCAL_ONLY_NOT_MODEL_ELIGIBILITY',
                 'input_diagnostic_source': 'diagnostic_summary.json',
                 'blockers': ['geometric_height_common_vertical_datum_and_branch_policy',
                              'hourly_output_aggregation_and_event_type_adaptation',
                              '700hPa_wind_path_motion_frequency_polarization_coefficients',
                              'rain_only_to_total_relative_target_alignment', 'uk_required_input_coverage']})
    status['data_scope']['local_test_membership_reverified_time_column_only'] = True
    need([m['id'] for m in status['models']] == ['proposed', 'bai_lstm', 'tabm', 'itu_combined', 'e_sst', 'rmop', 'nwp_chain'], 'exact_seven_families')
    mts = status['models'][6]['branches']['mts_seconds']
    need(mts['status'] == 'NOT_COMPUTED_USER_DECISION' and mts['numeric_results'] is None and mts['rank'] is None and not mts['awaiting_user_materials'] and not mts['required_for_current_delivery'], 'mts_decision_preserved')
    need(esst['numeric_results'] is None and esst['rank'] is None, 'no_esst_rank_from_candidate_coverage')
    emit('model_status.json', status)
    emit('input_readiness.json', {
        'family': 'E-SST', 'scope': 'LOCAL_INPUT_DIAGNOSTICS_ONLY', 'real_scoring_allowed': False,
        'verified_this_revision': ['all_340_bounded_raw_headers', 'declared_antenna_height_source',
                                  'continuous_UTC_time_and_rate_interval_mapping', 'all_687_test_time_keys',
                                  'liquid_difference_and_invalid_value_preservation',
                                  'all_available_rain_context_and_event_censoring',
                                  'all_profile_zero_crossing_candidates_independently_recomputed'],
        'unresolved_before_real_inference': esst['blockers'],
        'station_one_crossing_test_hours': 645, 'candidate_coverage_is_model_eligibility': False,
        'deg0l_missing_test_hours_with_single_profile_candidate': 108,
        'old_deg0l_imputed': False, 'test_labels_loaded': False,
        'uk_new_data_acquisition_authorized': False, 'new_predictions': 0,
        'official_geometric_conversion_source_found_but_not_applied': True})
    emit('sources_after_final.json', guards)
    emit('delivery_qa.json', {'status': 'PASS', 'utc': datetime.now(timezone.utc).isoformat(),
         'checks': checks, 'checks_count': len(checks), 'protected_sources': len(guards),
         'source_changes': changed, 'final_process_check': {'utc': '2026-10-17T16:22:12.9726109Z',
         'research_workers': 0, 'scope': 'CIM python/pythonw/matlab/octave/Rscript; not all external writers'},
         'required_seven_family_work_complete': False})
    files = {}
    for directory in (CODE, OUT):
        for p in sorted(directory.rglob('*')):
            if p.is_file() and '__pycache__' not in p.parts:
                files[str(p.relative_to(ROOT))] = {'sha256': sha(p), 'bytes': p.stat().st_size}
    emit('delivery_manifest.json', {'files': files, 'scope': 'Revision021 code and output, excluding this manifest and final marker'})
    need(all(sha(ROOT / p) == v['sha256'] for p, v in files.items()), 'sealed_files_readback')
    final = {'status': 'PASS', 'utc': datetime.now(timezone.utc).isoformat(),
             'local_phase': 'LOCAL_HEIGHT_TIME_AND_CONTINUOUS_RAIN_INPUT_DIAGNOSTICS_VERIFIED',
             'manifest_sha256': sha(OUT / 'delivery_manifest.json'), 'sealed_artifacts': len(files),
             'protected_sources': len(guards), 'protected_source_changes': 0,
             'execution_lock': 'RELEASED', 'lock_kind': 'logical_declaration_no_global_exclusivity',
             'new_real_predictions': 0, 'new_real_scores': 0,
             'seven_family_work_complete': False, 'all_seven_original_models_complete': False,
             'automation_modified': False, 'automation_pause_requested': False,
             'mts_status': 'NOT_COMPUTED_USER_DECISION'}
    emit('final_verification.json', final)
    print(json.dumps(final, ensure_ascii=False))


if __name__ == '__main__':
    main()
