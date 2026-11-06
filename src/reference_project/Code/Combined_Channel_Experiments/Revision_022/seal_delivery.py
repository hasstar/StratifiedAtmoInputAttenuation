"""Seal verified component outputs; no weather/model/prediction execution."""
import ast
import hashlib
import json
import msvcrt
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CODE = Path(__file__).resolve().parent
OUT = ROOT / 'Outputs/Experiments/Combined_Channel_Revision_022'
OLD = ROOT / 'Outputs/Experiments/Combined_Channel_Revision_021'
NEW_NAMES = ('sources_after.json', 'seal_command.json', 'delivery_qa.json',
             'delivery_manifest.json', 'final_verification.json')
DOCS = ('PROTOCOL.md', 'REPORT.md', 'FORMULA_AUDIT.md', 'QA.md', 'SOURCES.md',
        'MODEL_INPUT_MATRIX.md', 'HANDOFF.md', 'COMMANDS.md', 'CHANGE_RECORD.md')


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def emit(name, obj):
    with (OUT / name).open('x', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def main():
    for name in NEW_NAMES:
        assert not (OUT / name).exists(), ('output already exists', name)
    for name in DOCS:
        assert (OUT / name).is_file(), name
    with (OUT / 'execution.lock').open('r+b') as lock:
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        try:
            guards = read(OUT / 'sources_before.json')
            assert len(guards) == 1124
            after = {p: sha(p) for p in guards}
            assert after == guards, 'protected source drift'
            for p, digest in read(OUT / 'code_before.json').items():
                assert sha(p) == digest, ('producer changed after execution', p)
            for p in CODE.glob('*.py'):
                ast.parse(p.read_text(encoding='utf-8-sig'), filename=str(p))
            previous = read(OLD / 'final_verification.json')
            assert previous['status'] == 'PASS'
            assert sha(OLD / 'delivery_manifest.json') == previous['manifest_sha256']
            previous_files = read(OLD / 'delivery_manifest.json')['files']
            assert len(previous_files) == 39
            for p, info in previous_files.items():
                assert sha(ROOT / p) == info['sha256'], ('old sealed artifact changed', p)

            status = read(OUT / 'model_status.json')
            models = {m['id']: m for m in status['models']}
            assert list(models) == ['proposed', 'bai_lstm', 'tabm', 'itu_combined',
                                    'e_sst', 'rmop', 'nwp_chain']
            assert status['formal_family_count'] == len(models) == 7
            assert status['computed_hourly_family_count'] == 4
            for m in list(models.values())[:4]:
                assert m['hourly_status'] == 'COMPUTED_REUSED'
                assert (OUT / m['scores_source']).is_file()
            for mid in ('bai_lstm', 'tabm'):
                assert models[mid]['frozen_seeds_all_retained'] == [17, 29, 43]
                assert 'NOT_AUTHOR_PRETRAINED' in models[mid]['implementation']
            for mid in ('e_sst', 'rmop'):
                assert models[mid]['hourly_status'] == 'BLOCKED_NOT_COMPUTED'
                assert models[mid]['numeric_results'] is None and models[mid]['rank'] is None
            prm = models['nwp_chain']['branches']['prm_hourly']
            mts = models['nwp_chain']['branches']['mts_seconds']
            assert prm['status'] == 'BLOCKED_NOT_COMPUTED' and prm['numeric_results'] is None
            assert mts['status'] == 'NOT_COMPUTED_USER_DECISION'
            assert mts['numeric_results'] is None and mts['rank'] is None
            assert mts['required_for_current_delivery'] is False
            assert mts['awaiting_user_materials'] is False
            assert mts['resume_requires_new_explicit_user_decision'] is True
            assert status['all_seven_original_models_complete'] is False
            assert status['current_required_work_complete'] is False
            assert status['automation_modified'] is False
            for key in ('new_training', 'new_real_predictions', 'new_real_attenuation_physics_runs',
                        'new_real_scoring', 'new_weather_requests'):
                assert status[key] == 0, key
            ds = status['data_scope']
            assert [ds[k] for k in ('local_original_test_hours', 'uk_complete_station_hours',
                    'uk_qualified_target_hours', 'uk_common_physics_hours')] == [687,5664,5593,5538]
            assert ds['uk_source_station_days'] == {'chilbolton':143,'sparsholt':93}
            assert ds['changed'] is False

            qa = read(OUT / 'independent_qa.json')
            unit = read(OUT / 'unit_tests.json')
            inputs = read(OUT / 'input_summary.json')
            synthetic = read(OUT / 'synthetic_summary.json')
            assert qa['status'] == unit['status'] == 'PASS'
            assert unit['tests'] == 18 and unit['failures'] == unit['errors'] == []
            assert qa['synthetic']['cases'] == synthetic['cases'] == 864
            assert qa['synthetic']['numeric'] == synthetic['numeric'] == 792
            assert qa['synthetic']['refused_context'] == synthetic['refused_missing_context'] == 72
            assert qa['synthetic']['tolerance_db'] == 2e-10
            assert all(e < 2e-10 for e in qa['synthetic']['max_absolute_errors_db'].values())
            assert qa['real_inputs']['campaign_ledger_rows'] == inputs['campaign_hours'] == 7800
            assert qa['real_inputs']['test_members'] == inputs['test_members'] == 687
            assert qa['real_inputs']['geometric_candidates'] == inputs['geometric_candidates'] == 33514
            assert qa['real_inputs']['preserved_nan_positions'] == inputs['retained_null_positions'] == 356486
            assert inputs['labels_loaded'] is False and inputs['new_predictions'] == 0
            assert inputs['selected_height_branch'] is None and inputs['model_height_produced'] is False
            assert qa['coefficients']['old_training_entrypoint_executed'] is False
            assert qa['protected_sources_unchanged'] == len(guards)
            assert read(OUT / 'execution_complete.json')['changed_sources'] == 0

            ituout = ROOT / 'Outputs/Experiments/Combined_Channel_Revision_011'
            itucode = ROOT / 'Code/Combined_Channel_Experiments/Revision_011'
            itufinal = read(ituout / 'final_verification.json')
            pred_sha = itufinal['prediction_sha256']
            for name, digest in pred_sha.items():
                assert sha(ituout / name) == digest
            code_guard = {p: h for p,h in guards.items() if Path(p).parent == itucode}
            assert len(code_guard) == 5
            stop = ROOT / 'Outputs/Experiments/Combined_Channel_Revision_001/external/weather_gbs_extend_v011/STOP_REQUESTED.json'
            automation = Path('C:/Users/SDR_HH/.codex/automations/tvt/automation.toml')
            assert sha(stop) == status['stop_sha256']
            assert sha(automation) == status['automation_sha256']

            checked_links = []
            for name in DOCS:
                content = (OUT / name).read_text(encoding='utf-8')
                for link in re.findall(r'\]\(([^)]+)\)', content):
                    if link.startswith(('https://', 'http://', '#')):
                        continue
                    target = (OUT / link).resolve()
                    assert target.is_file() or target == OUT / 'final_verification.json', (name,link)
                    checked_links.append({'document': name, 'target': link})
            emit('sources_after.json', after)
            emit('seal_command.json', {'utc': now(), 'pid': os.getpid(),
                'executable': sys.executable, 'python': sys.version,
                'argv': sys.argv, 'mode': 'delivery_checks_and_sha_only',
                'pre_seal_cim_utc': '2026-10-17T16:43:23.214752Z',
                'pre_seal_cim_matching_workers': 0,
                'process_snapshot_scope': 'Python/pythonw/MATLAB/Octave/Rscript; not global exclusivity'})
            emit('delivery_qa.json', {'utc':now(), 'status':'PASS',
                'protected_sources':len(guards), 'changed_sources':0,
                'previous_sealed_artifacts_unchanged':len(previous_files),
                'producer_source_files_unchanged':len(read(OUT / 'code_before.json')),
                'revision011_calculation_sources_unchanged':len(code_guard),
                'revision011_prediction_sha256':pred_sha,
                'unit_test_groups':18,'synthetic_cases':864,
                'synthetic_numeric':792,'synthetic_context_refusals':72,
                'independent_qa_sha256':sha(OUT / 'independent_qa.json'),
                'checked_local_document_links':checked_links,
                'status_scope':'components_and_provenance_only_not_site_predictions_or_full_seven_models',
                'stop_unchanged':True,'automation_unchanged':True,
                'mts_status':mts['status'],'new_real_predictions':0,'new_real_scores':0,
                'seven_family_work_complete':False})
            files = {}
            for folder in (CODE, OUT):
                for p in sorted(folder.rglob('*')):
                    if not p.is_file() or '__pycache__' in p.parts:
                        continue
                    if p.name in ('delivery_manifest.json','final_verification.json'):
                        continue
                    files[str(p.relative_to(ROOT))] = {'sha256':sha(p),'bytes':p.stat().st_size}
            emit('delivery_manifest.json', {'files':files,
                'scope':'Revision022 code and outputs excluding this manifest and final marker'})
            for p, info in files.items():
                assert sha(ROOT / p) == info['sha256']
            assert {p:sha(p) for p in guards} == guards, 'source drift during seal'
            result = {'status':'PASS','utc':now(),
                'local_phase':'HOURLY_POWER_OPERATOR_AND_LOCAL_INPUT_COMPONENTS_VERIFIED',
                'manifest_sha256':sha(OUT / 'delivery_manifest.json'),
                'sealed_artifacts':len(files),'protected_sources':len(guards),
                'protected_source_changes':0,'new_real_predictions':0,'new_real_scores':0,
                'seven_family_work_complete':False,'all_seven_original_models_complete':False,
                'execution_lock':'RELEASED',
                'lock_kind':'local_byte_lock_and_logical_source_guard_no_global_exclusivity',
                'automation_modified':False,'automation_pause_requested':False,
                'mts_status':mts['status']}
        finally:
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
    result['lock_released_utc'] = now()
    emit('final_verification.json', result)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
