"""Retry sealing only; preserve first failed attempt and numerical outputs."""
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
    for name in ('sources_after_v002.json','seal_command_v002.json','delivery_qa_v002.json',
                 'delivery_manifest.json','final_verification.json'):
        assert not (OUT / name).exists(), name
    for name in ('PROTOCOL.md','REPORT.md','FORMULA_AUDIT.md','QA.md','SOURCES.md',
                 'MODEL_INPUT_MATRIX.md','HANDOFF.md','COMMANDS.md','CHANGE_RECORD.md',
                 'SEAL_ATTEMPT_001_FAILURE.md'):
        assert (OUT / name).is_file(), name
    attempt = {str(p):sha(p) for p in [CODE/'seal_delivery.py',OUT/'sources_after.json',
        OUT/'seal_command.json',OUT/'delivery_qa.json']}
    lockpath = OUT / 'execution.lock'
    lockinfo = {'sha256':sha(lockpath),'bytes':lockpath.stat().st_size}
    files = {}
    guards = read(OUT / 'sources_before.json')
    with lockpath.open('r+b') as lock:
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        try:
            assert len(guards) == 1124
            after = {p:sha(p) for p in guards}
            assert guards == after == read(OUT/'sources_after.json')
            for p, digest in read(OUT/'code_before.json').items():
                assert sha(p) == digest, p
            for p in CODE.glob('*.py'):
                ast.parse(p.read_text(encoding='utf-8-sig'), filename=str(p))
            firstqa = read(OUT/'delivery_qa.json')
            assert firstqa['status']=='PASS' and firstqa['changed_sources']==0
            assert firstqa['seven_family_work_complete'] is False
            assert sha(OUT/'independent_qa.json') == firstqa['independent_qa_sha256']
            for item in firstqa['checked_local_document_links']:
                target = (OUT / item['target']).resolve()
                assert target.is_file() or target == OUT/'final_verification.json'
            qa, unit = read(OUT/'independent_qa.json'), read(OUT/'unit_tests.json')
            assert qa['status'] == unit['status'] == 'PASS'
            assert unit['tests']==18 and unit['failures']==unit['errors']==[]
            syn = qa['synthetic']
            assert [syn[k] for k in ('cases','numeric','refused_context')] == [864,792,72]
            assert syn['tolerance_db']==2e-10
            assert all(e<2e-10 for e in syn['max_absolute_errors_db'].values())
            real = qa['real_inputs']
            assert [real[k] for k in ('wind_files','raw_wind_hours','corner_components_exact',
                'campaign_ledger_rows','test_members','geometric_candidates','preserved_nan_positions')
                ] == [24,7824,62592,7800,687,33514,356486]
            assert qa['protected_sources_unchanged']==len(guards)
            inputs = read(OUT/'input_summary.json')
            assert inputs['labels_loaded'] is False and inputs['new_predictions']==0
            assert inputs['model_height_produced'] is False and inputs['selected_height_branch'] is None
            assert read(OUT/'execution_complete.json')['changed_sources']==0
            status = read(OUT/'model_status.json')
            ids = [m['id'] for m in status['models']]
            assert ids == ['proposed','bai_lstm','tabm','itu_combined','e_sst','rmop','nwp_chain']
            assert status['formal_family_count']==7 and status['computed_hourly_family_count']==4
            for model in status['models'][:4]:
                assert model['hourly_status']=='COMPUTED_REUSED'
                assert (OUT/model['scores_source']).is_file()
            for model in status['models'][1:3]:
                assert model['frozen_seeds_all_retained']==[17,29,43]
                assert 'NOT_AUTHOR_PRETRAINED' in model['implementation']
            for model in status['models'][4:6]:
                assert model['hourly_status']=='BLOCKED_NOT_COMPUTED'
                assert model['numeric_results'] is None and model['rank'] is None
            branches = status['models'][6]['branches']
            prm, mts = branches['prm_hourly'], branches['mts_seconds']
            assert prm['status']=='BLOCKED_NOT_COMPUTED' and prm['numeric_results'] is None
            assert mts == {'status':'NOT_COMPUTED_USER_DECISION','numeric_results':None,
                'rank':None,'required_for_current_delivery':False,'awaiting_user_materials':False,
                'resume_requires_new_explicit_user_decision':True}
            for k in ('all_seven_original_models_complete','current_required_work_complete','automation_modified'):
                assert status[k] is False
            for k in ('new_training','new_real_predictions','new_real_attenuation_physics_runs',
                      'new_real_scoring','new_weather_requests'):
                assert status[k]==0
            ds = status['data_scope']
            assert [ds[k] for k in ('local_original_test_hours','uk_complete_station_hours',
                'uk_qualified_target_hours','uk_common_physics_hours')]==[687,5664,5593,5538]
            assert ds['uk_source_station_days']=={'chilbolton':143,'sparsholt':93}
            assert ds['changed'] is False
            old = ROOT/'Outputs/Experiments/Combined_Channel_Revision_021'
            oldfinal = read(old/'final_verification.json')
            assert sha(old/'delivery_manifest.json')==oldfinal['manifest_sha256']
            oldfiles = read(old/'delivery_manifest.json')['files']
            assert len(oldfiles)==39
            for p,info in oldfiles.items():
                assert sha(ROOT/p)==info['sha256'],p
            itu = ROOT/'Outputs/Experiments/Combined_Channel_Revision_011'
            for p,digest in firstqa['revision011_prediction_sha256'].items():
                assert sha(itu/p)==digest
            assert firstqa['revision011_calculation_sources_unchanged']==5
            stop = ROOT/'Outputs/Experiments/Combined_Channel_Revision_001/external/weather_gbs_extend_v011/STOP_REQUESTED.json'
            automation = Path('C:/Users/SDR_HH/.codex/automations/tvt/automation.toml')
            assert sha(stop)==status['stop_sha256'] and sha(automation)==status['automation_sha256']
            for p,digest in attempt.items():
                assert sha(p)==digest
            # Recheck all current document links, including appended failure notice.
            for p in OUT.glob('*.md'):
                for link in re.findall(r'\]\(([^)]+)\)', p.read_text(encoding='utf-8')):
                    if link.startswith(('https://','http://','#')):
                        continue
                    target=(p.parent/link).resolve()
                    assert target.is_file() or target==OUT/'final_verification.json', (p,link)
            emit('sources_after_v002.json',after)
            emit('seal_command_v002.json',{'utc':now(),'pid':os.getpid(),'executable':sys.executable,
                'argv':sys.argv,'python':sys.version,'mode':'delivery_checks_and_sha_only',
                'pre_seal_cim_utc':'2026-10-17T16:51:32.020888Z','matching_workers':0,
                'first_attempt_preserved_sha256':attempt})
            emit('delivery_qa_v002.json',{'utc':now(),'status':'PASS',
                'protected_sources':len(guards),'changed_sources':0,
                'old_sealed_artifacts_unchanged':len(oldfiles),
                'first_attempt_preserved_sha256':attempt,
                'independent_qa_sha256':sha(OUT/'independent_qa.json'),
                'producer_code_before_unchanged':True,'unit_test_groups':18,
                'synthetic_numeric':792,'synthetic_context_refusals':72,
                'stop_unchanged':True,'automation_unchanged':True,
                'new_real_predictions':0,'new_real_scores':0,
                'status_scope':'component_verification_not_site_model_completion',
                'seven_family_work_complete':False,'mts_status':mts['status'],
                'lock_fix':'hash lock before acquisition; verify all files again after release'})
            for folder in (CODE,OUT):
                for p in sorted(folder.rglob('*')):
                    if not p.is_file() or '__pycache__' in p.parts:
                        continue
                    if p.name in ('delivery_manifest.json','final_verification.json'):
                        continue
                    files[str(p.relative_to(ROOT))] = lockinfo if p==lockpath else {
                        'sha256':sha(p),'bytes':p.stat().st_size}
            emit('delivery_manifest.json',{'files':files,
                'scope':'Revision022 code and outputs, including failed seal attempt; excludes manifest and final marker'})
        finally:
            lock.seek(0)
            msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1)
    released=now()
    for p,info in files.items():
        assert sha(ROOT/p)==info['sha256'],p
        assert (ROOT/p).stat().st_size==info['bytes'],p
    assert {p:sha(p) for p in guards}==guards
    result={'status':'PASS','utc':now(),
        'local_phase':'HOURLY_POWER_OPERATOR_AND_LOCAL_INPUT_COMPONENTS_VERIFIED',
        'manifest_sha256':sha(OUT/'delivery_manifest.json'),'sealed_artifacts':len(files),
        'protected_sources':len(guards),'protected_source_changes':0,
        'execution_lock':'RELEASED','lock_released_utc':released,
        'lock_kind':'local_byte_lock_with_post_release_hash_check_not_global_exclusivity',
        'new_real_predictions':0,'new_real_scores':0,'seven_family_work_complete':False,
        'all_seven_original_models_complete':False,'automation_modified':False,
        'automation_pause_requested':False,'mts_status':mts['status'],
        'final_seal_attempt':'v002','first_failed_seal_attempt_preserved':True}
    emit('final_verification.json',result)
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':
    main()
