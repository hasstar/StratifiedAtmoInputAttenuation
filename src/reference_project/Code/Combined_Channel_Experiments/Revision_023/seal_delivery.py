"""Seal verified metadata support; neither physical inference nor new scoring."""
import ast
import hashlib
import json
import msvcrt
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
CODE=Path(__file__).resolve().parent
OUT=ROOT/'Outputs/Experiments/Combined_Channel_Revision_023'
OLD=ROOT/'Outputs/Experiments/Combined_Channel_Revision_022'


def now():return datetime.now(timezone.utc).isoformat()


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def emit(name,obj):
    with (OUT/name).open('x',encoding='utf-8') as f:
        json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')


def main():
    for n in ['seal_sources_after.json','seal_command.json','delivery_qa.json',
              'delivery_manifest.json','final_verification.json']:
        assert not (OUT/n).exists(),n
    required=['PROTOCOL.md','REPORT.md','QA.md','FORMULA_AUDIT.md','SOURCES.md',
        'MODEL_INPUT_MATRIX.md','COMMANDS.md','CHANGE_RECORD.md','HANDOFF.md',
        'ADAPTATION_PROPOSAL.md','QA_ATTEMPT_001_FAILURE.md']
    assert all((OUT/n).is_file() for n in required)
    guards=read(OUT/'sources_before.json')
    assert len(guards)==1175
    assert guards==read(OUT/'sources_after.json')=={p:sha(p) for p in guards}
    oldfiles=read(OLD/'delivery_manifest.json')['files']
    assert len(oldfiles)==38
    assert sha(OLD/'delivery_manifest.json')==read(OLD/'final_verification.json')['manifest_sha256']
    for p,item in oldfiles.items():assert sha(ROOT/p)==item['sha256']
    before=read(OUT/'code_before.json')
    for p,h in before.items():assert sha(p)==h
    for p in CODE.glob('*.py'):ast.parse(p.read_text(encoding='utf-8-sig'),filename=str(p))
    original=(CODE/'verify_independent.py').read_text(encoding='utf-8-sig').strip()
    corrected=(CODE/'verify_independent_v002.py').read_text(encoding='utf-8-sig').strip()
    assert corrected==original.replace('independent_qa.json','independent_qa_v002.json').replace(
        'pc.is_in(pa.array(bh),value_set=hourset).to_numpy()',
        'pc.is_in(pa.array(bh),value_set=hourset).to_numpy(zero_copy_only=False)')
    assert not (OUT/'independent_qa.json').exists()
    qa=read(OUT/'independent_qa_v002.json');unit=read(OUT/'unit_tests.json')
    assert qa['status']==unit['status']=='PASS'
    assert unit['tests']==10 and unit['failures']==unit['errors']==[]
    assert qa['source_rows_scanned']==132720993
    assert qa['hours_checked']==687
    assert qa['exact_timestamps_checked']==qa['exact_quality_flags_checked']==24196167
    assert qa['all_hour_metadata_fields_checked']==13 and qa['timestamp_histogram_exact'] is True
    assert qa['max_coverage_difference']==0.0 and qa['coverage_tolerance']==2e-15
    assert qa['protected_sources_unchanged']==1175
    assert qa['power_finiteness_rechecked'] is False
    assert qa['source_to_frozen_label_power_reconstruction']=='NOT_PERFORMED'
    summary=read(OUT/'sampling_summary.json')
    assert summary['raw_test_records']==24196233 and summary['duplicate_test_records']==66
    assert summary['quality_nonzero_records_retained']==1197
    assert summary['min_records']==34809 and summary['max_records']==35726
    assert summary['measured_power_values_loaded'] is False
    assert summary['new_predictions']==summary['new_scores']==0
    assert (OUT/'sampling_support.npz').stat().st_size==45162736
    status=read(OUT/'model_status.json')
    assert status['revision']=='023' and status['formal_family_count']==7
    assert status['computed_hourly_family_count']==4
    assert [m['id'] for m in status['models']]==['proposed','bai_lstm','tabm','itu_combined','e_sst','rmop','nwp_chain']
    for model in status['models'][:4]:
        assert model['hourly_status']=='COMPUTED_REUSED'
        assert (OUT/model['scores_source']).is_file()
    for model in status['models'][1:3]:
        assert model['frozen_seeds_all_retained']==[17,29,43]
        assert 'NOT_AUTHOR_PRETRAINED' in model['implementation']
    e=status['models'][4]
    assert e['valid_record_time_quality_support_verified'] is True
    assert e['real_target_valid_sampling_mask_aligned'] is False
    for model in status['models'][4:6]:
        assert model['hourly_status']=='BLOCKED_NOT_COMPUTED'
        assert model['numeric_results'] is None and model['rank'] is None
    prm=status['models'][6]['branches']['prm_hourly']
    mts=status['models'][6]['branches']['mts_seconds']
    assert prm['status']=='BLOCKED_NOT_COMPUTED' and prm['numeric_results'] is None and prm['rank'] is None
    assert mts=={'status':'NOT_COMPUTED_USER_DECISION','numeric_results':None,'rank':None,
        'required_for_current_delivery':False,'awaiting_user_materials':False,
        'resume_requires_new_explicit_user_decision':True}
    assert status['adaptation_decision']['status']=='AWAITING_USER_CONFIRMATION'
    assert status['adaptation_decision']['do_not_repeat_while_pending'] is True
    for n in ['all_seven_original_models_complete','current_required_work_complete','submission_ready','automation_modified']:
        assert status[n] is False
    for n in ['new_training','new_real_predictions','new_real_attenuation_physics_runs',
              'new_real_scoring','new_weather_requests']:
        assert status[n]==0
    scope=status['data_scope']
    assert [scope[n] for n in ['local_original_test_hours','uk_complete_station_hours',
        'uk_qualified_target_hours','uk_common_physics_hours']]==[687,5664,5593,5538]
    assert scope['uk_source_station_days']=={'chilbolton':143,'sparsholt':93}
    assert scope['changed'] is False
    itu=ROOT/'Outputs/Experiments/Combined_Channel_Revision_011'
    oldqa=read(OLD/'delivery_qa.json')
    for p,h in oldqa['revision011_prediction_sha256'].items():assert sha(itu/p)==h
    assert oldqa['revision011_calculation_sources_unchanged']==5
    itucode=[p for p in guards if 'Code\\Combined_Channel_Experiments\\Revision_011\\' in p]
    assert len(itucode)==5
    stop=ROOT/'Outputs/Experiments/Combined_Channel_Revision_001/external/weather_gbs_extend_v011/STOP_REQUESTED.json'
    automation=Path('C:/Users/SDR_HH/.codex/automations/tvt/automation.toml')
    assert sha(stop)==status['stop_sha256'] and sha(automation)==status['automation_sha256']
    links=[]
    for p in OUT.glob('*.md'):
        for href in re.findall(r'\]\(([^)]+)\)',p.read_text(encoding='utf-8')):
            if href.startswith(('https://','http://','#')):continue
            resolved=(p.parent/href).resolve()
            assert resolved.is_file() or resolved in [OUT/'delivery_manifest.json',OUT/'final_verification.json'],(p,href)
            links.append({'document':p.name,'target':href})
    lockpath=OUT/'execution.lock'
    lockinfo={'sha256':sha(lockpath),'bytes':lockpath.stat().st_size}
    files={}
    with lockpath.open('r+b') as lock:
        msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        try:
            after={p:sha(p) for p in guards};assert after==guards
            emit('seal_sources_after.json',after)
            emit('seal_command.json',{'utc':now(),'pid':os.getpid(),'argv':sys.argv,'executable':sys.executable,
                'python':sys.version,'pre_seal_cim_utc':'2026-10-17T17:17:29.9096757Z','matching_workers':0,
                'mode':'metadata_delivery_verification_and_sha_only'})
            emit('delivery_qa.json',{'utc':now(),'status':'PASS','protected_sources':1175,'source_changes':0,
                'revision022_sealed_artifacts_unchanged':38,'revision011_calculation_sources_unchanged':5,
                'revision011_prediction_sha256':oldqa['revision011_prediction_sha256'],
                'unit_test_groups':10,'independent_qa_sha256':sha(OUT/'independent_qa_v002.json'),
                'sampling_support_sha256':sha(OUT/'sampling_support.npz'),
                'failed_attempt_preserved':True,'qa_fix_exact_diff_verified':True,
                'checked_local_links':links,'pending_local_adaptation_decision':True,
                'stop_unchanged':True,'automation_unchanged':True,'new_predictions':0,'new_scores':0,
                'seven_family_work_complete':False,'mts_status':'NOT_COMPUTED_USER_DECISION'})
            for folder in (CODE,OUT):
                for p in sorted(folder.rglob('*')):
                    if not p.is_file() or '__pycache__' in p.parts:continue
                    if p.name in ['delivery_manifest.json','final_verification.json']:continue
                    files[str(p.relative_to(ROOT))]=lockinfo if p==lockpath else {'sha256':sha(p),'bytes':p.stat().st_size}
            emit('delivery_manifest.json',{'files':files,
                'scope':'Revision023 code/output including preserved first QA failure, excluding manifest/final marker'})
        finally:
            lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1)
    released=now()
    for p,item in files.items():
        assert sha(ROOT/p)==item['sha256'] and (ROOT/p).stat().st_size==item['bytes']
    assert {p:sha(p) for p in guards}==guards
    final={'status':'PASS','utc':now(),'local_phase':'EXACT_LOCAL_TEST_SAMPLING_SUPPORT_VERIFIED',
        'sealed_artifacts':len(files),'manifest_sha256':sha(OUT/'delivery_manifest.json'),
        'protected_sources':1175,'protected_source_changes':0,
        'execution_lock':'RELEASED','lock_released_utc':released,
        'lock_scope':'local byte lock plus post-release hashes, not global exclusivity',
        'new_real_predictions':0,'new_real_scores':0,'seven_family_work_complete':False,
        'all_seven_original_models_complete':False,'pending_local_adaptation_decision':True,
        'automation_modified':False,'automation_pause_requested':False,
        'mts_status':'NOT_COMPUTED_USER_DECISION','first_failed_qa_attempt_preserved':True}
    emit('final_verification.json',final)
    print(json.dumps(final,ensure_ascii=False))


if __name__=='__main__':main()
