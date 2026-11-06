"""Seal input census, with all seven families and uncomputed branches explicit."""
import ast
import hashlib
import json
import msvcrt
import os
import re
import sys
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
CODE=Path(__file__).resolve().parent
OUT=ROOT/'Outputs/Experiments/Combined_Channel_Revision_024'
OLD=ROOT/'Outputs/Experiments/Combined_Channel_Revision_023'


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def now():return datetime.now(timezone.utc).isoformat()
def emit(n,x):
    with (OUT/n).open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False)


def main():
    assert all(not (OUT/n).exists() for n in ['model_status.json','seal_sources_after.json','delivery_qa.json','delivery_manifest.json','final_verification.json','seal_command.json'])
    required=['PROTOCOL.md','REPORT.md','QA.md','FORMULA_AUDIT.md','MODEL_INPUT_MATRIX.md','MISSING_INPUT_PROPOSAL.md','SOURCES.md','HANDOFF.md','COMMANDS.md','CHANGE_RECORD.md']
    assert all((OUT/n).is_file() for n in required)
    guard=read(OUT/'sources_before.json');assert guard==read(OUT/'sources_after.json')=={p:sha(p) for p in guard}
    oldmanifest=read(OLD/'delivery_manifest.json')
    assert len(oldmanifest['files'])==34
    assert sha(OLD/'delivery_manifest.json')==read(OLD/'final_verification.json')['manifest_sha256']
    assert all(sha(ROOT/p)==v['sha256'] for p,v in oldmanifest['files'].items())
    assert all(sha(p)==h for p,h in read(OUT/'code_before.json').items())
    for p in CODE.glob('*.py'):ast.parse(p.read_text(encoding='utf-8'),filename=str(p))
    qa=read(OUT/'independent_qa.json');s=read(OUT/'summary.json');tests=read(OUT/'unit_tests.json')
    assert qa['status']==s['status']=='PASS_INPUT_AUDIT_ONLY'
    assert tests=={'status':'PASS','tests':10,'failures':[],'errors':[]}
    assert qa['evaluation_hours_exactly_checked']==s['evaluation_hours']==5664
    assert qa['station_days_exactly_checked']==s['evaluation_station_days']==236
    assert s['qualified_target_hours']==5593 and s['existing_physics_common_hours']==5538
    assert qa['max_rain_delta_mm_per_h']<=1e-12 and qa['all_event_boundaries_exact'] and qa['all_hourly_fields_exact']
    assert s['minimum_missing_evaluation_field_station_hours']==16992
    assert s['new_computable_esst_hours']==s['new_predictions']==s['new_scores']==s['weather_requests']==0
    assert s['actual_future_context_request_quantity'] is None
    status=read(OLD/'model_status.json')
    status.update(revision='024',scope='UK_EXISTING_INPUT_CENSUS_AND_CONTEXT_VERIFIED_ONLY',new_metadata_extraction_runs=1)
    status['adaptation_decision'].update(proposal='../Combined_Channel_Revision_023/ADAPTATION_PROPOSAL.md',question_asked_this_revision=False)
    status['data_scope']['new_uk_keywise_reconstruction_this_revision']=True
    status['data_scope']['local_test_membership_reverified_time_column_only']=False
    status['data_scope']['local_frozen_687_inherited_sha_verified']=True
    e=status['models'][4]
    e['sampling_support_source']='../Combined_Channel_Revision_023/sampling_support.npz'
    e['uk_input_audit_source']='summary.json'
    e['uk_input_status']='BLOCKED_MISSING_700HPA_U_V_AND_QUALIFIED_ZERO_DEGREE_HEIGHT'
    e['uk_existing_rain_context_hours']=7152
    e['uk_required_evaluation_hours_preserved']=5664
    e['uk_input_audit_independent_qa']='PASS_INPUT_AUDIT_ONLY'
    assert status['formal_family_count']==7 and status['computed_hourly_family_count']==4
    assert [x['id'] for x in status['models']]==['proposed','bai_lstm','tabm','itu_combined','e_sst','rmop','nwp_chain']
    assert status['models'][6]['branches']['mts_seconds']['status']=='NOT_COMPUTED_USER_DECISION'
    assert status['models'][6]['branches']['mts_seconds']['numeric_results'] is None
    assert status['models'][6]['branches']['mts_seconds']['awaiting_user_materials'] is False
    assert status['adaptation_decision']['status']=='AWAITING_USER_CONFIRMATION'
    assert status['all_seven_original_models_complete']==status['current_required_work_complete']==status['submission_ready']==status['automation_modified']==False
    assert status['new_training']==status['new_real_predictions']==status['new_real_attenuation_physics_runs']==status['new_real_scoring']==status['new_weather_requests']==0
    stop=ROOT/'Outputs/Experiments/Combined_Channel_Revision_001/external/weather_gbs_extend_v011/STOP_REQUESTED.json'
    automation=Path('C:/Users/SDR_HH/.codex/automations/tvt/automation.toml')
    assert sha(stop)==status['stop_sha256'] and sha(automation)==status['automation_sha256']
    itus={'predictions_local.parquet':'8efbb2d5981c0405940bfddbd7e7c6a3cdce4f62577f92efff46b1f6b0b81cae',
          'predictions_uk.parquet':'f91f9c7831a012ce29043eb477e54d4a4ddb5a5b4749f3558c081c2439e41408'}
    for n,h in itus.items():assert sha(ROOT/'Outputs/Experiments/Combined_Channel_Revision_011'/n)==h
    itucode=[p for p in guard if 'Code\\Combined_Channel_Experiments\\Revision_011\\' in p]
    assert len(itucode)==5
    links=[]
    for p in OUT.glob('*.md'):
        for href in re.findall(r'\]\(([^)]+)\)',p.read_text(encoding='utf-8')):
            if href.startswith(('http://','https://','#')):continue
            q=(p.parent/href).resolve()
            assert q.is_file() or q.name in ('delivery_manifest.json','final_verification.json','model_status.json'),(p,href)
            links.append({'document':p.name,'target':href})
    lockpath=OUT/'execution.lock'; lock_sha=sha(lockpath);lock_size=lockpath.stat().st_size
    with lockpath.open('r+b') as lock:
        msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        try:
            emit('model_status.json',status)
            after={p:sha(p) for p in guard};assert after==guard;emit('seal_sources_after.json',after)
            emit('seal_command.json',{'utc':now(),'pid':os.getpid(),'argv':sys.argv,'executable':sys.executable,'python':sys.version,'mode':'input_audit_seal_only'})
            emit('delivery_qa.json',{'utc':now(),'status':'PASS_INPUT_AUDIT_ONLY','protected_sources':len(guard),
                'source_changes':0,'revision023_sealed_artifacts_unchanged':34,
                'revision011_calculation_sources_unchanged':5,'revision011_prediction_sha256':itus,
                'independent_qa_sha256':sha(OUT/'independent_qa.json'),'unit_tests':10,
                'checked_links':links,'new_real_results':0,'pending_local_decision':True,
                'new_weather_authorization':False,'all_seven_complete':False})
            files={}
            for p in sorted([*CODE.iterdir(),*OUT.iterdir()]):
                if not p.is_file():continue
                files[str(p.relative_to(ROOT))]={'sha256':lock_sha if p==lockpath else sha(p),'bytes':lock_size if p==lockpath else p.stat().st_size}
            emit('delivery_manifest.json',{'utc':now(),'files':files,'self_and_final_excluded':True})
        finally:
            lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1)
    emit('final_verification.json',{'utc':now(),'status':'PASS','phase':'UK_INPUT_CENSUS_AND_RAIN_CONTEXT_VERIFIED',
        'sealed_files':len(files),'manifest_sha256':sha(OUT/'delivery_manifest.json'),
        'protected_sources':len(guard),'source_changes':0,'lock':'RELEASED',
        'new_predictions':0,'new_scores':0,'all_seven_complete':False,
        'pending_local_adaptation_decision':True,'weather_downloads':0,'automation_modified':False,
        'automation_paused':False,'mts_status':'NOT_COMPUTED_USER_DECISION'})
    print(json.dumps(read(OUT/'final_verification.json'),ensure_ascii=False))


if __name__=='__main__':main()
