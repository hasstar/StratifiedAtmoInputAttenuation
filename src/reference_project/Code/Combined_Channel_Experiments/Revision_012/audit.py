"""Offline, append-only frozen-prediction audit. No ML or network use."""
from pathlib import Path
import contextlib
import hashlib
import json
import os
import platform
import socket
import sys
from datetime import datetime, timezone

for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '2'
import numpy as np
import pandas as pd

CODE = Path(__file__).resolve().parent
ROOT = CODE.parents[2]
OUT = ROOT / 'Outputs/Experiments/Combined_Channel_Revision_012'
REV = lambda n: ROOT / f'Outputs/Experiments/Combined_Channel_Revision_{n:03d}'
RAW = Path('E:/0_Workflow_System/01_PhD/Data/era5_campaign_20250930_20260820/raw')
AUTO = Path('C:/Users/SDR_HH/.codex/automations/tvt/automation.toml')
STOP = REV(1) / 'external/weather_gbs_extend_v011/STOP_REQUESTED.json'
PDF = Path('C:/Users/SDR_HH/Desktop/Dynamical_Link_Budget_in_Satellite_Communications_at_Ka-Band_Testing_Radiometeorological_Forecasts_With_Hayabusa2_Deep-Space_Mission_Support_Data.pdf')
FAMILIES = {
    'ours_frozen_top10': ('本文冻结 Top10', ['ours_frozen_top10']),
    'bai2022_lstm_t0_adapted': ('Bai LSTM（既有 LOCAL 适配）', [f'bai2022_lstm_t0_adapted_seed{s}' for s in (17,29,43)]),
    'tabm2025_official_budgeted': ('TabM（既有 LOCAL 适配）', [f'tabm2025_official_budgeted_seed{s}' for s in (17,29,43)]),
    'itu_combined_geometric_approx': ('组合 ITU 近似', ['itu_combined_geometric_approx']),
}
BLOCKED = {'esst_hourly_adaptation': 'E-SST（小时输入适配待完成）',
           'rmop_atmospheric_branch': 'RMOP（大气路径分支待完成）',
           'nwp_prm_mts_chain': 'NWP–传播模型链（待完成）'}
MODELS = [m for _, runs in FAMILIES.values() for m in runs]
EVENTS = ('rain', 'dry_overcast')

def need(ok, message):
    if not ok: raise ValueError(message)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''): h.update(b)
    return h.hexdigest()

def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def now(): return datetime.now(timezone.utc).isoformat()

def emit(name, value):
    p = OUT / name
    need(p.resolve().is_relative_to(OUT.resolve()), 'Output escaped new version')
    with p.open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')

@contextlib.contextmanager
def lock():
    import msvcrt
    with (OUT/'execution.lock').open('a+b') as f:
        if f.tell() == 0: f.write(b'0'); f.flush()
        f.seek(0); msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        try: yield
        finally: f.seek(0); msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)

def offline():
    def deny(*a, **k): raise RuntimeError('This audit is offline')
    socket.socket.connect = deny
    socket.create_connection = deny

def check_guard(guard):
    for p, h in guard.items(): need(sha(ROOT/p) == h, f'Changed protected source: {p}')

def protect():
    guard = read(REV(11)/'sources_before.json')
    old_auto = guard.pop(str(AUTO))
    need(old_auto == '10f9dcb0e27d52c6c8494502760027655634e1d008a29440eef174b85b8b0f45', 'Unexpected historical automation guard')
    check_guard(guard)
    need(sha(AUTO) == 'c1de99bd5f18d721fb39f23362ccda4b9fbe690a38839fbfb67b7bd4cc258f8e', 'Unexpected automation change')
    need(sha(STOP) == '0d629e054904ca58dde38630af24899bf89620607410da97a9b49f259253864a', 'STOP changed')
    for p in list(REV(11).rglob('*')) + list((ROOT/'Code/Combined_Channel_Experiments/Revision_011').rglob('*')):
        if p.is_file() and p.suffix not in ('.lock', '.pyc'):
            guard[str(p)] = sha(p)
    extra = [AUTO, STOP, PDF, RAW/'download_manifest.json', RAW/'pressure_temperature_20251001_20251031.nc',
             REV(2)/'frozen_selection.json', ROOT/'Code/Combined_Channel_Experiments/Revision_002/MODEL_REGISTRY.md']
    selected = read(REV(2)/'frozen_selection.json')['models']
    for rec in selected:
        if rec['id'] in MODELS:
            p = REV(2)/rec['path']; need(sha(p)==rec['sha256'], 'Frozen checkpoint mismatch'); extra.append(p)
    extra += list(CODE.glob('*'))
    for p in extra:
        if p.is_file(): guard[str(p)] = sha(p)
    return guard

def score(y, p):
    y=np.asarray(y,dtype=float); p=np.asarray(p,dtype=float)
    need(y.ndim==1 and p.shape==y.shape and len(y)>0,'Invalid score shapes')
    need(np.isfinite(y).all() and np.isfinite(p).all(),'Nonfinite score inputs')
    e=p-y
    return {'n':len(y),'rmse_db':float(np.sqrt(np.mean(e*e))),
            'mae_db':float(np.mean(np.abs(e))),'bias_db':float(np.mean(e))}

def unique(d, keys): need(not d.duplicated(keys).any(), 'Duplicate sample identity')

def cohort_frames(local, uk):
    yield 'LOCAL_test', local
    yield 'UK_full_common', uk[uk.common_score_eligible]
    for station in ('chilbolton','sparsholt'):
        yield f'UK_{station}_full_common', uk[uk.common_score_eligible & uk.station.eq(station)]

def dates(d):
    return {s: sorted(g.date.unique().tolist()) for s,g in d.groupby('station')}

def input_audit(raw_local, uk):
    candidates = ['surface__zero_degree_level','p700__u_component_of_wind','p700__v_component_of_wind',
                  'surface__mean_convective_precipitation_rate','surface__mean_large_scale_precipitation_rate',
                  'surface__precipitation_type','surface__mean_total_precipitation_rate','surface__mean_snowfall_rate']
    fields=[]
    for domain,d in [('LOCAL_test',raw_local),('UK_full_completed',uk)]:
        for c in candidates:
            actual=c
            if domain.startswith('UK') and c.startswith('surface__mean_'): actual=c.removeprefix('surface__')
            present=actual in d
            finite=int(np.isfinite(d[actual].to_numpy(float)).sum()) if present else 0
            fields.append({'dataset':domain,'field':c,'actual_column':actual if present else None,
                           'present':present,'n_total':len(d),'n_finite':finite,'n_unavailable':len(d)-finite})
    manifest=read(RAW/'download_manifest.json')['campaign']
    variables=manifest['pressure_variables']
    return {'scope':'Frozen full-cohort tables and named LOCAL raw campaign; not an exhaustive search of all user storage.',
            'fields':fields,'local_raw_pressure_levels_hpa':manifest['pressure_levels_hpa'],
            'local_raw_pressure_variables':variables,'local_raw_sampling':manifest['sampling'],
            'local_raw_files_by_variable':{v:[p.name for p in sorted(RAW.glob(f'pressure_{v}_*.nc'))] for v in variables},
            'raw_grid_header_checked':{'file':'pressure_temperature_20251001_20251031.nc','hours':744,'levels':11,'latitude':4,'longitude':2,'units':'K','source_kind':'ERA5 analysis'},
            'raw_full_values_and_cross_field_alignment_qa':'NOT_RUN',
            'local_table_caution':'Random test hours are not contiguous storm input. Point LOS feature columns are not a vertical column. Raw small-area grids may support a separately verified reanalysis adaptation.',
            'uk_esst_candidate_missing_fields':['700hPa_u_m_per_s','700hPa_v_m_per_s','hourly_zero_C_height_m_AMSL','convective_precipitation_rate','large_scale_precipitation_rate'],
            'uk_candidate_station_field_hours':5*len(uk),'candidate_request_count':None,
            'acquisition_authorized':False,'candidate_note':'28,320 is a logical input-value inventory, not network request count. Exact source identities, time convention, static orography if h0 is AGL, event classification and padding must be fixed before asking for a download.',
            'rmop_missing':['verified GSDSU configuration/coefficients','compatible 3D thermodynamics and five hydrometeor fields, including graupel','verified path integration and fixed-reference mapping'],
            'nwp_missing':['paper PRM scattering coefficients at required frequencies and polarization','UK vertical thermodynamic/cloud/rain profiles','licensed original Italsat event library and MTS implementation/configuration'],
            'training':0,'weather_gets':0}

def main():
    offline()
    with lock():
        need(not (OUT/'calculation_complete.json').exists(),'Already calculated; do not rerun')
        guard=protect(); emit('sources_before.json',guard)
        emit('launch.json',{'utc':now(),'pid':os.getpid(),'command':sys.argv,'python':sys.version,
                           'platform':platform.platform(),'numpy':np.__version__,'pandas':pd.__version__,
                           'training':0,'inference':0,'weather_gets':0})
        local=pd.read_parquet(REV(11)/'predictions_local.parquet')
        uk=pd.read_parquet(REV(11)/'predictions_uk.parquet')
        raw_local=pd.read_parquet(ROOT/'Outputs/Datasets/Signal_Detectability_Revision_003/test.parquet')
        labels=pd.read_parquet(REV(8)/'weather_labels_all.parquet')
        raw_uk=pd.read_parquet(REV(10)/'expanded_dataset_with_itu_inputs.parquet')
        unique(local,['hour_bjt']); unique(uk,['station','utc_epoch_seconds'])
        unique(raw_uk,['station','utc_epoch_seconds']); unique(labels,['station','utc_epoch_seconds'])
        need(local.hour_bjt.equals(raw_local.hour_bjt),'LOCAL original test keys/order')
        need(np.array_equal(local.truth_db.to_numpy(),raw_local.clear_sky_attenuation_db.to_numpy()),'LOCAL truth')
        keys=['station','utc_epoch_seconds']
        a=uk.sort_values(keys).reset_index(drop=True)
        for source in (labels,raw_uk):
            b=source.sort_values(keys).reset_index(drop=True)
            need(a[keys].equals(b[keys]),'UK full key population changed')
        b=labels.sort_values(keys).reset_index(drop=True)
        for c in ('event','primary_score_eligible','physical_input_valid','common_score_eligible'):
            need(a[c].equals(b[c]),'UK mask/classification changed: '+c)
        need(len(local)==687 and len(uk)==5664,'Population changed')
        need(int(uk.primary_score_eligible.sum())==5593 and int(uk.common_score_eligible.sum())==5538,'Eligibility changed')
        need(np.array_equal(uk.common_score_eligible,uk.primary_score_eligible & uk.physical_input_valid),'Common mask logic')
        need((uk.hour_utc.astype('int64')//10**9).eq(uk.utc_epoch_seconds).all(),'UTC identity')
        need((uk.rain_valid_utc_end-uk.hour_utc).eq(pd.Timedelta(hours=1)).all(),'Rain interval identity')
        for d in (local,uk):
            need(np.isfinite(d.loc[d.common_score_eligible,MODELS].to_numpy()).all(),'Missing eligible prediction')
        local['station']='LOCAL'; local['date']=local.hour_bjt.dt.strftime('%Y-%m-%d'); local['time']=local.hour_bjt.astype(str)
        uk['date']=uk.hour_utc.dt.strftime('%Y-%m-%d'); uk['time']=uk.hour_utc.astype(str)
        need(uk.groupby('station').date.nunique().to_dict()=={'chilbolton':143,'sparsholt':93},'Complete-date coverage')
        need(uk.groupby(['station','date']).size().eq(24).all(),'Partial source date included')
        runs=[]; families=[]; membership=[]
        for cohort,frame in cohort_frames(local,uk):
            for event in EVENTS:
                d=frame[frame.event.eq(event)]
                coverage=dates(d)
                for row in d.itertuples(): membership.append({'cohort':cohort,'event':event,'station':row.station,'time':row.time})
                for fam,(name,models) in FAMILIES.items():
                    metrics=[]
                    for model in models:
                        rec={'cohort':cohort,'event':event,'family':fam,'model':model,**score(d.truth_db,d[model])}
                        runs.append(rec); metrics.append(rec)
                    rec={'cohort':cohort,'event':event,'family':fam,'name':name,'status':'SCORED_FROZEN',
                         'n':len(d),'n_runs':len(metrics),'dates_by_station':coverage,'rank':None}
                    for measure in ('rmse_db','mae_db','bias_db'):
                        vals=[m[measure] for m in metrics]
                        rec[measure+'_mean']=float(np.mean(vals)); rec[measure+'_sd']=float(np.std(vals,ddof=1)) if len(vals)>1 else None
                    families.append(rec)
                for fam,name in BLOCKED.items():
                    families.append({'cohort':cohort,'event':event,'family':fam,'name':name,'status':'BLOCKED_NOT_COMPUTED',
                                     'n':0,'n_runs':0,'eligible_target_n':len(d),'missing_prediction_n':len(d),'rank':None,
                                     'rmse_db_mean':None,'mae_db_mean':None,'bias_db_mean':None,'dates_by_station':{}})
        for r in families:
            if r['status']=='SCORED_FROZEN':
                r['rank']=1+sum(x['status']=='SCORED_FROZEN' and x['cohort']==r['cohort'] and x['event']==r['event'] and x['rmse_db_mean']<r['rmse_db_mean'] for x in families)
        ledger=[]; coverage=[]
        for domain,d in [('LOCAL_test',local),('UK_full_completed',uk)]:
            for row in d.itertuples():
                eligible=bool(row.common_score_eligible)
                reason='' if eligible else str(row.exclusion_reason) if not row.primary_score_eligible else 'negative_liquid_rain_estimate'
                ledger.append({'dataset':domain,'station':row.station,'time':row.time,'date':row.date,'event':row.event,
                               'primary_score_eligible':bool(getattr(row,'primary_score_eligible',True)),
                               'physical_input_valid':bool(getattr(row,'physical_input_valid',True)),
                               'common_score_eligible':eligible,'exclusion_reason':reason,
                               'truth_finite':bool(np.isfinite(row.truth_db))})
            for (station,date),g in d.groupby(['station','date']):
                coverage.append({'dataset':domain,'station':station,'date':date,'source_hours':len(g),
                                 'common_hours':int(g.common_score_eligible.sum()),
                                 'event_hours_all':{e:int(g.event.eq(e).sum()) for e in ('rain','dry_overcast','other')},
                                 'event_hours_scored':{e:int((g.event.eq(e)&g.common_score_eligible).sum()) for e in EVENTS}})
        for name,d,cols in [('selected_local.parquet',local,['hour_bjt']),('selected_uk.parquet',uk,keys+['hour_utc'])]:
            dest=OUT/name; need(not dest.exists(),'No overwrite')
            kept=list(dict.fromkeys(cols+['station','time','date','event','truth_db','common_score_eligible']+MODELS))
            with dest.open('xb') as f: d[kept].to_parquet(f,index=False)
        emit('run_metrics.json',runs); emit('seven_family_rankings.json',families)
        emit('sample_membership.json',membership); emit('all_hour_ledger.json',ledger); emit('coverage_by_date.json',coverage)
        emit('input_availability.json',input_audit(raw_local,uk))
        emit('dynamic_comparison_status.json',{'status':'BLOCKED_NOT_COMPUTED','full_seven_family_complete':False,
             'reason':'No verified E-SST/RMOP/PRM outputs or original MTS event library; no synthetic result substituted.',
             'permitted_next_design':'Keep hourly slow-component comparison separate. Full MTS requires CCDF/ACF/fade-duration/fade-slope statistical evaluation, not time-aligned second RMSE.',
             'forbidden':'No test-label-fitted target distributions, no full-MTS plus project-fast double counting, no new project fast CDF.',
             'common_fast_hybrid':'Possible separate project-adapted experiment only using the same pre-existing frozen fast component; not performed here.'})
        check_guard(guard)
        emit('calculation_complete.json',{'status':'COMPUTED_PENDING_QA','utc':now(),'protected_sources':len(guard),
             'run_metric_rows':len(runs),'family_rows':len(families),'scored_families':4,'blocked_families':3,
             'all_seven_complete':False,'new_training':0,'new_inference':0,'new_weather_gets':0,
             'output_sha256':{p.name:sha(p) for p in OUT.iterdir() if p.is_file() and p.suffix!='.lock'}})
        print(json.dumps({'run_rows':len(runs),'family_rows':len(families),'ledger_hours':len(ledger),'status':'COMPUTED_PENDING_QA'}))

if __name__=='__main__':
    try: main()
    except Exception:
        import traceback
        path=OUT/('failure_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%f')+'.json')
        with path.open('x',encoding='utf-8') as f: json.dump({'utc':now(),'traceback':traceback.format_exc()},f,indent=2)
        raise
