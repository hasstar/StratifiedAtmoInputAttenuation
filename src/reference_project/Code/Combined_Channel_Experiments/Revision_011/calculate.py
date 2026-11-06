"""Frozen-prediction comparison with the newly authorized physical combination."""
from support import *
from physical import FIELDS, components
import numpy as np
import pandas as pd
import re
import traceback

FAMILIES=['ours_frozen_top10','bai2022_mlp_adapted','bai2022_lstm_t0_adapted','tabm2025_official_budgeted']
PRIMARY='itu_combined_geometric_approx'
SENSITIVITY='itu_combined_p618_hourly_approx'
EVENTS=['rain','dry_overcast']
NAMES={'ours_frozen_top10':'本文冻结 Top10','bai2022_mlp_adapted':'Bai2022 MLP（既有适配版）',
       'bai2022_lstm_t0_adapted':'Bai2022 零历史 LSTM（既有适配版）',
       'tabm2025_official_budgeted':'TabM（既有本地适配版）',
       PRIMARY:'组合 ITU（几何雨路径，固定参考近似）',SENSITIVITY:'组合 ITU（P.618 小时雨路径敏感性）'}

def local_elevation():
    cp=np.cos(np.deg2rad(39.9))*np.cos(np.deg2rad(125-116.3))
    return float(np.rad2deg(np.arctan2(cp-6371/(6371+35786),np.sqrt(1-cp*cp))))

def score(y,p):
    need(len(y)>0 and np.isfinite(y).all() and np.isfinite(p).all(),'Invalid scoring sample')
    e=p-y
    return dict(n=len(e),rmse_db=float(np.sqrt(np.mean(e*e))),mae_db=float(np.mean(abs(e))),bias_db=float(np.mean(e)))

def main():
    offline()
    with lock():
        before=read(OUT/'sources_before.json'); check(before)
        refmanifest=read(OUT/'references_complete.json')
        for r in refmanifest['records']: need(sha(OUT/'references'/r['name'])==r['sha256'],'Reference changed')
        emit('calculation_contract.json',dict(utc=utc(),protocol_sha256=sha(CODE/'PROTOCOL.md'),
            code_sha256={p.name:sha(p) for p in CODE.glob('*.py')},primary=PRIMARY,sensitivity=SENSITIVITY,
            baseline_label_fitting=False,scintillation_included=False,model_training=False))
        local=pd.read_parquet(REV(6)/'predictions_local_all.parquet')
        local_raw=pd.read_parquet(ROOT/'Outputs/Datasets/Signal_Detectability_Revision_003/test.parquet')
        local=local.merge(local_raw[['hour_bjt']+FIELDS],on='hour_bjt',validate='one_to_one',how='left')
        local['event']=np.select([local.rain,local.dry_overcast],EVENTS,default='other')
        local['common_score_eligible']=True
        uk=pd.read_parquet(REV(6)/'predictions_uk_all.parquet')
        raw=pd.read_parquet(REV(10)/'expanded_dataset_with_itu_inputs.parquet')
        labels=pd.read_parquet(REV(8)/'weather_labels_all.parquet')
        keys=['station','utc_epoch_seconds']
        uk=uk.merge(raw[keys+FIELDS],on=keys,how='left',validate='one_to_one')
        uk=uk.merge(labels[keys+['event','tcc']],on=keys,how='left',validate='one_to_one')
        need(len(local)==687 and len(uk)==5664 and int(uk.common_score_eligible.sum())==5538,'Fixed sample population changed')
        params=read(REV(4)/'station_parameters.json')
        references={}
        result,references['LOCAL']=components(local,19.13,local_elevation(),0.)
        for k,v in result.items(): local[k]=v
        for station,g in uk.groupby('station'):
            s=params[station]
            result,references[station]=components(g,20.7,s['elevation_deg'],s['receiver_km_amsl'])
            for k,v in result.items(): uk.loc[g.index,k]=v
        models=['ours_frozen_top10']+[f'{name}_seed{seed}' for name in FAMILIES[1:] for seed in (17,29,43)]
        for frame in [local,uk]:
            frame[PRIMARY]=frame.gas_relative_db+frame.cloud_db+frame.itu838_geometric_proxy
            frame[SENSITIVITY]=frame.gas_relative_db+frame.cloud_db+frame.itu618_hourly_proxy
            need(np.isfinite(frame.loc[frame.common_score_eligible,models+[PRIMARY,SENSITIVITY]].to_numpy()).all(),'Nonfinite prediction on common mask')
        local.to_parquet(new(OUT/'predictions_local.parquet'),index=False)
        uk.to_parquet(new(OUT/'predictions_uk.parquet'),index=False)
        emit('fixed_reference_parameters.json',references)
        oldmetrics={(r['cohort'],r['event'],r['model']):r for r in read(REV(8)/'run_metrics.json')}
        cohorts=[('LOCAL_test',local),('UK_full_common',uk[uk.common_score_eligible])]
        cohorts += [(f'UK_{s}_full_common',uk[uk.common_score_eligible&uk.station.eq(s)]) for s in ('chilbolton','sparsholt')]
        runs=[]; memberships=[]; summaries=[]
        for cohort,frame in cohorts:
            for event in EVENTS:
                d=frame[frame.event.eq(event)]
                for r in d.itertuples(): memberships.append(dict(cohort=cohort,event=event,station=getattr(r,'station','LOCAL'),hour=str(getattr(r,'hour_bjt',getattr(r,'hour_utc',None)))))
                y=d.truth_db.to_numpy(float)
                for model in models+[PRIMARY,SENSITIVITY]:
                    row=dict(cohort=cohort,event=event,model=model,family=re.sub(r'_seed\d+$','',model),**score(y,d[model].to_numpy(float)))
                    if model in models:
                        old=oldmetrics[(cohort,event,model)]
                        need(old['n']==row['n'] and np.isclose(old['rmse_db'],row['rmse_db'],atol=1e-12,rtol=1e-12),'Frozen prior metric changed')
                    runs.append(row)
                summaries.append(dict(cohort=cohort,event=event,n=len(d),
                    mean_abs_gas_change_db=float(d.gas_relative_db.abs().mean()),mean_cloud_db=float(d.cloud_db.mean()),
                    mean_rain_db=float(d.itu838_geometric_proxy.mean()),negative_combined_hours=int((d[PRIMARY]<0).sum())))
        family=[]
        for (cohort,event,name),g in pd.DataFrame(runs).groupby(['cohort','event','family'],sort=False):
            family.append(dict(cohort=cohort,event=event,family=name,name=NAMES[name],n=int(g.n.iloc[0]),n_runs=len(g),
                rmse_db_mean=float(g.rmse_db.mean()),rmse_db_sd=float(g.rmse_db.std(ddof=1)) if len(g)>1 else None,
                mae_db_mean=float(g.mae_db.mean()),bias_db_mean=float(g.bias_db.mean()),primary_roster=name!=SENSITIVITY))
        for r in family:
            r['rank']=None if not r['primary_roster'] else 1+sum(q['primary_roster'] and q['cohort']==r['cohort'] and q['event']==r['event'] and q['rmse_db_mean']<r['rmse_db_mean'] for q in family)
        emit('run_metrics.json',runs); emit('family_rankings.json',family)
        emit('sample_membership.json',memberships); emit('component_diagnostics.json',summaries)
        emit('physical_input_audit.json',dict(local_hours=len(local),uk_hours=len(uk),uk_dates_by_station={s:int(g.hour_utc.dt.date.nunique()) for s,g in uk.groupby('station')},
             local_dewpoint_gt_temperature=int(local.dewpoint_above_temperature.sum()),uk_dewpoint_gt_temperature=int(uk.dewpoint_above_temperature.sum()),
             uk_rain_invalid_retained=int((~uk.physical_input_valid).sum()),new_imputation=0,new_clipping=0,
             local_rain_component='Reused frozen prior LOCAL component; original implementation clipped negative total-minus-snow and used training-frozen fallback for missing rain height. No such parameters refit here.',
             reference='P835-6 low-altitude state and exponential humidity-column approximation, not label-native absolute clear sky'))
        check(before)
        emit('calculation_complete.json',dict(status='COMPUTED_PENDING_INDEPENDENT_QA',utc=utc(),
             protected_sources=len(before),new_training=0,new_ml_inference=0,new_weather_downloads=0,
             physical_rows=len(local)+len(uk),family_rows=len(family),primary_families=5,
             prediction_sha256={n:sha(OUT/n) for n in ['predictions_local.parquet','predictions_uk.parquet']}))
        print(json.dumps([r for r in family if r['cohort'] in ['LOCAL_test','UK_full_common']],ensure_ascii=False,indent=2))

if __name__=='__main__':
    try: main()
    except Exception:
        emit('failures/calculation_'+stamp()+'.json',dict(utc=utc(),error=traceback.format_exc()))
        raise
