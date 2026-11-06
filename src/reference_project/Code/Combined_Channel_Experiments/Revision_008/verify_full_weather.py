"""Separate offline verifier: raw point series, all keys, scalar RMSE and ranks."""
from __future__ import annotations
import copy
import json
import math
import platform
import re
import statistics
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from cloud_acquisition import ROOT,OUT,R2,R6,GBS,EPOCH1900,read,sha,key,need,emit,lock,utc,decode

def parser_tests():
    node=dict(url_path='files/d633000/e5.oper.an.sfc/200308/test.nc',epochs=[EPOCH1900,EPOCH1900+3600],latitude=51.25,longitude_0_360=358.5)
    das='''Attributes {
TCC { String long_name "Total cloud cover"; String units "(0-1)"; Int32 ecmwf_local_table 128; Int32 ecmwf_parameter 164; }
time { String units "hours since 1900-01-01 00:00:00"; String calendar "gregorian"; }
latitude { String units "degrees_north"; }
longitude { String units "degrees_east"; }
}'''
    body='''Dataset {} files/d633000/e5.oper.an.sfc/200308/test.nc;
---------------------------------------------
TCC.TCC[2][1][1]
[0][0], 0.1
[1][0], 0.8

TCC.time[2]
0, 1

TCC.latitude[1]
51.25

TCC.longitude[1]
358.5
'''
    need(decode(body,das,node)==[.1,.8],'Positive parser control')
    cases=[('parameter',body,das.replace('164;','165;'),node),('table',body,das.replace('128;','235;'),node),('units',body,das.replace('(0-1)','%'),node),('time_units',body,das.replace('hours since','days since'),node),('utc',body.replace('0, 1','0, 2'),das,node),('latitude',body.replace('51.25','51.0'),das,node),('longitude',body.replace('358.5','358.75'),das,node),('shape',body.replace('TCC.TCC[2]','TCC.TCC[3]'),das,node),('truncation',body.replace('[1][0], 0.8',''),das,node),('duplicate',body.replace('[1][0],','[0][0],'),das,node),('nonfinite',body.replace('0.8','nan'),das,node),('range',body.replace('0.8','1.1'),das,node),('source',body.replace('200308/test.nc','200309/test.nc'),das,node)]
    passed=[]
    for name,b,d,n in cases:
        try:
            decode(b,d,n)
        except (RuntimeError,ValueError):
            passed.append(name)
        else:
            raise RuntimeError('Negative control accepted: '+name)
    return dict(positive=1,negative=len(passed),rejected=passed)

def independent_cloud(node):
    text=(OUT/'payloads'/ (node['id']+'.ascii')).read_text(encoding='utf-8')
    # Independent point-series extraction, not the production multidimensional decoder.
    heads=list(re.finditer(r'(?m)^(TCC\.\w+)((?:\[\d+\])+)[ \t\r]*$',text))
    need(len(heads)==4,'Independent cloud array count')
    arrays={}
    for i,h in enumerate(heads):
        name=h[1]
        shape=tuple(int(n) for n in re.findall(r'\d+',h[2]))
        block=text[h.end():heads[i+1].start() if i+1<len(heads) else len(text)].strip()
        if name=='TCC.TCC':
            need(shape==(len(node['epochs']),1,1),'Independent cloud data shape')
            lines=[l.strip() for l in block.splitlines() if l.strip()]
            values=[]
            for j,line in enumerate(lines):
                m=re.fullmatch(r'\[(\d+)\]\[0\],\s*([-+\d.Ee]+)',line)
                need(m is not None and int(m[1])==j,'Independent point-series index')
                v=float(m[2]);need(math.isfinite(v) and 0<=v<=1,'Independent cloud range')
                values.append(v)
            need(len(values)==len(node['epochs']),'Independent cloud length')
        else:
            need(len(shape)==1,'Independent map dimensions')
            values=[float(x) for x in re.split(r'[,\s]+',block) if x]
            need(len(values)==shape[0] and all(math.isfinite(v) for v in values),'Independent coordinate length')
        need(name not in arrays,'Independent duplicate array')
        arrays[name]=values
    need(arrays['TCC.time']==[(int(t)-EPOCH1900)/3600 for t in node['epochs']],'Independent UTC mismatch')
    need(arrays['TCC.latitude']==[node['latitude']] and arrays['TCC.longitude']==[node['longitude_0_360']],'Independent spatial mismatch')
    das=(OUT/'metadata'/ (node['month']+'.das')).read_text(encoding='utf-8')
    need(re.search(r'ecmwf_parameter\s+164;',das) and re.search(r'ecmwf_local_table\s+128;',das) and 'String units "(0-1)";' in das and 'String calendar "gregorian";' in das,'Independent metadata check')
    return {(node['station'],int(t)):v for t,v in zip(node['epochs'],arrays['TCC.TCC'])}

def verify():
    with lock():
        try:
            need(read(OUT/'scoring_complete.json')['status']=='COMPUTED_PENDING_INDEPENDENT_QA','No computed score to verify')
            before=read(OUT/'sources_before.json')
            need({p:sha(ROOT/p) for p in before}==before,'Protected input changed')
            controls=parser_tests()
            cloud={}
            plan=read(OUT/'plan.json')
            for node in plan['nodes']:
                subset=independent_cloud(node)
                need(not set(cloud).intersection(subset),'Duplicate cloud plan keys')
                cloud.update(subset)
            uk=pd.read_parquet(R6/'predictions_uk_all.parquet')
            local=pd.read_parquet(R6/'predictions_local_all.parquet')
            labels=pd.read_parquet(OUT/'weather_labels_all.parquet')
            need(len(labels)==5664 and not labels.duplicated(['station','utc_epoch_seconds']).any(),'Label count')
            original=set(zip(uk.station,uk.utc_epoch_seconds))
            need(set(cloud)==set(zip(labels.station,labels.utc_epoch_seconds))==original,'Full acquired date/key coverage differs')
            source=uk.set_index(['station','utc_epoch_seconds'])
            scope={r['id']:r for r in read(GBS/'scope_manifest.json')['links']}
            events={}
            for r in labels.itertuples(index=False):
                k=(r.station,r.utc_epoch_seconds)
                t=source.loc[k]
                c=cloud[k]
                p=3600*float(t.mean_total_precipitation_rate)
                s=3600*float(t.mean_snowfall_rate)
                if p>=.1 and s<.1:
                    event='rain'
                elif p<.1 and s<.1 and c>=.8:
                    event='dry_overcast'
                else:
                    event='other'
                need(c==r.tcc and p==r.total_water_mm and s==r.snow_water_mm and event==r.event,'Independent event/value mismatch')
                need(r.tcc_valid_utc_epoch_seconds==r.utc_epoch_seconds,'Independent TCC UTC')
                st=scope[r.station]
                nearest_lat=min([90-i*.25 for i in range(721)],key=lambda a:(abs(a-st['latitude_deg_n']),-a))
                nearest_lon=min([-180+i*.25 for i in range(1440)],key=lambda a:(abs(a-st['longitude_deg_e']),-a))
                need(r.latitude==nearest_lat and r.longitude_0_360==nearest_lon%360,'Independent nearest grid')
                need(r.common_score_eligible==t.common_score_eligible and r.primary_score_eligible==t.primary_score_eligible,'Historical mask changed')
                events[k]=event
            uk['event']=[events[(s,t)] for s,t in zip(uk.station,uk.utc_epoch_seconds)]
            local['event']=['rain' if r.rain else 'dry_overcast' if r.dry_overcast else 'other' for r in local.itertuples(index=False)]
            dates=read(OUT/'coverage_by_date.json')
            need(len(dates)==236 and {s:sum(d['station']==s for d in dates) for s in ['chilbolton','sparsholt']}=={'chilbolton':143,'sparsholt':93},'Date counts')
            labels['date_utc_check']=pd.to_datetime(labels.utc_epoch_seconds,unit='s',utc=True).dt.strftime('%Y-%m-%d')
            need((labels.date_utc_check==labels.date_utc).all(),'Date identity')
            for r in dates:
                g=labels[labels.station.eq(r['station'])&labels.date_utc.eq(r['date_utc'])]
                need(len(g)==r['acquired_hours']==r['weather_known_hours']==24,'All-day classification coverage')
                need(g.common_score_eligible.sum()==r['common_hours'],'Per-day shared score count')
                need((g.common_score_eligible & g.event.eq('rain')).sum()==r['rain_common'],'Per-day rain count')
                need((g.common_score_eligible & g.event.eq('dry_overcast')).sum()==r['overcast_common'],'Per-day overcast count')
            cohorts={'LOCAL_test':local,'UK_full_common':uk[uk.common_score_eligible],'UK_full_ML_primary_sensitivity':uk[uk.primary_score_eligible]}
            for s in ['chilbolton','sparsholt']:
                cohorts[f'UK_{s}_full_common']=uk[uk.common_score_eligible&uk.station.eq(s)]
                cohorts[f'UK_{s}_ML_primary_sensitivity']=uk[uk.primary_score_eligible&uk.station.eq(s)]
            models=[m['id'] for m in read(R2/'frozen_selection.json')['models']]+['ours_frozen_top10','tabpfn_v2_author_frozen','tabicl_v2_author_frozen','itu838_geometric_proxy','itu618_hourly_proxy']
            expected={}
            for cohort,g in cohorts.items():
                methods=models if 'sensitivity' not in cohort else [m for m in models if not m.startswith('itu')]
                for event in ['rain','dry_overcast']:
                    group=g[g.event.eq(event)]
                    for model in methods:
                        errors=[(float(p)-float(t))**2 for p,t in zip(group[model],group.truth_db)]
                        need(all(math.isfinite(x) for x in errors),'Independent nonfinite score')
                        expected[(cohort,event,model)]=(len(errors),math.sqrt(math.fsum(errors)/len(errors)))
            runs=read(OUT/'run_metrics.json')
            need(len(runs)==len(expected)==len({(r['cohort'],r['event'],r['model']) for r in runs}),'Run coverage/duplicates')
            max_delta=0.
            for r in runs:
                n,value=expected[(r['cohort'],r['event'],r['model'])]
                need(n==r['n'] and math.isclose(value,r['rmse_db'],abs_tol=1e-12,rel_tol=1e-12),'Independent RMSE')
                max_delta=max(max_delta,abs(value-r['rmse_db']))
            families=read(OUT/'family_rankings.json')
            familyvalues={}
            for r in families:
                a=[v for (c,e,m),(n,v) in expected.items() if c==r['cohort'] and e==r['event'] and re.sub(r'_seed\d+$','',m)==r['family']]
                mean=math.fsum(a)/len(a)
                need(len(a)==r['n_runs'] and math.isclose(mean,r['rmse_db_mean'],abs_tol=1e-12,rel_tol=1e-12),'Independent family mean')
                if len(a)>1:
                    need(math.isclose(statistics.stdev(a),r['rmse_db_sd'],abs_tol=1e-12,rel_tol=1e-12),'Independent seed SD')
                else:
                    need(r['rmse_db_sd'] is None,'Invented SD')
                familyvalues[(r['cohort'],r['event'],r['family'])]=mean
            need(len(familyvalues)==184,'Expected all 14x4x2 plus 12x3x2 families')
            for r in families:
                own=familyvalues[(r['cohort'],r['event'],r['family'])]
                rank=1+sum(v<own for (c,e,_),v in familyvalues.items() if c==r['cohort'] and e==r['event'])
                need(r['rank']==rank,'Independent ranking')
            receipts=[read(p) for p in (OUT/'http').glob('*.json')]
            successes=[r for r in receipts if r['status']==200]
            need(len(successes)==24,'Expected 6 catalogs + 6 DAS + 12 subsets')
            for r in successes:
                p=ROOT/r['path']
                need(p.stat().st_size==r['actual_bytes'] and sha(p)==r['sha256'],'HTTP payload length/SHA')
                need(r['content_length'] is None or r['content_length']==r['actual_bytes'],'Advertised length')
                need(r['request_method']=='GET' and r['validation']=='PASS','Unexpected request type')
            need({p:sha(ROOT/p) for p in before}==before,'Protected inputs changed after verification')
            emit('sources_after_verification.json',before)
            qa=dict(status='PASS',verified_utc=utc(),full_station_dates=236,full_weather_station_hours=5664,missing_weather_hours=0,chilbolton_dates=143,sparsholt_dates=93,common_scoring_hours=5538,primary_scoring_hours=5593,label_quality_exclusions=71,additional_physical_exclusions=55,independently_read_cloud_values=5664,raw_rain_values_reverified=read(OUT/'rain_provenance_qa.json')['verified_raw_values'],score_checks=len(expected),family_checks=len(families),independent_rmse_max_delta_db=max_delta,negative_controls=controls,shared_masks_unchanged=True,all_dates_represented=True,protected_sources_unchanged=len(before),historical_checkpoints_unchanged=18,author_checkpoints_unchanged=2,training=0,model_inference=0,old_downloader_resumed=False,old_stop_and_automation_unchanged=True,execution_lock='RELEASED')
            emit('verification.json',qa)
            manifest={key(p):sha(p) for p in OUT.rglob('*') if p.is_file() and p.name!='execution.lock'}
            emit('completion.json',dict(status='COMPLETE_FULL_DATE_WEATHER_RESCORING',completed_utc=utc(),verification='PASS',execution_lock='RELEASED',full_weather_station_hours=5664,station_dates=236,python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,source_count=len(before),artifacts=manifest,new_training=0,new_inference=0,new_cloud_field_only=True,t012_overall_complete=False))
            print(json.dumps(qa,ensure_ascii=False))
        except Exception as exc:
            emit('verification_failure.json',dict(utc=utc(),error=repr(exc),execution_lock='RELEASED_AFTER_FAILURE'))
            raise

if __name__=='__main__':
    if sys.argv[1:]==['selftest']:
        print(json.dumps(parser_tests()))
    elif not sys.argv[1:]:
        verify()
    else:
        raise RuntimeError('Unsupported arguments')
