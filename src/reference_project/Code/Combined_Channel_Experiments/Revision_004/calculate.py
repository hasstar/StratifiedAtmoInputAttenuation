"""Complete the missing UK physical proxies; reuse saved ML/LOCAL predictions."""
from common import *
from physics import *
import traceback
import netCDF4
import pandas as pd

MAIN=['itu838_geom_circular_nominal','itu618_proxy_circular_nominal']

def native_gate(links):
    sources=read(OLD/'native_label_sources.json');rows=[]
    require(len(sources)==236,'Expected236 native days')
    for item in sources:
        p=ROOT/item['source'];require(sha(p)==item['sha256'],'Native source SHA')
        link=links[item['station']]
        # The Windows netCDF C library cannot open this Unicode path reliably.
        # Exact source bytes in memory avoid renaming/copying or touching the source.
        with netCDF4.Dataset('native_memory',memory=p.read_bytes()) as ds:
            fields={}
            for key in ('frequency','latitude_receiver','longitude_receiver','satellite_position','receiver_altitude_amsl','elevation_angle'):
                v=ds[key];require(not np.ma.getmaskarray(v[:]).any(),'Masked native metadata')
                fields[key]={'value':float(v[:]),'units':v.units}
            pol={key:np.asarray(ds[key][:]).tobytes().decode('ascii').rstrip('\x00 ') for key in ('polarisation_transmit','polarisation_receive')}
            require(all(v=='Left Hand Circularly Polarized' for v in pol.values()),'Native circular polarization gate')
            require('Left Hand Circularly Polarized' in ds.title,'Native title polarization')
            expected={'frequency':(20.7,'GHz',1e-5),'latitude_receiver':(link['latitude_deg_n'],'degrees_north',1e-12),
                      'longitude_receiver':(link['longitude_deg_e']%360,'degrees_east',1e-12),
                      'satellite_position':(-link['satellite_longitude_deg_e'],'degrees_west',1e-12),
                      'receiver_altitude_amsl':(link['height_amsl_m'],'m',1e-12),
                      'elevation_angle':(link['source_coarse_elevation_deg'],'degrees',1e-12)}
            for key,(value,unit,tol) in expected.items():
                require(fields[key]['units']==unit and abs(fields[key]['value']-value)<=tol,'Native identity: '+key)
            rows.append({**item,'title':ds.title,'fields':fields,'polarization':pol})
    emit(OUT/'native_metadata_gate.json',{'status':'PASS','files':len(rows),'polarization':'source-confirmed matched left-hand circular; all236 files','rows':rows})

def masks(frame):
    common=frame.common_score_eligible.to_numpy(bool)
    selections={'UK_all':np.ones(len(frame),bool),'UK_original39':frame.period.eq('original39').to_numpy(),
                'UK_extension2003':frame.period.eq('extension2003').to_numpy()}
    for station in ('chilbolton','sparsholt'):
        selections['UK_'+station]=frame.station.eq(station).to_numpy()
        selections['UK_'+station+'_extension2003']=frame.station.eq(station).to_numpy()&frame.period.eq('extension2003').to_numpy()
    for cohort,selection in selections.items():
        yield cohort,'overall',common&selection
        for band in ('below_0.1','0.1_to_1','at_least_1'):
            yield cohort,'rain_'+band,common&selection&frame.rain_band.eq(band).to_numpy()

def scores(y,p):
    y=np.asarray(y,float);e=np.asarray(p,float)-y
    require(len(e)>0 and np.isfinite(e).all(),'Finite common scoring')
    variance=np.sum((y-np.mean(y))**2)
    return {'n':len(e),'rmse_db':float(np.sqrt(np.mean(e**2))),'mae_db':float(np.mean(np.abs(e))),
            'bias_db':float(np.mean(e)),'r2':float(1-np.sum(e**2)/variance) if variance>0 else None}

def save_metrics(frame,models,scenarios,local):
    registry=read(REV2/'frozen_selection.json')['models']
    families={r['id']:r['family'] for r in registry};families['ours_frozen_top10']='ours_frozen_top10'
    rows=[];counts=[]
    for cohort,group,mask in masks(frame):
        if not mask.any():continue
        d=frame.loc[mask]
        counts.append({'cohort':cohort,'group':group,'n':len(d),'utc_dates':d.hour_utc.dt.date.nunique(),
                       'station_days':len(d.assign(date=d.hour_utc.dt.date)[['station','date']].drop_duplicates())})
        for name in models+list(scenarios):
            physical=name in scenarios
            rows.append({'cohort':cohort,'group':group,'model':name,'family':name if physical else families[name],
                         'comparison':'additional_rain_inputs_physical_proxy' if physical else 'saved_top10_on_common_rain_valid_hours',
                         **scores(d.truth_db,d[name])})
    for group in ('overall','rain','heavy_rain','clear_dry','dry_overcast','other'):
        mask=np.ones(len(local),bool) if group=='overall' else ~(local.rain|local.clear_dry|local.dry_overcast) if group=='other' else local[group].to_numpy(bool)
        d=local.loc[mask]
        for name in models+['physics_geom_circular','physics_p618_hourly_circular']:
            rows.append({'cohort':'LOCAL_test','group':group,'model':name,'family':families.get(name,name),
                         'comparison':'reused_additional_inputs_physical_proxy' if name.startswith('physics_') else 'saved_top10',
                         **scores(d.truth_db,d[name])})
    metrics=pd.DataFrame(rows);metrics.to_csv(newpath(OUT/'metrics.csv'),index=False)
    family=metrics.groupby(['cohort','group','comparison','family'],sort=False).agg(runs=('model','size'),n=('n','first'),
        rmse_mean_db=('rmse_db','mean'),rmse_seed_sd_db=('rmse_db',lambda x:float(x.std(ddof=1)) if len(x)>1 else 0.),
        mae_mean_db=('mae_db','mean'),bias_mean_db=('bias_db','mean')).reset_index()
    family.to_csv(newpath(OUT/'family_summary.csv'),index=False)
    pd.DataFrame(counts).to_csv(newpath(OUT/'cohort_counts.csv'),index=False)
    return metrics

def bootstrap(frame):
    names=['ours_frozen_top10','train_mean']+MAIN;records=[]
    for cohort,group,mask in masks(frame):
        if group!='overall':continue
        d=frame.loc[mask];dates=d.hour_utc.dt.strftime('%Y-%m-%d');unique=sorted(dates.unique())
        errors=d[names].to_numpy(float)-d.truth_db.to_numpy(float)[:,None]
        counts=np.array([(dates==day).sum() for day in unique]);sums=np.array([(errors[dates.eq(day).to_numpy()]**2).sum(0) for day in unique])
        rng=np.random.default_rng(20261016);draws=rng.integers(0,len(unique),size=(2000,len(unique)))
        values=np.sqrt(sums[draws].sum(1)/counts[draws].sum(1)[:,None]);point=np.sqrt(sums.sum(0)/counts.sum())
        for i in (2,3):
            for j in (0,1):
                delta=values[:,i]-values[:,j]
                records.append({'cohort':cohort,'model':names[i],'reference':names[j],'n_dates':len(unique),'n':len(d),
                                'delta_rmse_db':float(point[i]-point[j]),'unadjusted_ci95_db':[float(x) for x in np.quantile(delta,[.025,.975])]})
    emit(OUT/'date_cluster_bootstrap.json',{'draws':2000,'seed':20261016,'cluster':'UTC date; both stations together; exploratory unadjusted, temporal dependence not modelled','negative_favors_model':True,'results':records})

def main():
    offline()
    with lock():
        guard=read(OUT/'source_sha256_before.json');protected_verify(guard)
        require(read(OUT/'references_complete.json')['status']=='PASS','Official reference gate')
        for row in read(OUT/'references_complete.json')['records']:require(sha(OUT/'references'/row['file'])==row['sha256'],'Reference SHA')
        plan=read(ACQ/'plan.json');links={r['id']:r for r in plan['links']};native_gate(links)
        lat,lon,h0=height_grid(OUT/'references/P839_4_components.zip')
        heights={key:{**climate_height(h0,(lat,lon),r['latitude_deg_n'],r['longitude_deg_e']),
                      'receiver_km_amsl':r['height_amsl_m']/1000,'frequency_GHz':20.7,
                      'elevation_deg':r['elevation_deg'],'latitude_deg':r['latitude_deg_n'],
                      'longitude_deg':r['longitude_deg_e'],'polarization':'circular_source_confirmed'} for key,r in links.items()}
        emit(OUT/'station_parameters.json',heights)
        frame=pd.read_parquet(OLD/'predictions_expanded.parquet');local=pd.read_parquet(REV2/'predictions/local_test.parquet')
        models=read(OLD/'evaluation_complete.json')['models']
        require(len(frame)==5664 and len(local)==687 and len(models)==20,'Fixed rows/models')
        rate=3600*(frame.mean_total_precipitation_rate-frame.mean_snowfall_rate)
        require(np.allclose(rate,frame.liquid_estimate_mm,rtol=0,atol=1e-14),'Existing rain conversion agreement')
        frame['rain_liquid_mm_per_h']=rate
        frame['rain_valid_utc_end']=frame.hour_utc+pd.Timedelta(hours=1)
        frame['physical_input_valid']=np.isfinite(rate)&(rate>=0)
        frame['common_score_eligible']=frame.primary_score_eligible&frame.physical_input_valid
        require(int(frame.primary_score_eligible.sum())==5593 and int(frame.common_score_eligible.sum())==5538,'Preregistered common cohort')
        require(int(frame.physical_input_valid.sum())==5608,'Physical-valid candidate count')
        scenarios={};parameters=[]
        for height,offset in HEIGHTS.items():
            for pol in POLARIZATIONS:
                columns={'geom':f'itu838_geom_{pol}_{height}','p618_hourly_proxy':f'itu618_proxy_{pol}_{height}'}
                for kind,name in columns.items():
                    frame[name]=np.nan;scenarios[name]={'kind':kind,'polarization':pol,'height':height,'height_offset_km':offset,'primary':pol=='circular' and height=='nominal','counterfactual_polarization':pol!='circular'}
                for station,setting in heights.items():
                    mask=frame.station.eq(station)
                    answer=attenuation(rate[mask].to_numpy(),20.7,setting['elevation_deg'],setting['latitude_deg'],setting['receiver_km_amsl'],setting['hR_km_amsl']+offset,pol)
                    for kind,name in columns.items():frame.loc[mask,name]=answer[kind]
                    parameters.append({'station':station,'height':height,'polarization':pol,'rain_height_km_amsl':setting['hR_km_amsl']+offset,'receiver_km_amsl':setting['receiver_km_amsl'],**{k:answer[k] for k in ('slant_km','k','alpha')}})
        require(len(scenarios)==18 and np.isfinite(frame.loc[frame.physical_input_valid,list(scenarios)]).all().all(),'18 scenario outputs')
        require(frame.loc[~frame.physical_input_valid,list(scenarios)].isna().all().all(),'Invalid rates remain invalid')
        emit(OUT/'scenarios.json',scenarios);emit(OUT/'physical_parameters.json',parameters)
        frame.to_parquet(newpath(OUT/'predictions_physical_and_reused.parquet'),index=False)
        exclusions=[]
        for station,d in frame.groupby('station'):
            exclusions.append({'station':station,'candidate_hours':len(d),'candidate_dates':d.hour_utc.dt.date.nunique(),
                               'old_primary_hours':int(d.primary_score_eligible.sum()),'rain_invalid_all':int((~d.physical_input_valid).sum()),
                               'rain_invalid_primary':int((d.primary_score_eligible&~d.physical_input_valid).sum()),
                               'common_hours':int(d.common_score_eligible.sum()),'zero_rain_common':int((d.common_score_eligible&d.rain_liquid_mm_per_h.eq(0)).sum())})
        emit(OUT/'coverage.json',{'station_counts':exclusions,'negative_not_clipped':True,'missing_labels_not_imputed':True})
        metrics=save_metrics(frame,models,scenarios,local);bootstrap(frame);protected_verify(guard)
        emit(OUT/'calculation_complete.json',{'status':'PENDING_INDEPENDENT_QA','time':now(),'source_count':len(guard),'ml_fit_calls':0,'ml_prediction_calls':0,'weather_downloads':0,
                                            'physical_scenarios':len(scenarios),'models_reused':models,'uk_all_rows':len(frame),'uk_common_hours':5538,'local_reused_hours':len(local),
                                            'metrics_rows':len(metrics),'code_sha256':{p.name:sha(p) for p in HERE.glob('*.py')},
                                            'predictions_sha256':sha(OUT/'predictions_physical_and_reused.parquet')})
        print('MAIN_RESULTS',metrics.loc[metrics.group.eq('overall')&metrics.cohort.isin(['UK_all','LOCAL_test'])&metrics.model.isin(MAIN+['ours_frozen_top10','train_mean','elastic_net_seed17','physics_geom_circular','physics_p618_hourly_circular'])].to_string(index=False),flush=True)
        print('HEIGHTS',heights,flush=True)

if __name__=='__main__':
    try:main()
    except Exception:
        emit(OUT/('failure_calculate_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'.json'),{'time':now(),'traceback':traceback.format_exc()});raise
