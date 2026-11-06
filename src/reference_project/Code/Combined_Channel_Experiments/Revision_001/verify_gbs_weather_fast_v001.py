"""Independent point/time/label verification; no import of first implementation."""
import hashlib, json, math, zipfile
from pathlib import Path
import netCDF4
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'Outputs/Experiments/Combined_Channel_Revision_001/external/validation_gbs_48h_v010'
OUT=BASE/'conditional_fast_independent_qa_001'
def digest(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def main():
    OUT.mkdir(exist_ok=False)
    old=BASE/'conditional_fast_001'
    protected=read(old/'protected_inputs.json')
    assert all(digest(ROOT/p)==h for p,h in protected.items())
    scope=read(BASE/'scope_manifest.json');frame=pd.read_parquet(old/'weather_labels.parquet')
    expected={};all_values={}
    for month in ['02','03','04']:
        p=BASE.parent/f'weather/GBS_England_2004-{month}_surface.download'
        with zipfile.ZipFile(p) as z:
            for name in z.namelist():
                if not name.endswith('.nc'):continue
                with netCDF4.Dataset('independent.nc',memory=z.read(name)) as nc:
                    for var in ['tcc','tp','sf']:
                        if var not in nc.variables:continue
                        item=nc.variables[var]
                        assert item.GRIB_paramId=={'tcc':164,'tp':228,'sf':144}[var]
                        assert item.units=={'tcc':'(0 - 1)','tp':'m','sf':'m of water equivalent'}[var]
                        for station in scope['links']:
                            lat=min(nc.variables['latitude'][:],key=lambda v:(abs(v-station['latitude_deg_n']),-v))
                            lon=min(nc.variables['longitude'][:],key=lambda v:(abs(v-station['longitude_deg_e']),-v))
                            ilat=nc.variables['latitude'][:].tolist().index(lat);ilon=nc.variables['longitude'][:].tolist().index(lon)
                            for ti,ts in enumerate(nc.variables['valid_time'][:]):
                                target=int(ts)-(0 if var=='tcc' else 3600)
                                if target not in scope['hours_utc_epoch_seconds']:continue
                                all_values[(station['id'],target,var)]=float(item[ti,ilat,ilon])
    max_delta=0.;unknown=[]
    for row in frame.itertuples(index=False):
        h=int(row.hour_utc.timestamp());vals=[]
        for v in ['tcc','tp','sf']:
            want=all_values.get((row.station,h,v),float('nan'));got=getattr(row,v)
            assert (np.isnan(want) and np.isnan(got)) or want==got
            assert getattr(row,v+'_valid_utc_epoch_s')==h+(0 if v=='tcc' else 3600)
            vals.append(want)
        c,p,s=vals;p*=1000;s*=1000
        if not np.isfinite(vals).all(): event='unknown';heavy=False;unknown.append((row.station,str(row.hour_utc)))
        else:
            conditions={'rain':p>=.1 and s<.1,'clear_dry':p<.1 and s<.1 and c<=.2,'dry_overcast':p<.1 and s<.1 and c>=.8}
            assert sum(conditions.values())<=1
            event=next((k for k,v in conditions.items() if v),'other');heavy=p>=1 and s<.1
        assert event==row.event and bool(heavy)==row.heavy_rain and row.weather_known==(event!='unknown')
    assert set(unknown)=={(s,f'2004-02-{d} 23:00:00+00:00') for s in scope['stations'] for d in ['21','28']}
    pred=pd.read_parquet(BASE/'evaluation_002/hourly_predictions.parquet')
    merged=pred.merge(frame,on=['station','hour_utc'],validate='one_to_one')
    cdf=read(old/'fast_compatibility_gate.json')
    assert cdf['status']=='PASS_FOR_FROZEN_MARGINAL_TRANSFER_DIAGNOSTIC'
    package=read(ROOT/'Outputs/Experiments/Fast_Fading_Revision_002/final_generator/model.json')
    for event,entry in package['components'].items():
        p=ROOT/'Outputs/Experiments/Fast_Fading_Revision_002'/entry['cdf_file_relative_to_revision_output']
        assert digest(p)==entry['cdf_sha256']
        with np.load(p) as a:
            weights=np.diff(np.r_[0.,a['cdf']]);assert np.all(weights>0) and a['cdf'][-1]==1
            # Literal compensated sum independent of primary BLAS dot.
            moment=math.fsum(float(w)*float(q) for w,q in zip(weights,a['q']))
            assert abs(moment-1)<1e-10
    assert all(digest(ROOT/p)==h for p,h in protected.items())
    result={'status':'PASS_WEATHER_AND_FROZEN_MARGINAL_TRANSFER_GATE','station_hours':len(frame),'known_hours':int(frame.weather_known.sum()),
       'primary_known_hours':int((merged.primary_score_eligible&merged.weather_known).sum()),'all_finite_known_hours':int((merged.finite_target_diagnostic_eligible&merged.weather_known).sum()),
       'unknown':unknown,'raw_weather_exact_match':True,'four_cdf_hashes_and_unit_power_recomputed':True,
       'measurement_equivalence_claimed':False,'new_fit_calls':0,'new_downloads':0,
       'primary_gate_sha256':digest(old/'fast_compatibility_gate.json'),'weather_labels_sha256':digest(old/'weather_labels.parquet'),
       'verifier_sha256':digest(__file__),'inputs_unchanged':True}
    with (OUT/'qa.json').open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    print(json.dumps(result),flush=True)
if __name__=='__main__': main()
