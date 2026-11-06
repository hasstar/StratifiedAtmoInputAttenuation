"""Read-only actual diagnostic/fast-package gate; no model/run/fit imports."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, io, json, math, traceback, zipfile
import netCDF4
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT/'Outputs/Experiments/Combined_Channel_Revision_001/external/validation_gbs_48h_v010'
OUT = BASE/'conditional_fast_001'
EXTERNAL = BASE.parent
FAST = ROOT/'Outputs/Experiments/Fast_Fading_Revision_002'
FIELDS = {'tcc': (164, '(0 - 1)', 'instant'), 'tp': (228, 'm', 'accum'), 'sf': (144, 'm of water equivalent', 'accum')}

def require(ok, message):
    if not ok: raise ValueError(message)

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for part in iter(lambda:f.read(1048576),b''): h.update(part)
    return h.hexdigest()

def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))

def emit(name,value):
    with (OUT/name).open('x',encoding='utf-8') as f: json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)

def classify(p,s,c):
    if not all(math.isfinite(x) for x in (p,s,c)): return 'unknown',False
    rain=p>=.1 and s<.1
    return ('rain' if rain else 'dry_overcast' if p<.1 and s<.1 and c>=.8 else
            'clear_dry' if p<.1 and s<.1 and c<=.2 else 'other'), bool(rain and p>=1.)

def grid(x): return math.floor(x*4+.5)/4

def weather_time(field,hour): return hour+(0 if field=='tcc' else 3600)

def quantile(q,cdf,p):
    return -10*np.log10(q[np.searchsorted(cdf,1-np.asarray(p),side='left')])

def controls():
    cases=[((0,0,0),('clear_dry',False)),((.099,0,.2),('clear_dry',False)),
           ((.1,0,.2),('rain',False)),((1.,0,1.),('rain',True)),((5.,0,0),('rain',True)),
           ((0,.1,.1),('other',False)),((.1,.1,0),('other',False)),
           ((0,0,.8),('dry_overcast',False)),((0,0,.7999),('other',False)),
           ((0,0,.2001),('other',False)),((float('nan'),0,0),('unknown',False)),
           ((0,float('nan'),0),('unknown',False)),((0,0,float('nan')),('unknown',False))]
    for args,want in cases: require(classify(*args)==want,'weather control')
    require(grid(51.125)==51.25 and grid(-1.375)==-1.25,'north/east midpoint')
    require(weather_time('tp',3600)==7200 and weather_time('sf',3600)==7200 and weather_time('tcc',3600)==3600,'validity alignment')
    require(np.array_equal(quantile(np.array([1.,10.]),np.array([.5,1.]),[.25,.75]),[-10.,0.]),'f quantile direction')
    require(classify(.1,.1,.8)[0]=='other','snow is not rain')
    emit('controls.json',{'status':'PASS','control_count':17,'new_fit_calls':0})

def main():
    controls()
    protected={}
    def pin(p, expected=None):
        p=Path(p); digest=sha(p)
        require(expected is None or expected==digest,'Source SHA changed: '+str(p))
        protected[str(p.relative_to(ROOT))]=digest
        return p
    scope=read(pin(BASE/'scope_manifest.json','43864c01e313b768ca21a8cbda396cda7a9a3dfbab71975f524a80089333f052'))
    pin(OUT/'PROTOCOL.md');pin(Path(__file__))
    pin(ROOT/'Code/Dataset_Build/propose_event_balanced_split.py')
    prediction=pin(BASE/'evaluation_002/hourly_predictions.parquet','c58822f6db3eea41e706b632610fdb1fddded0c9b401f48e72fdbf16e15061d8')
    pred=pd.read_parquet(prediction)
    require(len(pred)==1872 and pred.primary_score_eligible.sum()==1847,'Frozen evaluation scope')
    links={s['id']:s for s in scope['links']}
    hours=set(scope['hours_utc_epoch_seconds'])
    points={site:(grid(s['latitude_deg_n']),grid(s['longitude_deg_e'])) for site,s in links.items()}
    needed={(field,weather_time(field,h),lat,lon) for h in hours for lat,lon in points.values() for field in FIELDS}
    values={}; sources=[]
    for month in ['2004-02','2004-03','2004-04']:
        job=read(pin(EXTERNAL/f'weather_jobs/GBS_England_{month}_surface.json'))
        require(job['status']=='VERIFIED','Historical candidate not verified')
        path=pin(EXTERNAL/job['job']['target'],job['sha256'])
        require(path.stat().st_size==job['bytes'],'Historical bytes differ')
        with zipfile.ZipFile(path) as z:
            for member in z.namelist():
                if not member.endswith('.nc'): continue
                with netCDF4.Dataset('diagnostic.nc',memory=z.read(member)) as ds:
                    ds.set_auto_mask(False)
                    t=ds.variables['valid_time'];la=ds.variables['latitude'];lo=ds.variables['longitude']
                    require(t.units=='seconds since 1970-01-01' and la.units=='degrees_north' and lo.units=='degrees_east','UTC/coordinate units')
                    times=np.asarray(t[:],dtype=np.int64); lats=np.asarray(la[:]);lons=np.asarray(lo[:])
                    require(len(set(times))==len(times) and np.all(times%3600==0),'UTC identities')
                    for field,(pid,unit,step) in FIELDS.items():
                        if field not in ds.variables: continue
                        v=ds.variables[field]
                        require(int(v.GRIB_paramId)==pid and v.units==unit and v.GRIB_stepType==step,'diagnostic identity: '+field)
                        require(v.dimensions==('valid_time','latitude','longitude'),'diagnostic dimensions')
                        count=0
                        for lat,lon in sorted(set(points.values())):
                            iy=np.flatnonzero(lats==lat);ix=np.flatnonzero(lons==lon)
                            require(len(iy)==len(ix)==1,'Required nearest grid absent')
                            for it,utc in enumerate(times):
                                key=(field,int(utc),lat,lon)
                                if key not in needed: continue
                                value=float(v[it,int(iy[0]),int(ix[0])])
                                require(math.isfinite(value) and value!=getattr(v,'_FillValue',None),'Nonfinite diagnostic')
                                require((0<=value<=1) if field=='tcc' else value>=0,'Diagnostic physical range')
                                require(key not in values,'Duplicate diagnostic source for required key')
                                values[key]=(value,str(path.relative_to(ROOT)),member);count+=1
                        sources.append({'path':str(path.relative_to(ROOT)),'sha256':sha(path),'member':member,'field':field,'parameter_id':pid,'unit':unit,'step':step,'required_values_read':count})
    rows=[]
    for r in pred.itertuples(index=False):
        h=int(r.hour_utc.timestamp());lat,lon=points[r.station]
        record={'station':r.station,'hour_utc':r.hour_utc,'grid_latitude':lat,'grid_longitude':lon}
        for field in FIELDS:
            key=(field,weather_time(field,h),lat,lon); sample=values.get(key)
            record[field]=np.nan if sample is None else sample[0]
            record[field+'_valid_utc_epoch_s']=key[1]
            record[field+'_source']=None if sample is None else sample[1]
            record[field+'_member']=None if sample is None else sample[2]
        event,heavy=classify(record['tp']*1000,record['sf']*1000,record['tcc'])
        record.update(event=event,heavy_rain=heavy,weather_known=event!='unknown')
        rows.append(record)
    weather=pd.DataFrame(rows)
    require(len(weather)==1872 and not weather.duplicated(['station','hour_utc']).any(),'weather table shape')
    weather.to_parquet(OUT/'weather_labels.parquet',index=False)
    emit('weather_source_evidence.json',sources)
    emit('weather_availability_qa.json',{'status':'PASS_WITH_EXPLICIT_GAPS' if (~weather.weather_known).any() else 'PASS',
        'station_hours':len(weather),'known_station_hours':int(weather.weather_known.sum()),'unknown_station_hours':int((~weather.weather_known).sum()),
        'required_unique_grid_field_hours':len(needed),'available_unique_grid_field_hours':len(values),'missing_unique_grid_field_hours':len(needed-values.keys()),
        'counts':weather.groupby(['station','event']).size().to_dict() if False else [{'station':s,'event':e,'hours':len(g)} for (s,e),g in weather.groupby(['station','event'])],
        'missing':[{'station':r.station,'hour_utc':str(r.hour_utc),'tp_valid_utc':str(pd.to_datetime(r.tp_valid_utc_epoch_s,unit='s',utc=True)),
                   'missing_fields':[f for f in FIELDS if not math.isfinite(getattr(r,f))]} for r in weather.loc[~weather.weather_known].itertuples()],
        'missing_is_not_other':True,'new_downloads':0,'source_grid_values_not_interpolated':True,'weather_label_source':'existing ERA5 nearest surface diagnostic proxy, not observed rain gauge'})
    package_path=pin(FAST/'final_generator/model.json','b524c268054992953b63197eccd76888fde4de2e12a3d9c1ac6d3295647be714')
    package=read(package_path);pin(FAST/'final_generator/qa.json')
    require(package['candidate']=='block_0256' and package['block_length_s']==256,'Frozen dynamics identity')
    require(len(package['training_hours'])==2404 and all(not s.startswith('2004') for s in package['training_hours']),'Training/external hour overlap')
    cdf_checks={}
    for event,c in package['components'].items():
        path=pin(FAST/c['cdf_file_relative_to_revision_output'],c['cdf_sha256'])
        library=pin(FAST/'final_generator'/c['library_file'],c['library_sha256'])
        with np.load(path,allow_pickle=False) as a:
            q=a['q'];cdf=a['cdf'];weights=np.diff(np.r_[0.,cdf])
            require(q.ndim==1 and q.shape==cdf.shape and np.isfinite(q).all() and (q>0).all() and (np.diff(q)>=0).all(),'CDF positive sorted support')
            require(np.isfinite(cdf).all() and (weights>0).all() and cdf[-1]==1.,'CDF cumulative weights')
            moment=float(np.dot(weights,q));require(abs(moment-1)<1e-10,'Frozen power normalization')
            p=(np.arange(401)+.5)/401;f=quantile(q,cdf,p)
            require(np.isfinite(f).all() and np.all(np.diff(f)>=0),'Frozen quantiles')
        with np.load(library,allow_pickle=False) as a:
            require(a['q'].shape==a['records'].shape==(c['training_hours'],3600),'1s library layout')
            require(len(a['hours'])==c['training_hours'],'library hours')
        cdf_checks[event]={'cdf_sha256':c['cdf_sha256'],'library_sha256':c['library_sha256'],'support_count':len(q),'expected_power_ratio':moment,'training_hours':c['training_hours'],'eligible_256s_training_hours':c['eligible_hours']}
    native_evidence=read(pin(BASE/'evaluation_002/native_label_source_evidence.json'))
    native_checks=[]
    for entry in native_evidence:
        path=pin(ROOT/entry['source'],entry['sha256'])
        with netCDF4.Dataset('native.nc',memory=path.read_bytes()) as ds:
            ds.set_auto_mask(False);v=ds.variables['clearsky_attenuation'];tv=ds.variables['time']
            require(float(v.integration_period)==1 and v.units=='dB','GBS scale')
            require('end of integration period' in tv.long_name.lower(),'GBS interval alignment')
            t=np.asarray(tv[:],float);y=np.asarray(v[:],float)
            good=np.isfinite(y)&(y!=v._FillValue)&(y>=v.valid_min)&(y<=v.valid_max)&np.isfinite(t)&(t>0)&(t<=86400)
            require(np.all(t[good]==np.rint(t[good])) and np.unique(t[good]).size==good.sum(),'GBS regular integer timestamps')
            require(int(good.sum())==entry['valid_seconds'],'GBS unchanged eligibility')
            native_checks.append({'source':entry['source'],'integration_period_s':float(v.integration_period),'valid_seconds':int(good.sum()),'has_subsecond_record_counts':False})
    emit('fast_compatibility_gate.json',{'status':'PASS_FOR_FROZEN_MARGINAL_TRANSFER_DIAGNOSTIC',
        'not_measurement_equivalence':True,'source_and_destination':'LOCAL trained 1s power residual to GBS native 1s relative-clear-sky attenuation',
        'allowed':'Fixed CDF quantiles shifted by already frozen Top10 predictions; score directly against GBS native seconds on known-weather hours',
        'observation_weights':'GBS each valid native second equal weight; no fabricated LOCAL >=9 subsecond-count gate',
        'cdf_components':cdf_checks,'native_files':native_checks,'unknown_weather_not_scored':True,
        'new_fit_calls':0,'new_reference_fits':0,'new_cdf_fits':0,'new_downloads':0,
        'unresolved_measurement_metadata':['Receiver bandwidth not established','Native within-second averaging domain not established; compare native reported 1s attenuation, not identical power-measurement process'],
        'dynamic_evaluation_status':'pending separate evaluation/independent QA; no dynamic claims from marginal score'})
    require(all(sha(ROOT/p)==digest for p,digest in protected.items()),'Protected input changed')
    emit('protected_inputs.json',protected)
    emit('completion.json',{'status':'PASS_LOCAL_COMPATIBILITY_PENDING_INDEPENDENT_QA','completed_utc':datetime.now(timezone.utc).isoformat(),
        'known_station_hours':int(weather.weather_known.sum()),'unknown_station_hours':int((~weather.weather_known).sum()),'new_downloads':0,'new_fit_calls':0,
        'protected_inputs_unchanged':True,'output_sha256':{p.name:sha(p) for p in OUT.iterdir() if p.is_file()}})
    print(json.dumps({'status':'PASS_LOCAL_COMPATIBILITY_PENDING_INDEPENDENT_QA','known_station_hours':int(weather.weather_known.sum()),'unknown_station_hours':int((~weather.weather_known).sum()),'cdf_classes':list(cdf_checks)}),flush=True)

if __name__=='__main__':
    try: main()
    except Exception as e:
        emit('failure.json',{'status':'FAIL_PRESERVED','error':str(e),'traceback':traceback.format_exc()}); raise
