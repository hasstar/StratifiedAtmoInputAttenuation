"""Independent scalar equations, interpolation, geometry and metric verification."""
from common import *
import ast
import io
import math
import traceback
import zipfile
import numpy as np
import pandas as pd
import netCDF4
from scipy.interpolate import RegularGridInterpolator

def scalar_coeff(f,elev,pol):
    # Separate transcription from visually checked P838 Tables1–4.
    rows=[
        ([-5.33980,-.35351,-.23789,-.94158],[-.10008,1.26970,.86036,.64552],[1.13098,.45400,.15354,.16817],-.18961,.71147,True),
        ([-3.80595,-3.44965,-.39902,.50167],[.56934,-.22911,.73042,1.07319],[.81061,.51059,.11899,.27195],-.16398,.63297,True),
        ([-.14318,.29591,.32177,-5.37610,16.1721],[1.82442,.77564,.63773,-.96230,-3.29980],[-.55187,.19822,.13164,1.47828,3.43990],.67849,-1.95537,False),
        ([-.07771,.56727,-.20238,-48.2991,48.5833],[2.33840,.95545,1.14520,.791669,.791459],[-.76284,.54039,.26809,.116226,.116479],-.053739,.83433,False)]
    x=math.log10(f);ans=[]
    for a,b,c,m,d,exponentiate in rows:
        value=math.fsum(ai*math.exp(-((x-bi)/ci)**2) for ai,bi,ci in zip(a,b,c))+m*x+d
        ans.append(10**value if exponentiate else value)
    kh,kv,ah,av=ans;tau={'horizontal':0,'vertical':90,'circular':45}[pol]
    term=math.cos(math.radians(elev))**2*math.cos(math.radians(2*tau))
    k=.5*((1+term)*kh+(1-term)*kv)
    alpha=((1+term)*kh*ah+(1-term)*kv*av)/(2*k)
    return k,alpha

def scalar_paths(rain,f,elev,latitude,hs,hr,pol):
    if not math.isfinite(rain) or rain<0:return math.nan,math.nan
    if rain==0 or hr<=hs:return 0.,0.
    k,alpha=scalar_coeff(f,elev,pol);gamma=k*math.pow(rain,alpha)
    st=math.sin(math.radians(elev));ct=math.cos(math.radians(elev))
    ls=(hr-hs)/st;lg=ls*ct
    r=1/(1+.78*math.sqrt(lg*gamma/f)-.38*(1-math.exp(-2*lg)))
    zeta=math.degrees(math.atan((hr-hs)/(lg*r)))
    lr=lg*r/ct if zeta>elev else ls
    chi=36-abs(latitude) if abs(latitude)<36 else 0
    v=1/(1+math.sqrt(st)*(31*(1-math.exp(-elev/(1+chi)))*math.sqrt(lr*gamma)/math.pow(f,2)-.45))
    return gamma*ls,gamma*lr*v

def independent_mask(cohort,group,uk,local):
    if cohort=='LOCAL_test':
        d=local;mask=np.ones(len(d),bool)
        if group=='other':mask=~(d.rain|d.clear_dry|d.dry_overcast).to_numpy()
        elif group!='overall':mask=d[group].to_numpy(bool)
    else:
        d=uk;rain=3600*(d.mean_total_precipitation_rate-d.mean_snowfall_rate)
        mask=(np.isfinite(d.truth_db)&d.valid_seconds.ge(3420)&np.isfinite(rain)&rain.ge(0)).to_numpy()
        for station in ('chilbolton','sparsholt'):
            if station in cohort:mask &= d.station.eq(station).to_numpy()
        if 'extension2003' in cohort:mask &= d.hour_utc.dt.year.eq(2003).to_numpy()
        if 'original39' in cohort:mask &= d.hour_utc.dt.year.eq(2004).to_numpy()
        if group=='rain_below_0.1':mask &= rain.lt(.1).to_numpy()
        elif group=='rain_0.1_to_1':mask &= (rain.ge(.1)&rain.lt(1)).to_numpy()
        elif group=='rain_at_least_1':mask &= rain.ge(1).to_numpy()
        else:require(group=='overall','Unknown group')
    return d.loc[mask]

def scalar_metrics(y,p):
    y=list(map(float,y));p=list(map(float,p));n=len(y)
    e=[a-b for a,b in zip(p,y)];sse=math.fsum(v*v for v in e)
    ym=math.fsum(y)/n;sst=math.fsum((v-ym)**2 for v in y)
    return dict(n=n,rmse_db=math.sqrt(sse/n),mae_db=math.fsum(map(abs,e))/n,bias_db=math.fsum(e)/n,r2=1-sse/sst if sst>0 else math.nan)

def legacy_pure_coefficients():
    p=ROOT/'Code/Hourly_Attenuation_Experiments/Revision_016/run_itu_baseline.py'
    tree=ast.parse(p.read_text(encoding='utf-8-sig'));keep=[]
    for node in tree.body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='COEFFICIENT_TABLES' for t in node.targets):keep.append(node)
        if isinstance(node,ast.FunctionDef) and node.name=='itu_coefficients':keep.append(node)
    require(len(keep)==2,'Exactly literal tables and pure function; no old module imports')
    scope={'np':np,'MAIN_POL':'circular'}
    exec(compile(ast.Module(body=keep,type_ignores=[]),str(p)+'::pure_AST','exec'),scope)
    return scope['itu_coefficients']

def controls():
    import physics as production
    checks=[];legacy=legacy_pure_coefficients()
    official={19:(.08084,1.0691,.08642,.9930),20:(.09164,1.0568,.09611,.9847),21:(.1032,1.0447,.1063,.9771)}
    for f,(kh,ah,kv,av) in official.items():
        for pol,k,a in [('horizontal',kh,ah),('vertical',kv,av)]:
            got=scalar_coeff(f,0,pol)
            require(abs(got[0]-k)<(5.5e-6 if k<.1 else 5.5e-5) and abs(got[1]-a)<5.5e-5,'Official rounded P838 Table5')
            require(np.allclose(got,production.coefficients(f,0,pol),rtol=0,atol=2e-13),'Independent coefficient parity')
            checks.append(f'P838_Table5_{f}_{pol}')
    for f in (1.,19.13,20.7,40.,1000.):
        for elevation in (0.,28.04933017048484,90.):
            for pol in ('horizontal','vertical','circular'):
                expected=scalar_coeff(f,elevation,pol)
                require(np.allclose(expected,production.coefficients(f,elevation,pol),atol=3e-12,rtol=1e-12),'Scalar coefficients')
                require(np.allclose(expected,legacy(f,pol,elevation),atol=3e-12,rtol=1e-12),'Legacy pure coefficient parity')
                checks.append(f'coefficient_parity_{f}_{elevation}_{pol}')
    for label,args in [('invalid_frequency',(.9,28,'circular')),('invalid_elevation',(20.7,-1,'circular')),('unknown_polarization',(20.7,28,'other'))]:
        try:production.coefficients(*args)
        except RuntimeError:checks.append(label)
        else:raise RuntimeError('Negative control not rejected: '+label)
    r=production.attenuation(np.array([-1,np.nan,np.inf,0.,1.,10.]),20.7,28.,51.,.1,2.4,'circular')
    require(np.isnan(r['geom'][:3]).all() and np.isnan(r['p618_hourly_proxy'][:3]).all(),'Negative/nonfinite R not silently zeroed')
    require(r['geom'][3]==r['p618_hourly_proxy'][3]==0,'Exact zero-rain behavior')
    require(r['geom'][4]>0 and r['geom'][5]>r['geom'][4],'Positive path monotonic geometric control')
    checks.extend(['invalid_rain_preserved','zero_rain_zero','positive_rain_geometric'])
    below=production.attenuation(np.array([1.,10.]),20.7,28.,51.,3.,2.4,'circular')
    require((below['geom']==0).all() and (below['p618_hourly_proxy']==0).all(),'Rain layer below receiver')
    checks.append('rain_layer_below_receiver_zero')
    try:production.attenuation([1.],20.7,4.9,51.,.1,2.4,'circular')
    except RuntimeError:checks.append('unsupported_low_elevation_rejected')
    else:raise RuntimeError('Low-elevation branch must be rejected')
    return checks

def geometry_and_native(plan):
    records=read(OLD/'native_label_sources.json');summaries=[]
    for link in plan['links']:
        lat=math.radians(link['latitude_deg_n']);lon=math.radians(link['longitude_deg_e'])
        a=6378137.;e2=(1/298.257223563)*(2-1/298.257223563);N=a/math.sqrt(1-e2*math.sin(lat)**2)
        h=link['height_amsl_m'];receiver=np.array([(N+h)*math.cos(lat)*math.cos(lon),(N+h)*math.cos(lat)*math.sin(lon),(N*(1-e2)+h)*math.sin(lat)])
        radius=a+link['nominal_geo_altitude_km']*1000
        slon=math.radians(link['satellite_longitude_deg_e']);sat=np.array([radius*math.cos(slon),radius*math.sin(slon),0.])
        line=sat-receiver;line/=np.linalg.norm(line)
        up=np.array([math.cos(lat)*math.cos(lon),math.cos(lat)*math.sin(lon),math.sin(lat)])
        elevation=math.degrees(math.asin(np.dot(up,line)))
        require(np.allclose(receiver,link['origin_ecef_m'],rtol=0,atol=1e-6),'Independent WGS84 receiver')
        require(np.allclose(line,link['direction_ecef'],rtol=0,atol=1e-12) and abs(elevation-link['elevation_deg'])<1e-9,'Independent WGS84 nominal ray')
        count=0
        for item in records:
            if item['station']!=link['id']:continue
            with netCDF4.Dataset('independent_native',memory=(ROOT/item['source']).read_bytes()) as ds:
                for key in ('polarisation_transmit','polarisation_receive'):
                    require(b''.join(ds[key][:].tolist()).decode()=='Left Hand Circularly Polarized','Independent native polarization')
                require(abs(float(ds['frequency'][:])-20.7)<1e-5 and ds['frequency'].units=='GHz','Native frequency')
                require(float(ds['receiver_altitude_amsl'][:])==h and ds['receiver_altitude_amsl'].units=='m','Native altitude AMSL')
                require(float(ds['latitude_receiver'][:])==link['latitude_deg_n'],'Native latitude')
                require(float(ds['longitude_receiver'][:])%360==link['longitude_deg_e']%360,'Native longitude')
                require(float(ds['satellite_position'][:])==-link['satellite_longitude_deg_e'] and ds['satellite_position'].units=='degrees_west','Satellite longitude orientation')
                require(float(ds['elevation_angle'][:])==30.,'Native coarse elevation retained')
            count+=1
        summaries.append({'station':link['id'],'native_files':count,'elevation_deg_nominal_wgs84':elevation,'native_coarse_elevation_deg':30.,'polarization':'LHCP matched'})
    return summaries

def bootstrap_recompute(uk,records):
    groups={}
    for row in records:groups.setdefault(row['cohort'],[]).append(row)
    maxdiff=0.
    for cohort,items in groups.items():
        d=independent_mask(cohort,'overall',uk,None);date=d.hour_utc.dt.strftime('%Y-%m-%d');names=['ours_frozen_top10','train_mean','itu838_geom_circular_nominal','itu618_proxy_circular_nominal']
        aggregate=d[names].subtract(d.truth_db,axis=0).pow(2).assign(date=date).groupby('date',sort=True).sum()
        counts=d.assign(date=date).groupby('date',sort=True).size().to_numpy()
        rng=np.random.default_rng(20261016);samples=rng.integers(0,len(counts),size=(2000,len(counts)))
        # Independently aggregate resample multiplicities rather than direct sample indexing.
        frequencies=np.array([np.bincount(sample,minlength=len(counts)) for sample in samples])
        rmses=np.sqrt((frequencies@aggregate.to_numpy())/(frequencies@counts)[:,None])
        for row in items:
            expected=np.quantile(rmses[:,names.index(row['model'])]-rmses[:,names.index(row['reference'])],[.025,.975])
            diff=float(np.max(np.abs(expected-row['unadjusted_ci95_db'])));maxdiff=max(maxdiff,diff)
            require(diff<1e-11,'Independent date-bootstrap interval')
            m=scalar_metrics(d.truth_db,d[row['model']])['rmse_db'];r=scalar_metrics(d.truth_db,d[row['reference']])['rmse_db']
            require(abs((m-r)-row['delta_rmse_db'])<1e-12,'Paired bootstrap point estimate')
            require(row['n_dates']==len(counts) and row['n']==len(d),'Cluster and sample counts')
    return maxdiff

def main():
    offline()
    with lock():
        guard=read(OUT/'source_sha256_before.json');protected_verify(guard)
        gate=read(OUT/'calculation_complete.json');require(sha(OUT/'predictions_physical_and_reused.parquet')==gate['predictions_sha256'],'Saved prediction SHA')
        for filename,digest in gate['code_sha256'].items():require(sha(HERE/filename)==digest,'Executed code immutable')
        controls_pass=controls()
        uk=pd.read_parquet(OUT/'predictions_physical_and_reused.parquet');old=pd.read_parquet(OLD/'predictions_expanded.parquet');local=pd.read_parquet(REV2/'predictions/local_test.parquet')
        pd.testing.assert_frame_equal(uk[old.columns],old,check_exact=True)
        require(not uk[['station','hour_utc']].duplicated().any(),'Unique common hour keys')
        rate=(old.mean_total_precipitation_rate-old.mean_snowfall_rate)*3600
        require(np.array_equal(uk.rain_liquid_mm_per_h,rate),'Exact raw-rate conversion')
        require(np.array_equal(uk.rain_valid_utc_end,(uk.hour_utc+pd.Timedelta(hours=1))),'Rain valid-end interval')
        expected_primary=np.isfinite(old.truth_db)&old.valid_seconds.ge(3420)
        require(np.array_equal(expected_primary,old.primary_score_eligible),'Label eligibility reconstruction')
        valid=np.isfinite(rate)&(rate>=0);common=valid&expected_primary
        require(np.array_equal(valid,uk.physical_input_valid) and np.array_equal(common,uk.common_score_eligible),'Independent common masks')
        require(common.sum()==5538 and (expected_primary&~valid).sum()==55 and (~valid).sum()==56,'Exclusions/counts')
        with zipfile.ZipFile(OUT/'references/P839_4_components.zip') as z:
            require(z.testzip() is None,'Independent ZIP CRC')
            arrays={name:np.loadtxt(io.BytesIO(z.read(name))) for name in ['Lat.txt','Lon.txt','h0.txt']}
        interp=RegularGridInterpolator((arrays['Lat.txt'][:,0][::-1],arrays['Lon.txt'][0,:]),arrays['h0.txt'][::-1,:],bounds_error=True)
        plan=read(ACQ/'plan.json');stations=read(OUT/'station_parameters.json');scenarios=read(OUT/'scenarios.json');max_phys=0.;value_count=0
        for link in plan['links']:
            station=link['id'];p=stations[station]
            h0=float(interp([[link['latitude_deg_n'],link['longitude_deg_e']%360]])[0]);hr=h0+.36;hs=link['height_amsl_m']/1000
            require(abs(h0-p['h0_km_amsl'])<1e-12 and abs(hr-p['hR_km_amsl'])<1e-12,'Independent P839 interpolation/datum')
            require(abs(sum(c['weight'] for c in p['corners'])-1)<1e-14 and all(0<=c['weight']<=1 for c in p['corners']),'Convex interpolation weights')
            for c in p['corners']:
                require(arrays['h0.txt'][c['i'],c['j']]==c['h0_km_amsl'] and arrays['Lat.txt'][c['i'],c['j']]==c['latitude_deg'] and arrays['Lon.txt'][c['i'],c['j']]==c['longitude_deg'],'Map corner identity')
            d=uk.loc[uk.station.eq(station)]
            for name,s in scenarios.items():
                h=hr+s['height_offset_km'];index=0 if s['kind']=='geom' else 1
                values=np.array([scalar_paths(float(r),20.7,link['elevation_deg'],link['latitude_deg_n'],hs,h,s['polarization'])[index] for r in d.rain_liquid_mm_per_h])
                require(np.array_equal(np.isnan(values),d[name].isna()),'Invalid physical masks')
                diff=float(np.nanmax(np.abs(values-d[name].to_numpy())));max_phys=max(max_phys,diff)
                require(diff<2e-11,'Scalar full-field physical replay: '+name)
                value_count+=int(np.isfinite(values).sum())
        native=geometry_and_native(plan)
        metrics=pd.read_csv(OUT/'metrics.csv');max_metric=0.
        for row in metrics.to_dict('records'):
            d=independent_mask(row['cohort'],row['group'],uk,local);got=scalar_metrics(d.truth_db,d[row['model']])
            for key,value in got.items():
                if math.isnan(value):require(pd.isna(row[key]),'Undefined R2')
                else:
                    diff=abs(value-row[key]);max_metric=max(max_metric,diff)
                    require(diff<2e-10,'Independent metric '+key)
        family=pd.read_csv(OUT/'family_summary.csv')
        for row in family.to_dict('records'):
            part=metrics.loc[metrics.cohort.eq(row['cohort'])&metrics.group.eq(row['group'])&metrics.family.eq(row['family'])&metrics.comparison.eq(row['comparison'])]
            require(len(part)==row['runs'] and part.n.nunique()==1 and part.n.iloc[0]==row['n'],'Family membership/sample agreement')
            for source,target in [('rmse_db','rmse_mean_db'),('mae_db','mae_mean_db'),('bias_db','bias_mean_db')]:
                require(abs(math.fsum(part[source])/len(part)-row[target])<1e-12,'Family mean')
            sd=float(np.std(part.rmse_db,ddof=1)) if len(part)>1 else 0.
            require(abs(sd-row['rmse_seed_sd_db'])<1e-12,'Seed SD not CI')
        bootstrap_diff=bootstrap_recompute(uk,read(OUT/'date_cluster_bootstrap.json')['results'])
        protected_verify(guard)
        result={'status':'PASS','time':now(),'protected_sources':len(guard),'unchanged_saved_ml_and_labels':True,'native_files_rechecked':sum(s['native_files'] for s in native),
                'native_geometry':native,'controls':controls_pass,'physical_finite_values_independently_recomputed':value_count,'max_physical_abs_difference_db':max_phys,
                'metrics_rows_recomputed':len(metrics),'max_metric_abs_difference':max_metric,'family_summary_rows':len(family),'bootstrap_pairs':len(read(OUT/'date_cluster_bootstrap.json')['results']),
                'max_bootstrap_interval_abs_difference_db':bootstrap_diff,'uk_common_hours':int(common.sum()),'excluded_negative_primary':int((expected_primary&~valid).sum()),
                'fit_calls':0,'ml_prediction_calls':0,'weather_downloads':0,'source_sha256_after':guard,'verification_code_sha256':sha(Path(__file__))}
        emit(OUT/'independent_qa.json',result)
        print({k:v for k,v in result.items() if k not in ('source_sha256_after','controls')},flush=True)

if __name__=='__main__':
    try:main()
    except Exception:
        emit(OUT/('failure_verify_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'.json'),{'time':now(),'traceback':traceback.format_exc()});raise
