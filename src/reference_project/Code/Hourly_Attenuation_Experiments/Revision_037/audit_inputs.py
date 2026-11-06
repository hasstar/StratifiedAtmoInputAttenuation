"""Independent read-back of acquired inputs and unchanged UK cohort.

Does not call the producer decoder/geometry or any estimator.
"""
import re, math, json, itertools, hashlib, zipfile
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import netCDF4
from pyproj import Transformer
from freeze import ROOT, OUT, read, sha, now, emit, need, rel
from transport import stage_lock

def independent_ascii(text):
    sections=re.split(r'\n-{5,}\n',text)[1].strip().split('\n\n');out={}
    for section in sections:
        lines=[x.strip() for x in section.splitlines() if x.strip()]
        if not lines:continue
        name=lines[0].split('[')[0];sizes=tuple(map(int,re.findall(r'\[(\d+)\]',lines[0])))
        need(name not in out and sizes,'Independent DAP header')
        values={}
        if len(sizes)==1:
            a=np.fromstring(','.join(lines[1:]),sep=',');need(len(a)==sizes[0],'Independent vector size');out[name]=a;continue
        for line in lines[1:]:
            prefix,rest=line.split(',',1);idx=tuple(map(int,re.findall(r'\[(\d+)\]',prefix)))
            vals=np.fromstring(rest,sep=',');need(len(idx)==len(sizes)-1 and len(vals)==sizes[-1],'Independent row size')
            for last,value in enumerate(vals):
                index=np.ravel_multi_index((*idx,last),sizes)
                need(index not in values,'Independent duplicate index');values[index]=value
        need(set(values)==set(range(math.prod(sizes))),'Independent truncation')
        out[name]=np.array([values[i] for i in range(math.prod(sizes))]).reshape(sizes)
    return out

def audit_geometry(geom):
    links=read(ROOT/'Outputs/Experiments/Combined_Channel_Revision_001/external/validation_gbs_48h_v010/scope_manifest.json')['links']
    toxyz=Transformer.from_crs(4979,4978,always_xy=True);togeo=Transformer.from_crs(4978,4979,always_xy=True)
    for link in links:
        o=np.array(toxyz.transform(link['longitude_deg_e'],link['latitude_deg_n'],link['height_amsl_m']))
        satellite=np.array(toxyz.transform(link['satellite_longitude_deg_e'],0,35786000.));direction=(satellite-o)/np.linalg.norm(satellite-o)
        group=geom[geom.station.eq(link['id'])]
        xyz=o[:,None]+direction[:,None]*group.slant_m.to_numpy()[None,:]
        lon,lat,h=togeo.transform(*xyz)
        np.testing.assert_allclose(lat,group.latitude,rtol=0,atol=1e-8);np.testing.assert_allclose(lon,group.longitude,rtol=0,atol=1e-8)
        np.testing.assert_allclose(h,group.height_m,rtol=0,atol=1e-5)
        need(np.array_equal(360-np.floor(np.asarray(lat)*4+.5).astype(int),group.latitude_index),'Independent ray latitude cell')
        need(np.array_equal(np.floor(np.asarray(lon)*4+.5).astype(int)%1440,group.longitude_index),'Independent ray longitude cell')
    # Forward pressure reconstruction is separate from the producer's inverse-height calculation.
    for level,part in geom[geom.level_hpa.ne(1)].groupby('level_hpa'):
        need(part.height_m.nunique()==1,'Fixed-height constancy');h=float(part.height_m.iloc[0]);H=6356660*h/(6356660+h)
        p=101325.;temperature=288.15
        for low,high,lapse in [(0,11000,-.0065),(11000,20000,0),(20000,32000,.001),(32000,47000,.0028)]:
            delta=min(H,high)-low
            if delta<0:break
            temp2=temperature+lapse*delta
            p*=math.exp(-9.80665*28.9644*delta/(8314.32*temperature)) if lapse==0 else (temp2/temperature)**(-9.80665*28.9644/(8314.32*lapse))
            temperature=temp2
            if H<=high:break
        need(abs(p-level*100)<1e-5,'Independent standard pressure reconstruction')
    old=pd.read_parquet(ROOT/'Outputs/Experiments/Combined_Channel_Revision_003/geometry_checks.parquet')
    a=old[old.level_hpa.eq(1)].sort_values(['station','utc_epoch_seconds']);b=geom[geom.level_hpa.eq(1)].sort_values(['station','utc_epoch_seconds'])
    for column in old.columns:np.testing.assert_array_equal(a[column],b[column])

def main():
    with stage_lock('independent_input_qa'):
        folder=OUT/'inputs';acq=OUT/'acquisition';build=read(folder/'build.json')
        for p,d in build['outputs'].items():need(sha(folder/p)==d,'Input build output changed')
        original=pd.read_parquet(ROOT/'Outputs/Experiments/Combined_Channel_Revision_010/expanded_dataset_with_itu_inputs.parquet')
        merged=pd.read_parquet(folder/'uk_inputs_38.parquet');prov=pd.read_parquet(folder/'new_field_provenance.parquet')
        pd.testing.assert_frame_equal(merged[original.columns],original,check_exact=True)
        required=read(OUT/'required_fields_38.json');names=[r['feature'] for r in required];missing=[r['feature'] for r in required if not r['available_all_5664']]
        need(len(merged)==5664 and not merged.duplicated(['station','utc_epoch_seconds']).any(),'Frozen cohort unique')
        need(np.isfinite(merged[names].to_numpy(float)).all(),'38 actual inputs finite')
        need(len(prov)==27*5664 and not prov.duplicated(['station','utc_epoch_seconds','feature']).any(),'27 complete supplemental columns')
        need(set(prov.feature)==set(missing),'Exact supplemental feature set')
        plan=read(acq/'ncar_plan.json');all_raw={};ncar_count=0
        for n in plan['nodes']:
            p=acq/'ncar_values'/f'{n["id"]}.ascii';receipt=read(p.with_name(p.name+'.receipt.json'))
            need(sha(p)==receipt['sha256'] and receipt['url']==n['url'],'Independent NCAR receipt')
            raw=independent_ascii(p.read_text());v=n['variable'];timekey=v+'.time';t=raw[timekey].astype(np.int64)*3600-2208988800
            np.testing.assert_array_equal(t,n['epochs']);lat=raw[v+'.latitude'];lon=raw[v+'.longitude']
            np.testing.assert_array_equal(lat,90-np.array(n['latitude_indices'])*.25)
            np.testing.assert_array_equal(lon,np.array(n['longitude_indices'])*.25)
            levels=raw[v+'.level'].astype(int) if n['pressure'] else np.array([0]);np.testing.assert_array_equal(levels,n['levels'])
            actual=raw[v+'.'+v].reshape(len(t),len(levels),len(lat),len(lon))
            for it,il,ia,io in itertools.product(range(len(t)),range(len(levels)),range(len(lat)),range(len(lon))):
                key=(n['spec'],int(levels[il]),int(t[it]),float(lat[ia]),float((lon[io]+180)%360-180))
                need(key not in all_raw,'Independent duplicate grid value')
                all_raw[key]=(float(actual[it,il,ia,io]),rel(p),receipt['sha256']);ncar_count+=1
        demands=pd.read_parquet(acq/'ncar_demands.parquet')
        keyed=prov.set_index(['station','utc_epoch_seconds','feature'])
        for row in demands.itertuples(index=False):
            key=(row.spec,row.level_hpa,row.utc_epoch_seconds,90-row.latitude_index*.25,(row.longitude_index*.25+180)%360-180)
            value,source,digest=all_raw[key];p=keyed.loc[(row.station,row.utc_epoch_seconds,row.feature)]
            need(p.value==value and p.source==source and p.source_sha256==digest,'Independent raw value/provenance differs')
        cds=acq/'cds_three';cplan=read(cds/'plan.json');cds_values=0
        for j in cplan['jobs']:
            receipt=read(cds/'jobs'/j['id']/'download.json');p=cds/receipt['file'];need(sha(p)==receipt['sha256'],'Independent CDS receipt')
            if zipfile.is_zipfile(p):
                with zipfile.ZipFile(p) as z:
                    members=[n for n in z.namelist() if n.endswith('.nc')];need(len(members)==1,'CDS independent product count')
                    ds=netCDF4.Dataset('memory',memory=z.read(members[0]))
            else:ds=netCDF4.Dataset(p)
            with ds:
                np.testing.assert_array_equal(ds['valid_time'][:],j['epochs']);np.testing.assert_array_equal(ds['latitude'][:],[j['latitude']])
                np.testing.assert_array_equal((ds['longitude'][:]+180)%360-180,[j['longitude']])
                for v,column,pid,unit in [('kx','k_index',260121,'K'),('totalx','total_totals_index',260123,'K'),('ptype','precipitation_type',260015,'(Code table 4.201)')]:
                    x=ds[v];need(x.GRIB_paramId==pid and x.units==unit and x.GRIB_stepType=='instant','Independent CDS metadata')
                    for t,val in zip(j['epochs'],np.asarray(x[:]).ravel()):
                        record=keyed.loc[(j['station'],t,'surface__'+column)]
                        need(record.value==float(val) and record.source_sha256==receipt['sha256'],'Independent CDS value differs');cds_values+=1
        expected=pd.DataFrame([dict(station=r.station,utc_epoch_seconds=r.utc_epoch_seconds,feature=r.feature,value=r.value) for r in prov.itertuples(index=False)])
        wide=expected.pivot(index=['station','utc_epoch_seconds'],columns='feature',values='value')
        actual=merged.set_index(['station','utc_epoch_seconds'])
        pd.testing.assert_frame_equal(actual[missing].sort_index().sort_index(axis=1),wide.sort_index().sort_index(axis=1),check_names=False,check_exact=True)
        weather=pd.read_parquet(ROOT/'Outputs/Experiments/Combined_Channel_Revision_008/weather_labels_all.parquet')
        weather=merged[['station','utc_epoch_seconds','primary_score_eligible']].merge(weather.drop(columns='primary_score_eligible'),on=['station','utc_epoch_seconds'],validate='one_to_one')
        rain=(weather.total_water_mm>=.1)&(weather.snow_water_mm<.1)
        over=(weather.total_water_mm<.1)&(weather.snow_water_mm<.1)&(weather.tcc>=.8)
        need(np.array_equal(rain,weather.event.eq('rain')) and np.array_equal(over,weather.event.eq('dry_overcast')),'Independent weather rules')
        eligible=merged.primary_score_eligible
        need(int(eligible.sum())==5593 and int((rain&eligible).sum())==735 and int((over&eligible).sum())==2317,'Frozen scoring counts')
        need(merged[['station','hour_utc']].assign(day=merged.hour_utc.astype(str).str[:10])[['station','day']].drop_duplicates().shape[0]==236,'236 station days')
        geom=pd.read_parquet(acq/'geometry.parquet');audit_geometry(geom)
        emit(folder/'independent_input_qa.json',dict(status='PASS',utc=now(),hours=5664,scoring_hours=5593,rain=735,dry_overcast=2317,
            independent_ncar_raw_grid_values=ncar_count,independent_cds_raw_values=cds_values,new_station_hour_field_values=len(prov),
            input_sha256=sha(folder/'uk_inputs_38.parquet'),script_sha256=sha(Path(__file__)),
            old_columns_labels_masks_exactly_unchanged=True,independent_geometry_pressure_and_grid_replay=True,
            imputation_of_unacquired_fields=False,training_calls=0))
        print('INDEPENDENT_INPUT_QA_PASS',flush=True)
if __name__=='__main__':main()
