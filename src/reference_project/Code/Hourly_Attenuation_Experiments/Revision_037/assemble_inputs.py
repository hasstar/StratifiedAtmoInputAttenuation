"""Assemble all 27 acquired columns, preserving every historical UK column."""
import zipfile, re
import numpy as np
import pandas as pd
import netCDF4
from ncar_inputs import *
from cds_three import DEST as CDS, plan as cds_plan

CDS_FIELDS={'kx':('k_index',260121,'K'),'totalx':('total_totals_index',260123,'K'),
            'ptype':('precipitation_type',260015,'(Code table 4.201)')}

def read_cds(job,receipt):
    path=CDS/receipt['file'];need(sha(path)==receipt['sha256'],'CDS source SHA')
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as z:
            members=[x for x in z.namelist() if x.endswith('.nc')];need(len(members)==1,'Expected single CDS instant product')
            need(z.getinfo(members[0]).file_size<20*1024**2,'CDS payload size')
            ds=netCDF4.Dataset('memory',memory=z.read(members[0]))
    else:ds=netCDF4.Dataset(path)
    try:
        need(set(ds.dimensions)=={'valid_time','latitude','longitude'},'CDS dimensions')
        t=ds['valid_time'];need(t.units=='seconds since 1970-01-01','CDS time units')
        epochs=t[:].astype(np.int64);need(np.array_equal(epochs,job['epochs']),'CDS exact frozen UTC hours')
        need(ds['latitude'].units=='degrees_north' and ds['longitude'].units=='degrees_east','CDS spatial units')
        need(np.array_equal(ds['latitude'][:],[job['latitude']]),'CDS latitude')
        need(np.array_equal((ds['longitude'][:]+180)%360-180,[job['longitude']]),'CDS longitude')
        need(set(ds.variables)-{'valid_time','latitude','longitude','expver','number'}==set(CDS_FIELDS),'CDS extra/missing field')
        if 'expver' in ds.variables:need(set(str(x) for x in ds['expver'][:]).issubset({'0001','1'}),'Unexpected ERA5 experiment')
        rows=[];metadata={}
        for name,(field,pid,unit) in CDS_FIELDS.items():
            v=ds[name];need(v.dimensions==('valid_time','latitude','longitude'),'CDS variable dimensions')
            need(int(v.GRIB_paramId)==pid and v.GRIB_shortName==name and v.GRIB_typeOfLevel=='surface' and
                 v.GRIB_stepType=='instant' and v.units==unit,'CDS product identity/units')
            array=v[:];need(not np.ma.getmaskarray(array).any(),'CDS missing values');values=np.asarray(array).ravel()
            need(values.shape==epochs.shape and np.isfinite(values).all(),'CDS value coverage')
            if name=='ptype':need(np.equal(values,np.round(values)).all() and np.isin(values,[0,1,3,5,6,7,8]).all(),'Precipitation type code table')
            metadata[name]=dict(param_id=pid,units=unit,long_name=v.long_name,minimum=float(values.min()),maximum=float(values.max()))
            for t,x in zip(epochs,values):rows.append(dict(station=job['station'],utc_epoch_seconds=int(t),feature='surface__'+field,
                value=float(x),source=rel(path),source_sha256=receipt['sha256'],source_kind='CDS',latitude=job['latitude'],longitude=job['longitude']))
        return rows,metadata
    finally:ds.close()

def main():
    with stage_lock('assemble_inputs'):
        folder=OUT/'inputs';need(not folder.exists(),'Input build already exists');folder.mkdir()
        old=pd.read_parquet(DATA);nplan=read(ACQ/'ncar_plan.json');lookup={};source_records=[]
        need((ACQ/'ncar_download_complete.json').exists(),'NCAR download incomplete')
        for n in nplan['nodes']:
            raw=ACQ/'ncar_values'/f'{n["id"]}.ascii';das=ACQ/'ncar_metadata'/f'{fp(n["url_path"])[:24]}.das'
            rr=read(raw.with_name(raw.name+'.receipt.json'));dr=read(das.with_name(das.name+'.receipt.json'))
            need(sha(raw)==rr['sha256'] and rr['url']==n['url'] and sha(das)==dr['sha256'],'NCAR source SHA')
            values=decode(raw.read_text(),das.read_text(),n)
            for key,value in zip(node_keys(n),values):
                need(key not in lookup,'Duplicate NCAR value');lookup[key]=(value,rel(raw),rr['sha256'])
            source_records.append(dict(source=rel(raw),source_sha256=rr['sha256'],metadata=rel(das),metadata_sha256=dr['sha256'],url=n['url']))
        demands=pd.read_parquet(ACQ/'ncar_demands.parquet');rows=[]
        for r in demands.itertuples(index=False):
            key=(r.spec,r.level_hpa,r.utc_epoch_seconds,r.latitude_index,r.longitude_index);value,path,digest=lookup[key]
            rows.append(dict(station=r.station,utc_epoch_seconds=r.utc_epoch_seconds,feature=r.feature,value=value,
                source=path,source_sha256=digest,source_kind='NCAR',latitude=90-r.latitude_index*.25,
                longitude=((r.longitude_index*.25)+180)%360-180))
        cds_metadata={}
        for j in cds_plan()['jobs']:
            receipt=read(CDS/'jobs'/j['id']/'download.json');need(receipt['request_sha256']==j['request_sha256'],'CDS request identity')
            values,meta=read_cds(j,receipt);rows+=values;cds_metadata[j['id']]=meta
        provenance=pd.DataFrame(rows);keys=['station','utc_epoch_seconds','feature']
        need(len(provenance)==5664*27 and not provenance.duplicated(keys).any(),'27 complete new fields')
        expected={r['feature'] for r in read(OUT/'required_fields_38.json') if not r['available_all_5664']}
        need(set(provenance.feature)==expected,'Exact new field set')
        wide=provenance.pivot(index=['station','utc_epoch_seconds'],columns='feature',values='value').reset_index()
        need(len(wide)==5664 and np.isfinite(wide[sorted(expected)].to_numpy(float)).all(),'New field coverage')
        merged=old.merge(wide,on=['station','utc_epoch_seconds'],how='left',validate='one_to_one',sort=False)
        pd.testing.assert_frame_equal(merged[old.columns],old,check_exact=True)
        need(np.isfinite(merged[[r['feature'] for r in read(OUT/'required_fields_38.json')]].to_numpy(float)).all(),'All38 input fields')
        merged.to_parquet(folder/'uk_inputs_38.parquet',index=False)
        provenance.to_parquet(folder/'new_field_provenance.parquet',index=False)
        summary=provenance.groupby('feature').value.agg(['count','min','max']).reset_index()
        summary['negative_values']=summary.feature.map(provenance.assign(negative=provenance.value<0).groupby('feature').negative.sum())
        summary.to_csv(folder/'new_field_summary.csv',index=False,encoding='utf-8-sig',mode='x')
        emit(folder/'cds_metadata.json',cds_metadata);emit(folder/'ncar_sources.json',source_records)
        emit(folder/'build.json',dict(status='BUILT_PENDING_INDEPENDENT_INPUT_QA',utc=now(),hours=5664,eligible=5593,new_fields=27,
            all_old_columns_exactly_unchanged=True,source_sha256=read(OUT/'protected_sha256.json'),
            outputs={p.name:sha(p) for p in folder.iterdir() if p.is_file()},clipping=False,training_calls=0))
        print('UK_38_FIELDS_BUILT_PENDING_QA',flush=True)
if __name__=='__main__':main()
