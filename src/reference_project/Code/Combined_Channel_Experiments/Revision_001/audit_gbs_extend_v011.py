"""Independent calendar/allowed-grid request expansion; no downloads/models."""
import csv
import json
import math
from collections import Counter
from datetime import datetime,timezone
import weather_download_gbs_extend_v011 as d

def audit():
    p=d.app.load_plan()
    with (d.ROOT/'外部样例数据CEDA/GBS_20p7GHz/source_manifest.csv').open(encoding='utf-8-sig',newline='') as f:
        native={(r['site'],r['date_utc']) for r in csv.DictReader(f) if r['site'] in ('chilbolton','sparsholt')}
    expected={(s,int(datetime.fromisoformat(dt).replace(tzinfo=timezone.utc).timestamp())+3600*h) for s,dt in native for h in range(24)}
    for kind in ('rain','surface','pressure'):
        observed=Counter((s,t-(3600 if kind=='rain' else 0)) for q in p['parents'] if q['kind']==kind for s in q['link_ids'] for t in q['hours_utc_epoch'])
        assert observed==Counter(expected),kind
    assert len(p['links'])==2 and p['auxiliary_pairs']==[] and len(p['feature_names'])==10
    assert p['added_diagnostic_fields']==['mean_total_precipitation_rate','mean_snowfall_rate']
    # Independent raw array index -> exact field/UTC/level/cell expansion.
    sample_keys=set()
    for q in p['parents']:
        times=q['hours_utc_epoch']
        for s in q['link_ids']:
            link=next(x for x in p['links'] if x['id']==s)
            if q['kind']=='pressure':
                for level,variables in d.core.LEVEL_VARS.items():
                    cell=d.core.seed_cell(link,1) if level==1 else tuple(d.core.fixed_location(link,level)['cell'])
                    sample_keys.update((v,level,t,*cell) for v in variables for t in (times[0],times[-1]))
            else:
                cell=(math.floor(link['latitude_deg_n']/.25+.5),math.floor(link['longitude_deg_e']/.25+.5))
                variables=d.transport.RAIN if q['kind']=='rain' else d.core.SURFACE
                sample_keys.update((v,0,t,*cell) for v in variables for t in (times[0],times[-1]))
    observed=Counter(); e1900=int(datetime(1900,1,1,tzinfo=timezone.utc).timestamp())
    for node in d.transport.plan_requests(sample_keys):
        assert node['url'].startswith(d.transport.BASE) and '.ascii?' in node['url']
        c=node['coords']
        times=[e1900+3600*(a+b) for a in c['forecast_initial_time'] for b in c['forecast_hour']] if node['layout']=='forecast' else [e1900+3600*a for a in c['time']]
        if node['level_hpa']: assert c['level']==[node['level_hpa']]
        for t in times:
            for lat in c['latitude']:
                for lon in c['longitude']:
                    observed[(node['variable'],node['level_hpa'],t,round(lat*4),round(((lon+180)%360-180)*4))]+=1
    assert observed==Counter(sample_keys)
    result={'status':'PASS','plan_fingerprint':p['fingerprint'],'code_sha256':d.SHA(d.CODE),
            'native_station_days':len(native),'station_hours':len(expected),'calendar_counts':p['scope_counts'],
            'independent_request_samples':len(sample_keys),'old_protected_sha_checked':len(p['protected_files']),
            'scope':'Independent calendar/exact representative request identity audit; not full acquired field or model QA',
            'download_calls':0,'model_calls':0}
    d.io.immutable(d.DEST/'plan_independent_qa.json',result)
    print(json.dumps(result,indent=2))

if __name__=='__main__':audit()
