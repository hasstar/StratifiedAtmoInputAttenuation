"""Two-station, download-only continuation authorized 2026-10-16.

No old worker entry point is called. Frozen transport/geometry are read-only
dependencies; all mutable globals and outputs are redirected to this version.
"""
from __future__ import annotations
import argparse
import copy
import csv
import hashlib
import json
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone
from pathlib import Path
import netCDF4
import numpy as np
import weather_download_top10_ncar_v009 as previous
import ncar_rain_transport_v002 as transport

app = previous.app
io, core, legacy = app.io, app.core, app.legacy
ROOT, HERE, EXTERNAL = app.ROOT, app.HERE, app.EXTERNAL
DEST = EXTERNAL/'weather_gbs_extend_v011'
CODE = Path(__file__).resolve()
PLAN = DEST/'plan.json'
OLD9 = EXTERNAL/'weather_top10_ncar_v009'
OLD_SCOPE = EXTERNAL/'validation_gbs_48h_v010'
STATIONS = ('chilbolton','sparsholt')
R, SHA, FP, require = app.R, app.SHA, app.FP, app.require
app.DEST = io.DEST = DEST
app.CODE, app.PLAN, app.transport = CODE, PLAN, transport
raw_hours = core.parent_hours
raw_needs = legacy.acquisition_needs
raw_progress = previous.raw_progress

def hours(parent):
    if 'hours_utc_epoch' not in parent:
        return raw_hours(parent)
    hs = parent['hours_utc_epoch']
    require(hs == sorted(set(hs)) and len(hs) == parent['expected_hours'] and all(t%3600 == 0 for t in hs), 'Invalid exact hour mask')
    return hs
core.parent_hours = hours

def acquisition_needs(plan, parent, cache):
    selected = [x for x in plan['links'] if x['id'] in parent['link_ids']]
    require(len(selected) == len(parent['link_ids']) and set(parent['link_ids']) <= set(STATIONS), 'Station escaped scope')
    if parent['kind'] == 'rain':
        targets = set()
        records = []
        for link in selected:
            cell = core.grid_index(link['latitude_deg_n'], link['longitude_deg_e'])
            targets.update((v,0,t,*cell) for v in transport.RAIN for t in hours(parent))
            records.append({'link_id':link['id'], 'surface_cell':list(cell), 'hours_utc_epoch':hours(parent),
                            'beacon_interval_start_utc_epoch':[t-3600 for t in hours(parent)],
                            'interval':'[beacon hour, beacon hour + 1h); source timestamp is interval END'})
        require(all(np.isfinite(cache[k][0]) for k in targets if k in cache), 'Invalid rain cache value')
        return 'hourly_precipitation_and_snow_rates', targets-cache.keys(), records
    return raw_needs({**plan, 'links':selected}, parent, cache)
legacy.acquisition_needs = acquisition_needs

def load_cache(plan, parent):
    # Frozen loader supplies old CDS and this version's completed DAP values.
    cache, hashes = previous.raw_cache(plan, parent)
    wanted = set(hours(parent))
    pressure = parent['kind'] == 'pressure'
    for item in plan['ncar_reusable']:
        node = item['node']
        if bool(node['level_hpa']) != pressure:
            continue
        if not any(k[2] in wanted for k in transport.expected_keys(node)):
            continue
        record = previous.old_records(item)
        for key, value in record['records']:
            key = tuple(key)
            if key[2] in wanted and key not in cache:
                cache[key] = value, record['source']
        hashes[record['source']] = record['source_sha256']
    return cache, hashes
app.load_cache = load_cache

def source_inventory(links):
    root = ROOT/'外部样例数据CEDA/GBS_20p7GHz'
    manifest = root/'source_manifest.csv'
    with manifest.open(encoding='utf-8-sig', newline='') as f:
        rows = [r for r in csv.DictReader(f) if r['site'] in STATIONS]
    require(len({(r['site'],r['date_utc']) for r in rows}) == len(rows), 'Repeated native day')
    evidence = []
    for row in rows:
        path = root/row['relative_path']
        data = path.read_bytes()
        require(len(data)==int(row['bytes']) and hashlib.md5(data).hexdigest()==row['md5'], 'GBS native source corrupted')
        link = next(x for x in links if x['id']==row['site'])
        with netCDF4.Dataset('native.nc', memory=data) as ds:
            lat = float(ds['latitude_receiver'][:]); lon = float(ds['longitude_receiver'][:])
            require(ds['latitude_receiver'].units=='degrees_north' and ds['longitude_receiver'].units in ('degrees_east','degrees_west'), 'Native coordinate units')
            if ds['longitude_receiver'].units == 'degrees_west': lon = -lon
            lon = (lon+180)%360-180
            require(abs(lat-link['latitude_deg_n'])<1e-8 and abs(lon-link['longitude_deg_e'])<1e-8, 'Date-dependent station geometry needs review')
            require(float(ds['receiver_altitude_amsl'][:])==link['height_amsl_m'], 'Date-dependent antenna height')
            require(ds['receiver_altitude_amsl'].units=='m' and ds['frequency'].units=='GHz' and abs(float(ds['frequency'][:])-20.7)<1e-4, 'GBS antenna/frequency unit or value changed')
            satellite=float(ds['satellite_position'][:])
            require(ds['satellite_position'].units in ('degrees_west','degrees_east'),'Satellite longitude convention')
            if ds['satellite_position'].units=='degrees_west': satellite=-satellite
            require(abs((satellite+180)%360-180-link['satellite_longitude_deg_e'])<1e-8,'GBS satellite changed')
            require(ds['elevation_angle'].units=='degrees' and float(ds['elevation_angle'][:])==link['source_coarse_elevation_deg'],'GBS published pointing changed')
        evidence.append({'station':row['site'],'date_utc':row['date_utc'], 'path':str(path.relative_to(ROOT)),
                         'sha256':hashlib.sha256(data).hexdigest(), 'bytes':len(data)})
    return manifest, evidence

def make_parents(evidence, old_dates):
    # Identical per-station month/date masks share requests; no missing-day fill.
    grouped = defaultdict(lambda:defaultdict(set))
    for row in evidence:
        phase = 'baseline39' if row['date_utc'] in old_dates else 'extension'
        grouped[(phase,row['date_utc'][:7])][row['station']].add(row['date_utc'])
    result = []
    for (phase, month), sites in sorted(grouped.items()):
        masks = defaultdict(list)
        for site, dates in sites.items(): masks[tuple(sorted(dates))].append(site)
        for mask, sites_ in sorted(masks.items()):
            hh = sorted(int(datetime.fromisoformat(d).replace(tzinfo=timezone.utc).timestamp())+h*3600 for d in mask for h in range(24))
            for kind in ('rain','surface','pressure'):
                req = {'year':[month[:4]],'month':[month[5:]],'day':[d[-2:] for d in mask], 'time':[f'{h:02d}:00' for h in range(24)]}
                if kind=='pressure': req['pressure_level']=['1'] # Calendar flag only; never a submission template.
                result.append({'id':f'GBS_England_{month}_{phase}_{"-".join(sorted(sites_))}_{kind}',
                    'kind':kind, 'phase':phase, 'link_ids':sorted(sites_), 'dates_utc':list(mask), 'request':req,
                    'hours_utc_epoch':[t+3600 for t in hh] if kind=='rain' else hh, 'expected_hours':len(hh)})
    return result

def build_plan():
    require(not PLAN.exists() and not io.lock_held(), 'Plan already exists or writer occupied')
    old = app.verified_record(OLD9/'plan.json')
    require(SHA(OLD9/'plan.json')=='d857e59a91b672c4364d2f45a7f4441d0c06ab49861ae06d1c3d1e2a986911f5', 'Old plan changed')
    scope = R(OLD_SCOPE/'scope_manifest.json')
    require(SHA(OLD_SCOPE/'scope_manifest.json')=='43864c01e313b768ca21a8cbda396cda7a9a3dfbab71975f524a80089333f052','Frozen 39-day scope changed')
    links = [x for x in old['links'] if x['id'] in STATIONS]
    manifest, evidence = source_inventory(links)
    plan = copy.deepcopy(old); plan.pop('fingerprint')
    reusable = list(old['ncar_reusable'])
    for path in sorted((OLD9/'jobs').glob('*.json')):
        s = R(path)
        if s.get('status')!='VERIFIED_SUBSET': continue
        rp = OLD9/'records'/(s['job']['id']+'.json')
        item = {'node':s['job'], 'source_state':str(path.relative_to(ROOT)), 'state_sha256':SHA(path),
                'sha256':s['sha256'],'record_path':str(rp.relative_to(ROOT)), 'records_sha256':SHA(rp),
                'values':s['values'],'response_bytes':s['receipt']['bytes']}
        previous.old_records(item); reusable.append(item)
    sources = dict(old['source_sha256'])
    for p in (CODE,HERE/'ncar_rain_transport_v002.py',HERE/'test_gbs_extend_v011.py',HERE/'audit_gbs_extend_v011.py',HERE/'preflight_gbs_extend_v011.py'):
        sources[str(p)]=SHA(p)
    protected = dict(old['protected_files'])
    for p in OLD9.rglob('*'):
        if p.is_file(): protected[str(p.relative_to(ROOT))]=SHA(p)
    for row in evidence: protected[row['path']]=row['sha256']
    protected[str(manifest.relative_to(ROOT))]=SHA(manifest)
    protected[str((OLD_SCOPE/'scope_manifest.json').relative_to(ROOT))]=SHA(OLD_SCOPE/'scope_manifest.json')
    plan.update(version='gbs-two-station-extension-v011', authorization='Human 2026-10-16: extend dates and rain of existing two stations; download until told stop; no experiments',
        parents=make_parents(evidence,set(scope['dates_utc'])), links=links, native_days=evidence,
        ncar_reusable=reusable, source_sha256=sources, protected_files=protected,
        pointing_review_required_ids=[], predecessor_plan=str((OLD9/'plan.json').relative_to(ROOT)),
        scope='All locally inventoried native GBS days of the two authorized stations only; baseline39 first, then chronological expansion',
        added_diagnostic_fields=list(transport.RAIN), rain_source='NCAR hourly MTPR + MSR (ECMWF 235055/235031); raw rates retained; total is not liquid-only',
        rain_interval='Source valid UTC is end of hour. Requests t+1h cover each beacon interval [t,t+1h). 3600*rate yields mm/hour accumulation; no cumulative differencing.',
        rain_liquid_policy='Optional later liquid estimate = 3600*(MTPR-MSR); preserve negative differences and flag, no clamping or re-labelling as gauge truth',
        gauge_status='CEDA exact public file redirects to login (302); not downloaded, no token read or bypass. No minute-gauge completeness claim.',
        scope_counts={s:{'days':sum(x['station']==s for x in evidence),'station_hours':24*sum(x['station']==s for x in evidence),
                         'first':min(x['date_utc'] for x in evidence if x['station']==s),'last':max(x['date_utc'] for x in evidence if x['station']==s)} for s in STATIONS},
        stop_policy='STOP_REQUESTED.json or human stop: no new GET; allow already started bounded GET to settle, keep bytes. No restart while stop flag exists.',
        completion_policy='Finite native-date archive complete => idle monitoring, do not expand to other stations or unlabelled years; only human stop pauses tvt',
        new_fit_calls=0,predictions_executed=0)
    return app.signed(plan)

def progress(plan, verify=False):
    result = raw_progress(plan, verify)
    result.update(scope_counts=plan['scope_counts'], authorized_stations=list(STATIONS),
        stop_requested=(DEST/'STOP_REQUESTED.json').exists(), original_top10_columns=10,
        added_rain_raw_fields=list(transport.RAIN), reused_ncar_candidate_values=sum(x['values'] for x in plan['ncar_reusable']),
        reused_candidates_are_not_complete_station_hours=True, gauge_downloaded=False)
    units = [app.verified_record(p) for p in (DEST/'units').glob('*.json')]
    ids = {x['parent_id'] for x in units}
    result['acquisition_station_hours_by_kind'] = {kind:sum(p['expected_hours']*len(p['link_ids']) for p in plan['parents'] if p['kind']==kind and p['id'] in ids) for kind in ('rain','surface','pressure')}
    return result
app.progress = progress

def stopped(): return (DEST/'STOP_REQUESTED.json').exists()

raw_get_body = app.get_body
def get_body(url, partial, node):
    require(not stopped(), 'Human stop: no new GET is permitted')
    return raw_get_body(url, partial, node)
app.get_body = get_body

def run():
    require(not stopped(),'Human stop flag present; no restart')
    plan = app.load_plan()
    for name in ('preflight_qa.json','plan_independent_qa.json'):
        qa = R(DEST/name)
        require(qa['status']=='PASS' and qa['plan_fingerprint']==plan['fingerprint'] and qa['code_sha256']==SHA(CODE),'Matching preflight absent')
    progress(plan,True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        active={}
        while True:
            for future in list(active):
                if future.done(): future.result(); active.pop(future)
            if stopped():
                io.atomic(DEST/'progress.json',{**progress(plan),'state':'STOP_REQUESTED_DRAINING' if active else 'STOPPED_BY_USER'})
                if not active: return
                wait(active,timeout=1,return_when=FIRST_COMPLETED); continue
            pending=[n for n in app.nodes(plan) if app.state(n).get('status')!='VERIFIED_SUBSET']
            if any(app.state(n).get('status','').startswith('BLOCKED') for n in pending):
                io.atomic(DEST/'progress.json',progress(plan)); raise RuntimeError('Blocked subset: diagnose; no fallback to old full plans')
            if not pending and not active:
                if app.advance(plan):
                    io.immutable(DEST/'acquisition_complete.json',{'status':'FINITE_TWO_STATION_ARCHIVE_ACQUIRED_QA_PENDING',**progress(plan,True)})
                    io.atomic(DEST/'progress.json',progress(plan)); return
                pending=[n for n in app.nodes(plan) if app.state(n).get('status')!='VERIFIED_SUBSET']
            running={n['id'] for n in active.values()}
            for node in pending:
                if len(active)>=2 or stopped(): break
                if node['id'] not in running and app.state(node).get('next_retry_epoch',0)<=time.time():
                    active[pool.submit(app.download,node,plan)] = node
            io.atomic(DEST/'progress.json',progress(plan))
            if active: wait(active,timeout=5,return_when=FIRST_COMPLETED)
            else: time.sleep(2)

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('command',choices=('prepare','run','status','stop')); args=parser.parse_args()
    if args.command=='stop':
        io.atomic(DEST/'STOP_REQUESTED.json',{'utc':io.now(),'reason':'Direct human stop instruction'}); return
    if args.command=='status':
        print(json.dumps({**progress(app.load_plan(),True),'os_lock_held':io.lock_held()},indent=2)); return
    if args.command=='prepare':
        candidate=build_plan()
        with io.writer('gbs_extend_prepare'):
            io.immutable(PLAN,candidate)
        print(json.dumps({'plan_fingerprint':candidate['fingerprint'],'scope_counts':candidate['scope_counts'],'parents':len(candidate['parents'])})); return
    with io.writer('gbs_extend_run'): run()

if __name__=='__main__': main()
