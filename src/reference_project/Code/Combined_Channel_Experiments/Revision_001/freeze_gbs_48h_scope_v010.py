"""Freeze the human-authorized reduced scope, using local inputs only.

No network calls, downloader entry points, model loads, training, or predictions.
This is acquisition availability/provenance checking, not independent field QA.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

import weather_download_top10_ncar_v009 as existing

ROOT = Path(__file__).resolve().parents[3]
OLD = ROOT / 'Outputs/Experiments/Combined_Channel_Revision_001/external/weather_top10_ncar_v009'
OUT = ROOT / 'Outputs/Experiments/Combined_Channel_Revision_001/external/validation_gbs_48h_v010'
STATIONS = ('chilbolton', 'sparsholt')
DATES = [f'2004-02-{d:02d}' for d in (18,19,20,21,25,26,27,28)] + [f'2004-03-{d:02d}' for d in range(1,32)]
PAIRS = {1: ('geopotential', 'u_component_of_wind'), 5: ('ozone_mass_mixing_ratio', 'u_component_of_wind'),
         500: ('ozone_mass_mixing_ratio',), 650: ('u_component_of_wind',), 850: ('ozone_mass_mixing_ratio','relative_humidity')}
SURFACE = ('convective_available_potential_energy', 'maximum_2m_temperature_since_previous_post_processing')

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def emit(name, data):
    path = OUT / name
    # Exclusive creation: this script never overwrites a frozen output.
    with path.open('x', encoding='utf-8', newline='\n') as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')
    return {'path': str(path.relative_to(ROOT)), 'sha256': sha(path)}

def main():
    app = existing.app
    assert not app.io.lock_held(), 'Downloader lock still held'
    assert sha(OLD/'plan.json') == 'd857e59a91b672c4364d2f45a7f4441d0c06ab49861ae06d1c3d1e2a986911f5'
    assert not (OUT/'scope_manifest.json').exists(), 'Scope already frozen; inspect, do not rerun'
    stop = read(OUT/'stop_verification.json')
    assert stop['status'] == 'STOPPED_BY_HUMAN_SCOPE_REDUCTION' and stop['actual_downloaders_remaining'] == 0
    plan = app.load_plan()
    assert plan['auxiliary_pairs'] == []
    names = plan['feature_names']
    assert len(names) == len(set(names)) == 10
    assert set(names) == {f'p{level}__{var}' for level, variables in PAIRS.items() for var in variables} | {f'surface__{var}' for var in SURFACE}
    links = {x['id']: x for x in plan['links'] if x['region'] == 'GBS_England'}
    assert set(links) == set(STATIONS)
    hours = [int((datetime.fromisoformat(day).replace(tzinfo=timezone.utc)+timedelta(hours=h)).timestamp()) for day in DATES for h in range(24)]
    assert len(hours) == len(set(hours)) == 936
    tables = {(site,t): {} for site in STATIONS for t in hours}
    provenance = {key: {} for key in tables}
    unique_keys = set()
    key_sources = {}
    unit_evidence = []
    geometry_counts = Counter()
    max_residual = 0.0
    for month in ('02','03'):
        wanted = [t for t in hours if datetime.fromtimestamp(t,timezone.utc).month == int(month)]
        for kind in ('surface','pressure'):
            ident = f'GBS_England_2004-{month}_{kind}'
            parent = next(p for p in plan['parents'] if p['id'] == ident)
            unit_path = OLD/'units'/(ident+'.json')
            unit = app.verified_record(unit_path)
            assert unit['plan_fingerprint'] == plan['fingerprint']
            assert unit['hour_scope'] == len(wanted) and unit['invalid_rows_retained'] == 0
            assert app.core.parent_hours(parent) == wanted
            cache, source_hashes = app.load_cache(plan,parent)
            phase, missing, reconstructed = app.legacy.acquisition_needs(plan,parent,cache)
            assert not missing, f'Local acquisition incomplete: {ident}, {len(missing)}'
            assert app.FP(reconstructed) == app.FP(unit['records']), 'Acquisition geometry changed'
            records_seen = set()
            for rec in unit['records']:
                site = rec['link_id']
                assert site in STATIONS and rec['hours_utc_epoch'] == wanted
                level = rec.get('level_hpa',0)
                assert (site,level) not in records_seen
                records_seen.add((site,level))
                if kind == 'pressure':
                    assert not rec['prediction_geometry_review_pending']
                    assert len(rec['rows']) == len(wanted)
                for i,t in enumerate(wanted):
                    if kind == 'pressure':
                        g = rec['rows'][i]
                        assert g['status'] in ('STABLE_NEAREST','STABLE_FIXED_HEIGHT_NEAREST')
                        geometry_counts[g['status']] += 1
                        max_residual = max(max_residual, abs(g['height_residual_m']))
                        cell = g['cell']
                        variables = PAIRS[level]
                    else:
                        cell = rec['surface_cell']
                        variables = SURFACE
                    for var in variables:
                        feature = f'p{level}__{var}' if level else f'surface__{var}'
                        key = (var,level,t,*cell)
                        value, source = cache[key]
                        assert math.isfinite(value)
                        assert feature not in tables[site,t]
                        assert sha(ROOT/source) == source_hashes[source], 'Actual source SHA mismatch'
                        tables[site,t][feature] = float(value)
                        provenance[site,t][feature] = {'key': list(key), 'source': source}
                        unique_keys.add(key)
                        if key in key_sources:
                            assert key_sources[key] == source, 'Cross-station source priority differs'
                        key_sources[key] = source
            expected_records = {(s,l) for s in STATIONS for l in PAIRS} if kind == 'pressure' else {(s,0) for s in STATIONS}
            assert records_seen == expected_records
            unit_evidence.append({'parent_id':ident,'path':str(unit_path.relative_to(ROOT)),'sha256':sha(unit_path),
                                  'fingerprint':unit['fingerprint'],'calendar_hours':len(wanted),'local_missing_unique_grid_field_hours':0})
            print('LOCAL SCOPE COMPLETE',ident,len(wanted),flush=True)
    assert len(tables) == 1872 and all(set(row)==set(names) for row in tables.values())
    assert max_residual < 0.001
    sources = {source:sha(ROOT/source) for source in sorted(set(key_sources.values()))}
    rows = [{'station':s,'utc_epoch_seconds':t,'utc':datetime.fromtimestamp(t,timezone.utc).isoformat(),
             'values':{name:tables[s,t][name] for name in names},'provenance':provenance[s,t]} for s,t in sorted(tables)]
    table_ref = emit('local_top10_acquisition.json',{'feature_names':names,'rows':rows,'independent_field_qa':False,'model_compatible':None,
                                                   'new_fit_calls':0,'predictions_executed':0})
    now = datetime.now().astimezone()
    route = Counter('NCAR' if 'weather_top10_ncar' in source else 'existing_CDS' for source in key_sources.values())
    summary = {'status':'REDUCED_SCOPE_LOCAL_ACQUISITION_COMPLETE','checked_local':now.isoformat(),
               'calendar_dates':39,'calendar_hours_per_station':936,'stations':2,'complete_top10_station_hours':1872,
               'station_feature_hour_values':18720,'unique_grid_field_hours':len(unique_keys),'unique_grid_field_hours_by_source':dict(route),
               'missing_grid_field_hours':0,'additional_downloads_required':0,'max_reproduced_height_residual_m':max_residual,
               'geometry_status_counts':dict(geometry_counts),'source_file_count':len(sources),
               'scope_integrity_checks':'PASS; exact calendars/stations/ten fields/finite values/source SHA and reproduced acquisition geometry',
               'independent_field_qa':False,'target_model_compatibility':'PENDING','new_fit_calls':0,'predictions_executed':0,
               'qualification':'Reused acquisition decoder; not independent scientific field/model QA. No whole-old-plan completion claim.'}
    availability = emit('availability_qa.json',summary)
    manifest = {'version':'gbs_48h_v010','created_local':now.isoformat(),
                'authority':'2026-10-15 human requires download completion within two days, permits fewer stations/datasets for external validation',
                'download_deadline_anchor':stop['checked_local'],'download_deadline_not_after_local':(datetime.fromisoformat(stop['checked_local'])+timedelta(days=2)).isoformat(),
                'selection_rule':'All earliest completely acquired GBS England units available at scope change; no model results or weather/target values used to select',
                'dataset':'GBS','region':'GBS_England','frequency_ghz':20.7,'stations':list(STATIONS),'dates_utc':DATES,
                'hours_utc_epoch_seconds':hours,'feature_names':names,'auxiliary_pairs':[],
                'sampling':'Four fixed standard heights with forward station-satellite LOS nearest published quarter-degree grid; 1hPa positioned with own Top10 z; surface station nearest grid; no interpolation',
                'links':list(links.values()),'fixed_heights':plan['fixed_heights'],
                'selected_units':unit_evidence,'source_sha256':sources,'predecessor_plan_sha256':sha(OLD/'plan.json'),
                'predecessor_plan_fingerprint':plan['fingerprint'],'local_acquisition_table':table_ref,'availability_qa':availability,
                'old_scope_status':'REDUCED_BY_HUMAN_AUTHORITY_NOT_COMPLETED','old_downloader_run_authorization':'REVOKED; never resume any old full-plan run',
                'excluded':'Dundee, OpenSat, and other GBS dates/months; all prior files and failure evidence preserved',
                'completion_gates':['Independent actual field/UTC/unit/coordinates/SHA/geometry QA','Frozen GBS target and model compatibility','Zero-training inference and metrics with applicable probabilistic comparison','Independent evaluation QA, figures, Chinese report, progress'],
                'new_fit_calls':0,'predictions_executed':0,'overall_task_complete':False,'generator_sha256':sha(__file__)}
    manifest['fingerprint'] = app.FP(manifest)
    emit('scope_manifest.json',manifest)
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__ == '__main__':
    main()
