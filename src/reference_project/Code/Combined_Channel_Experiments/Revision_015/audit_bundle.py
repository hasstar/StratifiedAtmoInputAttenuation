"""One-shot offline evidence/coverage audit; not an RMOP or scoring implementation."""
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal, localcontext
import hashlib
import json
import math
import msvcrt
import os
import sys
import traceback

ROOT = Path(__file__).resolve().parents[3]
CODE = Path(__file__).resolve().parent
OUT = ROOT / 'Outputs/Experiments/Combined_Channel_Revision_015'
PREV = ROOT / 'Outputs/Experiments/Combined_Channel_Revision_014'
RAWQA = ROOT / 'Outputs/Experiments/Combined_Channel_Revision_013/attempt_002'
SCORE = ROOT / 'Outputs/Experiments/Combined_Channel_Revision_012'
PDF = Path('C:/Users/SDR_HH/Desktop/Dynamical_Link_Budget_in_Satellite_Communications_at_Ka-Band_Testing_Radiometeorological_Forecasts_With_Hayabusa2_Deep-Space_Mission_Support_Data.pdf')

def utc():
    return datetime.now(timezone.utc).isoformat()

def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))

def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def emit(name, value):
    with (OUT / name).open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')

def need(ok, msg):
    if not ok:
        raise RuntimeError(msg)

def check(entries):
    bad = [p for p, h in entries.items() if not Path(p).is_file() or sha(p) != h]
    need(not bad, f'Protected source mismatch: {bad}')

def main():
    need(not (OUT / 'launch.json').exists(), 'Already started; do not rerun or overwrite')
    guard = read(PREV / 'sources_before.json')
    check(guard)
    prior = read(PREV / 'delivery_manifest.json')
    for p, h in prior.items():
        path = ROOT / p
        need(sha(path) == h, f'Prior delivery changed: {p}')
        guard[str(path)] = h
    for p in [PREV / 'delivery_manifest.json', PREV / 'final_verification.json', PDF]:
        guard[str(p)] = sha(p)
    need(guard[str(PDF)] == 'f7ae0f8df59ced84c3b21bbb4378c7df998136ce7e0db8950187885a37e0184f', 'PDF changed')
    need(read(PREV / 'final_verification.json')['status'] == 'PASS', 'Prior QA failed')
    lock = (OUT / 'execution.lock').open('x+b')
    lock.write(b'0'); lock.flush(); lock.seek(0)
    msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    try:
        emit('launch.json', {'utc': utc(), 'pid': os.getpid(), 'command': sys.argv,
                             'executable': sys.executable, 'cwd': str(Path.cwd()), 'scope': 'offline audit only'})
        emit('sources_before.json', guard)
        emit('code_before.json', {str(p): sha(p) for p in CODE.iterdir() if p.is_file()})
        inv = read(RAWQA / 'file_inventory.json')
        raw = read(RAWQA / 'audit_complete.json')
        ledger = read(RAWQA / 'hourly_input_ledger.json')
        test = [r for r in ledger if r['original_test_key']]
        need(len(test) == 687 and all(r['nine_pressure_fields_all_grid_finite'] for r in test), 'Raw test field coverage')
        variables = sorted({v for item in inv for v in item['variables']})
        ids = {'clwc': 246, 'crwc': 75, 'ciwc': 247, 'cswc': 76, 't': 130, 'q': 133, 'z': 129}
        rows = []
        for v, param in ids.items():
            files = [i for i in inv if v in i['variables']]
            need(len(files) == 12, f'File count {v}')
            need(all(i['variables'][v]['parameter_id'] == param for i in files), f'Identity {v}')
            need(all(i['pressure_hpa'] == [1000,925,850,700,650,500,300,200,30,5,1] for i in files), f'Levels {v}')
            rows.append({'field': v, 'parameter_id': param, 'files': len(files),
                         'units': sorted({i['variables'][v]['units'] for i in files}),
                         'test_hours_raw_finite': 687, 'slant_input_accepted': False})
        need(len(variables) == 22 and len(inv) == 144, 'Inventory scope')
        emit('input_contract.json', {'utc': utc(), 'raw_inventory_sha256': sha(RAWQA/'file_inventory.json'),
             'local_raw_files': len(inv), 'local_variables': variables, 'known_fields': rows,
             'graupel': 'NOT_IN_AUDITED_INVENTORY', 'paper_hydrometeor_species': ['cloud','rain','ice','snow','graupel'],
             'actual_gsdsu_version': None, 'author_config_verified': False, 'licensed_package_verified': False,
             'output_attenuation_contract_verified': False, 'station_satellite_slant_qa': False,
             'rmop_complete_input_hours': None, 'rmop_status': 'BLOCKED_NOT_COMPUTED',
             'local_test_hours_retained': 687, 'weather_acquisition_authorized': False})
        allhours = read(SCORE / 'all_hour_ledger.json')
        cov = []
        for station in sorted({r['station'] for r in allhours}):
            selected = [r for r in allhours if r['station'] == station]
            cov.append({'station': station, 'hours': len(selected), 'dates': len({r['date'] for r in selected}),
                        'label_valid': sum(bool(r['primary_score_eligible']) for r in selected),
                        'old_physical_common': sum(bool(r['common_score_eligible']) for r in selected),
                        'events_common': dict(Counter(r['event'] for r in selected if r['common_score_eligible']))})
        need(len(allhours) == 6351, 'Frozen full scope changed')
        need(sum(c['hours'] for c in cov if c['station'] != 'LOCAL') == 5664, 'UK scope')
        need(sum(c['label_valid'] for c in cov if c['station'] != 'LOCAL') == 5593, 'UK label scope')
        need(sum(c['old_physical_common'] for c in cov if c['station'] != 'LOCAL') == 5538, 'UK common scope')
        emit('coverage_recheck.json', {'status': 'PASS', 'ledger_sha256': sha(SCORE/'all_hour_ledger.json'),
             'stations': cov, 'new_model_mask_created': False, 'metrics_recalculated': False})
        with localcontext() as ctx:
            ctx.prec = 60
            conversions = []
            for value in [1, 2, 10, 100]:
                expected = float(Decimal(10) * Decimal(value).log10())
                actual = 10 * math.log10(value)
                need(abs(expected-actual) < 1e-12, 'Power/dB conversion')
                conversions.append({'linear_power_loss': value, 'db': actual, 'decimal_expected_db': expected})
            average_db = (10 * math.log10(1) + 10 * math.log10(10))/2
            db_of_mean = 10 * math.log10((1+10)/2)
            need(abs(average_db-5) < 1e-12 and db_of_mean > 7.4, 'Aggregation distinction')
            need(10-12 == -2, 'Signed relative difference')
        emit('unit_sanity_qa.json', {'status': 'PASS', 'fixture': 'SYNTHETIC_ONLY', 'checks': 6,
             'conversions': conversions, 'mean_db': average_db, 'db_of_mean_linear_loss': db_of_mean,
             'signed_reference_difference_db': -2, 'validates_rmop': False})
        check(guard)
        emit('audit_complete.json', {'status': 'AUDIT_COMPLETE_MODEL_BLOCKED', 'utc': utc(),
             'protected_sources': len(guard), 'source_sha_unchanged': True,
             'new_predictions': 0, 'new_scores': 0, 'new_training': 0, 'weather_requests': 0,
             'software_installs': 0, 'seven_family_comparison_complete': False})
    except Exception:
        emit('failure.json', {'utc': utc(), 'traceback': traceback.format_exc()})
        raise
    finally:
        lock.seek(0); msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1); lock.close()
        emit('lock_release.json', {'utc': utc(), 'os_lock': 'RELEASED', 'pid': os.getpid()})
    print(json.dumps(read(OUT / 'audit_complete.json'), ensure_ascii=False))

if __name__ == '__main__':
    main()
