"""One-shot synthetic tests and source guards. Never loads model/data arrays."""
import hashlib
import json
import math
import msvcrt
import os
import platform
import subprocess
import traceback
from datetime import datetime, timezone
from pathlib import Path

import prm_kernel as k

ROOT = Path(__file__).resolve().parents[3]
CODE = Path(__file__).resolve().parent
OUT = ROOT / 'Outputs/Experiments/Combined_Channel_Revision_014'
PREVIOUS = ROOT / 'Outputs/Experiments/Combined_Channel_Revision_013'


def need(ok, msg):
    if not ok:
        raise RuntimeError(msg)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def save(name, data):
    with (OUT / name).open('x', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')


def process_evidence():
    command = '''$ErrorActionPreference='Stop'
$samples=foreach($i in 1..2){
 $rows=@(Get-CimInstance Win32_Process -Filter "Name = 'python.exe' OR Name = 'pythonw.exe'" -ErrorAction Stop | Select-Object ProcessId,ParentProcessId,ExecutablePath,CommandLine,CreationDate,KernelModeTime,UserModeTime)
 [pscustomobject]@{utc=[DateTime]::UtcNow.ToString('o');query_success=$true;processes=$rows}
 if($i -eq 1){Start-Sleep -Seconds 3}
}
$samples | ConvertTo-Json -Depth 6 -Compress
'''
    result = subprocess.run(['powershell.exe', '-NoProfile', '-NonInteractive',
                             '-Command', command], capture_output=True,
                            text=True, timeout=30)
    need(result.returncode == 0, 'CIM query failed: not absence evidence')
    samples = json.loads(result.stdout)
    for sample in samples:
        need(sample['query_success'], 'CIM unsuccessful')
        for proc in sample['processes']:
            need(proc['ProcessId'] == os.getpid(), 'Another Python process: review')
            need(proc['CommandLine'] and proc['ExecutablePath'], 'Identity unavailable')
    return samples


def guards():
    merged = {}
    for path in [PREVIOUS/'attempt_002/protected_sources_before.json',
                 PREVIOUS/'attempt_002/raw_sources_before.json',
                 PREVIOUS/'delivery_manifest.json']:
        for source, digest in read(path).items():
            source = str((ROOT/source).resolve())
            need(source not in merged or merged[source] == digest, 'Guard conflict')
            merged[source] = digest
        merged[str(path)] = sha(path)
    merged[str(PREVIOUS/'final_verification.json')] = sha(PREVIOUS/'final_verification.json')
    return merged


def validate_guards(items):
    for path, digest in items.items():
        need(sha(path) == digest, 'Source changed: ' + path)


def run():
    need(not (OUT/'launch.json').exists(), 'One-shot output exists')
    samples = process_evidence()
    save('launch.json', {'utc': datetime.now(timezone.utc).isoformat(),
                        'pid': os.getpid(), 'python': platform.python_version(),
                        'command': 'python.exe -B Code/Combined_Channel_Experiments/Revision_014/run_tests.py',
                        'process_samples': samples})
    source_guards = guards()
    validate_guards(source_guards)
    save('sources_before.json', source_guards)
    code_hash = {p.name: sha(p) for p in CODE.iterdir() if p.is_file()}
    save('code_before.json', code_hash)
    checks = []
    fixtures = []

    def check(name, condition):
        need(condition, name)
        checks.append(name)

    def reject(name, fn):
        try:
            fn()
        except k.ContractError:
            checks.append(name)
            return
        raise RuntimeError('Invalid input accepted: '+name)

    with (OUT/'execution.lock').open('x+b') as lock:
        lock.write(b'0'); lock.flush(); lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        try:
            check('explicit_kg_to_g', k.qi_to_g_per_kg([0, .001], unit='kg/kg') == (0., 1.))
            check('g_kg_identity', k.qi_to_g_per_kg([0, 1], unit='g/kg') == (0., 1.))
            reject('reject_ambiguous_mass_unit', lambda: k.qi_to_g_per_kg([1], unit='unknown'))
            reject('reject_negative_mass', lambda: k.qi_to_g_per_kg([-1], unit='kg/kg'))
            for q, p, t in [([0,1,2],[100000,90000,80000],[300,280,250]),
                            ([.1,.2,.3],[101325,75000,50000],[288.15,270,240])]:
                rho = k.density_g_per_m3(q,p,t,pressure_unit='Pa',temperature_unit='K')
                for a,b in [(1.,1.),(2.,2.)]:
                    ext = k.hydrometeor_extinction(rho,a=a,b=b)
                    for angle in (30.,90.):
                        z = [0.,1000.,3000.]
                        integral = k.column_integral(ext,z,angle,height_unit='m_above_antenna',elevation_unit='degree')
                        fixtures.append({'kind':'SYNTHETIC','q_g_kg':q,'p_pa':p,'t_k':t,
                                         'a':a,'b':b,'z_m_above_antenna':z,'theta_degree':angle,
                                         'rho_g_m3':rho,'k_db_km':ext,'attenuation_db':integral})
            check('synthetic_matrix_eight_cases', len(fixtures)==8)
            check('zero_density_zero_extinction', k.hydrometeor_extinction([0],a=2,b=.7)==(0.,))
            check('fractional_exponent_fixture', k.hydrometeor_extinction([4],a=2,b=.5)==(4.,))
            rho_call = lambda q,p,t,pu='Pa',tu='K': k.density_g_per_m3(q,p,t,pressure_unit=pu,temperature_unit=tu)
            reject('reject_pressure_hPa', lambda: rho_call([1],[1000],[300],pu='hPa'))
            reject('reject_temperature_C', lambda: rho_call([1],[100000],[27],tu='C'))
            reject('reject_shape_mismatch', lambda: rho_call([1,2],[100000],[300]))
            reject('reject_zero_temperature', lambda: rho_call([1],[100000],[0]))
            reject('reject_zero_pressure', lambda: rho_call([1],[0],[300]))
            reject('reject_negative_coefficient', lambda: k.hydrometeor_extinction([1],a=-1,b=1))
            reject('reject_zero_exponent', lambda: k.hydrometeor_extinction([0],a=1,b=0))
            reject('reject_nonfinite', lambda: k.hydrometeor_extinction([math.nan],a=1,b=1))
            reject('reject_bool_as_number', lambda: k.hydrometeor_extinction([True],a=1,b=1))
            integ = lambda v,z,ang=90: k.column_integral(v,z,ang,height_unit='m_above_antenna',elevation_unit='degree')
            check('metre_kilometre_conversion', integ([1,1],[0,1000]) == 1.)
            check('linear_profile_integral', integ([0,1,3],[0,1000,3000]) == 4.5)
            check('cosecant_30', abs(integ([1,1],[0,1000],30)-2.)<1e-14)
            check('zero_column', integ([0,0],[0,1000],30)==0.)
            reject('reject_below_antenna', lambda: integ([1,1],[-10,1000]))
            reject('reject_missing_surface', lambda: integ([1,1],[100,1000]))
            reject('reject_duplicate_height', lambda: integ([1,1],[0,0]))
            reject('reject_descending_height', lambda: integ([1,1,1],[0,1000,500]))
            reject('reject_empty_column', lambda: integ([],[]))
            reject('reject_single_node', lambda: integ([1],[0]))
            reject('reject_zero_elevation', lambda: integ([1,1],[0,1000],0))
            reject('reject_beyond_zenith', lambda: integ([1,1],[0,1000],91))
            reject('reject_negative_extinction', lambda: integ([-1,1],[0,1000]))
            reject('reject_implicit_radians', lambda: k.column_integral([1,1],[0,1000],1,height_unit='m_above_antenna',elevation_unit='rad'))
            comps = {'oxygen':[.1,.1], 'water_vapour':[.2,.2], 'cloud':[.3,.3], 'rain':[.4,.4]}
            paths = k.four_component_paths(comps,[0,1000],90)
            check('component_sum', paths['total_absolute_dB']==1.)
            reject('reject_extra_hydrometeor', lambda: k.four_component_paths({**comps,'graupel':[0,0]},[0,1000],90))
            check('retain_negative_relative', k.relative_attenuation(.1,.3)<0)
            check('fixed_reference_zero', k.relative_attenuation(.3,.3)==0)
            ready = read(OUT/'readiness.json')
            for station,freq,pol in [('LOCAL',19.13,'UNVERIFIED'),('Chilbolton',20.7,'LHCP'),('Sparsholt',20.7,'LHCP')]:
                reject('real_scoring_blocked_'+station, lambda f=freq,p=pol: k.require_real_scoring_contract(ready,frequency_ghz=f,polarization=p))
            schema = {key:True for key in k.REAL_GATES}
            schema.update(fixture_kind='SYNTHETIC',frequency_ghz=19.7,polarization='LHCP',coefficient_sha256='fixture-only',source_url='synthetic://test')
            reject('synthetic_never_real_model', lambda: k.require_real_scoring_contract(schema,frequency_ghz=19.7,polarization='LHCP'))
            schema['fixture_kind']='SCHEMA_TEST_ONLY'
            reject('reject_frequency_substitution', lambda: k.require_real_scoring_contract(schema,frequency_ghz=20.7,polarization='LHCP'))
            reject('reject_polarization_substitution', lambda: k.require_real_scoring_contract(schema,frequency_ghz=19.7,polarization='RHCP'))
            for key in k.REAL_GATES:
                reduced = {**schema,key:False}
                reject('require_gate_'+key, lambda c=reduced: k.require_real_scoring_contract(c,frequency_ghz=19.7,polarization='LHCP'))
            validate_guards(source_guards)
            need(all(sha(CODE/name)==digest for name,digest in code_hash.items()),'Code changed')
            save('synthetic_cases.json', {'warning':'Synthetic numerical fixtures only; no author coefficients or observed data.', 'cases':fixtures,'component_fixture':{'inputs':comps,'z':[0,1000],'theta':90,'outputs':paths}})
            save('test_results.json', {'status':'PASS','utc':datetime.now(timezone.utc).isoformat(),
                                       'checks':checks,'check_count':len(checks),'synthetic_cases':len(fixtures),
                                       'protected_source_count':len(source_guards),'source_sha_unchanged':True,
                                       'new_real_predictions':0,'new_scores':0,'new_training':0,'weather_requests':0})
            print(json.dumps({'status':'PASS','checks':len(checks),'synthetic_cases':len(fixtures),'protected_sources':len(source_guards)}))
        finally:
            lock.seek(0); msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1)
            save('lock_release.json', {'utc':datetime.now(timezone.utc).isoformat(),'pid':os.getpid(),'status':'RELEASED'})


if __name__ == '__main__':
    try:
        run()
    except Exception as exc:
        if not (OUT/'failure.json').exists():
            save('failure.json', {'utc':datetime.now(timezone.utc).isoformat(),'error':str(exc),'traceback':traceback.format_exc()})
        raise
