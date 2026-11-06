"""Independent high-precision oracle. Does not import prm_kernel or run_tests."""
import ast
import hashlib
import json
import math
import msvcrt
from datetime import datetime, timezone
from decimal import Decimal, localcontext
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CODE = Path(__file__).resolve().parent
OUT = ROOT/'Outputs/Experiments/Combined_Channel_Revision_014'


def need(ok, msg):
    if not ok: raise ValueError(msg)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):digest.update(block)
    return digest.hexdigest()


def main():
    need(not (OUT/'independent_qa.json').exists(), 'One-shot verifier output exists')
    source_guard=read(OUT/'sources_before.json')
    for path,digest in source_guard.items():need(sha(path)==digest,'Changed protected source')
    for name,digest in read(OUT/'code_before.json').items():need(sha(CODE/name)==digest,'Changed code')
    result=read(OUT/'test_results.json')
    need(result['status']=='PASS','Tests not PASS')
    data=read(OUT/'synthetic_cases.json')
    errors=[]
    scalar_checks=0
    def compare(actual, expected):
        nonlocal scalar_checks
        delta=abs(float(actual)-float(expected))
        need(math.isfinite(float(actual)) and delta<=1e-12*max(1,abs(float(expected))), 'Numerical mismatch')
        errors.append(delta);scalar_checks+=1
    with localcontext() as ctx:
        ctx.prec=60
        d=lambda v:Decimal(str(v))
        for case in data['cases']:
            need(case['kind']=='SYNTHETIC','Unmarked fixture')
            q,p,t=[list(map(d,case[name])) for name in ('q_g_kg','p_pa','t_k')]
            rho=[qj*pj/(Decimal('287.05')*tj) for qj,pj,tj in zip(q,p,t)]
            ext=[d(case['a'])*(r**int(case['b'])) for r in rho]
            for actual,expected in zip(case['rho_g_m3'],rho):compare(actual,expected)
            for actual,expected in zip(case['k_db_km'],ext):compare(actual,expected)
            z=list(map(d,case['z_m_above_antenna']))
            # Algebraically independent exact polynomial-area expression.
            area=Decimal(0)
            for i in range(len(z)-1):
                dz=(z[i+1]-z[i])/Decimal(1000)
                slope=(ext[i+1]-ext[i])/dz
                area+=ext[i]*dz+slope*dz*dz/Decimal(2)
            sine={30.:Decimal('.5'),90.:Decimal(1)}[case['theta_degree']]
            compare(case['attenuation_db'],area/sine)
        component=data['component_fixture']
        for name in ('oxygen','water_vapour','cloud','rain'):
            # This fixture is a constant 1 km vertical column.
            compare(component['outputs'][name],d(component['inputs'][name][0]))
        compare(component['outputs']['total_absolute_dB'],Decimal(1))
    need(len(data['cases'])==8 and scalar_checks==61,'Fixture coverage changed')
    checks=result['checks']
    for name in ('real_scoring_blocked_LOCAL','real_scoring_blocked_Chilbolton',
                 'real_scoring_blocked_Sparsholt','synthetic_never_real_model',
                 'reject_frequency_substitution','reject_polarization_substitution',
                 'reject_missing_surface','reject_negative_extinction','retain_negative_relative'):
        need(name in checks,'Missing contract rejection test: '+name)
    readiness=read(OUT/'readiness.json')
    need(readiness['coefficient_numeric_bundle'] is False,'No actual coefficient bundle obtained')
    need(readiness['new_real_predictions']==readiness['new_scores']==readiness['new_training']==readiness['weather_requests']==0,'Scope breach')
    tree=ast.parse((CODE/'prm_kernel.py').read_text(encoding='utf-8'))
    imports=[node.names[0].name for node in ast.walk(tree) if isinstance(node,ast.Import)]
    need(imports==['math'],'Kernel import scope')
    need(not any(isinstance(n,ast.ImportFrom) for n in ast.walk(tree)),'Unexpected import')
    prohibited={'open','eval','exec','fit','partial_fit','fit_transform','predict','train','backward'}
    calls={n.func.id if isinstance(n.func,ast.Name) else n.func.attr if isinstance(n.func,ast.Attribute) else '' for n in ast.walk(tree) if isinstance(n,ast.Call)}
    need(not calls.intersection(prohibited),'Prohibited kernel call')
    with (OUT/'execution.lock').open('r+b') as f:
        msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
        f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_UNLCK,1)
    qa={'status':'PASS','utc':datetime.now(timezone.utc).isoformat(),
        'scope':'Synthetic equation-kernel tests only; no complete PRM validation',
        'decimal_precision':60,'independent_scalar_checks':scalar_checks,
        'max_absolute_difference':max(errors),'source_sha_unchanged':True,
        'protected_source_count':len(source_guard),'os_lock':'RELEASED',
        'kernel_sha256':sha(CODE/'prm_kernel.py'),
        'verifier_sha256':sha(__file__),
        'test_results_sha256':sha(OUT/'test_results.json'),
        'synthetic_cases_sha256':sha(OUT/'synthetic_cases.json'),
        'new_real_predictions':0,'new_scores':0,'seven_family_comparison_complete':False}
    with (OUT/'independent_qa.json').open('x',encoding='utf-8') as f:
        json.dump(qa,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
    print(json.dumps(qa,ensure_ascii=True))


if __name__=='__main__':main()
