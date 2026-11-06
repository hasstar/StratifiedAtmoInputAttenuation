"""Independent scalar arithmetic, immutable predictions and common-sample audit.

Expected answers do not call physical.py. That module is imported only as the
system under test for negative/identity controls. No models or network access.
"""
from support import *
import bisect
import math
import statistics
import numpy as np
import pandas as pd

FIELDS = ['surface__2m_temperature','surface__2m_dewpoint_temperature',
          'surface__surface_pressure','surface__total_column_water_vapour',
          'surface__total_column_cloud_liquid_water']

def parse_coefficients(path):
    return [tuple(map(float, line.split())) for line in path.read_text().splitlines() if line.strip()]

def linear(f, rows):
    frequencies = [r[0] for r in rows]
    i = bisect.bisect_left(frequencies, f)
    if frequencies[i] == f:
        return rows[i][1:]
    lo, hi = rows[i-1], rows[i]
    w = (f-lo[0])/(hi[0]-lo[0])
    return tuple((1-w)*lo[j]+w*hi[j] for j in range(1,5))

def oxygen(f, total, temp, rho, rows):
    theta = 300/temp
    vapour = rho*temp/216.7
    dry = total-vapour
    terms = []
    for centre,a1,a2,a3,a4,a5,a6 in rows:
        strength = a1*1e-7*dry*theta**3*math.exp(a2*(1-theta))
        width = math.hypot(a3*1e-4*(dry*theta**(.8-a4)+1.1*vapour*theta), .0015)
        delta = (a5+a6*theta)*1e-4*total*theta**.8
        minus = ((width-delta*(centre-f))/((centre-f)**2+width**2))
        plus = ((width-delta*(centre+f))/((centre+f)**2+width**2))
        terms.append(strength*(f/centre)*(minus+plus))
    d = 5.6e-4*total*theta**.8
    continuum = f*dry*theta**2*(6.14e-5/(d*(1+(f/d)**2))+
                       1.4e-12*dry*theta**1.5/(1+1.9e-5*f**1.5))
    return .1820*f*(math.fsum(terms)+continuum)

def gas_reference(f, elevation, p, t, rho, v, data):
    o,p1,p2 = data
    a,b,c,d = linear(f,p1)
    height = a+b*t+c*p+d*rho
    a,b,c,d = linear(f,p2)
    kv = a+b*rho+c*t+d*p
    sine = math.sin(math.radians(elevation))
    dry = oxygen(f,p,t,rho,o)*height/sine
    wet = kv*v/sine
    return dry,wet,height,kv

def cloud_reference(f, elevation, liquid):
    theta = 300/273.75
    eps0 = 77.66+103.3*(theta-1)
    eps1,eps2 = .0671*eps0,3.52
    fp = 20.20-146*(theta-1)+316*(theta-1)**2
    fs = 39.8*fp
    # Complex double-Debye form, independently rearranged from the real/imag code.
    eps = (eps0-eps1)/(1+1j*f/fp)+(eps1-eps2)/(1+1j*f/fs)+eps2
    loss = -eps.imag
    kl = .819*f*loss/((2+eps.real)**2+loss**2)
    correction = (.1522*math.exp(-((f-(-23.9589))**2)/3299.1)+
                  11.51*math.exp(-((f-219.2096)**2)/2759500)-10.4912)
    return liquid*kl*correction/math.sin(math.radians(elevation))

def fixed_reference(h):
    hp = 6356.766*h/(6356.766+h)
    t = 288.15-6.5*hp
    return t, 1013.25*(t/288.15)**(34.1632/6.5),7.5*math.exp(-h/2),15*math.exp(-h/2)

def metrics(y,p):
    errors = [float(a)-float(b) for a,b in zip(p,y)]
    n = len(errors)
    return dict(n=n,rmse_db=math.sqrt(math.fsum(e*e for e in errors)/n),
                mae_db=math.fsum(abs(e) for e in errors)/n,bias_db=math.fsum(errors)/n)

def main():
    offline()
    with lock():
        protected = read(OUT/'sources_before.json'); check(protected)
        contract = read(OUT/'calculation_contract.json')
        for name,expected in contract['code_sha256'].items():
            need(sha(CODE/name)==expected,'Calculation code changed: '+name)
        need(sha(CODE/'PROTOCOL.md')==contract['protocol_sha256'],'Protocol changed')
        for row in read(OUT/'references_complete.json')['records']:
            need(sha(OUT/'references'/row['name'])==row['sha256'],'Official source changed')
        data = (read(OUT/'references/oxygen_table_from_P676_13.json')['rows'],
                parse_coefficients(OUT/'references/P676_13_part1.txt'),
                parse_coefficients(OUT/'references/P676_13_part2.txt'))
        local = pd.read_parquet(OUT/'predictions_local.parquet')
        uk = pd.read_parquet(OUT/'predictions_uk.parquet')
        previous = {'LOCAL':pd.read_parquet(REV(6)/'predictions_local_all.parquet'),
                    'UK':pd.read_parquet(REV(6)/'predictions_uk_all.parquet')}
        for name,frame in [('LOCAL',local),('UK',uk)]:
            pd.testing.assert_frame_equal(frame[previous[name].columns],previous[name],check_exact=True)
        labels = pd.read_parquet(REV(8)/'weather_labels_all.parquet')
        merged = uk.merge(labels[['station','utc_epoch_seconds','event']],on=['station','utc_epoch_seconds'],validate='one_to_one',suffixes=('_actual','_expected'))
        need((merged.event_actual==merged.event_expected).all(),'UK weather labels changed')
        rawlocal = pd.read_parquet(ROOT/'Outputs/Datasets/Signal_Detectability_Revision_003/test.parquet')
        rawuk = pd.read_parquet(REV(10)/'expanded_dataset_with_itu_inputs.parquet')
        pd.testing.assert_frame_equal(local[['hour_bjt']+FIELDS],rawlocal[['hour_bjt']+FIELDS],check_exact=True)
        pd.testing.assert_frame_equal(uk[['station','utc_epoch_seconds']+FIELDS],rawuk[['station','utc_epoch_seconds']+FIELDS],check_exact=True)
        need(len(local)==687 and len(uk)==5664 and int(uk.common_score_eligible.sum())==5538,'Scope changed')
        refs = read(OUT/'fixed_reference_parameters.json')
        errors = {}
        count = 0
        for name,frame in [('LOCAL',local)]+list(uk.groupby('station')):
            r = refs[name]
            t0,p0,r0,v0 = fixed_reference(r['height_km'])
            g0 = math.fsum(gas_reference(r['frequency_GHz'],r['elevation_deg'],p0,t0,r0,v0,data)[:2])
            need(abs(g0-r['gas_absolute_db'])<2e-12,'Fixed gas reference mismatch')
            for row in frame.to_dict('records'):
                t,td,pa,v,l = [float(row[k]) for k in FIELDS]
                p = pa*.01; dew = td-273.15
                ef = 1+.0001*(7.2+p*(.0320+5.9e-6*dew**2))
                e = ef*6.1121*math.exp((18.678-dew/234.5)*dew/(257.14+dew))
                rho = 216.7*e/t
                dry,wet,height,kv = gas_reference(r['frequency_GHz'],r['elevation_deg'],p,t,rho,v,data)
                expected = dict(oxygen_db=dry,water_vapour_db=wet,gas_absolute_db=dry+wet,
                    gas_reference_db=g0,gas_relative_db=dry+wet-g0,
                    cloud_db=cloud_reference(r['frequency_GHz'],r['elevation_deg'],l),
                    vapour_pressure_hpa=e,vapour_density_gm3=rho,
                    oxygen_equivalent_height_km=height,water_mass_absorption_db_per_kgm2=kv)
                for k,value in expected.items():
                    delta=abs(float(row[k])-value)
                    errors[k]=max(errors.get(k,0),delta)
                    need(delta<2e-11,'Independent scalar mismatch: '+k)
                for col,rain in [('itu_combined_geometric_approx','itu838_geometric_proxy'),('itu_combined_p618_hourly_approx','itu618_hourly_proxy')]:
                    if math.isfinite(row[rain]):
                        need(abs(row[col]-(expected['gas_relative_db']+expected['cloud_db']+row[rain]))<2e-11,'Combined mismatch')
                    else:
                        need(pd.isna(row[col]),'Invalid rain was filled')
                count += 1
        cohorts = {'LOCAL_test':local,'UK_full_common':uk[uk.common_score_eligible]}
        for station in ('chilbolton','sparsholt'):
            cohorts['UK_'+station+'_full_common']=uk[uk.common_score_eligible & uk.station.eq(station)]
        recalculated=[]
        for old in read(OUT/'run_metrics.json'):
            frame=cohorts[old['cohort']]; d=frame[frame.event.eq(old['event'])]
            need(np.isfinite(d[old['model']]).all(),'Per-model omission')
            expected=metrics(d.truth_db,d[old['model']])
            for k,value in expected.items(): need(abs(old[k]-value)<2e-12,'Metric mismatch: '+k)
            recalculated.append(dict(cohort=old['cohort'],event=old['event'],family=old['family'],**expected))
        for old in read(OUT/'family_rankings.json'):
            vals=[r['rmse_db'] for r in recalculated if all(r[k]==old[k] for k in ['cohort','event','family'])]
            need(abs(statistics.mean(vals)-old['rmse_db_mean'])<2e-12,'Family metric mismatch')
            if len(vals)>1: need(abs(statistics.stdev(vals)-old['rmse_db_sd'])<2e-12,'Seed SD mismatch')
        import physical as target
        controls=[]
        def reject(name,fn):
            try: fn()
            except (RuntimeError,ValueError): controls.append(dict(name=name,outcome='rejected')); return
            raise RuntimeError('Invalid input accepted: '+name)
        for f in (0,351,float('nan')):
            reject('gas_frequency_'+str(f),lambda f=f:target.gas(f,30,1000,290,7,15))
        for el in (0,4.99,91,float('nan')):
            reject('gas_elevation_'+str(el),lambda el=el:target.gas(20,el,1000,290,7,15))
        for v in (-1,float('nan')):
            reject('water_column_'+str(v),lambda v=v:target.gas(20,30,1000,290,7,v))
            reject('cloud_column_'+str(v),lambda v=v:target.cloud(20,30,v))
        for f in (0,201,float('nan')):
            reject('cloud_frequency_'+str(f),lambda f=f:target.cloud(f,30,1))
        reject('negative_temperature',lambda:target.vapour_from_dewpoint(-1,280,1000))
        reject('out_of_range_dewpoint',lambda:target.vapour_from_dewpoint(290,200,1000))
        reject('water_pressure_exceeds_total',lambda:target.vapour_from_dewpoint(290,280,1))
        reject('negative_reference_altitude',lambda:target.fixed_reference(-1))
        for f in (19.13,20.7):
            need(target.cloud(f,30,0)==0,'Zero liquid')
            g30=target.gas(f,30,1000,290,7,15)
            g90=target.gas(f,90,1000,290,7,15)
            need(abs(float(g30['gas_absolute_db'])-2*float(g90['gas_absolute_db']))<1e-12,'Gas sine scaling')
            need(abs(float(target.cloud(f,30,1))-2*float(target.cloud(f,90,1)))<1e-12,'Cloud sine scaling')
            need(target.gas(f,30,1000,290,7,0)['water_vapour_db']==0,'Zero water column')
        for table in data[1:]:
            for j in (0,20,100,-1):
                need(tuple(linear(table[j][0],table))==tuple(table[j][1:]),'Official coefficient knots')
        negative_counts={name:int((df.itu_combined_geometric_approx<0).sum()) for name,df in [('LOCAL',local),('UK',uk)]}
        need(sum(negative_counts.values())>0,'Unexpected absence of negative un-clipped relative attenuation')
        check(protected)
        emit('verification.json',dict(status='PASS',utc=utc(),auditor='independent scalar arithmetic in the same agent; not a separate human reviewer',
            all_physical_rows_checked=count,maximum_absolute_error=errors,
            run_metrics_checked=len(recalculated),official_source_files_verified=6,
            frozen_prediction_tables_exact=True,raw_five_fields_exact=True,common_masks_unchanged=True,
            negative_controls=controls,positive_controls=['zero liquid','zero water vapour column','gas slant scaling','cloud slant scaling','coefficient knots','fixed reference recomputed','negative relative attenuation retained'],
            negative_predictions_retained=negative_counts,protected_files_verified=len(protected),
            training_calls=0,ml_inference_calls=0,weather_requests=0,
            verification_script_sha256=sha(Path(__file__))))
        print('PASS: independent scalar arithmetic for '+str(count)+' hours; '+str(len(recalculated))+' metric rows; '+str(len(controls))+' negative controls; all '+str(len(protected))+' protected source hashes unchanged.')

if __name__=='__main__': main()
