"""P.676-13 instantaneous gas and P.840-9 cloud columns. No learned parameters."""
import numpy as np
from support import OUT, read, need

FIELDS = ['surface__2m_temperature','surface__2m_dewpoint_temperature',
          'surface__surface_pressure','surface__total_column_water_vapour',
          'surface__total_column_cloud_liquid_water']

def tables():
    oxygen=np.asarray(read(OUT/'references/oxygen_table_from_P676_13.json')['rows'],float)
    part1=np.loadtxt(OUT/'references/P676_13_part1.txt')
    part2=np.loadtxt(OUT/'references/P676_13_part2.txt')
    need(oxygen.shape==(44,7) and part1.shape==(700,5) and part2.shape==(699,5),'Official table shapes')
    need(np.array_equal(part1[:,0],np.sort(np.r_[np.arange(1,350.1,.5),118.75])),'Part1 frequencies')
    need(np.array_equal(part2[:,0],np.arange(1,350.1,.5)),'Part2 frequencies')
    need(all(np.isfinite(t).all() for t in (oxygen,part1,part2)),'Nonfinite reference table')
    return oxygen,part1,part2

def vapour_from_dewpoint(temperature, dewpoint, total_hpa):
    t=np.asarray(dewpoint,float)-273.15
    temperature=np.asarray(temperature,float); total_hpa=np.asarray(total_hpa,float)
    need(np.isfinite(t).all() and ((t>=-40)&(t<=50)).all(),'Dewpoint outside P453 water expression domain')
    need(np.isfinite(temperature).all() and (temperature>0).all(),'Invalid temperature')
    # P.453-14 Eq.9 evaluated at dewpoint (saturation over water, including enhancement).
    enhancement=1+1e-4*(7.2+total_hpa*(.0320+5.9e-6*t*t))
    e=enhancement*6.1121*np.exp((18.678-t/234.5)*t/(257.14+t))
    rho=216.7*e/temperature
    need(np.isfinite(rho).all() and (rho>=0).all() and (total_hpa>e).all(),'Invalid vapour/dry pressure')
    return e,rho

def oxygen_specific(f,total_hpa,temperature,rho,oxygen):
    total_hpa,temperature,rho=np.broadcast_arrays(total_hpa,temperature,rho)
    theta=300/temperature; e=rho*temperature/216.7; p=total_hpa-e
    need(np.isfinite([f]).all() and 1<=f<=350,'P676 frequency domain')
    need(np.isfinite(p).all() and (p>0).all() and (temperature>0).all() and (rho>=0).all(),'P676 state')
    fi,a1,a2,a3,a4,a5,a6=(oxygen[:,i] for i in range(7))
    th=theta[...,None]; pp=p[...,None]; ee=e[...,None]
    width=np.sqrt((a3*1e-4*(pp*th**(.8-a4)+1.1*ee*th))**2+2.25e-6)
    delta=(a5+a6*th)*1e-4*(pp+ee)*th**.8
    shape=f/fi*((width-delta*(fi-f))/((fi-f)**2+width**2)+(width-delta*(fi+f))/((fi+f)**2+width**2))
    strength=a1*1e-7*pp*th**3*np.exp(a2*(1-th))
    d=5.6e-4*(p+e)*theta**.8
    continuum=f*p*theta**2*(6.14e-5/(d*(1+(f/d)**2))+1.4e-12*p*theta**1.5/(1+1.9e-5*f**1.5))
    return .1820*f*(np.sum(strength*shape,axis=-1)+continuum)

def coefficients(f,table):
    need(1<=f<=350,'Coefficient interpolation domain')
    return np.asarray([np.interp(f,table[:,0],table[:,j]) for j in range(1,5)])

def gas(f,elevation,total_hpa,temperature,rho,column,reference_tables=None):
    need(np.isfinite(elevation) and 5<=elevation<=90,'Slant elevation outside 5..90 degrees')
    oxygen,part1,part2=tables() if reference_tables is None else reference_tables
    p,t,r,v=np.broadcast_arrays(total_hpa,temperature,rho,column)
    need(np.isfinite(v).all() and (v>=0).all(),'Invalid integrated water vapour')
    a,b,c,d=coefficients(f,part1)
    height=a+b*t+c*p+d*r
    a,b,c,d=coefficients(f,part2)
    kv=a+b*r+c*t+d*p
    specific=oxygen_specific(f,p,t,r,oxygen)
    sine=np.sin(np.deg2rad(elevation))
    dry=specific*height/sine; wet=kv*v/sine
    need(np.isfinite(dry).all() and np.isfinite(wet).all() and (height>0).all() and (kv>=0).all() and (dry>=0).all() and (wet>=0).all(),'Invalid P676 output')
    return dict(oxygen_db=dry,water_vapour_db=wet,gas_absolute_db=dry+wet,
                oxygen_equivalent_height_km=height,water_mass_absorption_db_per_kgm2=kv)

def liquid_coefficient(f):
    need(np.isfinite(f) and 1<=f<=200,'P840 frequency domain')
    t=273.75; th=300/t
    e0=77.66+103.3*(th-1); e1=.0671*e0; e2=3.52
    fp=20.20-146*(th-1)+316*(th-1)**2; fs=39.8*fp
    real=(e0-e1)/(1+(f/fp)**2)+(e1-e2)/(1+(f/fs)**2)+e2
    imag=f*(e0-e1)/(fp*(1+(f/fp)**2))+f*(e1-e2)/(fs*(1+(f/fs)**2))
    kl=.819*f/(imag*(1+((2+real)/imag)**2))
    return kl*(.1522*np.exp(-(f+23.9589)**2/3299.1)+11.51*np.exp(-(f-219.2096)**2/2759500)-10.4912)

def cloud(f,elevation,column):
    column=np.asarray(column,float)
    need(5<=elevation<=90 and np.isfinite(column).all() and (column>=0).all(),'Invalid cloud inputs')
    return liquid_coefficient(f)*column/np.sin(np.deg2rad(elevation))

def fixed_reference(height_km):
    need(0<=height_km<=1,'Reference implemented only for the low-altitude sites in this protocol')
    hp=6356.766*height_km/(6356.766+height_km)
    t=288.15-6.5*hp
    p=1013.25*(288.15/t)**(-34.1632/6.5)
    rho=7.5*np.exp(-height_km/2)
    # Integral of the declared exponential humidity profile; no measured-label fit.
    # The high-altitude 2e-6 mixing-ratio floor in full P835 is not integrated here.
    return dict(height_km=height_km,temperature_K=t,total_pressure_hpa=p,
                vapour_density_gm3=rho,water_column_kgm2=2*rho,
                water_column_definition='2-km exponential reference integral; high-altitude mixing-ratio floor omitted')

def components(frame,frequency,elevation,height):
    values=frame[FIELDS].to_numpy(float)
    need(np.isfinite(values).all(),'Missing physical field; no imputation allowed')
    t,td,p,v,l=values.T; p=p/100
    e,rho=vapour_from_dewpoint(t,td,p)
    result=gas(frequency,elevation,p,t,rho,v)
    ref=fixed_reference(height)
    ref_gas=gas(frequency,elevation,ref['total_pressure_hpa'],ref['temperature_K'],ref['vapour_density_gm3'],ref['water_column_kgm2'])
    ref.update(gas_absolute_db=float(ref_gas['gas_absolute_db']),frequency_GHz=frequency,elevation_deg=elevation)
    result.update(gas_reference_db=np.full(len(frame),ref['gas_absolute_db']),
                  gas_relative_db=result['gas_absolute_db']-ref['gas_absolute_db'],
                  cloud_db=cloud(frequency,elevation,l),vapour_pressure_hpa=e,vapour_density_gm3=rho,
                  dewpoint_above_temperature=td>t)
    return result,ref
