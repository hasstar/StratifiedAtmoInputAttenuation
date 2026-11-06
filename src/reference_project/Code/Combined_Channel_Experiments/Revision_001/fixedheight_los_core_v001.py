"""Human-approved fixed standard-height LOS approximation for four layers.

Only natural Top10 fields may be newly requested. No auxiliary z, interpolation,
weather fitting, or vertical-station-column substitution. 1 hPa z is Top10 and
retains the frozen hourly nearest-grid iteration. Other four layers use the
US Standard Atmosphere pressure-height relation, never actual hourly heights.
"""
from __future__ import annotations
import math
import nearest_los_core_v002 as base

GRID, LEVEL_VARS, SURFACE = base.GRID, base.LEVEL_VARS, base.SURFACE
require, geometry, original = base.require, base.geometry, base.original
grid_index, intersect_heights = base.grid_index, base.intersect_heights
parent_hours, tiles, expand_request = base.parent_hours, base.tiles, base.expand_request
seed_cell = base.seed_cell
FIXED_LEVELS = (5, 500, 650, 850)
SEED_HEIGHT = {1: base.SEED_HEIGHT[1]}
STANDARD_SOURCE = 'https://ntrs.nasa.gov/api/citations/20050207438/downloads/20050207438.pdf'
STANDARD_DEFINITION = 'NASA/TM-2005-213659 Appendix A, US Standard Atmosphere 1976; fixed idealized atmosphere, not observed ERA5 height'
# Appendix A equations A.1--A.5, metres/Pa/K; constants are not fitted.
G0, GAS_R, MOLAR_MASS, EARTH_RADIUS_M = 9.80665, 8314.32, 28.9644, 6356660.
BASE_HEIGHTS_M = (0., 11000., 20000., 32000., 47000.)
LAPSE_K_PER_M = (-.0065, 0., .001, .0028)


def standard_height(level_hpa):
    """Analytic inversion of the published standard atmosphere, not interpolation."""
    require(level_hpa in FIXED_LEVELS, 'Only the four authorized fixed layers')
    wanted = float(level_hpa)*100.
    p, temperature = 101325., 288.15
    for low, high, lapse in zip(BASE_HEIGHTS_M[:-1], BASE_HEIGHTS_M[1:], LAPSE_K_PER_M):
        next_temperature = temperature + lapse*(high-low)
        if lapse == 0:
            next_pressure = p*math.exp(-G0*MOLAR_MASS*(high-low)/(GAS_R*temperature))
        else:
            next_pressure = p*(next_temperature/temperature)**(-G0*MOLAR_MASS/(GAS_R*lapse))
        if next_pressure <= wanted <= p:
            if lapse == 0:
                H = low-GAS_R*temperature/(G0*MOLAR_MASS)*math.log(wanted/p)
            else:
                H = low+temperature/lapse*((wanted/p)**(-GAS_R*lapse/(G0*MOLAR_MASS))-1.)
            height = EARTH_RADIUS_M*H/(EARTH_RADIUS_M-H)
            return {'pressure_hpa':level_hpa, 'geopotential_height_m':H,
                    'geometric_height_m':height, 'basis':STANDARD_DEFINITION}
        p, temperature = next_pressure, next_temperature
    raise RuntimeError('Pressure outside authorized standard atmosphere range')


FIXED_HEIGHTS = {p:standard_height(p) for p in FIXED_LEVELS}


def fixed_location(link, level):
    require(level in FIXED_LEVELS, 'Not a fixed layer')
    h = FIXED_HEIGHTS[level]['geometric_height_m']
    if h <= float(link['height_amsl_m']):
        return {'status':'INVALID_FIXED_HEIGHT_NOT_ABOVE_ANTENNA', 'height_m':h,
                'fixed_height_approximation':True, 'actual_hourly_pressure_height_verified':False}
    lat, lon, distance, residual = intersect_heights(link, [h])
    return {'status':'STABLE_FIXED_HEIGHT_NEAREST', 'cell':list(grid_index(lat[0],lon[0])),
            'intersection_lat':float(lat[0]), 'intersection_lon':float(lon[0]),
            'height_m':h, 'slant_m':float(distance[0]), 'height_residual_m':float(residual[0]),
            'fixed_height_approximation':True, 'actual_hourly_pressure_height_verified':False,
            'height_basis':STANDARD_DEFINITION}


def trace(link, level, hours, lookup):
    if level == 1:
        return base.trace(link, level, hours, lookup)
    record = fixed_location(link, level)
    targets = set()
    if record['status']=='STABLE_FIXED_HEIGHT_NEAREST':
        targets = {(v,level,t,*record['cell']) for t in hours for v in LEVEL_VARS[level]}
    return [dict(record) for _ in hours], set(), targets


def allowed_pair(variable, level):
    return (level == 0 and variable in SURFACE) or (level in LEVEL_VARS and variable in LEVEL_VARS[level])


def validate_request(request):
    require('grid' not in request, 'Regridding is forbidden')
    levels = request.get('pressure_level',['0'])
    require(len(levels)==1, 'Cross-level Cartesian requests forbidden')
    level = int(levels[0])
    require(bool(request['variable']) and len(set(request['variable']))==len(request['variable']), 'Duplicate/empty variables')
    require(all(allowed_pair(v,level) for v in request['variable']), 'Only natural Top10 pairs; auxiliary z forbidden')


def demands_to_requests(parent, demands):
    demands = set(demands)
    require(all(allowed_pair(v,p) for v,p,*_ in demands), 'Auxiliary or non-Top10 demand forbidden')
    result = base.demands_to_requests(parent, demands)
    for request in result:
        validate_request(request)
    return result
