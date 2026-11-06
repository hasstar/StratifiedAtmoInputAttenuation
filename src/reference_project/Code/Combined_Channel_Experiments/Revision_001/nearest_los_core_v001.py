"""Nearest published ERA5 grid, never bilinear or vertical-column substitution.

The height guesses below ONLY seed an iterative, hourly geopotential lookup.
No target weather parameter is synthesized. Unsupported/cycling rows stay invalid.
"""
from __future__ import annotations
import math
from collections import defaultdict, Counter
from datetime import datetime, timezone
from functools import lru_cache
import numpy as np
import top10_los_geometry_v003 as geometry

GRID = 0.25
LEVEL_VARS = {1: ('geopotential', 'u_component_of_wind'),
              5: ('ozone_mass_mixing_ratio', 'u_component_of_wind'),
              500: ('ozone_mass_mixing_ratio',), 650: ('u_component_of_wind',),
              850: ('ozone_mass_mixing_ratio', 'relative_humidity')}
SURFACE = ('convective_available_potential_energy',
           'maximum_2m_temperature_since_previous_post_processing')
SEED_HEIGHT = {1: 48000., 5: 36000., 500: 5600., 650: 3600., 850: 1500.}
MAX_ITERATIONS = 8


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


@lru_cache(maxsize=1)
def original():
    return geometry.original()


def grid_index(lat, lon):
    """Coordinate-wise nearest published grid; exact ties north/east.

This explicitly specifies regular-grid nearest-neighbour (not geodesic NN).
"""
    require(np.isfinite([lat, lon]).all() and -90 <= lat <= 90, 'Invalid coordinate')
    lon = (float(lon) + 180) % 360 - 180
    return int(math.floor(float(lat)/GRID + .5)), (int(math.floor(lon/GRID + .5)) + 720) % 1440 - 720


def intersect_heights(link, heights):
    heights = np.asarray(heights, dtype=float)
    require(np.isfinite(heights).all() and np.all(heights >= link['height_amsl_m'])
            and np.all(heights <= 80000), 'Height outside forward 0--80 km support')
    origin = np.asarray(link['origin_ecef_m'], float)
    direction = np.asarray(link['direction_ecef'], float)
    require(abs(np.linalg.norm(direction)-1) < 1e-10, 'Non-unit ray')
    module = original()
    lo = np.zeros(heights.shape); hi = np.full(heights.shape, 500000.)
    require(np.all(module._ecef_to_geodetic(origin + hi[:, None]*direction)[2] > heights), 'No forward bracket')
    for _ in range(42):
        mid = (lo + hi)/2
        h = module._ecef_to_geodetic(origin + mid[:, None]*direction)[2]
        lower = h < heights
        lo = np.where(lower, mid, lo); hi = np.where(lower, hi, mid)
    slant = (lo + hi)/2
    lat, lon, h = module._ecef_to_geodetic(origin + slant[:, None]*direction)
    require(np.max(np.abs(h-heights), initial=0) < .001, 'Ray height residual exceeds 1 mm')
    return lat, lon, slant, h-heights


def seed_cell(link, level):
    lat, lon, _, _ = intersect_heights(link, [SEED_HEIGHT[level]])
    return grid_index(lat[0], lon[0])


def parent_hours(parent):
    r = parent['request']
    require(len(r['year']) == len(r['month']) == 1, 'Parent must remain one calendar month')
    values = [int(datetime(int(r['year'][0]), int(r['month'][0]), int(d), int(t[:2]),
                           tzinfo=timezone.utc).timestamp()) for d in r['day'] for t in r['time']]
    require(len(values) == len(set(values)) == parent['expected_hours'], 'Original hour scope differs')
    return sorted(values)


def trace(link, level, hours, lookup):
    """lookup(var, level, epoch, i, j) -> (value, source) or None.

    Returns records, missing exact z samples, and actual target demands. Only a
    stable hourly z -> ray -> same nearest cell may produce a target demand.
    """
    seed = seed_cell(link, level)
    cells = [seed for _ in hours]; visited = [set() for _ in hours]
    records = [None for _ in hours]; missing = set(); targets = set()
    active = list(range(len(hours)))
    for iteration in range(MAX_ITERATIONS):
        available = []
        for k in active:
            i, j = cells[k]; key = ('geopotential', level, hours[k], i, j)
            item = lookup(*key)
            if item is None:
                missing.add(key)
                records[k] = {'status': 'WAITING_Z', 'cell': [i,j], 'iteration': iteration}
                continue
            z, source = item; H = float(z)/9.80665
            height = 6371229.*H/(6371229.-H) if np.isfinite(H) and H < 6371229. else float('nan')
            if not np.isfinite(height) or not link['height_amsl_m'] <= height <= 80000.:
                records[k] = {'status':'INVALID_PRESSURE_HEIGHT', 'cell':[i,j], 'z':float(z) if np.isfinite(z) else None,
                              'source':source, 'iteration':iteration}
                continue
            available.append((k, float(z), height, source))
        if not available:
            break
        lat, lon, slant, residual = intersect_heights(link, [v[2] for v in available])
        active = []
        for n, (k, z, height, source) in enumerate(available):
            updated = grid_index(lat[n], lon[n]); old = cells[k]
            if updated == old:
                records[k] = {'status':'STABLE_NEAREST', 'cell':list(old), 'intersection_lat':float(lat[n]),
                              'intersection_lon':float(lon[n]), 'height_m':height, 'slant_m':float(slant[n]),
                              'height_residual_m':float(residual[n]), 'z':z, 'source':source, 'iteration':iteration+1}
                targets.update((v, level, hours[k], *old) for v in LEVEL_VARS[level])
            elif updated in visited[k]:
                records[k] = {'status':'INVALID_NN_CELL_CYCLE','cell':list(old), 'next_cell':list(updated),
                              'iteration':iteration+1, 'source':source}
            else:
                visited[k].add(old); cells[k] = updated; active.append(k)
    for k in active:
        if records[k] is None:
            records[k] = {'status':'INVALID_NN_NO_CONVERGENCE', 'cell':list(cells[k]), 'iteration':MAX_ITERATIONS}
    require(all(r is not None for r in records), 'Unaccounted geometry row')
    return records, missing, targets


def tiles(cells):
    """Exact disjoint rectangles: no hole filling, no extra grid points."""
    remaining = set(cells); result = []
    while remaining:
        i, j = min(remaining); east = j
        while (i, east+1) in remaining:
            east += 1
        north = i
        while all((north+1, x) in remaining for x in range(j, east+1)):
            north += 1
        rect = {(y,x) for y in range(i,north+1) for x in range(j,east+1)}
        remaining.difference_update(rect)
        result.append((north, j, i, east))
    return result


def demands_to_requests(parent, demands):
    """Pack exact field/level/cell/hour demands, no variable-level cross-products.

    Identical cell masks are tiled; identical per-day hour masks share dates.
    Never fills missing dates, missing hours, empty cells or unused variable pairs.
    """
    demands = set(demands); by_cell = defaultdict(set)
    allowed = set(parent_hours(parent)); r = parent['request']
    for var, level, epoch, i, j in demands:
        require(epoch in allowed, 'Demand outside frozen calendar')
        require((level == 0 and var in SURFACE) or (level in LEVEL_VARS and
                (var in LEVEL_VARS[level] or var == 'geopotential')), 'Unapproved field-level pair')
        by_cell[(level, i, j)].add((var, epoch))
    groups = defaultdict(set)
    for (level, i, j), fields in by_cell.items():
        # Combine variables only when their exact hour sets coincide.
        variable_hours = defaultdict(set)
        for var, epoch in fields:
            variable_hours[var].add(epoch)
        hours_vars = defaultdict(list)
        for var, hours in variable_hours.items():
            hours_vars[tuple(sorted(hours))].append(var)
        for hours, variables in hours_vars.items():
            daily = defaultdict(list)
            for epoch in hours:
                dt = datetime.fromtimestamp(epoch, timezone.utc)
                daily[dt.strftime('%d')].append(dt.strftime('%H:%M'))
            equal_masks = defaultdict(list)
            for day, times in daily.items():
                equal_masks[tuple(sorted(times))].append(day)
            for times, days in equal_masks.items():
                groups[(level, tuple(sorted(variables)), tuple(sorted(days)), times)].add((i,j))
    requests = []
    for (level, variables, days, times), cells in sorted(groups.items()):
        for north, west, south, east in tiles(cells):
            req = {'product_type':['reanalysis'], 'variable':list(variables),
                   'year':list(r['year']), 'month':list(r['month']), 'day':list(days), 'time':list(times),
                   'area':[north*GRID,west*GRID,south*GRID,east*GRID],
                   'data_format':'netcdf', 'download_format':'unarchived'}
            if level:
                req['pressure_level'] = [str(level)]
            requests.append(req)
    actual = Counter(key for req in requests for key in expand_request(req))
    require(actual == Counter(demands), 'Packing introduced missing, duplicate or extra samples')
    return requests


def expand_request(req):
    n,w,s,e = req['area']; level = int(req.get('pressure_level',['0'])[0])
    require(len(req.get('pressure_level', ['0'])) == 1, 'Cross-level Cartesian request forbidden')
    require(all(abs(x/GRID-round(x/GRID)) < 1e-9 for x in (n,w,s,e)), 'Non-published grid requested')
    require(n >= s and e >= w, 'Bad rectangle')
    for d in req['day']:
        for tm in req['time']:
            epoch = int(datetime(int(req['year'][0]),int(req['month'][0]),int(d),int(tm[:2]),tzinfo=timezone.utc).timestamp())
            for i in range(round(s/GRID), round(n/GRID)+1):
                for j in range(round(w/GRID), round(e/GRID)+1):
                    for var in req['variable']:
                        yield var, level, epoch, i, j
