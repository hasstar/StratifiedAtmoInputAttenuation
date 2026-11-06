"""Isolated reuse of frozen DAP decoder, plus hourly precipitation/snow rates.

Original Top10 is unchanged. New rates are separate rain diagnostics, not model
inputs. Mean rates cover the hour ENDING at valid UTC, not since forecast start.
"""
import importlib.util
import re
from pathlib import Path

spec = importlib.util.spec_from_file_location('ncar_top10_private_v011', Path(__file__).with_name('ncar_top10_transport_v001.py'))
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
RAIN = ('mean_total_precipitation_rate', 'mean_snowfall_rate')
base.FIELDS.update({RAIN[0]: ('MTPR',55,'mtpr','ll025sc',('kg m**-2 s**-1',)),
                    RAIN[1]: ('MSR',31,'msr','ll025sc',('kg m**-2 s**-1',))})
original_locate, original_contract = base.locate, base.metadata_contract

def locate(variable, level, epoch):
    if variable not in RAIN:
        return original_locate(variable, level, epoch)
    base.check(level == 0, 'Rain diagnostics must be surface')
    file, layout, start, temporal = original_locate(base.core.SURFACE[1], 0, epoch)
    _, pid, short, _, _ = base.FIELDS[variable]
    file = file.replace('sfc.minmax', 'sfc.meanflux').replace('128_201_mx2t', f'235_{pid:03d}_{short}')
    return file, layout, start, temporal

def metadata_contract(body, node):
    result = original_contract(body, node)
    if node['variable'] in RAIN:
        block = re.search(r'\b' + node['nc_variable'] + r'\s*\{([^{}]*)\}', body, re.S)[1]
        base.check(re.search(r'ecmwf_local_table\s+235\s*;', block), 'Wrong ECMWF rain table')
        expected = {'MTPR':'Mean total precipitation rate', 'MSR':'Mean snowfall rate'}[node['nc_variable']]
        base.check(f'String long_name "{expected}";' in block, 'Wrong rain parameter identity')
        result.update(ecmwf_local_table=235, param_id=235000+result['parameter_id'],
                      processing_period='one hour ending at valid UTC; no cumulative differencing',
                      quantity='water-equivalent rate; raw kg m-2 s-1')
    return result

base.locate = locate
base.metadata_contract = metadata_contract

def __getattr__(name):
    return getattr(base, name)
