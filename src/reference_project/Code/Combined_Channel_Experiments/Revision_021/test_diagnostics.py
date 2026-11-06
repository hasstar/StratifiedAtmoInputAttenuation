"""Small hand-derived boundary tests of the new diagnostic helpers, not Rev020 reruns."""
import importlib.util
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[2] / 'Outputs/Experiments/Combined_Channel_Revision_021'
spec = importlib.util.spec_from_file_location('rev021_diag', HERE / 'diagnose_inputs.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
checks = []


def check(name, condition):
    checks.append({'name': name, 'pass': bool(condition)})
    assert condition, name


def profile(t, h=None, valid=None):
    t = np.array(t, float)
    h = np.arange(len(t)) * 1000. if h is None else np.array(h, float)
    valid = np.ones(len(t), bool) if valid is None else np.array(valid, bool)
    return module.crossings(t[None, None], h[None, None], valid[None, None])


q = profile([275.15, 271.15])
check('linear_zero_at_500_geopotential_m', q['candidate_geopotential_m'][0, 0, 0] == 500)
check('width_1000', q['bracket_width_geopotential_m'][0, 0, 0] == 1000)
check('all_cold_not_zero_height', np.isnan(profile([270, 269])['candidate_geopotential_m']).all())
check('all_warm_not_extrapolated', profile([280, 279])['warm_to_cold_count'].item() == 0)
q = profile([275, 271, 275, 271])
check('both_warm_to_cold_branches_retained', q['warm_to_cold_count'].item() == 2)
check('reverse_crossing_recorded', q['cold_to_warm_count'].item() == 1)
check('below_surface_pair_rejected', profile([275, 271], valid=[False, True])['warm_to_cold_count'].item() == 0)
check('missing_middle_not_bridged', profile([275, np.nan, 271], valid=[True, False, True])['warm_to_cold_count'].item() == 0)
q = profile([275, 271], h=[1000, 500])
check('descending_height_rejected', q['warm_to_cold_count'].item() == 0)
check('descending_height_flagged', q['nonascending_pair_count'].item() == 1)
q = profile([273.15, 270])
check('exact_zero_separately_flagged', q['exact_zero_count'].item() == 1)
check('exact_zero_not_silently_chosen', q['warm_to_cold_count'].item() == 0)
ids, events = module.wet_events(np.arange(9) * 3600, np.array([0, 1, 10, 0, 11, np.nan, 2, 0, 1.]))
check('four_event_fragments', len(events) == 4)
check('bounded_event_complete', events[0]['complete'])
check('exact_ten_is_not_convective', events[0]['hourly_peak_type_candidate'] == 'stratiform')
check('over_ten_candidate_convective', events[1]['hourly_peak_type_candidate'] == 'convective')
check('missing_rain_not_dry_boundary', events[1]['right_censored'] and events[2]['left_censored'])
check('last_event_right_censored', events[3]['right_censored'])
check('invalid_and_dry_have_no_event', np.array_equal(ids[[0, 3, 5, 7]], [-1] * 4))
with (OUT / 'unit_checks.json').open('x', encoding='utf-8') as f:
    json.dump({'status': 'PASS', 'checks': checks, 'n': len(checks),
               'scope': 'new Revision021 diagnostic functions only'}, f, indent=2)
print(f'{len(checks)} diagnostic boundary checks PASS')
