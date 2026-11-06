"""Approved LOCAL engineering adapter. Pure functions; no labels/I/O/fitting."""
import math
import numpy as np


def gate(candidates_m, up_crossings, exact_zeros, bad_pairs, wind_mps,
         elevation_deg, rain, event_ids, events, anchor, antenna_m=117.5):
    """Conservative full-hour stratiform-window contract from Revision023."""
    reasons = []
    h = np.asarray(candidates_m, dtype=float)
    if h.ndim != 1 or not np.isfinite(h).all():
        raise ValueError('finite candidate list required; missing slots removed explicitly')
    if len(h) == 0: reasons.append('ZERO_LEVEL_NO_DOWN_CROSSING')
    if len(h) + up_crossings > 1: reasons.append('ZERO_LEVEL_MULTIPLE_CROSSINGS')
    if exact_zeros: reasons.append('ZERO_LEVEL_EXACT_ZERO_AMBIGUITY')
    if bad_pairs: reasons.append('ZERO_LEVEL_NONASCENDING_PROFILE')
    if not math.isfinite(wind_mps) or wind_mps <= 0: reasons.append('WIND_INVALID')
    if not math.isfinite(elevation_deg) or not 0 < elevation_deg <= 90:
        raise ValueError('elevation domain')
    result = dict(reasons=reasons, regime=None, height_km=None,
                  conservative_lookahead_s=None, used_lookahead_s=None,
                  context_bins=None, wet_event_ids=[])
    if reasons: return result
    hs = (float(h[0]) + 360.0 - antenna_m) / 1000.0
    if hs < 0:
        reasons.append('EQUIVALENT_LAYER_BELOW_ANTENNA'); return result
    ws = 0.0 if elevation_deg == 90 else 1000.0 * hs / (wind_mps * math.tan(math.radians(elevation_deg)))
    if not math.isfinite(ws):
        reasons.append('TIME_GEOMETRY_NONFINITE'); return result
    n = int(math.ceil(1.0 + ws / 3600.0))
    result.update(conservative_lookahead_s=ws, context_bins=n)
    if anchor < 0 or anchor + n > len(rain):
        reasons.append('MISSING_FORWARD_CONTEXT'); return result
    values = np.asarray(rain[anchor:anchor+n], dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        reasons.append('INVALID_LIQUID_RAIN_CONTEXT'); return result
    ids = sorted({int(event_ids[anchor+i]) for i, value in enumerate(values) if value > 0})
    result['wet_event_ids'] = ids
    if any(i < 0 or i >= len(events) for i in ids):
        raise ValueError('wet bin missing event identity')
    if any(not events[i]['complete'] for i in ids): reasons.append('CENSORED_WET_EVENT')
    types = {events[i]['hourly_peak_type_candidate'] for i in ids}
    if types - {'stratiform', 'convective'}: raise ValueError('unknown rain type')
    if len(types) > 1: reasons.append('MIXED_RAIN_REGIMES')
    if reasons: return result
    regime = next(iter(types)) if types else 'dry_window'
    actual_h = hs if regime != 'convective' else (float(h[0]) - antenna_m) / 1000.0
    if actual_h < 0:
        reasons.append('EQUIVALENT_LAYER_BELOW_ANTENNA'); return result
    width = 0.0 if elevation_deg == 90 else 1000.0 * actual_h / (wind_mps * math.tan(math.radians(elevation_deg)))
    result.update(regime=regime, height_km=actual_h, used_lookahead_s=width)
    return result


def record_attenuations(rates, offset_ms, *, height_km, wind_mps, elevation_deg, k, alpha):
    """Forward box integral using local prefix antiderivative, no interpolation."""
    r, offsets = np.asarray(rates, dtype=float), np.asarray(offset_ms)
    if r.ndim != 1 or not len(r) or not np.isfinite(r).all() or (r < 0).any():
        raise ValueError('nonnegative finite rain bins required')
    if offsets.dtype.kind not in 'iu' or offsets.ndim != 1 or not len(offsets):
        raise ValueError('nonempty integer millisecond offsets required')
    if (offsets < 0).any() or (offsets >= 3600000).any() or (np.diff(offsets.astype(np.int64)) <= 0).any():
        raise ValueError('unique increasing timestamps inside output hour required')
    params = (height_km, wind_mps, elevation_deg, k, alpha)
    if not all(math.isfinite(v) for v in params) or height_km < 0 or wind_mps <= 0 or not 0 < elevation_deg <= 90 or k < 0 or alpha <= 0:
        raise ValueError('physical parameter domain')
    gamma = k * np.power(r, alpha)
    if not np.isfinite(gamma).all(): raise ValueError('nonfinite specific attenuation')
    if height_km == 0: return np.zeros(len(offsets))
    if elevation_deg == 90: return np.full(len(offsets), gamma[0] * height_km)
    theta = math.radians(elevation_deg)
    width = 1000.0 * height_km / (wind_mps * math.tan(theta))
    if not math.isfinite(width) or 3600.0 + width > len(r) * 3600.0:
        raise ValueError('entire output hour lacks weather context')
    prefix = np.r_[0.0, np.cumsum(gamma * 3600.0)]
    t = offsets.astype(float) / 1000.0

    def primitive(u):
        j = np.floor(u / 3600.0).astype(np.int64)
        if (j < 0).any() or (j > len(r)).any(): raise ValueError('integral outside domain')
        # The appended zero is used only at the exact supplied right endpoint.
        return prefix[j] + np.r_[gamma, 0.0][j] * (u - j * 3600.0)

    a = wind_mps / (1000.0 * math.cos(theta)) * (primitive(t + width) - primitive(t))
    if not np.isfinite(a).all() or (a < 0).any(): raise ValueError('invalid attenuation, never clip')
    return a


def power_mean_db(attenuations):
    a = np.asarray(attenuations, dtype=float)
    if a.ndim != 1 or not len(a) or not np.isfinite(a).all():
        raise ValueError('finite nonempty attenuation vector required')
    low = float(a.min())
    return low - 10.0 * math.log10(float(np.mean(np.exp(-(a-low) * math.log(10.0)/10.0))))
