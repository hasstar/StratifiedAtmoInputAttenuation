"""Hourly POWER-mean adapter for Revision020's frozen single-layer kernel.

This is an engineering adapter, not minute observations or author software.
H and v are fixed within the output hour. No labels, fitting, file writes,
default physical coefficients or real-site eligibility are provided here.
"""
import importlib.util
import math
from pathlib import Path

_source = Path(__file__).resolve().parent.parent / 'Revision_020/esst_hourly_kernel.py'
_spec = importlib.util.spec_from_file_location('frozen_esst020', _source)
legacy = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(legacy)
ContractError = legacy.ContractError
InsufficientContextError = legacy.InsufficientContextError
DB_TO_NEPER = math.log(10.0) / 10.0


def geometric_height_m(geopotential_height_m, *, earth_radius_m):
    """Explicit-radius spherical approximation; vertical datum unchanged."""
    h = legacy._number(geopotential_height_m, 'geopotential_height_m')
    r = legacy._number(earth_radius_m, 'earth_radius_m')
    if r <= 0 or h >= r:
        raise ContractError('requires radius>0 and height<radius')
    value = h / (1.0 - h / r)
    if not math.isfinite(value):
        raise ContractError('height conversion overflow')
    return value


def hourly_power_mean(rain_rates, sample_time_seconds, anchor_bin, **parameters):
    """Exact hour mean under piecewise-constant rain and fixed hourly H/v.

    Arguments and units are identical to Revision020.attenuation_at_bin_start.
    Requires enough rain context for EVERY instant in the requested hour.
    Returns rain-only effective attenuation of mean linear transmission,
    alongside a diagnostic arithmetic dB mean. Neither is a site prediction
    unless a separate real-input and target-alignment contract is satisfied.
    """
    anchor = legacy.attenuation_at_bin_start(
        rain_rates, sample_time_seconds, anchor_bin, **parameters)
    rates = legacy._hourly_grid(rain_rates, sample_time_seconds,
        timestep_seconds=parameters['timestep_seconds'], rain_unit=parameters['rain_unit'])
    height, wind = float(parameters['height_above_antenna']), float(parameters['wind_speed'])
    angle, k, alpha = (float(parameters[x]) for x in ('elevation', 'k', 'alpha'))
    # Relative clock avoids subtracting epoch-size numbers in fractional steps.
    left, right = anchor_bin * 3600.0, (anchor_bin + 1) * 3600.0
    if height == 0 or angle == 90.0:
        return {'power_mean_attenuation_db': anchor, 'arithmetic_mean_attenuation_db': anchor,
                'anchor_attenuation_db': anchor, 'min_attenuation_db': anchor,
                'max_attenuation_db': anchor, 'lookahead_seconds': 0.0,
                'segments': 1, 'hour_duration_seconds': 3600.0}
    theta = math.radians(angle)
    width = 1000.0 * height / (wind * math.tan(theta))
    factor = wind / (1000.0 * math.cos(theta))
    if not math.isfinite(width) or width <= 0 or not math.isfinite(factor):
        raise ContractError('unrepresentable time geometry')
    if right + width > len(rates) * 3600.0:
        raise InsufficientContextError('full output hour requires additional forward rain context')
    gamma = []
    try:
        for rate in rates:
            g = k * rate ** alpha if rate > 0 and k > 0 else 0.0
            if not math.isfinite(g):
                raise ContractError('nonfinite rain extinction')
            gamma.append(g)
    except OverflowError as exc:
        raise ContractError('rain extinction overflow') from exc

    def value_at(t):
        last = t + width
        i = max(0, int(math.floor(t / 3600.0)))
        stop = min(len(rates), int(math.ceil(last / 3600.0)))
        parts = []
        for j in range(i, stop):
            overlap = min(last, (j + 1) * 3600.0) - max(t, j * 3600.0)
            if overlap > 0:
                parts.append(gamma[j] * overlap)
        answer = factor * math.fsum(parts)
        if not math.isfinite(answer):
            raise ContractError('integrated attenuation overflow')
        return answer

    breaks = {left, right}
    for j in range(anchor_bin + 1, min(len(rates), int(math.ceil((right + width) / 3600.0))) + 1):
        for t in (j * 3600.0, j * 3600.0 - width):
            if left < t < right:
                breaks.add(t)
    breaks = sorted(breaks)
    values = [value_at(t) for t in breaks]
    logs, db_parts = [], []
    for a, b, y0, y1 in zip(breaks, breaks[1:], values, values[1:]):
        fraction = (b - a) / 3600.0
        delta = DB_TO_NEPER * abs(y1 - y0)
        # Mean exp(-c*A) for linear A. Use low endpoint + stable expm1.
        correction = math.log(-math.expm1(-delta) / delta) if delta > 0 else 0.0
        logs.append(math.log(fraction) - DB_TO_NEPER * min(y0, y1) + correction)
        db_parts.append(fraction * (y0 + y1) / 2.0)
    largest = max(logs)
    log_mean = largest + math.log(math.fsum(math.exp(x - largest) for x in logs))
    result = -log_mean / DB_TO_NEPER
    if not math.isfinite(result):
        raise ContractError('nonfinite effective attenuation')
    return {'power_mean_attenuation_db': result,
            'arithmetic_mean_attenuation_db': math.fsum(db_parts),
            'anchor_attenuation_db': anchor,
            'min_attenuation_db': min(values), 'max_attenuation_db': max(values),
            'lookahead_seconds': width, 'segments': len(logs), 'hour_duration_seconds': 3600.0}
