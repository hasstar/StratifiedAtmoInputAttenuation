"""E-SST single-layer *hourly-input numerical kernel*, not a site adapter.

Source: Luini et al., DOI 10.1109/TAP.2020.2981682, PDF pp. 5-6,
Fig. 7 and Eqs. (8)-(10). The paper uses one-minute rain observations.
This module instead integrates piecewise-constant 3600-second rain bins.
It neither reconstructs minute observations nor validates that adaptation.

Rain: mm/h; velocity: m/s; path height: km above antenna; angle: degrees.
k has units (dB/km)/(mm/h)**alpha and must be supplied explicitly.
The coefficients used by tests are artificial, NOT P.838 site coefficients.
No files, network, fitting, random generators, labels or measured-channel
history are used. Output is rain attenuation only, not total relative fade.
"""

import math
from collections.abc import Mapping, Sequence


class ContractError(ValueError):
    """An explicit numerical or metadata contract was not satisfied."""


class InsufficientContextError(ContractError):
    """The forward frozen-storm window exceeds supplied weather context."""


REAL_SCORING_GATES = (
    "hourly_adaptation_explicitly_accepted",
    "liquid_rain_rate_and_accumulation_interval_verified",
    "utc_bins_and_hour_duration_verified",
    "continuous_context_and_complete_events_verified",
    "zero_level_and_antenna_common_datum_verified",
    "wind_700_hpa_and_path_motion_verified",
    "frequency_polarization_and_geometry_verified",
    "p838_coefficients_and_units_verified",
    "rain_vs_total_relative_target_aligned",
)


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(f"{name} must be a finite real number, not bool")
    value = float(value)
    if not math.isfinite(value):
        raise ContractError(f"{name} must be finite")
    return value


def _hourly_grid(rain_rates: Sequence, sample_time_seconds: Sequence, *,
                 timestep_seconds, rain_unit):
    if rain_unit != "mm/h":
        raise ContractError("rain_unit must be mm/h (liquid rain rate)")
    if _number(timestep_seconds, "timestep_seconds") != 3600.0:
        raise ContractError("this adaptation requires actual 3600-second bins")
    rates = tuple(_number(x, "rain rate") for x in rain_rates)
    times = tuple(_number(x, "sample time") for x in sample_time_seconds)
    if not rates or len(rates) != len(times):
        raise ContractError("nonempty rain and bin-start time arrays must match")
    if any(x < 0 for x in rates):
        raise ContractError("negative rain is invalid; no clipping to zero")
    if any(b - a != 3600.0 for a, b in zip(times, times[1:])):
        raise ContractError("missing, repeated or non-hourly bins are forbidden")
    return rates


def classify_hourly_events(rain_rates, sample_time_seconds, *,
                           timestep_seconds, rain_unit):
    """Events use whole zero-rain hourly bins as >=1 h dry separators.

    Peak >10 mm/h => convective; otherwise stratiform. Edge events are
    flagged as censored; this function does NOT certify their full peaks.
    Hourly peaks are not the minute peaks used by the original method.
    """
    rates = _hourly_grid(rain_rates, sample_time_seconds,
                         timestep_seconds=timestep_seconds, rain_unit=rain_unit)
    events = []
    start = None
    for index in range(len(rates) + 1):
        wet = index < len(rates) and rates[index] > 0.0
        if wet and start is None:
            start = index
        if not wet and start is not None:
            peak = max(rates[start:index])
            events.append({
                "start_bin": start, "stop_bin_exclusive": index,
                "peak_mm_per_h": peak,
                "regime": "convective" if peak > 10.0 else "stratiform",
                "left_censored": start == 0,
                "right_censored": index == len(rates),
            })
            start = None
    return events


def equivalent_height_above_antenna_km(zero_level_height, antenna_height, *,
                                      absolute_height_unit, regime):
    """Convert an explicitly common AMSL datum to the ray's above-antenna H.

    Caller must establish that conversion; an ERA5 AGL field cannot be
    relabeled AMSL. Stratiform: h0+0.36 km; convective: h0, then antenna
    subtraction. Below-antenna zero levels are rejected, never clipped.
    """
    if absolute_height_unit != "m_AMSL":
        raise ContractError("both heights must use a verified common m_AMSL datum")
    zero = _number(zero_level_height, "zero_level_height")
    antenna = _number(antenna_height, "antenna_height")
    if zero < antenna:
        raise ContractError("zero-degree level below antenna requires separate physics")
    if regime not in ("stratiform", "convective"):
        raise ContractError("event regime must be explicitly classified")
    height = (zero - antenna) / 1000.0 + (0.36 if regime == "stratiform" else 0.0)
    if not math.isfinite(height):
        raise ContractError("equivalent path height must remain finite")
    return height


def attenuation_at_bin_start(rain_rates, sample_time_seconds, anchor_bin, *,
                             wind_speed, height_above_antenna, elevation,
                             k, alpha, timestep_seconds, rain_unit,
                             wind_unit, height_unit, angle_unit):
    """Integrate the single-layer ray over forward piecewise-constant bins.

    Wind and equivalent height are held fixed at this anchor. Motion is
    assumed along the horizontal ray, as in the paper's frozen-storm model.
    A = sec(theta) * integral_0^(H/tan(theta)) k R(l)**alpha dl.
    No missing-weather extrapolation, border replication or zero padding.
    At exactly 90 degrees the analytic limit is k R(anchor)**alpha * H.
    This bin-start output is NOT automatically an hourly-mean attenuation.
    """
    rates = _hourly_grid(rain_rates, sample_time_seconds,
                         timestep_seconds=timestep_seconds, rain_unit=rain_unit)
    if isinstance(anchor_bin, bool) or not isinstance(anchor_bin, int):
        raise ContractError("anchor_bin must be an integer")
    if not 0 <= anchor_bin < len(rates):
        raise ContractError("anchor_bin outside input")
    if (wind_unit, height_unit, angle_unit) != ("m/s", "km_above_antenna", "degree"):
        raise ContractError("explicit wind/path-height/elevation units do not match")
    wind = _number(wind_speed, "wind_speed")
    height = _number(height_above_antenna, "height_above_antenna")
    angle = _number(elevation, "elevation")
    coefficient = _number(k, "k")
    exponent = _number(alpha, "alpha")
    if wind <= 0 or height < 0 or not 0 < angle <= 90 or coefficient < 0 or exponent <= 0:
        raise ContractError("requires v>0, H>=0, 0<elevation<=90, k>=0 and alpha>0")

    def gamma(rate):
        if coefficient == 0.0 or rate == 0.0:
            return 0.0
        try:
            value = coefficient * rate ** exponent
        except OverflowError as exc:
            raise ContractError("rain extinction overflow") from exc
        if not math.isfinite(value):
            raise ContractError("rain extinction must remain finite")
        return value

    if height == 0.0:
        return 0.0
    if angle == 90.0:
        result = gamma(rates[anchor_bin]) * height
    else:
        theta = math.radians(angle)
        horizontal_span = height / math.tan(theta)
        bin_width = wind * 3600.0 / 1000.0
        available_bins = len(rates) - anchor_bin
        if not math.isfinite(horizontal_span) or not math.isfinite(bin_width) or bin_width <= 0:
            raise ContractError("unrepresentable horizontal geometry")
        if horizontal_span > available_bins * bin_width:
            raise InsufficientContextError("forward weather window is incomplete")
        full_bins = min(int(horizontal_span / bin_width), available_bins)
        remainder = horizontal_span - full_bins * bin_width
        pieces = [gamma(rates[anchor_bin + offset]) * bin_width
                  for offset in range(full_bins)]
        if remainder > 0:
            if full_bins >= available_bins:
                raise InsufficientContextError("partial final bin is unavailable")
            pieces.append(gamma(rates[anchor_bin + full_bins]) * remainder)
        try:
            result = math.fsum(pieces) / math.cos(theta)
        except OverflowError as exc:
            raise ContractError("integrated attenuation overflow") from exc
    if not math.isfinite(result):
        raise ContractError("integrated attenuation must remain finite")
    return result


def require_real_scoring_contract(evidence_gates: Mapping):
    """Fail closed; test success alone never opens the site-scoring gate.

    True must be backed by an independently reviewed provenance record.
    This guard validates declarations only, not the evidence itself.
    """
    unknown = set(evidence_gates) - set(REAL_SCORING_GATES)
    if unknown:
        raise ContractError("unknown evidence gates: " + ", ".join(sorted(unknown)))
    missing = [name for name in REAL_SCORING_GATES if evidence_gates.get(name) is not True]
    if missing:
        raise ContractError("real E-SST scoring BLOCKED: " + ", ".join(missing))
