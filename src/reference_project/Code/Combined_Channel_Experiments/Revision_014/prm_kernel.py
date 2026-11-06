"""Explicit PRM algebra only; no calibrated coefficients or real-data adapter.

Reference equations: Biscarini et al. (2024), DOI 10.1109/TAP.2024.3455567.
This module is NOT the authors' implementation and cannot produce a validated
comparison without a separately verified physical contract. No I/O or fitting.
"""
import math

R_DRY = 287.05
COMPONENTS = ('oxygen', 'water_vapour', 'cloud', 'rain')
REAL_GATES = (
    'coefficient_numeric_bundle', 'coefficient_source_and_license',
    'particle_distribution_shape_orientation', 'frequency_polarization_contract',
    'temperature_density_validity_domain', 'gas_model_and_units',
    'hydrometeor_mass_basis', 'geometric_column_surface_top',
    'input_field_utc_qa', 'fixed_physical_clear_reference',
)


class ContractError(ValueError):
    pass


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ContractError(name + ': numeric scalar required')
    if not math.isfinite(value):
        raise ContractError(name + ': nonfinite')
    return float(value)


def _vector(values, name, minimum=0.0):
    vals = tuple(_number(v, name) for v in values)
    if not vals or any(v < minimum for v in vals):
        raise ContractError(name + ': empty or below allowed minimum')
    return vals


def qi_to_g_per_kg(values, *, unit):
    """Numerical unit conversion only: does not convert moist/dry mass bases."""
    if unit not in ('g/kg', 'kg/kg'):
        raise ContractError('Explicit Qi mass unit required')
    vals = _vector(values, 'Qi')
    factor = 1000.0 if unit == 'kg/kg' else 1.0
    result = tuple(v * factor for v in vals)
    return _vector(result, 'Qi_g_per_kg')


def density_g_per_m3(qi_g_kg, pressure, temperature, *, pressure_unit,
                     temperature_unit):
    """Paper Eq.2, with Qi's denominator convention left to the source contract."""
    if pressure_unit != 'Pa' or temperature_unit != 'K':
        raise ContractError('Requires pressure Pa and temperature K')
    q = _vector(qi_g_kg, 'Qi_g_kg')
    p = _vector(pressure, 'pressure')
    t = _vector(temperature, 'temperature')
    if len({len(q), len(p), len(t)}) != 1 or min(p) <= 0 or min(t) <= 0:
        raise ContractError('Shape mismatch or nonpositive p/T')
    return _vector((a*b/(R_DRY*c) for a,b,c in zip(q,p,t)), 'rho_g_m3')


def hydrometeor_extinction(rho_g_m3, *, a, b):
    """Eq.1. a is in (dB/km)/(g/m3)**b. No defaults or coefficient fitting."""
    a = _number(a, 'a')
    b = _number(b, 'b')
    if a < 0 or b <= 0:
        raise ContractError('Require a>=0 and b>0; zero density must give zero')
    rho = _vector(rho_g_m3, 'rho_g_m3')
    try:
        result = tuple(a * r ** b for r in rho)
    except OverflowError as exc:
        raise ContractError('Extinction overflow') from exc
    return _vector(result, 'extinction_dB_per_km')


def column_integral(k_db_km, height, elevation, *, height_unit, elevation_unit):
    """Our trapezoid implementation of Eq.5 on a supplied complete column.

    No extrapolation or implicit near-surface infill. 0<theta<=90 is an algebraic
    domain, not a claim that plane-parallel propagation is accurate throughout.
    """
    if height_unit != 'm_above_antenna' or elevation_unit != 'degree':
        raise ContractError('Explicit geometric-height and angle units required')
    k = _vector(k_db_km, 'k_dB_km')
    z = _vector(height, 'geometric_height_m')
    theta = _number(elevation, 'elevation_degree')
    if len(k) != len(z) or len(z) < 2:
        raise ContractError('Matched column with at least two nodes required')
    if z[0] != 0 or any(b <= a for a,b in zip(z,z[1:])):
        raise ContractError('Surface node must be zero; heights strictly ascending')
    if not 0 < theta <= 90:
        raise ContractError('Elevation outside (0,90] degrees')
    integral = math.fsum((a+b)*0.5*(d-c)/1000.0
                         for a,b,c,d in zip(k,k[1:],z,z[1:]))
    result = integral / math.sin(math.radians(theta))
    return _number(result, 'path_attenuation_dB')


def four_component_paths(components, height, elevation):
    """Each of four node profiles is supplied; gas is NOT inferred from ITU."""
    if set(components) != set(COMPONENTS):
        raise ContractError('Exactly oxygen, water_vapour, cloud and rain required')
    height = tuple(height)
    paths = {key: column_integral(components[key], height, elevation,
                                 height_unit='m_above_antenna',
                                 elevation_unit='degree') for key in COMPONENTS}
    paths['total_absolute_dB'] = math.fsum(paths.values())
    return paths


def relative_attenuation(absolute_db, fixed_gas_reference_db):
    """Approved comparison convention, not an original PRM calibration."""
    a = _number(absolute_db, 'absolute_attenuation')
    ref = _number(fixed_gas_reference_db, 'fixed_gas_reference')
    if a < 0 or ref < 0:
        raise ContractError('Absolute attenuation/reference must be nonnegative')
    return a - ref  # Negative relative attenuation is intentionally retained.


def require_real_scoring_contract(contract, *, frequency_ghz, polarization):
    """Schema gate only, never a replacement for independent scientific QA."""
    if contract.get('fixture_kind') == 'SYNTHETIC':
        raise ContractError('Synthetic fixtures cannot be scored as a paper model')
    missing = [key for key in REAL_GATES if contract.get(key) is not True]
    if missing:
        raise ContractError('Unresolved real-scoring gates: ' + ', '.join(missing))
    target_f = _number(frequency_ghz, 'target_frequency_GHz')
    source_f = _number(contract.get('frequency_ghz'), 'coefficient_frequency_GHz')
    if target_f <= 0 or abs(target_f-source_f) > 1e-9:
        raise ContractError('No implicit coefficient frequency interpolation')
    if polarization not in ('LHCP', 'RHCP', 'H', 'V'):
        raise ContractError('Explicit supported polarization required')
    if polarization != contract.get('polarization'):
        raise ContractError('Coefficient polarization mismatch')
    if not contract.get('coefficient_sha256') or not contract.get('source_url'):
        raise ContractError('Coefficient identity/source absent')
    return True
