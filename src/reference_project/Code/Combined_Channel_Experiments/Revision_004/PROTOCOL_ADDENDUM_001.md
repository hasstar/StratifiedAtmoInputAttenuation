# Input-evidence correction before physical results

2026-10-16, after opening native headers but BEFORE physical calculations/results.
Both first source files explicitly state Left Hand Circularly Polarized in the
title AND polarisation_transmit AND polarisation_receive fields. The earlier
Revision003 missing-polarization statement was an inspection gap, not evidence
that these data lack polarization metadata. Correct that statement prospectively.

Require every one of the236 native source files used by the expanded comparison
to agree on20.7GHz and matched left-hand circular polarization, and receiver
coordinates/AMSL altitude/satellite location to agree with the frozen plan.
If all pass, the main circular scenario is SOURCE-CONFIRMED, not assumed.
Horizontal/vertical scenarios remain the already planned counterfactual controls,
not plausible unknown-polarization scenarios or candidates for selecting a winner.
No target, model output or error has been used to make this correction.

Native elevation is a coarse30degrees, whereas the already frozen nominal WGS84
GEO geometry gives approximately28.05/28.09degrees. Retain that predeclared nominal
geometry for consistency; do not claim it is an orbital-ephemeris measurement.
All height scenarios, rain masks, sample/metric definitions and no-training rules
in PROTOCOL.md remain unchanged. The static P839 map is still approximate annual
rain height, not an observed hourly zero-degree height.

Reference formula pages rendered/read visually: P838-3 printed pages1–3 and6;
P839-4 printed page1; P618-14 printed pages7–8. Coefficient table and fraction/
exponent/trigonometric placement have been checked against those rendered pages.
