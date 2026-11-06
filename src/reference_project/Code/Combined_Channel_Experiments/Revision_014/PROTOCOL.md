# Revision014 — NWP–PRM equation kernel and coefficient-source audit

T012; 2026-10-17; holder=/root; thread=01a0bda1-92d1-7832-9508-60f4e0ec1634. This is a bounded continuation of Revision013, not a new task or a completed seventh model. The current seven-family list and all no-training/no-weather-acquisition constraints remain in force.

## Scope fixed before execution

READ: frozen Revision011/012 results and Revision013 QA/source manifests; public primary literature and software documentation. WRITE: only new Code/Combined_Channel_Experiments/Revision_014 and Outputs/Experiments/Combined_Channel_Revision_014, recursively. No old scripts are imported or run. No raw weather or signal values are read for a model prediction; protected files are only SHA-read. Model checkpoints, predictions, labels, old metrics, STOP_REQUESTED and tvt are not changed.

Implement only the explicit density, power-law, component-sum and finite-column integral algebra. All numerical fixtures are SYNTHETIC, not author coefficients, not observed channels and not model results. A missing coefficient bundle is not assigned a default. Separate readiness gates must reject real scoring while coefficient provenance, frequency/polarization, gas implementation, mass convention, column boundaries and input/UTC validation remain unresolved. No coefficient regression, numerical fitting, seed selection or installation is permitted.

The implemented quadrature is explicitly piecewise-linear trapezoidal integration of supplied node extinction. This is our numerical choice, not a claim about the authors' unpublished integration implementation. Heights are geometric metres relative to the antenna, starting at exactly zero and strictly increasing; extinction is dB/km. No extrapolation, sorting, clipping or automatic pressure/geopotential conversion. A caller must supply a physically justified surface node and top boundary. The analytical cosecant operation is not a new WGS84 slant ray-tracer and is not asserted accurate at arbitrary low elevation.

## Preflight, reuse and conflicts

2026-10-17T04:54:49.3249142Z and 04:54:52.4050995Z: two successful elevated CIM queries, all Python/pythonw processes empty. The first hash command subsequently failed because a guessed filename used local_predictions rather than the actual predictions_local; that diagnostic did not write any file. The exact discovered files were then checked: Revision011 local/UK prediction SHA remain 8efbb2d5981c0405940bfddbd7e7c6a3cdce4f62577f92efff46b1f6b0b81cae and f91f9c7831a012ce29043eb477e54d4a4ddb5a5b4749f3558c081c2439e41408. STOP remains 0d629e054904ca58dde38630af24899bf89620607410da97a9b49f259253864a.

04:57:17.4559924Z: all 765 entries in the two prior guard lists match (763 distinct paths; two overlap), and all 35 Revision013 delivery entries match. Both new directories absent. Current visible task inventory shows only this thread active in this project; unavailable/unregistered external writers are not proved absent. BLOCKERS remains an unreviewed empty template, not evidence of no blockers. Root README/T012/PROGRESS hashes unchanged; latest heartbeat and Rev013 handoff override historical download/ten-model/paused clauses.

| Conflict | Assessment and bound |
|---|---|
| C1/C2 | LOW within new isolated output; historical/raw inputs read-only and SHA guarded. |
| C3 | LOW: new pure standard-library kernel; no old model or downloader imports. |
| C4 | MEDIUM: formula availability does not establish coefficients or physical applicability; fail-closed scoring contract. |
| C5 | LOW: no ranking-driven selection and no label use. |
| C6 | LOW: short serial CPU tests, no GPU/background worker/weather requests. |
| C7 | CONFIRMED/MEDIUM: coefficient and physical-input gaps block actual PRM scores; synthetic algebra tests independent. |
| C8 | LOW: reuse all four scored families and Rev013 field QA; no duplicate calculations. |

Decision=PARALLEL_ALLOWED only for this independent, bounded scope. Execution Lock=LOCKED for this new version during authoring/testing; a real one-byte OS lock is acquired and released by the test and verifier processes. Final handoff releases logical ownership. Do not delete or infer ownership from old lock files.

## Acceptance

Independent analytic/Decimal checks, invalid-input rejection, source/output SHA and publication checks must pass. Publish source lookup ledger, equation contract, numerical QA and Chinese handoff/report. Mark no new real predictions/scores, no training, no weather requests and seven-family comparison incomplete. Do not rerun one-shot entries after their output exists; fixes use new names and retain failures. Scheduler remains unchanged.
