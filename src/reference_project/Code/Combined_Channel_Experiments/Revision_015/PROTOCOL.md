# Revision015 — RMOP propagation-branch implementation contract

T012 continuation, 2026-10-17. Holder=/root; thread=01a0bda1-92d1-7832-9508-60f4e0ec1634. This is source/input verification, not a new experiment or a completed RMOP implementation.

## Scope / reuse

READ: supplied RMOP PDF, NASA public GSDSU/NU-WRF documentation, Revision012 coverage and scores, Revision013 raw-field audit, Revision014 manifests and handoff. Previous rules and results are reused only after matching SHA. Root README/T012/Revision001 PROGRESS have unchanged SHA; their historical download, training and model lists do not override the current human decisions. Revision011 predictions and all historical files remain read-only.

WRITE: only the previously absent Code/Combined_Channel_Experiments/Revision_015 and Outputs/Experiments/Combined_Channel_Revision_015 recursively. Preserve extraction/rendering intermediates as evidence. No generated predictions, fitting, normalization, physical-reference estimation, model imports, weather requests, software installation, license acceptance, repository access request, author messaging or scheduler modification. Do not derive graupel from snow, insert zeros, shrink evaluation masks, or rename the ITU approximation RMOP.

## Preflight / Conflict Check

Two elevated CIM samples at 05:10:11.3529573Z / 05:10:14.4336546Z returned no python/pythonw/GSDSU.x/wrf.exe/wsl.exe processes. This is a bounded process check, not a claim about all possible external writers. Current visible thread inventory: only this thread active in this project among six recent tasks; unregistered/unavailable external work is not proved absent. BLOCKERS is still an unreviewed template. E drive free 165167230976 bytes. No gfortran/ifort/ifx/make/mpif90/nc-config resolved in this PowerShell PATH; this does not prove no installation elsewhere.

05:11:07.8158319Z: all 795 protected source SHAs and all 19 Revision014 delivery hashes matched; both Revision015 directories absent. Supplied PDF SHA f7ae0f8df59ced84c3b21bbb4378c7df998136ce7e0db8950187885a37e0184f. STOP SHA remains 0d629e054904ca58dde38630af24899bf89620607410da97a9b49f259253864a. Revision011 local/UK predictions match the prior frozen hashes.

| Class | Status / risk | Control |
|---|---|---|
| C1/C2 | POTENTIAL / LOW in isolated scope | New directories only; source SHA guarded before/after. |
| C3 | NONE / LOW within audited scope | No existing code imports or edits; audit-only standard-library tooling. |
| C4 | CONFIRMED / MEDIUM | Native versus adapted geometry, units and output are not equivalent; unresolved items prevent scoring. |
| C5 | NONE / LOW | Same seven families and frozen masks; no performance-based choices. |
| C6 | POTENTIAL / LOW | Short serial PDF/metadata checks; no GPU, WRF simulation or background job. |
| C7 | CONFIRMED / MEDIUM | Licensed package, author configuration, complete physical input and output contract missing. |
| C8 | NONE / LOW in this delta | Reuse raw QA and scores; only audit previously unresolved RMOP details. |

Decision=PARALLEL_ALLOWED solely for this independent output scope. Execution Lock=LOCKED during authoring/audit, started after the preflight above; final_verification.json governs release. File presence is not OS ownership. audit_bundle.py takes a one-byte OS lock while generating its records and refuses existing outputs. No old execution locks are removed or modified.

## Acceptance

Visually inspect PDF method/Table II and Appendix A; distinguish paper statements from generic software manual and from derived equations. Record exact known/missing inputs without pretending raw metadata are validated slant profiles. Reconcile coverage against frozen ledger, verify all protected inputs and retained new outputs, and deliver Chinese report, implementation matrix, QA and handoff. No need to rerun previous scores. All seven and dynamic comparison remain incomplete; do not pause tvt on this stage completion.
