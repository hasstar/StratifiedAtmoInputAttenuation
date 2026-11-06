# T015 transport repair v003 — 2026-10-21

This is a transport-only continuation of Revision_037, not a new experiment or a new acquisition plan. The human explicitly authorized the proposed bounded repair in this same thread: “快点整吧。只要能下载，什么操作都可以”. This supersedes the 01:56 hold awaiting explicit repair authorization, not the scientific or credential boundaries.

## Scope / preservation

- Original 2012 parent requests, exact axes/UTC/parameters/levels/grids, 236 station-days, 27 inputs, 12 CDS jobs, models and each LOCAL RFE path remain immutable. No fitting, interpolation, substitute values, dates, sites, new environment, paid service, old queue access or old STOP changes.
- Original 18 Revision_037 files are read-only and pinned by the last continuation guard. Its 16 plan source/input SHA entries and plan SHA 92c2f404c4b2e4c9840857b829f10cd807417be412493824fb5285562eddf1b2 must match. All successful original receipts and failures are preserved.
- New code is isolated here. New hourly response bytes and receipts go under the existing task output's acquisition/ncar_hourly_v003. Completed parent composites are explicitly DERIVED_TIME_CONCAT, not claimed to be an HTTP daily response. They occupy previously absent parent cache paths only, retain all raw hourly hashes, and must pass both the original decoder and independent raw-hour read-back before counting.

## Bounded acquisition

One local acquisition worker, one request at a time. Reuse exact verified existing parent cache. For known problematic 5 hPa temperature parents, split directly into their existing individual hours. For other missing parents try the original exact URL once with 60-second read limit; only a transient network/408/429/5xx error allows the deterministic hourly fallback. This new bound does not change v002. Each hourly GET has connect=10/read=180 seconds and at most two attempts separated by 15 seconds; an exhausted hour stops the worker for inspection. Nontransient HTTP, identity, coordinate, SHA, incomplete-body or numeric failures stop immediately. No blind worker restart or bypassing failed input.

All child URLs are derived only by selecting one existing time index from that parent; remaining axes are identical. Each child is checked against DAS parameter/units plus exact time, level, grid, shape, finite values. Parent concatenation is in frozen time order with no arithmetic on values. Incomplete parents never increment the 2012 counter. Existing hourly fragments resume only after receipt/URL/bytes/SHA/decoder checks.

The existing new CDS plan and exact registered pending ID are reused. Poll/submit/download via unchanged cds_three.step, serially between NCAR requests, no more than once per 60 seconds. At most one unresolved CDS job, no old tasks, no ambiguous POST retries; credential and signed-URL suppression unchanged.

## Verification / continuation

Offline tests must cover all 2012 layouts, old real payload replay, wrong UTC/grid/parameter, missing/duplicate hours, truncation and tampered fragment/hash rejection. Synthetic fixtures are explicitly tests and never acquisition input. Live repaired parent requires complete independent fragment-to-parent equality before acceptance. The source manifest pins all repair sources and original sources; it is sealed after tests and before network use.

The new passive continuation waits for exact v003 PID/create-time/command and requires both the original complete-plan manifest and a PASS transport-provenance audit. It does not restart failures. Then unchanged CDS drain, assembly, independent full input QA, 100 frozen inferences, independent metric QA, six figures and result report run strictly serially. A transport test or download success is not full input QA. Final six plots require actual visual inspection.

## Conflict Check / reuse

2026-10-21 09:38 +08, Codex thread 01a0bda1-92d1-7832-9508-60f4e0ec1634. Root README fully read including supplementary section; task, project/WF01, control blockers/registry, original protocols/HANDOFF and relevant source inspected. Host scan: zero Revision_037 pipeline instances; old 39108/36632 absent, both OS locks free, E: 127.44 GiB free. Read-only check: original 18 source/protocol and 16 source/input hashes PASS; 480 parents/457 DAS/23856 grid values revalidated, continuous checkpoint next e4d6a675661c3fed62c6cb5d; CDS 3 payloads/4 identities/one pending valid. Unregistered sessions UNKNOWN; locks and output existence rechecked at execution.

C1/C3 POTENTIAL, MEDIUM: new isolated files only, fixed HANDOFF/task updates with verified complete before/after copies. C2 NONE within inspected targets: no overwrite of data/results; only absent payloads/append-only receipts. C4 POTENTIAL, MEDIUM: independent fragment provenance/equality gate. C5 NONE: latest direct human permission for transport repair; original science preserved. C6 POTENTIAL, MEDIUM: one sequential worker, existing OS execution/continuation locks, no GPU, anonymous bounded requests and original CDS policy. C7 CONFIRMED: full input QA prerequisite for inference; incomplete inputs keep downstream waiting. C8 CONFIRMED: ADAPT/COMBINE frozen parsers, geometry, CDS implementation and delivery stages; no reimplementation of models or new library. Existing code + live single-hour evidence sufficient for this local adaptation; no new external tool research/install needed. Decision: PARALLEL_ALLOWED only for isolated code/test writing; live acquisition strictly single-worker after sealed QA/gates.

## Execution Lock

Holder: T015, same thread. From 2026-10-21 09:42 +08, logical WRITE on this new Revision_038 code and tests, new v003 receipts/logs/QA, task HANDOFF and T015 fixed state record (complete before/after snapshots). Existing models/raw files/old code/plans READ ONLY. OS execution lock for network/testing acquisition; continuation lock for new passive relay. Release code-edit occupation when manifest sealed; running worker retains its own process/OS locks until completion/failure. Do not edit executed v003 sources; a later fix must be separately versioned.

Status at registration: GENERATED / tests and live repair not yet run. No completion or speed guarantee.
