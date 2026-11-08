# Revision042 — T017 execution-only Windows sharing repair

2026-10-26. My decision: “快点修复”. This addendum supersedes the direct Revision041 launch entry only; the T017/Revision041 scientific protocol and manifest remain unchanged.

## Diagnosis and scope

At 15:54:55 Beijing time PID41856/create_time1792987763.7064028 failed in Revision031/core.update → os.replace while replacing the mutable progress.json. Forty states k50 through k11 passed QA. The last visible epoch is51, not necessarily the saved checkpoint epoch. A monitoring read overlapped this error; a reader lacking FILE_SHARE_DELETE is a plausible cause, not an identified historical handle owner. Preserve this uncertainty and the original failure.

Reuse Decision=ADAPT: use the bounded Windows-sharing retry approach already tested in Revision033/execution.py, with unique temporary names and explicit retry evidence. No import or launch of the old T014 CLI. Revision042 changes only four mutable execution JSON files (progress, state_count, status, active_worker) via an in-memory adapter. Fourteen attempts with capped exponential delay total at most5.25seconds of sleep. Only PermissionError with Windows codes5/32/33 is retried. Persistent denial and unrelated errors propagate; failed temporaries and retry events remain visible. Committed results/models/manifests/QA and scientific functions are not patched.

Monitoring MUST open live state files with FileShare.ReadWrite | FileShare.Delete and dispose handles promptly. Do not use ordinary Get-Content/Get-FileHash on mutable files while a writer is active. Stable committed QA/results may be read normally. Do not inspect live recovery.pt while it is being updated.

## Recovery gates and execution lock

Parent: Tasks/T017_TabM_Surface_Only_RFE.md; session01a0bda1-92d1-7832-9508-60f4e0ec1634. This is the same scientific run, not a new experiment or new RFE path.

WRITE: new Code/Hourly_Attenuation_Experiments/Revision_042 and Outputs/Experiments/Hourly_Attenuation_Revision_042_Surface_Execution_Repair; on inspected recovery, only the original Revision041 mutable state/checkpoint slots and unfinished scientific outputs. READ: all completed states, old failures, frozen code/input/model assets and manifest. No old STOP_REQUESTED/UK/weather/fast-variation/other-model task is resumed. No environment, service, automation, chat or subagent is created.

Conflict Check: C1 confirmed original mutable-state write failure, addressed by shared-delete readers and bounded retries; C2/C3 existing scientific artifacts and executed sources stay read-only; C4/C5 same method/config/splits/seed/epoch contract (seed17, epochs54); C6 single existing CPU2thread worker, real PID+creation+command and OS writer.lock verified; C7 checkpoint and source integrity required; C8 reuse40 accepted states plus completed k10 fold roles. Risk=MEDIUM after gates; Decision=sequential inspected recovery. Lock=RESERVED for new repair files; existing writer.lock remains the exclusive scientific writer lock, not deleted or bypassed.

Before continuation: confirm exact prior worker gone, lock free, no other project Python worker, no effective new T017 STOP; hash all original pinned sources; verify all40 QA/state/evaluation digests and existing fit/model/permutation files. Reconstruct k10/full-training identity from the frozen path, exact training hours and config. Verify checkpoint identity, contiguous history, optimizer/generator/RNG and weights; replay saved-epoch training metrics without any new fit. Preserve checkpoint, failed temporary, attempts, failure/log/status evidence as SHA-verified copies before mutation.

Preparation runs synthetic file-I/O tests only: recreate the old Windows denial, release a real conflicting handle and check retry success; confirm a delete-sharing reader permits replacement while still open; verify persistent denial and unrelated errors propagate without changing the previous target. Record zero scientific fits during preparation. Freeze repair source hashes and QA, then launch repair_surface.py run once. Its guarded call to unchanged Revision041/run_surface.run retains original fitting, selection, independent QA, reporting and stop behavior. Unsaved epochs are deterministically replayed from the saved optimizer/RNG; checkpoint choice is not selected by performance.

## Acceptance and handoff

Verify the resumed identity and held OS lock, a fresh progress timestamp, successful completion of the interrupted k10 full fit/state, no new failure, and unchanged pre-repair committed files. Full experiment completion still requires all50 states, export/numerical QA and actual visual inspection. Restart is not experiment completion. A failed repair requires inspection/new repair revision, never an automatic whole-run retry. Historical source files and failures remain unchanged.
