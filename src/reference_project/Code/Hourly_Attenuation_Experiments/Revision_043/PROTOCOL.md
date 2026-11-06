# Revision043 — corrected Windows-reader regression test, T017 continuation

2026-10-26, same user authorization “快点修复”. Read Revision041/PROTOCOL.md and Revision042/PROTOCOL.md. Those scientific and operational boundaries remain in force except for the test assumption corrected below. Current legal entry is Revision043/repair_surface.py; do not execute Revision042 again.

## Evidence-driven correction

Revision042 prepare successfully passed prior-process/resource checks, original source hashing, forty-state/path/model checks, and epoch50 checkpoint identity/optimizer/RNG/metric replay before its synthetic test failed. No new scientific training started. Its test expected os.replace to succeed immediately with an existing target reader using FILE_SHARE_DELETE; this actual Windows system returned WinError5. Therefore share-delete access alone is not claimed to guarantee replacement while the reader remains open. Keep that failed test and all temporaries under the Revision042 repair output.

Revision043 reuses the already-tested immutable Revision042 atomic_state.py unchanged: unique temp file, fourteen bounded Windows5/32/33 retries, every retry failure/recovery recorded, permanent/non-sharing failures propagated. Its corrected regression closes the shared-delete reader promptly and requires bounded retry success. This matches production monitoring: explicit ReadWrite|Delete sharing AND prompt disposal, with bounded writer retry. It does not weaken persistence/error/previous-target tests or any scientific QA. The repair entry is a versioned copy of042 with only output/provenance/test changes.

## Execution lock / scope / reuse

Parent task T017; session01a0bda1-92d1-7832-9508-60f4e0ec1634. Decision=ADAPT, sequential inspected recovery. WRITE=new Revision043 code/audit, plus the same authorized incomplete Revision041 results and mutable state/checkpoint after all gates. READ=executed041/042 sources, original inputs/manifests, completed40 states and all historical failures. No old scientific files/results/model/manifest edits, no task/worker duplication, no changed method, no environment/service/chat/automation/subagent. Exact process+OS-lock/resource checks are performed again during prepare and run.

Prepare again verifies all frozen source SHA and committed artifacts, reconstructs the exact k10/full-fit identity, checks epoch50 saved optimizer/RNG and training metric replay, preserves failed evidence and checkpoint, and emits immutable repair manifest/QA/resume authorization. Runtime rechecks all those hashes before calling unchanged041 run with its original exclusive writer lock. Existing forty states and k10 folds are reused; unsaved epochs resume from51, not from scratch. Code041 and the reused042 atomic function are not modified.

Acceptance: real shared-handle tests, checkpoint/source/committed-result QA; then actual resumed worker identity and held lock, progress advancement beyond the failure and k10 state QA. Stop/QA/report/visual-review requirements remain unchanged. A restart is not full experiment completion. No blind retry of another failure.

Correction record: Revision042's failed one-attempt test is evidence, not a scientific data failure. The original15:54 reader owner remains unproven; overlap with the monitoring read is acknowledged, not presented as causal certainty.
