# Revision 027 progress

2026-10-11: user authorized continuing experiments toward simultaneous overall/rain/dry-overcast improvement for the full-input overall test winner. First establish a correctly paired three-seed comparison; no test-driven new tuning and no preset positive result.

- Reuse: Revision 026 full models and Revision 014 training functions; 50/156 input lists remain fixed.
- Process check: no matching hourly/fast Python workers before setup.
- Six pure synthetic checks PASS with research sample reads=0. Frozen fingerprint: 334aa66de5e15c240f5280ae423317474521d85834eb4c4f434ea1c251c18788.
- fit-base worker PID33648 started 2026-10-11 22:43:27 BJT and exited normally at 22:44:27. All 30 base weights and records complete. Only LightGBM deprecated-interface warnings, no numerical failure.
- verify-fit PASS: no test target read during fit, saved/reloaded delta <=1.34e-15 dB. Paired evaluate complete, zero fits during evaluation.
- Independent QA PASS: 161 unchanged source digests, 60 replayed weights, 30 paired evaluations, 900 metric recomputations, 41,220 hourly predictions and 300 seed/input/weather rows.
- Complete diagnostic acceptance NOT_MET. Full-input overall winner remains BiLSTM: base/full mean RMSE overall 0.321304/0.285673, rain 0.364908/0.416094, dry-overcast 0.489845/0.437779 dB. All three rain seed differences are negative. Do not claim user objective achieved.
- Independent report/interval audit is run separately with audit_report.py, preserving frozen training and scoring code. The full result is retained as a failed hypothesis, not deleted.
- All old experiments/raw data are READ ONLY. User subsequently authorized restoring 15-minute checks for the separately planned Revision_028, not repeating Revision_027.

Execution and result status will be updated only from actual logs and saved artifacts.
