# v1.1.0: Physical Reproduction Inputs And Entry Points

The first release supported inspection of saved records but omitted inputs needed by physical evaluation. This release adds 36 selected policy files, two runtime helper modules, the screening record, five fixed voltage-context caches, exact frozen operating points for the AC-outage audit, and the final workflow protocols and spatial profiles. The repository now contains 44 checkpoint files and 54 indexed evaluation configurations.

`physical_reproduce.py` materializes an independent runtime for each invocation, validates distributed input bytes, and launches policy inference and physical solvers. Its execution receipt distinguishes actual solver/model calls from saved-record inspection. The shared final optimizer, gate, cases, acceptance checks and trained tensor values are unchanged.

The manifest now describes the exact LF bytes stored by Git. The previous working-tree CRLF manifest could fail after a fresh clone. `public_reproduce.py verify` remains a saved-record check, separate from physical reruns.

Physical validation covers all four original policies on 75 cases, one-repeat final workflows on 75 historical and 100 spatial cases, the complete 39/118-bus common evaluations, 882 explicit AC-outage pairs, and a reduced SAC optimizer-update test. Details and limits are in `PHYSICAL_VALIDATION.json` and `PHYSICAL_REPRODUCTION.md`. Fresh dependency installation, full raw-input retraining, all 54 evaluation configurations, and exact timing replication are not claimed.
