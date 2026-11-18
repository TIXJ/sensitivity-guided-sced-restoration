# v1.2.0: Additional Small-System Training Seeds

Adds the original-protocol 39-bus and 118-bus correction-policy runs for training seeds 1 and 2, with four validation-selected actors, frozen partition identifiers, per-case physical outcomes, validation histories and execution receipts. New `small-seed` and `train-small` stages provide fresh physical evaluation and correction-policy training, separate from saved-record checks. The graph predictors, data partitions, physical environments, controls and rewards are retained. Seed-level variability is computed over the original run and the two additions, with all unsuccessful cases retained.

The small-system tables distinguish original-run results from added-seed results. The 39-bus validation-subset description follows the original script's first 15 fixed validation IDs. Existing 300-bus results, workflows, paired timing and cost evidence are unchanged. The v1.1.1, v1.1.0 and v1.0.0 tags are retained unchanged.

The physical diagnostic counter now recognizes the small-system predictor class names when counting graph-network forwards. This changes instrumentation only, not predictions, policy actions or physical checks.

## v1.1.1: Windows Runtime Path Length

Fresh public-clone validation found that the 32-character runtime identifier could exceed Windows path limits when the repository itself has a long path. Runtime directories now use atomically allocated eight-character identifiers. This changes filesystem layout only, not model inputs, policy actions or numerical algorithms. The v1.1.0 and v1.0.0 tags remain unchanged.

## v1.1.0: Physical Reproduction Inputs And Entry Points

The first release supported inspection of saved records but omitted inputs needed by physical evaluation. This release adds 36 selected policy files, two runtime helper modules, the screening record, five fixed voltage-context caches, exact frozen operating points for the AC-outage audit, and the final workflow protocols and spatial profiles. The repository now contains 44 checkpoint files and 54 indexed evaluation configurations.

`physical_reproduce.py` materializes an independent runtime for each invocation, validates distributed input bytes, and launches policy inference and physical solvers. Its execution receipt distinguishes actual solver/model calls from saved-record inspection. The shared final optimizer, gate, cases, acceptance checks and trained tensor values are unchanged.

The manifest now describes the exact LF bytes stored by Git. The previous working-tree CRLF manifest could fail after a fresh clone. `public_reproduce.py verify` remains a saved-record check, separate from physical reruns.

Physical validation covers all four original policies on 75 cases, one-repeat final workflows on 75 historical and 100 spatial cases, the complete 39/118-bus common evaluations, 882 explicit AC-outage pairs, and a reduced SAC optimizer-update test. Details and limits are in `PHYSICAL_VALIDATION.json` and `PHYSICAL_REPRODUCTION.md`. Fresh dependency installation, full raw-input retraining, all 54 evaluation configurations, and exact timing replication are not claimed.
