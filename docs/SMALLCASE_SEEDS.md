# Additional Small-System Correction-Policy Seeds

| System | Original run | Seed 1 | Seed 2 | Mean pass rate | Sample SD (percentage points) |
| --- | --- | --- | --- | --- | --- |
| IEEE 39-bus | 30/30 | 30/30 | 24/30 | 93.3% | 11.5 |
| IEEE 118-bus | 20/20 | 20/20 | 20/20 | 100.0% | 0.0 |

The added correction-policy runs retain the original fixed data partitions, frozen predictors, rewards, controls, 100 outer iterations, ten episodes per iteration and 30 gradient steps per outer update after replay starts. Validation selects the checkpoints. The 39-bus validator uses the first 15 fixed validation IDs; the 118-bus validator uses all 16 validation cases.

The statistics use three seed-level rates, with the same 30 or 20 test cases reused across seeds. All unsuccessful cases enter the stored step and timing summaries. The existing baseline and action-masking comparisons retain the original policies. Newly collected times are descriptive and do not replace the 300-bus paired timing or cost evidence.

Physical evaluation: `python code/analysis/physical_reproduce.py small-seed --case 39 --seed 1 --name seed39_1` (also use case 118 and seed 2).

Correction-policy training: `python code/analysis/physical_reproduce.py train-small --case 39 --seed 1 --name train39_1`. The `--smoke-training` option performs two updates and evaluates one case for an execution check; it is excluded from these results.

Selected actors are listed in `data/protocols/smallcase_seed_configurations.json`. Per-case outcomes, complete training logs, original-run provenance hashes, validation trajectories and actual optimizer/solver/model-call receipts are in `results/smallcase_multiseed_v1/`. Their distribution bytes are separately bound to `MANIFEST.json`.
