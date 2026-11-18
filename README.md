# Sensitivity-guided SCED restoration

Reproducibility materials for **Sensitivity-Guided Low-Dimensional Redispatch for Base-Case Alternating-Current Feasibility and Linear N-1 Contingency Screening**, manuscript IJEPES-D-26-03365.

Authors: Kaiting Lai, Yigong Xie, Wenchong Fang, Danli Xu, Yuxuan Liu, Yuanwei Yang, and Changtao Liao.

## Contents

- `code/pipeline/`: preserved prediction, training and evaluation scripts; root-level dependencies are in `code/pipeline/root/`.
- `code/analysis/`: revision experiment scripts, the real-simulation entry point `physical_reproduce.py`, and separate saved-record checks in `public_reproduce.py`.
- `data/cases/`, `data/partitions/`, `data/matrices/`: modified test cases, stored candidate datasets, partition identifiers and derived numerical matrices.
- `models/`: graph predictors and stored correction policies. Both the earlier v26 policy and the v31 policy used for the revised original-policy evaluation are identified separately.
- `results/`: stored numerical records, final paired measurements and table-tabulation intermediates. Different experiment protocols are not pooled.
- `docs/REPRODUCE.md`: table/figure-to-script mapping, commands, scope and expected numerical checks.
- `MANIFEST.json`: SHA256 checksums of this sanitized distribution. Original-run hashes in the provenance records refer to the original author-held files, not relocated/sanitized copies.

## Install And Run Physical Experiments

The measured environment used Python 3.10, pandapower 3.4, PyTorch 2.9, NumPy and SciPy. Gurobi-dependent label-generation scripts require a separately obtained Gurobi installation and license; the repository distributes neither. For saved-record checks, no optimizer or GPU is required.

```sh
python -m pip install -r requirements.txt
python code/analysis/physical_reproduce.py original --method sac --name sac75
python code/analysis/physical_reproduce.py historical --repeats 3 --timing --name historical_timing
python code/analysis/physical_reproduce.py spatial --repeats 3 --timing --name spatial_timing
```

These commands load model weights, compute policy actions, run AC power flows and line-outage checks, and invoke the shared optimizer where required. They do not read saved experimental outcomes. Each run verifies the distribution hashes and creates an independent runtime snapshot under `_runtime/<uuid>/ED_RL/`. Follow [the physical reproduction guide](docs/PHYSICAL_REPRODUCTION.md) for all four policies, small systems, explicit outages, training, and timing scope.

Fresh physical checks reproduced 44/75, 43/75, 45/75 and 30/75 for SAC, DDPG, TD3 and generator-wise SAC. One-repeat final workflow checks reproduced identical accepted IDs: 60/75 historical and 93/100 spatial in both arms. The small-system policy checks passed 30/30 and 20/20. The explicit outage rerun reproduced all 882 pair-level outcomes (536 pass, 46 nonconvergent, 300 voltage violations). See [the validation record](docs/PHYSICAL_VALIDATION.json). Instrumented checks are correctness checks, not reproductions of the paper's measured seconds. The `--timing` option omits per-call instrumentation and produces new, hardware-dependent measurements.

Saved-record verification and table export remain available separately:

```sh
python code/analysis/public_reproduce.py verify
python code/analysis/public_reproduce.py tables --output generated_tables
```

Those two commands inspect historical records; they are not physical experiment reproduction.

## Additional Small-System Training Seeds (v1.2.0)

The 39-bus and 118-bus correction policies include additional training seeds 1 and 2, alongside the original run. Each new run retains the fixed data partitions, frozen graph predictor, original controls/rewards, 100 outer iterations, ten episodes per iteration and validation-based checkpoint selection. Selected actor files are indexed in `data/protocols/smallcase_seed_configurations.json`; per-case physical results, validation histories and optimizer-step receipts are under `results/smallcase_multiseed_v1/`.

```sh
python code/analysis/physical_reproduce.py small-seed --case 39 --seed 1 --name seed39_1
python code/analysis/physical_reproduce.py small-seed --case 118 --seed 2 --name seed118_2
python code/analysis/physical_reproduce.py train-small --case 39 --seed 1 --name retrain39_1
python code/analysis/physical_reproduce.py train-small --case 118 --seed 1 --smoke-training --name smoke118_1
```

`small-seed` runs policy inference and fresh physical checks on every selected test case. `train-small` executes the original correction-policy trainer with the fixed predictor; the smoke option uses two training iterations and one evaluation case as an execution check, not a paper result. The training adapter uses an outcome-free test-ID record to check partition identity. New measured times are descriptive and do not replace the paper's paired 300-bus timing experiment. The original v1.1.1 tag is unchanged. See [the small-system seed record](docs/SMALLCASE_SEEDS.md) and [its physical validation](docs/SMALLCASE_SEED_VALIDATION.json).

Large result JSON files are losslessly compressed as `.json.gz`; the portable entry point reads either form. Standard `gzip` or Python's `gzip` module can expand them when inspecting the historical scripts. Checkpoint path metadata has been relocated without changing any tensor data block.

## Reproduction Scope

Full retraining from raw inputs and a fresh dependency installation have **not been validated**. A one-iteration 300-bus SAC training check performed real critic and actor optimizer steps. The additional small-system correction-policy runs retain their frozen predictors. The original 44 checkpoints and 54 frozen evaluation configurations are indexed in [the asset coverage report](docs/ASSET_COVERAGE.md); the four added small-system actors have a separate configuration index. Not all 54 original configurations have been rerun. The physical wrapper restores the historical relative layout inside its isolated snapshot. Historical provenance hashes remain unchanged; the relocated execution is bound to the current distribution manifest. Original archival scripts with author-layout or historical-hash assumptions should not be run by disabling their assertions.

The original policy accepts 44/75 unfiltered cases. Its 27/27 result and 0.788 s timing concern the screened, acceptance-matched subset. The final eight-step gated workflow is a different configuration: matched workflows accept 60/75 historical and 93/100 spatial cases. Historical average timing is 1.912 versus 1.725 s; the spatial paired saving interval contains zero. Base-case AC checks plus linear outage screening do not establish full AC contingency security.

## License and Citation

Author-owned code: MIT. Author-owned derived data, weights and results: CC BY 4.0. Third-party test-case/software rights remain unchanged; see `LICENSE` and `docs/DATA_DICTIONARY.md`. Cite the manuscript and this version using `CITATION.cff`. Contact: 1181413980@qq.com.
