# Physical Reproduction

## Scope And Current Status

This guide distinguishes new physical executions from inspection of saved records. Commands are run from the repository root unless a different working directory is explicitly stated. All paths are repository-relative.

The current completed checks used an existing environment, not a newly installed environment. **A clean dependency installation has not been tested. Full retraining from raw inputs has not been completed or validated.** The commands below are not a claim that all manuscript simulations have been rerun.

| Physical experiment | Current verified outcome | Validation limit |
| --- | --- | --- |
| Original v31 SAC policy, all 75 upstream test cases | 44/75 accepted | One complete evaluation in the existing environment |
| Original DDPG policy, all 75 upstream test cases | 43/75 accepted | Per-case acceptance, failure reasons, and steps match saved evidence |
| Original TD3 policy, all 75 upstream test cases | 45/75 accepted | Per-case acceptance, failure reasons, and steps match saved evidence |
| Original SAC-68D policy, all 75 upstream test cases | 30/75 accepted | Per-case acceptance, failure reasons, and steps match saved evidence |
| Historical final workflow, all 75 cases | Direct and selected arms both accept 60/75 | One repeat per case and arm, not the three-repeat timing experiment |
| Spatial final workflow, all 100 profiles | Direct and selected arms both accept 93/100 | One repeat per case and arm, not the three-repeat timing experiment |
| Complete 39-bus common evaluation | Prediction 12/30; search 30/30; policy 30/30 | New complete 90-row method/case evaluation; not every separate action-mask experiment |
| Complete 118-bus common evaluation | Prediction 0/20; search 20/20; policy 20/20 | New complete 60-row method/case evaluation; not every separate action-mask experiment |
| Explicit 882-pair AC-outage audit | 536 passes, 46 nonconvergent pairs, 300 converged voltage-violation pairs | Pair-level classifications match saved evidence exactly; not full-population AC security |
| Screened 27-case optimization baselines | Repair 27/27; base-case AC-OPF 0/27; preventive LODF-SCOPF 27/27 | Case-level acceptance matches saved evidence; timings are new measurements |
| Reduced SAC training smoke run | Two Adam optimizer steps; Q loss 342.878 and policy loss 1.890 | One outer iteration with one episode, not full training |
| All 54 frozen configuration keys | Not all physically rerun | A complete key mapping and valid command syntax do not establish evaluation coverage |
| Full SAC, deterministic-policy, or GAT retraining | Not completed/validated | Parser-correct commands are provided where supported; this is not training validation |

The completed one-repeat historical/spatial checks were instrumented correctness checks and must not be used for runtime comparison. Audit hooks and counters add overhead. The separate `--timing` mode below removes those per-call hooks, but its results are still **new, hardware-dependent timings, not paper constants**. Environment, caches, warm-up, and hardware can change measured times. One-repeat acceptance agreement does not validate three-repeat paired timing means, confidence intervals, or the paper's timing advantage.

## Saved-Record Checks Are Not Physical Runs

```sh
python code/analysis/public_reproduce.py verify
python code/analysis/public_reproduce.py tables --output generated_tables
```

`verify` checks distributed bytes and recomputes selected numerical summaries from saved records. `tables` exports saved table tabulations. Neither command trains a policy, runs a new AC power flow, or solves a new optimization problem. See `docs/REPRODUCE.md` for the saved-record/table mapping and `docs/ASSET_COVERAGE.md` for checkpoint hashes.

## Physical CLI And Isolation

```sh
python code/analysis/physical_reproduce.py --help
python code/analysis/physical_reproduce.py prepare
```

The positional stages are `prepare`, `original`, `small`, `train`, `historical`, `spatial`, and `ac-audit`. `prepare` verifies the distribution manifest and materializes inputs without executing an experiment. Every invocation creates a fresh `_runtime/<uuid>/ED_RL/` directory; the UUID is a new hexadecimal identifier. Reusing `--name` on a later invocation does not resume the previous UUID's run.

The wrapper reconstructs the historical relative directory layout within that snapshot. Distributed datasets, partitions, policies, protocol inputs, and caches are copied there. New output does not overwrite the distributed `models/` or saved `results/`. The materialized input inventory is `_runtime/<uuid>/ED_RL/execution_inputs.json`.

The experiment child is launched through `_physical_bootstrap.py` with Python isolated mode (`-I`) and a cleared `PYTHONPATH`. In the default instrumented mode, file-read auditing rejects undeclared external experiment inputs, with exceptions for interpreter/dependency files, operating-system files, and the declared Numba cache. This is an input-audit mechanism, not an operating-system sandbox. In `--timing` mode, this per-call read audit is disabled; distribution-wide verification and per-file checks during materialization remain enabled, as do runtime-input hash checks before and after execution. The default shared Numba cache is in the system temporary directory; `SCED_NUMBA_CACHE` can select a cache location. Other model/plot caches are directed into the runtime snapshot.

| Option | Actual scope |
| --- | --- |
| `--max-samples 0` | All selected cases for evaluation stages; a positive value selects a prefix, not a random sample |
| `--name NAME` | One output-directory name; not a filesystem path |
| `--method sac/full68/ddpg/td3` | Base-policy choice for `original`; `train` accepts only `sac` |
| `--configuration KEY` | Frozen checkpoint and argument mapping for `original` only |
| `--scope test/val` | `original` uses the upstream test/validation partition, each with 75 IDs; this is not the screened 27/25 partition |
| `--case 39/118` | Small-system common evaluator |
| `--repeats N` | Historical/spatial workflows; default 3 |
| `--timing` | Historical/spatial only; disables per-call auditing and solver/neural/optimizer counters while retaining input-integrity checks |
| `--seed`, `--family`, `--dimension`, `--shifts` | SAC training options; they do not select a different frozen actor during `original` |
| `--updates`, `--episodes-per-update`, `--smoke-training` | SAC training budget and reduced smoke settings |

Do not silently bypass manifest, checkpoint, or provenance assertions. A failed integrity check requires investigation; it is not evidence of a completed physical run.

## Original 75-Case Policy Evaluations

```sh
python code/analysis/physical_reproduce.py original --method sac --scope test --max-samples 0 --name original75
python code/analysis/physical_reproduce.py original --method full68 --scope test --max-samples 0 --name full68_original75
python code/analysis/physical_reproduce.py original --method ddpg --scope test --max-samples 0 --name ddpg_original75
python code/analysis/physical_reproduce.py original --method td3 --scope test --max-samples 0 --name td3_original75
```

These invoke `evaluate_revision.py` with the selected distributed base checkpoint. `sac` means the revised v31 original policy, not the earlier v26 actor. Complete existing-environment runs accept 44/75 for SAC, 30/75 for SAC-68D, 43/75 for DDPG, and 45/75 for TD3. The comparator runs also match the saved per-case acceptance, constraint flags, and step counts. This does not establish that every frozen configuration or training seed has been physically rerun.

For a limited physical check, explicitly cap the number of cases:

```sh
python code/analysis/physical_reproduce.py original --method sac --max-samples 1 --name original_smoke
```

Results are written under the printed runtime root at `GAT-SAC-SCED-Framework/first revise/experiment_results/<name>/<method>_<scope>.json`. A configuration-key run uses `<configuration>_<scope>.json` instead.

## Frozen Configuration Keys

`data/protocols/policy_evaluation_specs.json` is authoritative. `--configuration` selects both a checkpoint and its frozen evaluation arguments, including method, basis family, dimensions/split, initialization, stress, voltage limits, radius, and AC-check cadence. The wrapper checks the checkpoint SHA-256 before launch. Configuration arguments replace the base `--method` selection; do not use unrelated top-level training flags to customize a frozen evaluation.

```sh
python code/analysis/physical_reproduce.py original --configuration full_sac --scope test --max-samples 0 --name frozen_original75
python code/analysis/physical_reproduce.py original --configuration sac_seed1_full --scope test --max-samples 0 --name sac_seed1_test75
python code/analysis/physical_reproduce.py original --configuration modes32_seed2_v1_full --scope test --max-samples 0 --name modes_seed2_test75
python code/analysis/physical_reproduce.py original --configuration random32_full --scope test --max-samples 0 --name random_seed42_test75
python code/analysis/physical_reproduce.py original --configuration dim16_full --scope test --max-samples 0 --name dim16_test75
```

The keys currently supplied are:

| Group | Exact keys |
| --- | --- |
| Base policies | `full_sac`, `full_full68`, `full_ddpg`, `full_td3` |
| Additional policy seeds | `sac_seed1_full`, `sac_seed2_full`, `full68_seed1_full`, `full68_seed2_full`, `ddpg_seed1_full`, `ddpg_seed2_full`, `td3_seed1_full`, `td3_seed2_full` |
| Matched 32D bases | `modes32_full`, `modes32_seed1_v1_full`, `modes32_seed2_v1_full`, `random32_full`, `random32_seed1_v1_full`, `random32_seed2_v1_full` |
| Other action controls | `keys32_full`, `dim16_full`, `dim24_full`, `dim40_full`, `split8_24_full`, `split20_12_full`, `radius20_full`, `radius60_full` |
| Reward controls | `n1_half_full`, `n1_double_full`, `voltage_half_full`, `voltage_double_full`, `bonus_half_full`, `bonus_double_full` |
| AC cadence | `ac_every1_full`, `ac_every10_full` |
| Tight-voltage evaluations | `voltage_0.94_1.08`, `voltage_0.95_1.05`, `tight_sac_094_108_seed42_v1_full` |
| Initializers | `start_economic`, `start_economic_v2`, `start_label`, `start_label_v2`, `start_previous`, `start_previous_v2` |
| Stresses | `stress_demand-high`, `stress_demand-low`, `stress_reactive`, `stress_regional`, `stress_renewable`, `stress_renewable_v2` |
| Historically named smoke records | `smoke_sac`, `smoke_full68`, `smoke_ddpg`, `smoke_td3`, `smoke_regional` |

A `smoke_*` key does not automatically limit the number of cases: add `--max-samples 1` when a one-case check is intended. Frozen evaluation keys do not retrain a policy. For the random basis, the construction seed remains 20260721; policy-training seeds 42/1/2 are distinct from random-basis construction variability.

Not all 54 configuration keys have been physically rerun. Their presence and parser-checked commands document available selections, not completed simulations.

## Small-System Evaluations

```sh
python code/analysis/physical_reproduce.py small --case 39 --max-samples 0 --name small39_full
python code/analysis/physical_reproduce.py small --case 118 --max-samples 0 --name small118_full
```

The common evaluator runs prediction-only, one-step myopic search, and the selected policy on the same system-specific IDs. It measures reset/prediction, initial checks, correction, and independent final verification with resident-model setup excluded. The complete commands select 30 cases for 39 buses and 20 for 118 buses, producing 90 and 60 method/case rows respectively. A positive `--max-samples` limits cases, not total method rows.

Both complete common evaluations have now run. For 39 buses, prediction-only accepts 12/30, search 30/30, and policy 30/30. For 118 buses, prediction-only accepts 0/20, search 20/20, and policy 20/20. These commands do not train small-system policies or reproduce every separate action-masking experiment. Outputs are `<name>/case39.json` or `<name>/case118.json` within runtime `experiment_results/`.

## Historical And Spatial Workflows

Use the separate timing mode for new complete three-repeat measurements:

```sh
python code/analysis/physical_reproduce.py historical --max-samples 0 --repeats 3 --timing --name historical_three_repeats_timing
python code/analysis/physical_reproduce.py spatial --max-samples 0 --repeats 3 --timing --name spatial_three_repeats_timing
```

`historical` selects the 75 frozen upstream test IDs. `spatial` selects the 100 stored spatial profiles. Each case/repeat evaluates both `direct` and `selected` arms, with rotating arm order. The frozen workflow uses the eight-step gate and dual `pf`/`flat` optimizer initialization. Complete three-repeat outputs contain 450 historical rows and 600 spatial rows, keyed by `(repeat, sample_id, arm)`. Do not pool these populations or discard failed cases.

`--timing` disables the per-call file-read audit, `runpp`/`runopp` wrappers, actor/predictor/neural-layer hooks, and optimizer-step hook. It retains manifest verification, hash-checked materialization, runtime-input hash checks before and after execution, and the output receipt. Timing receipts have `instrumented: false` and null call counters; they do not provide a per-file read trace. This trade-off removes instrumentation overhead but does not prove that no undeclared external read occurred. New timings remain hardware- and environment-dependent measurements, not constants that must equal the manuscript's values. The three-repeat timing commands are instructions, not a claim that those new measurements have completed.

For the one-repeat configurations corresponding to the currently verified acceptance outcomes:

```sh
python code/analysis/physical_reproduce.py historical --max-samples 0 --repeats 1 --name historical_one_repeat
python code/analysis/physical_reproduce.py spatial --max-samples 0 --repeats 1 --name spatial_one_repeat
```

One-repeat outputs contain 150 and 200 rows. Current verified acceptance is 60/75 in both historical arms and 93/100 in both spatial arms. These completed instrumented checks are for correctness, not runtime comparison, and are not completion of the three-repeat timing commands above. Output filenames are `<name>/historical.json` and `<name>/spatial.json` within runtime `experiment_results/`.

## Explicit 882-Pair AC Audit

```sh
python code/analysis/physical_reproduce.py ac-audit --max-samples 0 --name ac_audit882
```

This command uses the 44 frozen original accepted operating points in `data/operating_points/ac_audit_original.json`; it does not regenerate them by rerunning SAC. The underlying audit defaults are `--top-k 3` and `--full-per-demand 1`; the wrapper does not expose overrides for these two arguments.

One lexicographically selected point in each of three demand groups receives all 253 valid outages. Each of the other 41 points receives its three highest-scoring outages, giving `3 * 253 + 41 * 3 = 882` operating-point/outage pairs. The AC solve attempts use `results` initialization with `flat` fallback. A pair can therefore generate multiple `runpp` calls, and base-point checks generate additional calls. The receipt counter is not expected to equal 882.

`--max-samples` limits operating points before selection, not outage pairs; a capped run is not the full 882-pair audit. Output is `<name>/ac_outages.json` within runtime `experiment_results/`; the audit also writes `excluded_contingencies.json` at the runtime `experiment_results/` level. The complete new audit has now produced exactly 536 passes, 46 nonconvergent pairs, and 300 converged voltage-violation pairs, with pair-level classifications matching saved evidence. These selected-outage diagnostics do not establish full AC contingency security or population-wide reliability.

## SAC Training: Smoke Versus Full Budget

The wrapper's `train` stage supports **SAC only**. It rejects `--method ddpg`, `--method td3`, and `--method full68`; a generator-wise SAC training run uses `--method sac --family full68` instead.

```sh
python code/analysis/physical_reproduce.py train --method sac --smoke-training --seed 42 --name sac_training_smoke
python code/analysis/physical_reproduce.py train --method sac --family physics --dimension 32 --shifts 12 --updates 100 --episodes-per-update 10 --seed 42 --name sac_training_full
```

Both commands use the distributed screened 167-sample dataset, the 115/25/27 partition, and the frozen 300-bus graph predictor. The full command requests 100 outer iterations with 10 episodes each, with the underlying 50-step episode cap and validation-based checkpoint selection. Successful episodes may terminate early; equal episode budgets do not imply equal transition counts.

`--smoke-training` overrides the budget to one update and one episode, with learning-starts 4, batch size 4, one gradient step, and validation interval 1. It is not full training and does not establish convergence or seed robustness. Replay updates still depend on collecting enough transitions. It does not disable the underlying final selected-checkpoint test evaluation or reduce the default episode step cap, so it is not necessarily just one short forward pass.

Training outputs are under `<name>/sac_train/` within runtime `experiment_results/`, including configurations, trajectories, and checkpoint files produced by the underlying trainer. The actual SAC smoke run completed with nonzero optimizer updates and logged Q loss 342.878 and policy loss 1.890. This demonstrates a reduced learning execution, not full-budget training, convergence, or reproduction of a selected manuscript checkpoint. An additional test of the newly added `optimizer_step_calls` receipt counter is still running; no completed counter-test result or exact counter value is claimed here. The wrapper trains a new policy from the supplied labeled dataset with a frozen predictor; it does not perform end-to-end raw-label generation and predictor retraining.

## Screened Optimization Baselines

```sh
python code/analysis/physical_reproduce.py baseline --baseline repair --name screened_repair
python code/analysis/physical_reproduce.py baseline --baseline acopf --name screened_acopf
python code/analysis/physical_reproduce.py baseline --baseline scopf --name screened_scopf
```

These select the screened 27-case test benchmark, not all 75 upstream cases. `repair` uses the frozen screening IDs but recomputes the repair rather than consuming stored rescue controls. `acopf` and `scopf` use their original evaluation scripts and constraints. A positive `--max-samples` is a reduced smoke check. A solver failure is retained as a failed case, not silently replaced by a historical result. Gurobi-dependent paths require a separately obtained license. The wrapper sets the legacy OPF entry points' requested `NUMBA_DISABLE_JIT` default before the instrumentation imports pandapower, preserving their initialization order.

## Original Deterministic And GAT Training CLIs

The following commands were checked against the original parsers, not executed as full training validations. They are **not** additional `physical_reproduce.py train` modes. First create a materialized layout:

```sh
python code/analysis/physical_reproduce.py prepare
```

Change the working directory to the printed `_runtime/<uuid>/ED_RL/` root before running one of the commands below. Prepare a fresh runtime for each independent experiment and use a new output directory; do not overwrite copied selected-model directories. The paths in this subsection are relative to that materialized `ED_RL/` root, not to the distribution root.

```sh
python paper_results/code/train_deterministic_n1_aligned.py DDPG --dataset case300_active_required_fixed_slack_v1.json --split-file paper_results/case300_fixed_slack_v1_split.json --gat-checkpoint gat_case300_fixed_slack_v1/gatv10_predictor_best.pth --output-dir rl_ddpg_public_seed42 --seed 42 --updates 100 --episodes-per-update 10
python paper_results/code/train_deterministic_n1_aligned.py TD3 --dataset case300_active_required_fixed_slack_v1.json --split-file paper_results/case300_fixed_slack_v1_split.json --gat-checkpoint gat_case300_fixed_slack_v1/gatv10_predictor_best.pth --output-dir rl_td3_public_seed42 --seed 42 --updates 100 --episodes-per-update 10
python paper_results/code/train_gat_v10.py --dataset case300_sced_candidates_fixed_slack_v1.json --split-file paper_results/case300_fixed_slack_v1_candidate_split.json --output-dir gat_public_retrain
```

The deterministic algorithm positional argument is uppercase `DDPG` or `TD3`. Its parser supports `--seed`, `--updates`, `--episodes-per-update`, and the explicit data/checkpoint paths above. The GAT parser exposes only `--dataset`, `--split-file`, and `--output-dir`: do not append invented `--seed`, `--epochs`, or `--updates` flags. Its current training loop uses a maximum of 300 epochs and early-stopping patience 40. The candidate split is applied to complete samples, yielding 350/75/75, not the screened correction split.

These direct original-script commands bypass the physical wrapper/bootstrap and therefore do not automatically produce its execution receipt or enforce its input-read audit. They are supplied as explicit original CLI recipes, not as instrumented clean-machine reproduction claims. A newly trained predictor is written to a new directory; this does not automatically replace the frozen predictor used by later wrapper runs. No claim is made that retraining yields byte-identical weights or reproduces the original training trajectory.

## Receipts And Completion Checks

For wrapper-launched physical experiments, receipts are written under `_runtime/<uuid>/ED_RL/GAT-SAC-SCED-Framework/first revise/experiment_results/execution_receipts/<script>_<timestamp>.json`. `prepare` alone does not launch the bootstrap or produce a physical-execution receipt.

Inspect both the stage result and its receipt. A zero process exit code, a nonempty output, or a positive call counter alone does not establish the scientific result. Check the result's `complete` flag where supplied, expected case/row counts, both workflow arms, retained failures, selected checkpoint/configuration, and receipt integrity fields.

| Receipt field | Meaning and interpretation |
| --- | --- |
| `instrumented` | True for default correctness runs; false for `--timing` runs |
| `runpp_calls` | Instrumented AC power-flow call attempts, including failed attempts and retries; not the number of accepted cases; null in timing mode |
| `runopp_calls` | Instrumented AC optimization call attempts; positive values are expected for optimizer workflows, not every stage; null in timing mode |
| `policy_forward_calls` | Calls to instrumented policy actor forwards; includes setup/validation calls, not just accepted corrections; null in timing mode |
| `gat_forward_calls`, `neural_layer_forward_calls` | Class-name-based predictor/layer instrumentation; a predictor class not explicitly recognized can leave the GAT-specific counter at zero; null in timing mode |
| `optimizer_step_calls` | Newly added counter for instrumented `torch.optim.Adam.step` calls; its additional smoke test is still running; null in timing mode |
| `completed` | Target script returned normally and the final materialized-input check did not force failure; not an acceptance or convergence guarantee |
| `denied_inputs` | Undeclared external read attempts detected by the default audit hook; an empty list in timing mode does not establish absence of external reads |
| `changed_materialized_inputs` | Inputs modified or removed during execution; nonempty forces receipt completion false |
| `optimizer_step_calls` | Actual Adam step calls in instrumented training; a positive count does not demonstrate convergence |
| `input_files` | Non-code runtime input files observed by the audit hook; not collected in timing mode |
| `distribution_manifest_sha256`, `verified_distribution_files`, `materialized_input_files` | Distribution verification and snapshot inventory identifiers/counts |
| `output_sha256` | Hash of the requested `--output` file when it exists; supplied in both instrumented and timing modes |

A pure explicit AC audit need not call a policy or an optimizer. Initially accepted small-system cases can skip policy correction. Interpret counters against the selected stage rather than requiring every counter to be positive. Null timing counters mean unavailable instrumentation, not zero solver/model activity. The optimizer counter observes Adam step calls, not every possible optimizer implementation; complete operating-system isolation is not claimed.

Use the recorded environment in `requirements.txt` as the dependency reference, not as evidence of a tested fresh installation. Gurobi-dependent optimization and label-generation paths need an independently installed, licensed Gurobi environment. Keep new measured timings and receipts separate from saved paper evidence. Full retraining, clean installation, evaluation of all 54 configurations, the additional optimizer-counter test, and the requested three-repeat timing workflows remain unvalidated or incomplete and must not be reported as complete. The completed small-system, 882-pair AC, base-comparator, and reduced SAC smoke checks do not remove those limits.
