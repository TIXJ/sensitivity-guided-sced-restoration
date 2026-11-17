# Scientific Asset Coverage

## Scope And Validation Boundary

This audit describes the repository tree containing the completed asset additions. It does not assert that an earlier public release or downloaded clone contains those additions. All paths below are relative to the repository root.

The checks were independent file inventories, SHA-256 comparisons, JSON-content comparisons, and deterministic split reconstruction. No training or physical experiment was executed for this audit. **Full retraining from raw inputs has not been validated.** Asset availability and agreement with recorded provenance do not establish a clean-machine physical rerun, reproduce measured runtimes, or show that all simulations were rerun. Physical runner validation is separate from this document.

## Checkpoint Inventory

| Category | Files | Verification |
| --- | ---: | --- |
| Revision-run selected actors, `models/revision_runs/*/best_actor.pth` | 31 | Exact source-byte and recorded evaluation SHA-256 matches for all 31 |
| Seed-42 SAC-68D, DDPG, and TD3 comparator selected actors | 3 | Exact source-byte and seed-audit SHA-256 matches |
| Selected 39-bus and 118-bus policies | 2 | Exact source-byte and common-evaluation input SHA-256 matches |
| Revised original-policy v31 selected actor | 1 | Exact source-byte and seed-audit SHA-256 match |
| Earlier v26 SAC policy files | 4 | Exact source-byte matches; distinct from v31 |
| Graph predictors for 300, 39, and 118 buses | 3 | Both small-system files match source bytes; the sanitized 300-bus container preserves all 73 tensor-storage blocks exactly |
| **Total checkpoint files (`.pth`)** | **44** | **36 restored policies; 37 evaluation-linked selected policies including v31** |

The 44-file count includes the earlier v26 best, best-training, final, and latest actors; these are not four additional training seeds. The v26 files must not substitute for the v31 policy evaluated in the revised 44/75 result.

## Experiment Coverage Matrix

| Experiment or input | Repository assets | Coverage and manuscript support |
| --- | --- | --- |
| 300-bus upstream population | `data/cases/case300_sced_candidates_fixed_slack_v1.json`; `data/partitions/case300_fixed_slack_v1_candidate_split.json` | Present. All 600 stored samples match original sample contents; 500 complete samples use the 350/75/75 partition. Supports `tab:revision_full_methods` and unfiltered seed/basis comparisons. |
| 300-bus screened correction population | `data/cases/case300_active_required_fixed_slack_v1.json`; `data/partitions/case300_fixed_slack_v1_split.json` | Present. All 167 sample contents and the 115/25/27 partition IDs match originals. Supports policy training and screened action/algorithm comparisons. Do not interchange these IDs with upstream IDs. |
| Frozen screening membership | `data/partitions/case300_fixed_slack_v1_screen.json` | Present. Original screening records preserve upstream indices, partition membership, and the selected status used by deterministic repair. Its evaluator recomputes control actions rather than consuming saved rescue controls. |
| 300-bus network and numerical security model | `data/matrices/case300_network.json`; `data/matrices/security_matrices.npz`; `data/matrices/settings.json` | Present. Network JSON is semantically identical to its original snapshot; the numerical archive is byte-identical. Includes the stored sensitivity/security arrays and reference basis. |
| 300-bus graph predictor | `models/gat_case300_fixed_slack_v1/gatv10_predictor_best.pth`; `code/pipeline/train_gat_v10.py` | Present. The serialized metadata was sanitized; original whole-file hashes are not interchangeable with the distribution hash. All 73 tensor-storage blocks match the original. |
| SAC-32D, seeds 42/1/2 | `models/rl_sac_fixed_slack_v1_v31_p40_voltage_sensitivity/best_actor.pth`; `models/revision_runs/sac_seed1/best_actor.pth`; `models/revision_runs/sac_seed2/best_actor.pth` | All selected actors present and exact. Supports `tab:revision_seeds` and the mixed-basis arm of `tab:final_basis_seeds`: 44/45/41 accepted upstream cases. |
| SAC-68D, seeds 42/1/2 | `models/rl_full68_fixed_slack_v1_p40_raw_action_v4/best_actor.pth`; `models/revision_runs/full68_seed1/best_actor.pth`; `models/revision_runs/full68_seed2/best_actor.pth` | All present and exact, including the zero-acceptance run. Supports `tab:action_compression`, `tab:revision_full_methods`, and seed counts 30/43/0. |
| DDPG, seeds 42/1/2 | `models/rl_ddpg_fixed_slack_v1_p40_raw_action_v9/best_actor.pth`; `models/revision_runs/ddpg_seed1/best_actor.pth`; `models/revision_runs/ddpg_seed2/best_actor.pth` | All present and exact. Supports `tab:rl_compare`, unfiltered comparison, and seed counts 43/45/0. |
| TD3, seeds 42/1/2 | `models/rl_td3_fixed_slack_v1_p40_unified_noise_v5/best_actor.pth`; `models/revision_runs/td3_seed1/best_actor.pth`; `models/revision_runs/td3_seed2/best_actor.pth` | All present and exact. Supports `tab:rl_compare`, unfiltered comparison, and seed counts 45/45/0. |
| System-wide 32D basis, seeds 42/1/2 | `models/revision_runs/modes32/best_actor.pth`; `models/revision_runs/modes32_seed1_v1/best_actor.pth`; `models/revision_runs/modes32_seed2_v1/best_actor.pth` | All present and exact. Supports `tab:final_basis_seeds` and `fig:final_basis_validation`: 45/43/45. |
| Random balanced 32D basis, seeds 42/1/2 | `models/revision_runs/random32/best_actor.pth`; `models/revision_runs/random32_seed1_v1/best_actor.pth`; `models/revision_runs/random32_seed2_v1/best_actor.pth` | All present and exact. Supports the matched-basis comparison: 34/38/33. Construction seed remains fixed at 20260721; policy seeds are not independent random-basis constructions. |
| Other action, reward, cadence, and tight-voltage controls | Remaining `models/revision_runs/` actors in the hash matrix below | Present and exact for every top-level revision run with a selected actor. Supports `tab:revision_ablations`, `tab:revision_cadence`, and the recorded tight-voltage policy diagnostic. |
| 39-bus inference and training inputs | `data/cases/case39_routeb_rescuable_dataset.json`; `models/gat39_results/gat39_best.pth`; `models/rl39_routeb_main/best_model.pth`; `code/pipeline/train_rl39_v1.py` | Present. All 200 sample contents match originals. Public split logic reconstructs 140/30/30 and exactly the recorded test IDs. Supports `tab:case39`, `tab:case39_action`, and common small-system acceptance/timing. |
| 118-bus inference and training inputs | `data/cases/case118_routeb_rescuable_dataset.json`; `models/gatv118_results/gatv118_predictor_best.pth`; `models/rl118_routeb_main/best_model.pth`; `code/pipeline/train_rl118_v1.py` | Present. All 120 sample contents match originals. Public split logic reconstructs 84/16/20 and exactly the recorded test IDs. Supports `tab:case118`, action masking, and common small-system acceptance/timing. |
| Small-system network and predictor helpers | `code/analysis/legacy_dependencies/generate_case39_gt_v1.py`; `code/analysis/legacy_dependencies/generate_case118_gt_v1.py`; `code/analysis/legacy_dependencies/train_gat39_v1.py`; `code/analysis/legacy_dependencies/train_gat118_v1.py`; `code/pipeline/case39_route_b_utils.py` | Present. Networks are constructed from pandapower benchmark cases and included modification code; no separate external network file was identified as a required missing asset. |
| Historical voltage context | `data/voltage_context/none/case300_reset_voltage_jacobian_d16000_v1.npz` through `case300_reset_voltage_jacobian_d16800_v1.npz`, in 200-MW increments; `code/pipeline/reset_voltage_jacobian_fast.py` | All five numerical cache files match original bytes; the helper is present. This is not a cache bank for every newly generated or stressed condition. |
| Frozen original controls for explicit AC diagnostics | `data/operating_points/ac_audit_original.json` | Present. All 44 accepted operating points exactly match the selected original fields, including dispatch and voltage controls; the recorded source hash matches. Supports auditing the selected AC-outage diagnostic, not a new outage simulation. |
| Final workflow and spatial-condition records | `data/protocols/final_deployment.json`; `data/protocols/spatial_conditions.json`; corresponding records under `results/` | Present. Supplies stored protocol/condition inputs; this inventory does not validate runner-to-policy mapping or reproduce final historical/spatial timings. |

The core training implementations are present in `code/pipeline/train_sac_n1_active_aligned.py` and `code/pipeline/train_deterministic_n1_aligned.py`; revision basis construction is in `code/analysis/revision_common.py`. Per-run configuration and split companions are included where present in the source runs. Recorded trajectories remain under `results/`, with seed-42 comparator trajectories also under their model directories. Metadata presence is not evidence that fresh training reproduces a selected checkpoint.

## Exact Evaluation-Linked SHA-256 Matrix

Every row below passed comparison against both the original selected checkpoint bytes and the recorded evaluation checkpoint hash. For revision runs, the evidence is `checkpoint_sha256` in `results/<run>_full.json` or its lossless `.json.gz` representation. The seed-42 rows also match `results/seed_audit.json`. Small-system evidence is the actor entry in `input_hashes` of `results/evaluation_contract_v2/case39_common.json.gz` or `case118_common.json.gz`.

| Repository checkpoint | SHA-256 |
| --- | --- |
| `models/rl_sac_fixed_slack_v1_v31_p40_voltage_sensitivity/best_actor.pth` | `8861bcfb4eab1db6f362a94cabf1babab4580f0b89831cb485dde569582c5514` |
| `models/rl_full68_fixed_slack_v1_p40_raw_action_v4/best_actor.pth` | `18ba4d045478c88f70ee1a6b0da04d6fc616699cb3f11f29421e92874d7605ff` |
| `models/rl_ddpg_fixed_slack_v1_p40_raw_action_v9/best_actor.pth` | `41b937bfd64f206b2f0083b03f56210dddb2aa62dd8bc111fd299a183824ef9d` |
| `models/rl_td3_fixed_slack_v1_p40_unified_noise_v5/best_actor.pth` | `30ed9d0da66cd9ceebc6176c5489a853af9424c81cd0f4932a9f1f974faec669` |
| `models/rl39_routeb_main/best_model.pth` | `b5b1d8b8883d7c28fd24656c076b36de01a55b9cab41c66e283ab5b58f459e93` |
| `models/rl118_routeb_main/best_model.pth` | `b35b3210240c8126f857c0aa741625c0e54da371df25e975c2e0cb3a4fbe5bc1` |
| `models/revision_runs/ac_every1/best_actor.pth` | `3c8ed70f2e7e75273cf3a06563d6302eb893dc527693cef8f3cde11d9788e3e9` |
| `models/revision_runs/ac_every10/best_actor.pth` | `08646ffbd07ee90f38616dea8f1e6d0d342be62763f53e2ef6b5c146b9accd29` |
| `models/revision_runs/bonus_double/best_actor.pth` | `95b3aa06e8c9395fbae2b32b8f8a7d8d936aefc347bd8c28e4604ffa40855611` |
| `models/revision_runs/bonus_half/best_actor.pth` | `ea950274b66a830f90e886c68982ac42ec5018a22e351d6acafcf14ad62d9dc5` |
| `models/revision_runs/ddpg_seed1/best_actor.pth` | `8592dfc07b90f51c32784c87d2ddfd6adf27f6065b86ea08672b4e8229b8c8e7` |
| `models/revision_runs/ddpg_seed2/best_actor.pth` | `72c2a63bbccea601bde007222158b49207854c44f460bab75bd4d8396ad60382` |
| `models/revision_runs/dim16/best_actor.pth` | `e62dbb7bbdda758f186f829ce8f6c73a8fc5b6b0bffa30fb1c57b9e7732ef693` |
| `models/revision_runs/dim24/best_actor.pth` | `33099f7d5562462da0616df4519ed029f33bd94ff62b492422edd4ab32d765a1` |
| `models/revision_runs/dim40/best_actor.pth` | `5640e86ed57789dc53bc6f9bcb452dfec2cd8463883ff944b7bb4686f2d10a3c` |
| `models/revision_runs/full68_seed1/best_actor.pth` | `6bd234febadf46784a2f85b1e99f6e8fb824b70a74d1b3c2fca70a1a29d4fff9` |
| `models/revision_runs/full68_seed2/best_actor.pth` | `f48b8eada4e39eb68edc0acbcb3656c92ac3e3b19415b9a2dc554d43ed065576` |
| `models/revision_runs/keys32/best_actor.pth` | `c1fc36044d2b255080da7598c8f56062ea9d3b7c2b3fac63d2c49af36cdf9a6c` |
| `models/revision_runs/modes32/best_actor.pth` | `8a6b241ad927112e2a033ba8b9ce14c8edc1eacd5883c8a571b7d1ca9a9900af` |
| `models/revision_runs/modes32_seed1_v1/best_actor.pth` | `b839fd016bf5fe5df6261222465d00ad034359533aa6840f6bc4f4958d7d5f87` |
| `models/revision_runs/modes32_seed2_v1/best_actor.pth` | `0eacade1abaaf2598b7dc1ed7cfd6ea08340ad8e73feef3f41760c10db78c2fa` |
| `models/revision_runs/n1_double/best_actor.pth` | `d566076ee42798cb5838b36290fc17006a32fe42a70d85d07a61785625a7d588` |
| `models/revision_runs/n1_half/best_actor.pth` | `19168bc89527bbf30d5a50b00d27bbbf6e7b23cbbc42157bce6d38dda6eeb19a` |
| `models/revision_runs/radius20/best_actor.pth` | `ba386d74d566b04f998bba33def78c1cb8272b946a41b512c29ac5ee6317dac6` |
| `models/revision_runs/radius60/best_actor.pth` | `dfdbe9a0ed1b1bef55a62d26e0f576884c9e766d08729c8d2cd2528f1ef9a029` |
| `models/revision_runs/random32/best_actor.pth` | `942bde676e10d241adc0aefc090c3f8d78352ef2a833354f501aa5c766228bee` |
| `models/revision_runs/random32_seed1_v1/best_actor.pth` | `2d8b79f3845a3529c980675669cc6a349418a16311642cd4759555e69b9d4a4c` |
| `models/revision_runs/random32_seed2_v1/best_actor.pth` | `c4593128c15f044fd0fb38be97e059364ff639d1ec0cf971c7a99b922b6bdf16` |
| `models/revision_runs/sac_seed1/best_actor.pth` | `de6736932439193038d262d17668820d89363c117c99ae2ca960886c6e78d6d7` |
| `models/revision_runs/sac_seed2/best_actor.pth` | `d677c488969705001334f0e27a4f37f6180f781c5bee1ccb4a4d62c98f9e2990` |
| `models/revision_runs/split20_12/best_actor.pth` | `fc4535fb1e2e8097a5b02dc296872be7e414eaf755a80a354c229c68131cf403` |
| `models/revision_runs/split8_24/best_actor.pth` | `9c1203b35dfc45259086534dba711a7b3cd4ac324bae7462bc483e981238fd98` |
| `models/revision_runs/td3_seed1/best_actor.pth` | `230777e8db17987cc5e29feb1a5b48e78f59b1f555037cf48a42d1e5c5b1569f` |
| `models/revision_runs/td3_seed2/best_actor.pth` | `e0175f1a3f67a48f658a359e7aa7e8aa79d29d3df6a01f202a368089a9913e1e` |
| `models/revision_runs/tight_sac_094_108_seed42_v1/best_actor.pth` | `ffa20e079236eb9cbf5a5f1179db22d0ede51bba297fee8b285ed0b9b5df37cb` |
| `models/revision_runs/voltage_double/best_actor.pth` | `f6aa1160060cf4af5b8685976bdd823f1574a023f85d1cfc7e7e54bc4ce3beae` |
| `models/revision_runs/voltage_half/best_actor.pth` | `5beb98bb39e1f8442bb59fd92028ef7bbb7b556639c6ffe679095577119f8d8a` |

## Other Checkpoint Distribution Hashes

These seven files complete the 44-file inventory. The 300-bus predictor hash identifies the sanitized distribution container, not the original container. The other six files match original source bytes. These rows are not additional evaluation-linked policies in the preceding matrix.

| Repository checkpoint | Distribution SHA-256 |
| --- | --- |
| `models/gat39_results/gat39_best.pth` | `13fc97607468483f95d4dcb4d6f8906b7e8af1e6746686f47be625c1f9335a8f` |
| `models/gatv118_results/gatv118_predictor_best.pth` | `63402ef50ae34f4122461d1216655e943832bd61e458ef97b929202b57ec43ca` |
| `models/gat_case300_fixed_slack_v1/gatv10_predictor_best.pth` | `fe044910e4e3feb454e750a909e4b5b539ca9477e2875fc630e9926041a42523` |
| `models/rl_sac_fixed_slack_v1_v26_p40_v020/best_actor.pth` | `f5912acb9ade876f9550db2afb4b0e8a4d14d6fc3551b46f15e59c5e995f0c8e` |
| `models/rl_sac_fixed_slack_v1_v26_p40_v020/best_train_actor.pth` | `af1fe93ec9da8ae926b4696db6091157ef94aef3585e595b812a3455d95199b9` |
| `models/rl_sac_fixed_slack_v1_v26_p40_v020/final_actor.pth` | `011679ad6056697bbc516ab294fdee4e91f32bb76b93989014a778fd716a2d7a` |
| `models/rl_sac_fixed_slack_v1_v26_p40_v020/latest_actor.pth` | `57a840c06b0f527ce1bb956ffb709e40c96e6c5a72c9de413beefe62b371a87b` |

## Remaining Requirements

No remaining hard missing dataset, partition, network asset, or selected checkpoint was identified within the audited experiment scope. This is an asset-coverage conclusion, not a claim of complete executable reproducibility. Python/package installation, correct policy/runner contracts, relocated provenance handling, and fresh physical validation remain separate requirements. Original provenance assertions must not be silently disabled when sanitized files have different distribution hashes.

Label generation and optimization baselines requiring Gurobi still require an external installation and license; neither is distributed. Small-system partitions are deterministic definitions rather than separately frozen split JSON files, and their reconstructed test IDs were checked against saved evaluation rows. No full retraining validation or claim that all simulations were rerun is made here.
