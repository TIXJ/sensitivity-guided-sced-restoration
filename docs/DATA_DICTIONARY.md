# Data Dictionary

## Cases and Matrices

`data/cases/case*_*.json` contains stored samples and dataset metadata for IEEE-derived 39-, 118- and 300-bus experiments. The `sample_id` field identifies an operating case. Integer train/validation/test identifiers index the matching dataset, not another file with a similar name. The 500-candidate upstream split and the 167-case screened split are separate datasets; test IDs must not be interchanged.

`data/partitions/` preserves split files. `data/matrices/case300_network.json` is the modified pandapower network snapshot; `security_matrices.npz` contains the stored numerical sensitivity/security arrays. `settings.json`, the original `manifest.json` and `README.txt` describe that preserved package. The original manifest hashes can differ from sanitized JSON files; use root `MANIFEST.json` for this distribution's integrity.

The modified cases derive from IEEE benchmark networks. Authors' modifications and computed quantities are CC BY 4.0 to the extent of authors' rights. Upstream data and pandapower/PYPOWER rights remain unchanged.

## Weights

`models/gat_case300_fixed_slack_v1/`, `gat39_results/` and `gatv118_results/` contain graph-attention weights. The earlier `rl_sac_fixed_slack_v1_v26_p40_v020` directory is not the revised v31 policy. The v31 best actor and system-specific small-case actors are separate entries. Loading PyTorch checkpoints requires trusted-file handling; these files are supplied by the authors and listed in `MANIFEST.json`.

## Results

Large JSON records use lossless gzip compression (`.json.gz`). Paths below identify the logical JSON name; the portable reader resolves either representation.

- `full_sac.json`: 75 original unfiltered policy evaluation rows; `passed` is the adopted verifier outcome, not full AC contingency security.
- `dual_verified_once_v2/dual.json`: historical final-configuration rows. `(repeat, sample_id, arm)` is a unique key. Arms `direct` and `selected` denote direct optimization and gated correction plus fallback.
- `spatial_dual_final_v1/results.json`: corresponding final spatial-profile measurements; all preparation/solver failures remain.
- `runtime_sec`: measured end-to-end time for that recorded workflow. Average repeats within each case before paired case resampling. Never pool the two populations.
- `scheduled_cost`: controllable-generation cost, excluding reference-source exchange. Exchange-priced costs use the stated hypothetical import/export prices.
- `final_paired_timing_cases.csv`: one case-level mean per population and sample ID, with direct time, gated time and direct-minus-gated saving.
- `final_paired_timing_audit.json`: seed 2026101805, 10000 bootstrap resamples, recorded paired intervals and original-source hashes.
- `ac_audit_summary.json`: selected operating-point/outage diagnostics, not a random population reliability estimate.
- `reported_tables.json`: the saved tabulation intermediates for all manuscript tables. They support exact table-snippet export, not a new independent physical evaluation.

Absolute machine paths and internal host references have been relocated or redacted. Numerical samples, metrics and physical settings are preserved. Original provenance hashes and sanitized-distribution hashes are explicitly distinguished.
