# Reproduce and Inspect the Reported Results

## Physical Experiments

Use `code/analysis/physical_reproduce.py` to compute policy actions, run AC/LODF checks, and invoke optimization from distributed inputs. [PHYSICAL_REPRODUCTION.md](PHYSICAL_REPRODUCTION.md) contains commands, checked outcomes, full-training recipes, and timing modes. [PHYSICAL_VALIDATION.json](PHYSICAL_VALIDATION.json) records the fresh physical checks separately from historical paper results.

## Saved-Record Entry Points

From the repository root:

```sh
python code/analysis/public_reproduce.py verify
python code/analysis/public_reproduce.py tables --output generated_tables
```

These commands validate stored-record numerical summaries and export all saved table snippets. They do not train a policy, call a power-flow solver, or establish a clean-machine physical rerun. A complete raw-input retraining has not been validated in this repository.

## Main Table/Figure Mapping

Use the LaTeX labels because pagination and table numbers can change. All table snippets, including configuration/literature tables, are exported by `public_reproduce.py tables`.

| Table/figure label | Recorded input | Original numerical/plot script |
| --- | --- | --- |
| `tab:baseline`, `tab:restoration_baseline_v26` | `results/original_screened/`, `results/reported_tables.json` | pipeline `eval_case300_deterministic_repair.py`, optimization evaluators and saved original pipeline audit |
| `tab:revision_full_methods`, `tab:revision_unfiltered` | `results/full_*.json` | `code/analysis/evaluate_revision.py`; the portable `verify` command checks the 44/75 SAC rows |
| `tab:revision_ablations`, `tab:action_compression` | action-control `*_full.json`, saved tabulations | `code/analysis/train_revision.py`, `evaluate_revision.py` |
| `tab:final_basis_seeds`, `fig:final_basis_validation` | `results/*seed*_full.json`, `results/*/training_results.json` | `code/analysis/run_basis_seed_completion.py`, `final_evidence_documents.py` |
| `tab:revision_seeds`, `tab:revision_seed_metrics`, `fig:revision_seed_validation` | stored seed-level records and trajectories | original seed collection/plot scripts in `code/analysis/` |
| `tab:revision_cadence` | `results/ac_every*_full.json` and saved tabulations | cadence controls via `train_revision.py` / `evaluate_revision.py` |
| `tab:review_followup_runtime`, `fig:final_paired_timing` | `results/dual_verified_once_v2/dual.json`, `results/spatial_dual_final_v1/results.json`, case-level CSV | `remeasure_verified_dual.py`, `remeasure_spatial_dual.py`, `analyze_verified_dual.py`, `analyze_spatial_dual.py`, `final_evidence_documents.py`; portable `verify` recalculates means and intervals |
| `tab:final_route_groups`, `tab:final_workflow_exchange` | historical final records | `analyze_deployment_accounting.py` |
| `tab:final_strong`, `tab:final_independent`, `tab:short_deployment` | respective stored summaries and tabulations | `compare_strong_optimizer.py`, `evaluate_independent_conditions.py`, deployment follow-up scripts |
| `tab:sequence_runtime` | recorded sequence summary and tabulations | `analyze_reviewer_followup.py --balanced-sequence` |
| `tab:revision_exchange` | stored exchange-cost records | `code/analysis/` exchange-cost analysis scripts |
| `tab:revision_paired_ac`, `fig:revision_ac_error` | `results/ac_contingencies.json`, `results/ac_audit_summary.json`, paired AC records | `ac_contingency_audit.py`, `audit_final_ac_anchor.py` and AC-analysis scripts |
| `tab:revision_stress`, `tab:final_initializer`, `tab:revision_fallback` | stored condition/initialization/fallback records | `evaluate_revision.py`, `compare_initialization_workflows.py`, joint-fallback scripts |
| `tab:case39`, `tab:case118`, small-system action and timing tables | `results/case39_*.json`, `results/case118_*.json` | small-case evaluation scripts in `code/analysis/` and `code/pipeline/` |

The physical wrapper creates and verifies the required relative layout in an independent runtime snapshot. Archived scripts also document earlier protocols and document-generation steps; they are not all standalone portable commands. Keep their historical provenance hashes intact. Use the supported physical entry points and their distribution-bound inventory rather than disabling old assertions. Timings on another machine are new measurements, not expected exact reproductions.

## Expected Saved-Record Results

- Screened 27 cases: all three restoration procedures accept 27/27; original complete times 0.788 s, 3.604 s and 3.075 s.
- Original unfiltered policy: 44/75.
- Final historical comparison: both accept the same 60/75; direct/gated means 1.912/1.725 s; 9.8% mean reduction; paired 95% interval [0.104, 0.282] s.
- Final spatial comparison: both accept the same 93/100; mean paired saving 0.040 s; interval [-0.003, 0.088] s includes zero.
- Selected explicit AC checks: 882 operating-point/outage pairs, comprising 536 passes, 46 nonconvergent solves and 300 converged voltage violations.

The portable check recomputes the unfiltered policy count, both final paired comparisons and AC summary from stored records. The screened timing values and the remaining exact table entries are preserved in their original audit/tabulation files. No failed case is removed to improve a reported mean.
