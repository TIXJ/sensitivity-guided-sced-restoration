# Sensitivity-guided SCED restoration

Reproducibility materials for **Sensitivity-Guided Low-Dimensional Redispatch for Base-Case Alternating-Current Feasibility and Linear N-1 Contingency Screening**, manuscript IJEPES-D-26-03365.

Authors: Kaiting Lai, Yigong Xie, Wenchong Fang, Danli Xu, Yuxuan Liu, Yuanwei Yang, and Changtao Liao.

## Contents

- `code/pipeline/`: preserved prediction, training and evaluation scripts; root-level dependencies are in `code/pipeline/root/`.
- `code/analysis/`: revision experiment/analysis scripts and the portable saved-record entry point `public_reproduce.py`.
- `data/cases/`, `data/partitions/`, `data/matrices/`: modified test cases, stored candidate datasets, partition identifiers and derived numerical matrices.
- `models/`: graph predictors and stored correction policies. Both the earlier v26 policy and the v31 policy used for the revised original-policy evaluation are identified separately.
- `results/`: stored numerical records, final paired measurements and table-tabulation intermediates. Different experiment protocols are not pooled.
- `docs/REPRODUCE.md`: table/figure-to-script mapping, commands, scope and expected numerical checks.
- `MANIFEST.json`: SHA256 checksums of this sanitized distribution. Original-run hashes in the provenance records refer to the original author-held files, not relocated/sanitized copies.

## Environment and Three-Step Check

The measured environment used Python 3.10, pandapower 3.4, PyTorch 2.9, NumPy and SciPy. Gurobi-dependent label-generation scripts require a separately obtained Gurobi installation and license; the repository distributes neither. For saved-record checks, no optimizer or GPU is required.

```sh
python -m pip install -r requirements.txt
python code/analysis/public_reproduce.py verify
python code/analysis/public_reproduce.py tables --output generated_tables
```

The second command verifies distribution hashes, the 44/75 original-policy result, final paired runtimes and confidence intervals, matched accepted IDs, and the 882-pair AC diagnostic. The third exports every saved manuscript table snippet from stored tabulation intermediates. It reconstructs the saved tabulations, not an independent recomputation of every table from physical simulations.

Large result JSON files are losslessly compressed as `.json.gz`; the portable entry point reads either form. Standard `gzip` or Python's `gzip` module can expand them when inspecting the historical scripts. Checkpoint path metadata has been relocated without changing any tensor data block.

## Reproduction Scope

Full retraining from raw inputs has **not been validated in this repository**. Original scripts retain their research-layout assumptions; relocation of filesystem paths is not a validation of their physical execution. Saved protocols retain original-run provenance hashes, and those checks must not be silently disabled to make relocated code run. The portable entry point above checks stored results without invoking those historical protocol-bound runners.

The original policy accepts 44/75 unfiltered cases. Its 27/27 result and 0.788 s timing concern the screened, acceptance-matched subset. The final eight-step gated workflow is a different configuration: matched workflows accept 60/75 historical and 93/100 spatial cases. Historical average timing is 1.912 versus 1.725 s; the spatial paired saving interval contains zero. Base-case AC checks plus linear outage screening do not establish full AC contingency security.

## License and Citation

Author-owned code: MIT. Author-owned derived data, weights and results: CC BY 4.0. Third-party test-case/software rights remain unchanged; see `LICENSE` and `docs/DATA_DICTIONARY.md`. Cite the manuscript and this version using `CITATION.cff`. Contact: 1181413980@qq.com.
