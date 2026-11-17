"""Compatibility entry point for the IEEE 39-bus GAT trainer.

The maintained implementation lives at the project root because the RL scripts
import its model classes by module name.  This wrapper keeps the experiment
command path under paper_results/code without duplicating training logic.
"""
from pathlib import Path
import importlib.util
import sys

PROJ = Path(r".")
ROOT_SCRIPT = PROJ / "train_gat39_v1.py"

if not ROOT_SCRIPT.exists():
    raise FileNotFoundError(f"Missing root trainer: {ROOT_SCRIPT}")

sys.path.insert(0, str(PROJ))
spec = importlib.util.spec_from_file_location("_root_train_gat39_v1", ROOT_SCRIPT)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)

if __name__ == "__main__":
    module.train()
