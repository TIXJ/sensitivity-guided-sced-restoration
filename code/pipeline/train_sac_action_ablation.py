"""Fair active-action ablation using the frozen V26 training implementation.

``physics32`` runs the proposed 32-D PTDF/LODF basis. ``random32`` uses a
deterministic random balanced orthonormal basis of the same dimension and
aligns the two gradient feature blocks with that basis. ``full68`` uses 68
generator-wise active-power outputs and expresses the gradient features in the
same generator-wise coordinates.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
ALIGNED = ROOT / "paper_results" / "code" / "train_sac_n1_active_aligned.py"


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--active-encoding",
        choices=["physics32", "random32", "full68"],
        required=True,
    )
    known, forwarded = parser.parse_known_args()
    if "--output-dir" not in forwarded:
        default_dirs = {
            "physics32": "rl_action_ablation_physics32_v26",
            "random32": "rl_action_ablation_random32_fixedsplit_v26",
            "full68": "rl_action_ablation_full68_v26",
        }
        forwarded += ["--output-dir", default_dirs[known.active_encoding]]

    old_argv = sys.argv[:]
    sys.argv = [str(ALIGNED), *forwarded]
    try:
        spec = importlib.util.spec_from_file_location("_v26_action_ablation", ALIGNED)
        aligned = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(aligned)
    finally:
        sys.argv = old_argv

    if known.active_encoding == "random32":
        class Random32Env(aligned.CleanN1Env):
            def __init__(self, samples, sample_ids, dataset_meta):
                self._random_action_basis = None
                super().__init__(samples, sample_ids, dataset_meta)

            def _build_orthogonal_physics_action_basis(self):
                if self._random_action_basis is not None:
                    return self._random_action_basis
                rng = aligned.np.random.default_rng(20260721)
                raw = rng.standard_normal((self.N, 32))
                raw -= raw.mean(axis=0, keepdims=True)
                basis, _ = aligned.np.linalg.qr(raw, mode="reduced")
                for col in range(basis.shape[1]):
                    pivot = int(aligned.np.argmax(aligned.np.abs(basis[:, col])))
                    if basis[pivot, col] < 0.0:
                        basis[:, col] *= -1.0
                self._random_action_basis = basis
                return basis

        aligned.CleanN1Env = Random32Env
    elif known.active_encoding == "full68":
        class Full68Env(aligned.CleanN1Env):
            def __init__(self, samples, sample_ids, dataset_meta):
                aligned.core.SCEDEnvV21.__init__(
                    self,
                    samples,
                    sample_ids,
                    dataset_meta=dataset_meta,
                    action_encoding="full68",
                    action_basis_seed=20260721,
                    voltage_action_mode="incremental",
                    voltage_absolute_range_pu=aligned.ARGS.voltage_target_range_pu,
                    observation_mode="normalized_compact",
                    generator_upper_margin=0.99,
                    include_bus_voltage_feedback=True,
                    initial_acpf_on_reset=True,
                )
                self._voltage_group_indices = [
                    aligned.np.asarray([idx], dtype=int) for idx in range(self.N)
                ]
                # Full68 replaces only the active-power interface and therefore
                # bypasses CleanN1Env.__init__. Recreate the V31 voltage-context
                # cache used by CleanN1Env._get_obs().
                self._voltage_jacobian_memory = {}
                self._voltage_jacobian_dir = (
                    aligned.PROJECT
                    / "paper_results"
                    / ".ac_voltage_jacobian_cache"
                )
                self._voltage_jacobian_dir.mkdir(parents=True, exist_ok=True)
                self._generatorwise_observation_basis = None

            def _build_orthogonal_physics_action_basis(self):
                # The full interface has one actor output per generator. Its
                # gradient features must therefore use those same coordinates,
                # rather than the proposed 32-D basis. Centering removes the
                # uniform component that is eliminated by box-and-sum recovery.
                if self._generatorwise_observation_basis is None:
                    basis = aligned.np.eye(self.N, dtype=aligned.np.float64)
                    basis -= 1.0 / float(self.N)
                    self._generatorwise_observation_basis = basis
                return self._generatorwise_observation_basis

        aligned.CleanN1Env = Full68Env

    aligned.main()

    # Correct the inherited metadata labels without touching training results.
    output_dir = pathlib.Path(aligned.OUTPUT_DIR)
    for config_path in (output_dir / "config.json", output_dir / "training_results.json"):
        if not config_path.exists():
            continue
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        config = payload.get("config", payload)
        encoding_labels = {
            "physics32": "physics32orth",
            "random32": "random32_balanced_orthonormal",
            "full68": "full68_generatorwise_balanced",
        }
        config["action_encoding"] = encoding_labels[known.active_encoding]
        config["active_action_dim"] = 68 if known.active_encoding == "full68" else 32
        config["ablation_control"] = (
            "same_dimension_basis_structure"
            if known.active_encoding == "random32"
            else "active_action_interface_comparator"
        )
        config["observation_contract"] = (
            "same_base_observation_with_basis_aligned_gradients"
            if known.active_encoding == "random32"
            else "generatorwise_active_gradient_features_aligned_with_full68_action"
        )
        voltage_context_dim = int(config.get("voltage_action_dim", 0)) * int(
            config.get("voltage_context_features_per_control", 0)
        )
        config["base_observation_dim"] = (
            int(config["obs_dim"])
            - (2 * int(config["active_action_dim"]) + 3)
            - voltage_context_dim
        )
        config["action_aligned_observation_dim"] = 2 * int(config["active_action_dim"]) + 3
        config_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
