"""Diagnose case39 Route-B alignment without running RL training."""
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJ = Path(r".")
sys.path.insert(0, str(PROJ))

from case39_route_b_utils import CASE39_VMIN as VMIN, CASE39_VMAX as VMAX, apply_case39_route_b_limits


def project_box_sum(x, lo, hi, target_sum, max_iter=100):
    x = np.asarray(x, dtype=np.float64).copy()
    lo = np.asarray(lo, dtype=np.float64)
    hi = np.asarray(hi, dtype=np.float64)
    y = np.clip(x, lo, hi)
    target_sum = float(np.clip(target_sum, float(lo.sum()), float(hi.sum())))
    for _ in range(max_iter):
        err = target_sum - float(y.sum())
        if abs(err) < 1e-7:
            break
        if err > 0:
            free = y < hi - 1e-9
            if not np.any(free):
                break
            y[free] += err / float(np.count_nonzero(free))
            y = np.minimum(y, hi)
        else:
            free = y > lo + 1e-9
            if not np.any(free):
                break
            y[free] += err / float(np.count_nonzero(free))
            y = np.maximum(y, lo)
    return np.clip(y, lo, hi)


def run_acpf(dispatch, demand_mw, gen_vm=None, ext_vm=None):
    import pandapower as pp
    from pandapower.networks import case39

    net = case39()
    apply_case39_route_b_limits(net)
    pmax = net.gen.max_p_mw.values.astype(float)
    net.gen.p_mw = np.clip(np.asarray(dispatch, dtype=float), 0.0, pmax)
    if gen_vm is not None:
        net.gen.vm_pu = np.asarray(gen_vm, dtype=float)[:len(net.gen)]
    if ext_vm is not None and len(net.ext_grid):
        net.ext_grid.vm_pu = float(np.asarray(ext_vm, dtype=float).reshape(-1)[0])
    scale = float(demand_mw) / float(net.load.p_mw.sum())
    net.load.p_mw *= scale
    net.load.q_mvar *= scale
    net.bus.min_vm_pu = VMIN
    net.bus.max_vm_pu = VMAX
    try:
        pp.runpp(net, calculate_voltage_angles=True, enforce_q_lims=True, numba=False, max_iteration=100)
        if not net.converged:
            return {"conv": False, "vv": 999, "vgap": 999.0, "vmin": 0.0, "vmax": 0.0, "ov": 999}
        vm = net.res_bus.vm_pu.values
        low = np.maximum(VMIN - vm, 0.0)
        high = np.maximum(vm - VMAX, 0.0)
        loading = net.res_line.loading_percent.values
        ov = int((loading > 100).sum())
        if len(net.res_trafo):
            ov += int((net.res_trafo.loading_percent.values > 100).sum())
        return {
            "conv": True,
            "vv": int((low > 0).sum() + (high > 0).sum()),
            "vgap": float(max(low.max(initial=0.0), high.max(initial=0.0))),
            "vmin": float(vm.min()),
            "vmax": float(vm.max()),
            "ov": ov,
        }
    except Exception as exc:
        return {"conv": False, "vv": 999, "vgap": 999.0, "vmin": 0.0, "vmax": 0.0, "ov": 999, "err": str(exc)[:80]}


def main():
    from pandapower.networks import case39
    from train_gat39_v1 import GATPredictor, ObsBuilder, get_features

    with open(PROJ / "case39_routeb_rescuable_dataset.json", "r", encoding="utf-8") as f:
        samples = json.load(f)["samples"]

    net = case39()
    apply_case39_route_b_limits(net)
    pmin = net.gen.min_p_mw.values.astype(float)
    pmax = net.gen.max_p_mw.values.astype(float)
    base_gen_vm = net.gen.vm_pu.values.astype(float)
    base_ext_vm = net.ext_grid.vm_pu.values.astype(float) if len(net.ext_grid) else np.array([1.0])

    gf = get_features()
    src, dst = [], []
    for i in range(gf["n_gen"]):
        for j in range(gf["n_gen"]):
            if i != j:
                src.append(i)
                dst.append(j)
    model = GATPredictor(n_units=gf["n_gen"], edge_index=torch.LongTensor([src, dst]))
    ckpt = torch.load(PROJ / "gat39_results" / "gat39_best.pth", map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    ob = ObsBuilder(gf)

    print("idx gw gat_l1 label-rescue-l1 base-vm-mae | label_base | gat_base | label_rescue_v | rescue_pv")
    for idx, s in enumerate(samples[:20]):
        demand = float(s["demand_mw"])
        label = np.asarray(s["dispatch"], dtype=float)
        rescue_p = np.asarray(s.get("rescue_dispatch", label), dtype=float)
        rescue_v = np.asarray(s.get("rescue_gen_vm_pu", base_gen_vm), dtype=float)
        rescue_ext = np.asarray(s.get("rescue_ext_grid_vm_pu", base_ext_vm), dtype=float)
        nf, gfa = ob.build(demand)
        with torch.no_grad():
            gat = model(torch.FloatTensor(nf).unsqueeze(0), torch.FloatTensor(gfa).unsqueeze(0)).numpy()[0]
        gat = project_box_sum(gat, pmin, pmax * 0.99, demand)

        def fmt(x):
            return f"conv={int(x['conv'])} vv={x['vv']} gap={x['vgap']:.4f} v=[{x['vmin']:.3f},{x['vmax']:.3f}] ov={x['ov']}"

        print(
            f"{idx:02d} {demand/1000.0:.1f} {np.abs(gat-label).sum():.1f} "
            f"{np.abs(label-rescue_p).sum():.3g} {np.abs(base_gen_vm-rescue_v).mean():.4f} | "
            f"{fmt(run_acpf(label, demand, base_gen_vm, base_ext_vm))} | "
            f"{fmt(run_acpf(gat, demand, base_gen_vm, base_ext_vm))} | "
            f"{fmt(run_acpf(label, demand, rescue_v, rescue_ext))} | "
            f"{fmt(run_acpf(rescue_p, demand, rescue_v, rescue_ext))}"
        )


if __name__ == "__main__":
    main()
