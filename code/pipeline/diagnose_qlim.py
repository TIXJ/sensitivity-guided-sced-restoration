"""
Diagnose the lookup contradiction: run each test-demand training label through
AC-PF BOTH with and without enforce_q_lims, to confirm the label-generation vs
evaluation AC-PF mismatch is what makes 'complete-SCED' labels fail at high load.
"""
import sys, json, numpy as np, copy
from pathlib import Path
PROJ = Path(r".")
sys.path.insert(0, str(PROJ))
import pandapower as pp
from pandapower.networks import case300
from train_rlv35_complete import split_samples_stratified

data = json.load(open(str(PROJ / 'case300_gt_v10_dataset.json')))
samples = [s for s in data['samples'] if s.get('is_complete_sced')]
train_ids, _, test_ids = split_samples_stratified(samples)

# lowest-cost training label per demand level
lookup = {}
for sid in train_ids:
    s = samples[sid]; dmw = int(s['demand_mw'])
    P = np.array(s['dispatch_68'], dtype=np.float64)
    cost = float(s.get('sced_cost_per_h', 1e9))
    if dmw not in lookup or cost < lookup[dmw][1]:
        lookup[dmw] = (P, cost)

net_base = case300()
PMAX = net_base.gen.max_p_mw.values.astype(np.float64)

def acpf(P, enforce_q):
    net = case300()
    net.gen.p_mw = np.clip(P, 0.0, PMAX * 0.99)
    tg = float(P.sum()); bl = net.load.p_mw.sum()
    ls = tg / bl
    net.load.p_mw = net.load.p_mw * ls; net.load.q_mvar = net.load.q_mvar * ls
    net.bus.min_vm_pu = 0.88; net.bus.max_vm_pu = 1.10
    try:
        pp.runpp(net, calculate_voltage_angles=True, enforce_q_lims=enforce_q, numba=False)
        if not net.converged: return False, -1, -1
        vv = int((net.res_bus.vm_pu < 0.88).sum())
        vmin = float(net.res_bus.vm_pu.min())
        return True, vv, vmin
    except Exception:
        return False, -1, -1

print(f"{'GW':>3} {'sid':>4} | {'no-Qlim (gen)':>22} | {'Qlim=True (eval)':>22}")
print("-"*60)
noq_pass = 0; q_pass = 0; total = 0
for sid in test_ids:
    dmw = int(samples[sid]['demand_mw']); gw = dmw // 1000
    if dmw not in lookup: continue
    total += 1
    P = lookup[dmw][0]
    c0, vv0, vm0 = acpf(P, False)   # generation-style
    c1, vv1, vm1 = acpf(P, True)    # eval-style
    ok0 = c0 and vv0 == 0; ok1 = c1 and vv1 == 0
    noq_pass += ok0; q_pass += ok1
    if gw >= 21:
        print(f"{gw:>3} {sid:>4} | conv={str(c0):>5} vv={vv0:>2} vmin={vm0:.3f} {'PASS' if ok0 else 'FAIL':>4} | "
              f"conv={str(c1):>5} vv={vv1:>2} vmin={vm1:.3f} {'PASS' if ok1 else 'FAIL':>4}")

print("-"*60)
print(f"Labels passing AC-PF WITHOUT enforce_q_lims (generation setting): {noq_pass}/{total}")
print(f"Labels passing AC-PF WITH    enforce_q_lims (evaluation setting): {q_pass}/{total}")
